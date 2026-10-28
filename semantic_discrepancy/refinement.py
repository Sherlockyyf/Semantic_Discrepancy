"""Correlation Refinement (CR) and Noisy Feature Refinement (NFR).

Both refinements take accurate features from a *single-step* DDIM inversion of the reference
image (one U-Net pass on the clean latent at timestep tau = 1 with an empty prompt) and plug them
into every denoising step of SDEdit:

* CR  (Eq. 7): the self-attention maps A of the selected decoder layers are replaced by the
  inversion's maps, which removes the erroneous correlations between unrelated regions.
* NFR (Eq. 8): at decoder layer gamma, the skip connection h^{L-gamma} and the output of the residual
  block f^gamma are replaced by the inversion's (noise-free) features; the self-attention that
  follows then works on clean features (a softer constraint than CR).

Decoder ("up") layers are numbered 0..11 as in the paper and in the original LDM code
(``output_blocks``): layer ``i`` is ``unet.up_blocks[i // 3]``, resnet / attention ``i % 3``.
Only layers 3..11 of Stable Diffusion 1.x/2.x have self-attention.

Everything is implemented with attention processors and forward hooks, so the diffusers U-Net is
used unmodified.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch


@dataclass
class RefinementConfig:
    cr_layers: Sequence[int] = (3, 4, 5)     # decoder layers with refined self-attention maps
    nfr_layers: Sequence[int] = (5,)         # decoder layers with refined (noise-free) features
    nfr_min_timestep: int = 0                # NFR is applied while t > nfr_min_timestep
    inversion_timestep: int = 1              # tau: timestep of the single DDIM-inversion step

    @classmethod
    def from_gamma(cls, gamma=6, first_cr_layer=3, **kwargs):
        """The paper's parameterisation: CR on layers [first_cr_layer, gamma) and NFR on layer gamma - 1.

        With the default ``gamma = 6`` this gives CR on layers 3-5 and NFR on the 6th decoder layer
        (index 5), the setting used in our experiments.
        """
        return cls(cr_layers=tuple(range(first_cr_layer, gamma)), nfr_layers=(gamma - 1,), **kwargs)


def _expand_to(stored: torch.Tensor, batch: int) -> torch.Tensor:
    """Repeat inversion features (batch B) for a batch of k * B samples (e.g. classifier-free guidance)."""
    if stored.shape[0] == batch:
        return stored
    if batch % stored.shape[0] != 0:
        raise RuntimeError(f'Cannot broadcast stored features of batch {stored.shape[0]} to {batch}')
    return stored.repeat(batch // stored.shape[0], *([1] * (stored.ndim - 1)))


class FeatureBank:
    """Holds the inversion features and the current mode (``off`` / ``record`` / ``inject``)."""

    def __init__(self):
        self.mode = 'off'
        self.timestep: Optional[int] = None
        self.features: Dict[str, torch.Tensor] = {}

    def clear(self):
        self.features.clear()


class RefinedAttnProcessor:
    """Self-attention that records its attention map, or replaces it by the recorded one (CR)."""

    def __init__(self, bank: FeatureBank, key: str):
        self.bank = bank
        self.key = key

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 *args, **kwargs):
        residual = hidden_states
        if getattr(attn, 'spatial_norm', None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None
                                          else encoder_hidden_states.shape)
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if getattr(attn, 'group_norm', None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))

        if self.bank.mode == 'inject' and self.key in self.bank.features:
            attention_probs = _expand_to(self.bank.features[self.key], value.shape[0]).to(value.dtype)
        else:
            query = attn.head_to_batch_dim(attn.to_q(hidden_states))
            key = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            if self.bank.mode == 'record':
                self.bank.features[self.key] = attention_probs.detach()

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


@dataclass
class _Installed:
    processors: List = field(default_factory=list)   # (attention module, original processor)
    hooks: List = field(default_factory=list)        # torch hook handles


class SemanticDiscrepancyRefiner:
    """Installs CR / NFR on a diffusers ``UNet2DConditionModel``.

    Usage::

        refiner = SemanticDiscrepancyRefiner(unet, RefinementConfig())
        with refiner.recording():                       # single-step DDIM inversion
            unet(x0, tau, encoder_hidden_states=empty_prompt)
        for t in timesteps:                             # SDEdit sampling
            with refiner.injecting(t):
                noise_pred = unet(latents, t, ...)
        refiner.remove()
    """

    def __init__(self, unet, config: RefinementConfig = RefinementConfig()):
        self.unet = unet
        self.config = config
        self.bank = FeatureBank()
        self._installed = _Installed()
        self.block_size = len(unet.up_blocks[0].resnets)            # layers_per_block + 1
        self.num_decoder_layers = self.block_size * len(unet.up_blocks)
        self._skip_layout = self._build_skip_layout()
        self.install()

    # ------------------------------------------------------------------ layout
    def _build_skip_layout(self):
        """Index of every skip connection -> (down block, position in its output tuple); index 0 is conv_in."""
        layout = [None]
        for block in self.unet.down_blocks:
            n_outputs = len(block.resnets) + (1 if getattr(block, 'downsamplers', None) else 0)
            layout += [(block, p) for p in range(n_outputs)]
        return layout

    def decoder_layer(self, i):
        if not 0 <= i < self.num_decoder_layers:
            raise ValueError(f'Decoder layer must be in [0, {self.num_decoder_layers - 1}], got {i}')
        return self.unet.up_blocks[i // self.block_size], i % self.block_size

    def skip_index(self, i):
        """Skip connection consumed by decoder layer ``i`` (``L - i`` in the paper's notation)."""
        return len(self._skip_layout) - 1 - i

    # ------------------------------------------------------------ installation
    def install(self):
        self.remove()
        for i in self.config.cr_layers:
            block, j = self.decoder_layer(i)
            if not hasattr(block, 'attentions') or block.attentions is None:
                raise ValueError(f'Decoder layer {i} has no self-attention (CR needs layers >= {self.block_size})')
            for b, transformer_block in enumerate(block.attentions[j].transformer_blocks):
                attn = transformer_block.attn1
                self._installed.processors.append((attn, attn.processor))
                attn.set_processor(RefinedAttnProcessor(self.bank, f'cr{i}_{b}'))

        for i in self.config.nfr_layers:
            block, j = self.decoder_layer(i)
            s = self.skip_index(i)
            if s < 1:
                raise ValueError(f'NFR is not supported on the last decoder layer ({i})')
            down_block, position = self._skip_layout[s]
            self._installed.hooks.append(down_block.register_forward_hook(self._skip_hook(f'skip{i}', position)))
            self._installed.hooks.append(block.resnets[j].conv2.register_forward_hook(self._feature_hook(f'res{i}')))

    def remove(self):
        for attn, processor in self._installed.processors:
            attn.set_processor(processor)
        for handle in self._installed.hooks:
            handle.remove()
        self._installed = _Installed()

    def _nfr_active(self):
        t = self.bank.timestep
        return self.bank.mode == 'inject' and (t is None or t > self.config.nfr_min_timestep)

    def _skip_hook(self, key, position):
        def hook(module, inputs, output):
            hidden_states, res_samples = output
            if self.bank.mode == 'record':
                self.bank.features[key] = res_samples[position].detach()
            elif self._nfr_active() and key in self.bank.features:
                res_samples = list(res_samples)
                current = res_samples[position]
                res_samples[position] = _expand_to(self.bank.features[key], current.shape[0]).to(current.dtype)
                return hidden_states, tuple(res_samples)
            return output
        return hook

    def _feature_hook(self, key):
        def hook(module, inputs, output):
            if self.bank.mode == 'record':
                self.bank.features[key] = output.detach()
            elif self._nfr_active() and key in self.bank.features:
                return _expand_to(self.bank.features[key], output.shape[0]).to(output.dtype)
            return output
        return hook

    # ------------------------------------------------------------------- modes
    @contextmanager
    def _mode(self, mode, timestep=None):
        previous = (self.bank.mode, self.bank.timestep)
        self.bank.mode, self.bank.timestep = mode, timestep
        try:
            yield self
        finally:
            self.bank.mode, self.bank.timestep = previous

    def recording(self):
        """Store the features of the next U-Net call (the single-step DDIM inversion)."""
        self.bank.clear()
        return self._mode('record')

    def injecting(self, timestep=None):
        """Replace the features of the next U-Net call by the recorded ones."""
        t = int(timestep.flatten()[0]) if torch.is_tensor(timestep) else timestep
        return self._mode('inject', t)

    def clear(self):
        self.bank.clear()
