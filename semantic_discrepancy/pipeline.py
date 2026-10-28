"""SDEdit image-to-image translation with Stable Diffusion, plus our training-free refinements.

    pipe = SemanticDiscrepancyPipeline.from_pretrained()          # SD 1.5 + ControlNet (Canny)
    out = pipe(image, prompt='realistic photo', strength=0.5, return_baseline=True)
    out['sdedit'][0].save('sdedit.png'); out['ours'][0].save('ours.png')

Algorithm 1 of the paper:
  1. single-step DDIM inversion of the reference latent x0 (t = tau, empty prompt) -> store the
     self-attention maps (CR) and the residual / skip features (NFR) of the selected decoder layers;
  2. SDEdit: x_T = sqrt(a_T) x0 + sqrt(1 - a_T) eps, then DDIM sampling where the stored features
     replace the erroneous ones at every step.
"""
import os
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from .refinement import RefinementConfig, SemanticDiscrepancyRefiner
from .utils import canny_edges, load_image

DEFAULT_BASE_MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
DEFAULT_CONTROLNET = 'lllyasviel/sd-controlnet-canny'


class SemanticDiscrepancyPipeline:
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, controlnet=None,
                 refinement: Optional[RefinementConfig] = None, device='cuda', dtype=torch.float16):
        self.vae, self.text_encoder, self.tokenizer = vae, text_encoder, tokenizer
        self.unet, self.scheduler, self.controlnet = unet, scheduler, controlnet
        self.device, self.dtype = torch.device(device), dtype
        for module in (vae, text_encoder, unet, controlnet):
            if module is not None:
                module.to(self.device)
                if getattr(module, 'dtype', dtype) != dtype:
                    module.to(dtype=dtype)
                module.eval().requires_grad_(False)
        self.refiner = SemanticDiscrepancyRefiner(unet, refinement or RefinementConfig())

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, base_model=DEFAULT_BASE_MODEL, controlnet: Optional[str] = DEFAULT_CONTROLNET,
                        vae: Optional[str] = None, refinement: Optional[RefinementConfig] = None,
                        device: Optional[str] = None, dtype: Optional[torch.dtype] = None):
        """``base_model``: a diffusers repo id / folder, or a single ``.ckpt`` / ``.safetensors`` file
        (e.g. a Civitai checkpoint such as Realistic Vision). ``controlnet=None`` disables ControlNet."""
        from diffusers import AutoencoderKL, ControlNetModel, DDIMScheduler, StableDiffusionPipeline

        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = dtype or (torch.float16 if str(device).startswith('cuda') else torch.float32)
        kwargs = dict(torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
        if os.path.isfile(base_model):
            sd = StableDiffusionPipeline.from_single_file(base_model, **kwargs)
        else:
            sd = StableDiffusionPipeline.from_pretrained(base_model, **kwargs)
        scheduler = DDIMScheduler.from_config(sd.scheduler.config)
        vae_model = AutoencoderKL.from_pretrained(vae, torch_dtype=dtype) if vae else sd.vae
        cn = ControlNetModel.from_pretrained(controlnet, torch_dtype=dtype) if controlnet else None
        return cls(vae_model, sd.text_encoder, sd.tokenizer, sd.unet, scheduler, cn, refinement, device, dtype)

    def set_refinement(self, config: RefinementConfig):
        self.refiner.remove()
        self.refiner = SemanticDiscrepancyRefiner(self.unet, config)

    # --------------------------------------------------------------- encoders
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        ids = self.tokenizer(texts, padding='max_length', max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors='pt').input_ids
        return self.text_encoder(ids.to(self.device))[0]

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], generator=None) -> torch.Tensor:
        x = np.stack([np.asarray(im, dtype=np.float32) / 127.5 - 1 for im in images])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device, self.dtype)
        latents = self.vae.encode(x).latent_dist.sample(generator=generator)
        return latents * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> List[Image.Image]:
        x = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        x = ((x.float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
        return [Image.fromarray(a) for a in x.permute(0, 2, 3, 1).cpu().numpy()]

    # ------------------------------------------------------------ main steps
    @torch.no_grad()
    def record_inversion_features(self, x0: torch.Tensor):
        """Single-step DDIM inversion (x0 -> x1): one U-Net pass at t = tau with an empty prompt."""
        empty = self.encode_text([''] * x0.shape[0])
        t = torch.full((x0.shape[0],), self.refiner.config.inversion_timestep, device=self.device, dtype=torch.long)
        with self.refiner.recording():
            self.unet(x0, t, encoder_hidden_states=empty)

    @torch.no_grad()
    def sdedit(self, x0, prompt_embeds, strength, num_inference_steps, guidance_scale, seed=0, control=None,
               controlnet_conditioning_scale=0.9, refine=False):
        """SDEdit sampling. With ``refine=True`` the recorded inversion features are injected (CR + NFR)."""
        batch = x0.shape[0]
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        init_steps = min(int(num_inference_steps * strength), num_inference_steps)
        timesteps = self.scheduler.timesteps[num_inference_steps - init_steps:]
        if len(timesteps) == 0:
            return x0

        generator = torch.Generator('cpu').manual_seed(seed)
        noise = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(self.device, x0.dtype)
        latents = self.scheduler.add_noise(x0, noise, timesteps[:1].repeat(batch))

        do_cfg = guidance_scale > 1
        for t in timesteps:
            latent_in = torch.cat([latents] * 2) if do_cfg else latents
            latent_in = self.scheduler.scale_model_input(latent_in, t)
            down_res, mid_res = None, None
            if control is not None:
                down_res, mid_res = self.controlnet(latent_in, t, encoder_hidden_states=prompt_embeds,
                                                    controlnet_cond=control,
                                                    conditioning_scale=controlnet_conditioning_scale,
                                                    return_dict=False)
            if refine:
                with self.refiner.injecting(t):
                    noise_pred = self.unet(latent_in, t, encoder_hidden_states=prompt_embeds,
                                           down_block_additional_residuals=down_res,
                                           mid_block_additional_residual=mid_res, return_dict=False)[0]
            else:
                noise_pred = self.unet(latent_in, t, encoder_hidden_states=prompt_embeds,
                                       down_block_additional_residuals=down_res,
                                       mid_block_additional_residual=mid_res, return_dict=False)[0]
            if do_cfg:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return latents

    @torch.no_grad()
    def __call__(self, image: Union[str, Image.Image, Sequence[Image.Image]], prompt: str, negative_prompt: str = '',
                 strength: float = 0.5, num_inference_steps: int = 20, guidance_scale: float = 7.5, seed: int = 0,
                 size: int = 512, center_crop: bool = True, controlnet_conditioning_scale: float = 0.9,
                 canny_thresholds=(50, 100), refine: bool = True, return_baseline: bool = False):
        """Translate one image (or a list of equally sized images).

        Returns a dict with ``'ours'`` (and ``'sdedit'`` when ``return_baseline``) -> list of PIL images.
        Both branches share the reference latent and the SDEdit noise, so they differ only by CR / NFR.
        """
        images = [image] if isinstance(image, (str, Image.Image)) else list(image)
        images = [load_image(im, size, center_crop) for im in images]
        batch = len(images)

        x0 = self.encode_images(images, generator=torch.Generator('cpu').manual_seed(seed))
        text = self.encode_text([prompt] * batch)
        prompt_embeds = torch.cat([self.encode_text([negative_prompt] * batch), text]) if guidance_scale > 1 else text

        control = None
        if self.controlnet is not None:
            edges = np.stack([np.asarray(canny_edges(im, *canny_thresholds), dtype=np.float32) / 255 for im in images])
            control = torch.from_numpy(edges).permute(0, 3, 1, 2).to(self.device, self.dtype)
            if guidance_scale > 1:
                control = torch.cat([control] * 2)

        common = dict(prompt_embeds=prompt_embeds, strength=strength, num_inference_steps=num_inference_steps,
                      guidance_scale=guidance_scale, seed=seed, control=control,
                      controlnet_conditioning_scale=controlnet_conditioning_scale)
        out = {'input': images}
        if return_baseline or not refine:
            out['sdedit'] = self.decode_latents(self.sdedit(x0, refine=False, **common))
        if refine:
            self.record_inversion_features(x0)
            out['ours'] = self.decode_latents(self.sdedit(x0, refine=True, **common))
            self.refiner.clear()
        return out
