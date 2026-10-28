"""Visualise the self-attention maps behind semantic discrepancy (Sec. IV of the paper).

The self-attention map of one decoder layer is projected to 3 channels with PCA (as in PnP) and
shown for the single-step DDIM inversion and for the SDEdit denoising steps, without and with our
refinements. Regions with the same colour attend to each other: in SDEdit, unrelated regions start
to share colours (erroneous correlations), while the inversion keeps the layout of the reference.

    python scripts/visualize_attention.py --input photo.jpg --style realistic --layer 8 --output outputs/attn
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantic_discrepancy import RefinementConfig  # noqa: E402
from semantic_discrepancy.pipeline import (DEFAULT_BASE_MODEL, DEFAULT_CONTROLNET,  # noqa: E402
                                           SemanticDiscrepancyPipeline)
from semantic_discrepancy.refinement import FeatureBank, RefinedAttnProcessor  # noqa: E402
from semantic_discrepancy.utils import canny_edges, load_image, make_comparison  # noqa: E402


class AttentionProbe:
    """Attention processor that records the self-attention map of one decoder layer at every U-Net call.

    It wraps the processor already installed on that layer, so CR keeps working when the probed layer
    is one of the refined layers (the recorded map is then the injected inversion map).
    """

    def __init__(self, unet, layer, block_size=3):
        block = unet.up_blocks[layer // block_size]
        self.attn = block.attentions[layer % block_size].transformer_blocks[0].attn1
        self.original = self.attn.processor
        self.maps = []
        self.attn.set_processor(self)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 *args, **kwargs):
        orig = self.original
        refined = isinstance(orig, RefinedAttnProcessor)
        if refined and orig.bank.mode == 'inject' and orig.key in orig.bank.features:
            self.maps.append(orig.bank.features[orig.key].float().cpu())
            return orig(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        bank = FeatureBank()
        bank.mode = 'record'
        out = RefinedAttnProcessor(bank, 'probe')(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        self.maps.append(bank.features['probe'].float().cpu())
        if refined and orig.bank.mode == 'record':  # keep the refiner's own recording working
            orig.bank.features[orig.key] = bank.features['probe']
        return out

    def remove(self):
        self.attn.set_processor(self.original)


def pca_image(attn_map, heads, size):
    """attention probs [(B*heads), N, N] -> RGB PCA visualisation of the last sample."""
    maps = attn_map[-heads:]                                  # conditional branch of the last sample
    n = maps.shape[-1]
    x = maps.permute(1, 0, 2).reshape(n, -1)                  # 'h n m -> n (h m)'
    x = x - x.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=3, center=False)
    proj = (x @ v[:, :3]).numpy()
    proj = (proj - proj.min(0)) / (proj.max(0) - proj.min(0) + 1e-8)
    h = w = int(round(n ** 0.5))
    h, w = (size[1] * h // max(size), size[0] * w // max(size)) if h * w != n else (h, w)
    img = Image.fromarray((proj.reshape(h, w, 3) * 255).astype(np.uint8))
    return img.resize(size, Image.NEAREST)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='outputs/attention')
    parser.add_argument('--style', default='realistic')
    parser.add_argument('--styles_file', default='configs/styles.yaml')
    parser.add_argument('--layer', type=int, default=8, help='decoder layer to visualise (3..11)')
    parser.add_argument('--show_steps', type=int, nargs='+', default=[0, 5, 9], help='denoising step indices')
    parser.add_argument('--strength', type=float, default=0.5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--base_model', default=DEFAULT_BASE_MODEL)
    parser.add_argument('--controlnet', default=DEFAULT_CONTROLNET)
    parser.add_argument('--cr_layers', type=int, nargs='*', default=[3, 4, 5])
    parser.add_argument('--nfr_layers', type=int, nargs='*', default=[5])
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    with open(args.styles_file, 'r', encoding='utf-8') as f:
        preset = yaml.safe_load(f)[args.style]
    prompt = ', '.join(p for p in (preset['prompt'], preset.get('added_prompt', '')) if p)
    controlnet = None if args.controlnet.lower() == 'none' else args.controlnet
    pipe = SemanticDiscrepancyPipeline.from_pretrained(
        args.base_model, controlnet=controlnet, device=args.device,
        refinement=RefinementConfig(cr_layers=tuple(args.cr_layers), nfr_layers=tuple(args.nfr_layers)))
    heads = pipe.unet.up_blocks[args.layer // 3].attentions[args.layer % 3].transformer_blocks[0].attn1.heads

    image = load_image(args.input)
    x0 = pipe.encode_images([image], generator=torch.Generator('cpu').manual_seed(args.seed))
    embeds = torch.cat([pipe.encode_text([preset.get('negative_prompt', '')]), pipe.encode_text([prompt])])
    control = None
    if pipe.controlnet is not None:
        edges = np.asarray(canny_edges(image), dtype=np.float32)[None] / 255
        control = torch.from_numpy(edges).permute(0, 3, 1, 2).to(pipe.device, pipe.dtype).repeat(2, 1, 1, 1)
    kwargs = dict(prompt_embeds=embeds, strength=args.strength, num_inference_steps=args.steps, guidance_scale=7.5,
                  seed=args.seed, control=control)

    panels, labels = [image], ['Reference']
    probe = AttentionProbe(pipe.unet, args.layer)
    pipe.record_inversion_features(x0)                       # single-step DDIM inversion
    panels.append(pca_image(probe.maps[-1], heads, image.size))
    labels.append('DDIM inv. (t=1)')
    for refine, name in ((False, 'SDEdit'), (True, 'Ours')):
        probe.maps.clear()
        latents = pipe.sdedit(x0, refine=refine, **kwargs)
        for s in args.show_steps:
            if s < len(probe.maps):
                panels.append(pca_image(probe.maps[s], heads, image.size))
                labels.append(f'{name} step {s}')
        panels.append(pipe.decode_latents(latents)[0])
        labels.append(f'{name} result')
    probe.remove()

    os.makedirs(args.output, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.output, f'{name}_layer{args.layer}.jpg')
    make_comparison(panels, labels, height=256).save(out_path, quality=92)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
