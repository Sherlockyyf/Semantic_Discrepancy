"""Image-to-image translation with SDEdit + CR/NFR (single image or a whole folder).

    # one image, with the SDEdit baseline for comparison
    python scripts/translate.py --input path/to/image.jpg --style realistic --strength 0.5 --compare

    # a dataset folder (sub-folders are mirrored in the output, finished images are skipped)
    python scripts/translate.py --input data/imagenet-r --output outputs/imagenet-r --style realistic --compare

    # custom prompt, custom checkpoint, no ControlNet, different refinement layers
    python scripts/translate.py --input photo.jpg --prompt "oil painting" --controlnet none \
        --base_model path/to/realisticVisionV20_v20.safetensors --cr_layers 3 4 --nfr_layers 5

For each input ``<name>`` the script writes ``<name>_ours.png``; with ``--compare`` also
``<name>_sdedit.png`` and ``<name>_compare.jpg`` (reference | SDEdit | ours).
"""
import argparse
import os
import sys

import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantic_discrepancy import RefinementConfig  # noqa: E402
from semantic_discrepancy.pipeline import (DEFAULT_BASE_MODEL, DEFAULT_CONTROLNET,  # noqa: E402
                                           SemanticDiscrepancyPipeline)
from semantic_discrepancy.utils import list_images, make_comparison  # noqa: E402


def build_prompts(args):
    preset = {}
    if args.style:
        with open(args.styles_file, 'r', encoding='utf-8') as f:
            presets = yaml.safe_load(f)
        if args.style not in presets:
            raise SystemExit(f'Unknown style "{args.style}". Available: {", ".join(presets)}')
        preset = presets[args.style]
    prompt = args.prompt if args.prompt is not None else preset.get('prompt', '')
    added = args.added_prompt if args.added_prompt is not None else preset.get('added_prompt', '')
    negative = args.negative_prompt if args.negative_prompt is not None else preset.get('negative_prompt', '')
    return ', '.join(p for p in (prompt, added) if p), negative


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, help='image file or folder (searched recursively)')
    parser.add_argument('--output', default='outputs')
    parser.add_argument('--style', default=None, help='prompt preset from configs/styles.yaml')
    parser.add_argument('--styles_file', default='configs/styles.yaml')
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--added_prompt', default=None)
    parser.add_argument('--negative_prompt', default=None)
    parser.add_argument('--strength', type=float, default=0.5, help='SDEdit noise strength t0 in [0, 1]')
    parser.add_argument('--steps', type=int, default=20, help='DDIM steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--size', type=int, default=512, help='shorter side; 0 keeps the original size')
    parser.add_argument('--keep_aspect', action='store_true', help='no square crop (crop to multiples of 64)')
    parser.add_argument('--base_model', default=DEFAULT_BASE_MODEL, help='diffusers repo/folder or .safetensors/.ckpt')
    parser.add_argument('--vae', default=None, help='optional VAE, e.g. stabilityai/sd-vae-ft-mse')
    parser.add_argument('--controlnet', default=DEFAULT_CONTROLNET, help='ControlNet repo, or "none"')
    parser.add_argument('--controlnet_scale', type=float, default=0.9)
    parser.add_argument('--canny', type=int, nargs=2, default=[50, 100], metavar=('LOW', 'HIGH'))
    parser.add_argument('--cr_layers', type=int, nargs='*', default=[3, 4, 5], help='decoder layers for CR')
    parser.add_argument('--nfr_layers', type=int, nargs='*', default=[5], help='decoder layers for NFR')
    parser.add_argument('--nfr_min_timestep', type=int, default=0)
    parser.add_argument('--compare', action='store_true', help='also run the SDEdit baseline')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    prompt, negative_prompt = build_prompts(args)
    print(f'prompt: "{prompt}"\nnegative prompt: "{negative_prompt}"')
    refinement = RefinementConfig(cr_layers=tuple(args.cr_layers), nfr_layers=tuple(args.nfr_layers),
                                  nfr_min_timestep=args.nfr_min_timestep)
    controlnet = None if args.controlnet.lower() == 'none' else args.controlnet
    pipe = SemanticDiscrepancyPipeline.from_pretrained(args.base_model, controlnet=controlnet, vae=args.vae,
                                                       refinement=refinement, device=args.device)

    root = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
    for path in tqdm(list_images(args.input), desc='Translating'):
        rel = os.path.splitext(os.path.relpath(path, root))[0]
        out_base = os.path.join(args.output, rel)
        if not args.overwrite and os.path.isfile(out_base + '_ours.png'):
            continue
        os.makedirs(os.path.dirname(out_base) or '.', exist_ok=True)
        out = pipe(path, prompt, negative_prompt, strength=args.strength, num_inference_steps=args.steps,
                   guidance_scale=args.guidance_scale, seed=args.seed, size=args.size,
                   center_crop=not args.keep_aspect, controlnet_conditioning_scale=args.controlnet_scale,
                   canny_thresholds=tuple(args.canny), return_baseline=args.compare)
        out['ours'][0].save(out_base + '_ours.png')
        if args.compare:
            out['sdedit'][0].save(out_base + '_sdedit.png')
            out['input'][0].save(out_base + '_input.png')
            make_comparison([out['input'][0], out['sdedit'][0], out['ours'][0]],
                            ['Reference', 'SDEdit', 'Ours']).save(out_base + '_compare.jpg', quality=92)
    print(f'Done. Results in {args.output}')


if __name__ == '__main__':
    main()
