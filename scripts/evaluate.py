"""CLIP-I / CLIP-T / SSIM-S between reference images and translations produced by translate.py.

    python scripts/evaluate.py --results outputs/coco --methods ours sdedit --style_prefix "cartoon style of "

Reference images are the ``<name>_input.png`` files written by ``translate.py --compare``; use
``--references`` to point to the original dataset folder instead (matched by relative path / name).
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantic_discrepancy.metrics import FaithfulnessMetrics  # noqa: E402
from semantic_discrepancy.utils import list_images, load_image  # noqa: E402


def find_reference(result_path, method, args):
    base = result_path[:-len(f'_{method}.png')]
    if args.references is None:
        return base + '_input.png'
    rel = os.path.relpath(base, args.results)
    for ext in ('.jpg', '.JPEG', '.jpeg', '.png'):
        candidate = os.path.join(args.references, rel + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', required=True, help='output folder of translate.py')
    parser.add_argument('--references', default=None, help='folder with the original images (optional)')
    parser.add_argument('--methods', nargs='+', default=['ours', 'sdedit'])
    parser.add_argument('--style_prefix', default='', help='prepended to the BLIP caption for CLIP-T')
    parser.add_argument('--size', type=int, default=512, help='preprocessing of --references images')
    parser.add_argument('--no_clip_t', action='store_true', help='skip BLIP captioning / CLIP-T')
    parser.add_argument('--output', default=None, help='json file with the per-image and mean scores')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    metrics = FaithfulnessMetrics(device=args.device, use_clip_t=not args.no_clip_t)
    summary, per_image = {}, {}
    for method in args.methods:
        files = [p for p in list_images(args.results) if p.endswith(f'_{method}.png')]
        scores = []
        for path in tqdm(files, desc=method):
            ref_path = find_reference(path, method, args)
            if ref_path is None or not os.path.isfile(ref_path):
                continue
            reference = Image.open(ref_path) if args.references is None else load_image(ref_path, args.size)
            s = metrics(reference, Image.open(path), args.style_prefix)
            scores.append(s)
            per_image.setdefault(method, {})[os.path.relpath(path, args.results)] = s
        if not scores:
            print(f'[{method}] no result images found')
            continue
        summary[method] = {k: float(np.mean([s[k] for s in scores])) for k in scores[0] if k != 'caption'}
        summary[method]['n'] = len(scores)
        print(f'[{method}] ' + ', '.join(f'{k}={v:.4f}' for k, v in summary[method].items() if k != 'n') +
              f' (n={len(scores)})')
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'per_image': per_image}, f, indent=2)


if __name__ == '__main__':
    main()
