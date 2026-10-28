"""Gradio demo: SDEdit vs. SDEdit + our refinements (CR & NFR).

    python app.py                                   # SD 1.5 + ControlNet (Canny), http://127.0.0.1:7860
    python app.py --base_model path/to/realisticVisionV20_v20.safetensors --share
"""
import argparse

import gradio as gr
import torch
import yaml

from semantic_discrepancy import RefinementConfig
from semantic_discrepancy.pipeline import DEFAULT_BASE_MODEL, DEFAULT_CONTROLNET, SemanticDiscrepancyPipeline


def build_demo(args):
    with open(args.styles_file, 'r', encoding='utf-8') as f:
        presets = yaml.safe_load(f)
    controlnet = None if args.controlnet.lower() == 'none' else args.controlnet
    pipe = SemanticDiscrepancyPipeline.from_pretrained(args.base_model, controlnet=controlnet, vae=args.vae)

    def fill_preset(style):
        p = presets[style]
        return ', '.join(x for x in (p['prompt'], p.get('added_prompt', '')) if x), p.get('negative_prompt', '')

    def run(image, prompt, negative_prompt, strength, steps, guidance, seed, cr_layers, nfr_layers, cn_scale):
        if image is None:
            raise gr.Error('Please upload a reference image.')
        pipe.set_refinement(RefinementConfig(cr_layers=tuple(int(i) for i in cr_layers),
                                             nfr_layers=tuple(int(i) for i in nfr_layers)))
        out = pipe(image, prompt, negative_prompt, strength=strength, num_inference_steps=int(steps),
                   guidance_scale=guidance, seed=int(seed), controlnet_conditioning_scale=cn_scale,
                   return_baseline=True)
        return out['sdedit'][0], out['ours'][0]

    first = next(iter(presets))
    with gr.Blocks(title='Semantic Discrepancy') as demo:
        gr.Markdown('# Dissecting and Mitigating Semantic Discrepancy in Stable Diffusion for I2I Translation\n'
                    'Training-free: single-step DDIM inversion features fix the erroneous correlations of SDEdit.')
        with gr.Row():
            with gr.Column():
                image = gr.Image(type='pil', label='Reference image')
                style = gr.Dropdown(list(presets), value=first, label='Style preset')
                prompt = gr.Textbox(value=fill_preset(first)[0], label='Prompt')
                negative = gr.Textbox(value=fill_preset(first)[1], label='Negative prompt')
                strength = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label='Noise strength (SDEdit t0)')
                with gr.Accordion('Advanced', open=False):
                    steps = gr.Slider(10, 50, value=20, step=1, label='DDIM steps')
                    guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label='Guidance scale')
                    seed = gr.Number(value=0, precision=0, label='Seed')
                    cr = gr.CheckboxGroup([str(i) for i in range(3, 12)], value=['3', '4', '5'],
                                          label='CR layers (decoder self-attention)')
                    nfr = gr.CheckboxGroup([str(i) for i in range(1, 11)], value=['5'], label='NFR layers')
                    cn_scale = gr.Slider(0.0, 2.0, value=0.9, step=0.05, label='ControlNet scale',
                                         visible=pipe.controlnet is not None)
                button = gr.Button('Translate', variant='primary')
            with gr.Column():
                sdedit = gr.Image(type='pil', label='SDEdit')
                ours = gr.Image(type='pil', label='Ours (CR + NFR)')
        style.change(fill_preset, inputs=style, outputs=[prompt, negative])
        button.click(run, inputs=[image, prompt, negative, strength, steps, guidance, seed, cr, nfr, cn_scale],
                     outputs=[sdedit, ours])
    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=DEFAULT_BASE_MODEL)
    parser.add_argument('--controlnet', default=DEFAULT_CONTROLNET)
    parser.add_argument('--vae', default=None)
    parser.add_argument('--styles_file', default='configs/styles.yaml')
    parser.add_argument('--server_name', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    build_demo(args).queue().launch(server_name=args.server_name, server_port=args.port, share=args.share)
