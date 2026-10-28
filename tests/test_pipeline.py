"""End-to-end test of the pipeline with tiny random components (no download, CPU)."""
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import AutoencoderKL, ControlNetModel, DDIMScheduler
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel

from semantic_discrepancy import RefinementConfig
from semantic_discrepancy.pipeline import SemanticDiscrepancyPipeline
from test_refinement import tiny_unet


class DummyTokenizer:
    model_max_length = 77

    def __call__(self, texts, padding=None, max_length=77, truncation=True, return_tensors='pt'):
        ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        for i, text in enumerate(texts):
            tokens = [sum(map(ord, w)) % 997 + 2 for w in text.split()][:max_length]
            if tokens:
                ids[i, :len(tokens)] = torch.tensor(tokens)
        return SimpleNamespace(input_ids=ids)


def tiny_pipeline(controlnet=True):
    torch.manual_seed(0)
    unet = tiny_unet()
    vae = AutoencoderKL(down_block_types=('DownEncoderBlock2D',) * 4, up_block_types=('UpDecoderBlock2D',) * 4,
                        block_out_channels=(8, 8, 8, 8), latent_channels=4, norm_num_groups=4, sample_size=64)
    text_encoder = CLIPTextModel(CLIPTextConfig(vocab_size=1000, hidden_size=32, intermediate_size=37,
                                                num_hidden_layers=2, num_attention_heads=4, projection_dim=32,
                                                bos_token_id=0, eos_token_id=2, pad_token_id=1))
    scheduler = DDIMScheduler(beta_schedule='scaled_linear', beta_start=0.00085, beta_end=0.012, clip_sample=False,
                              set_alpha_to_one=False, steps_offset=1)
    cn = ControlNetModel.from_unet(unet) if controlnet else None
    return SemanticDiscrepancyPipeline(vae, text_encoder, DummyTokenizer(), unet, scheduler, cn,
                                       RefinementConfig(), device='cpu', dtype=torch.float32)


def test_translate_with_baseline():
    pipe = tiny_pipeline()
    image = Image.fromarray((np.random.RandomState(0).rand(80, 96, 3) * 255).astype(np.uint8))
    out = pipe(image, 'realistic photo', 'ugly', strength=0.6, num_inference_steps=5, size=64, return_baseline=True)
    assert out['ours'][0].size == (64, 64) and out['sdedit'][0].size == (64, 64)
    assert not np.array_equal(np.asarray(out['ours'][0]), np.asarray(out['sdedit'][0]))
    assert not pipe.refiner.bank.features  # cleared after use
    # same seed -> same result
    again = pipe(image, 'realistic photo', 'ugly', strength=0.6, num_inference_steps=5, size=64)
    assert np.array_equal(np.asarray(again['ours'][0]), np.asarray(out['ours'][0]))


def test_without_controlnet_and_strength_zero():
    pipe = tiny_pipeline(controlnet=False)
    image = Image.fromarray((np.random.RandomState(1).rand(64, 64, 3) * 255).astype(np.uint8))
    out = pipe(image, 'cartoon', strength=0.0, num_inference_steps=4, size=64, guidance_scale=1.0)
    assert out['ours'][0].size == (64, 64)
