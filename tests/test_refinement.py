"""Unit tests of CR / NFR on a tiny randomly initialised U-Net with the Stable Diffusion layout (CPU, seconds)."""
import pytest
import torch
from diffusers import UNet2DConditionModel

from semantic_discrepancy import RefinementConfig, SemanticDiscrepancyRefiner


def tiny_unet():
    torch.manual_seed(0)
    return UNet2DConditionModel(
        sample_size=16, in_channels=4, out_channels=4, layers_per_block=2, block_out_channels=(32, 32, 64, 64),
        down_block_types=('CrossAttnDownBlock2D',) * 3 + ('DownBlock2D',),
        up_block_types=('UpBlock2D',) + ('CrossAttnUpBlock2D',) * 3,
        cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8).eval()


@pytest.fixture(scope='module')
def unet():
    return tiny_unet()


def inputs(seed, batch=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 4, 16, 16, generator=g), torch.randn(batch, 77, 32, generator=g)


def run(unet, x, t, emb):
    with torch.no_grad():
        return unet(x, torch.tensor([t] * x.shape[0]), encoder_hidden_states=emb).sample


def test_layout(unet):
    refiner = SemanticDiscrepancyRefiner(unet, RefinementConfig())
    try:
        assert refiner.num_decoder_layers == 12
        assert refiner.skip_index(5) == 6 and refiner.skip_index(6) == 5
        block, j = refiner.decoder_layer(5)
        assert block is unet.up_blocks[1] and j == 2
        assert len(refiner._skip_layout) == 12
    finally:
        refiner.remove()


def test_inactive_refiner_does_not_change_the_unet(unet):
    x, emb = inputs(1)
    reference = run(unet, x, 500, emb)
    refiner = SemanticDiscrepancyRefiner(unet, RefinementConfig(cr_layers=(3, 4, 5, 8), nfr_layers=(5, 7)))
    try:
        assert torch.allclose(run(unet, x, 500, emb), reference, atol=1e-4)
    finally:
        refiner.remove()
    assert torch.allclose(run(unet, x, 500, emb), reference, atol=0)  # processors restored


def test_injecting_the_same_features_is_identity(unet):
    x, emb = inputs(2)
    refiner = SemanticDiscrepancyRefiner(unet, RefinementConfig(cr_layers=(3, 4, 5), nfr_layers=(5,)))
    try:
        reference = run(unet, x, 1, emb)
        with refiner.recording():
            run(unet, x, 1, emb)
        assert set(refiner.bank.features) == {'cr3_0', 'cr4_0', 'cr5_0', 'skip5', 'res5'}
        with refiner.injecting(1):
            assert torch.allclose(run(unet, x, 1, emb), reference, atol=1e-4)
    finally:
        refiner.remove()


@pytest.mark.parametrize('config', [RefinementConfig(cr_layers=(3, 4, 5), nfr_layers=()),
                                    RefinementConfig(cr_layers=(), nfr_layers=(5,)),
                                    RefinementConfig.from_gamma(6)])
def test_injection_changes_the_prediction(unet, config):
    x_ref, emb_ref = inputs(3)
    x, emb = inputs(4)
    refiner = SemanticDiscrepancyRefiner(unet, config)
    try:
        plain = run(unet, x, 600, emb)
        with refiner.recording():
            run(unet, x_ref, 1, emb_ref)
        with refiner.injecting(600):
            refined = run(unet, x, 600, emb)
        assert not torch.allclose(refined, plain, atol=1e-3)
    finally:
        refiner.remove()


def test_features_are_broadcast_to_the_guidance_batch(unet):
    x_ref, emb_ref = inputs(5)
    x, emb = inputs(6)
    refiner = SemanticDiscrepancyRefiner(unet, RefinementConfig.from_gamma(6))
    try:
        with refiner.recording():
            run(unet, x_ref, 1, emb_ref)
        with refiner.injecting(700):
            single = run(unet, x, 700, emb)
            double = run(unet, torch.cat([x, x]), 700, torch.cat([emb, emb]))
        assert torch.allclose(double[:1], single, atol=1e-4) and torch.allclose(double[1:], single, atol=1e-4)
    finally:
        refiner.remove()


def test_nfr_min_timestep(unet):
    x_ref, emb_ref = inputs(7)
    x, emb = inputs(8)
    outputs = {}
    for name, cfg in {'cr_only': RefinementConfig(cr_layers=(3, 4), nfr_layers=()),
                      'gated': RefinementConfig(cr_layers=(3, 4), nfr_layers=(5,), nfr_min_timestep=500)}.items():
        refiner = SemanticDiscrepancyRefiner(unet, cfg)
        try:
            with refiner.recording():
                run(unet, x_ref, 1, emb_ref)
            with refiner.injecting(400):  # t <= nfr_min_timestep: NFR inactive
                outputs[name] = run(unet, x, 400, emb)
        finally:
            refiner.remove()
    assert torch.allclose(outputs['cr_only'], outputs['gated'], atol=1e-5)


def test_invalid_layers(unet):
    with pytest.raises(ValueError):
        SemanticDiscrepancyRefiner(unet, RefinementConfig(cr_layers=(1,), nfr_layers=()))   # no attention
    with pytest.raises(ValueError):
        SemanticDiscrepancyRefiner(unet, RefinementConfig(cr_layers=(), nfr_layers=(12,)))
