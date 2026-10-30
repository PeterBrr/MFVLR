import copy
import math

import pytest
import torch
from torch import nn

from mfvlr.vision_decoder import VisionDecoder
from mfvlr.objectives import forgery_localization_loss
from mfvlr_research.pilot_variant import (ResizeConvStage, ResizeConvDecoder, language_visual_feature,
                                         weighted_mask_loss, variant_settings)


SETTINGS = {'variant': 'resize_conv_v1', 'resize_mode': 'nearest',
            'language_visual_gradients': 'stop_all', 'mask_background_weight': 1.,
            'mask_foreground_weight': 4., 'mask_loss_weight': 1.}


@pytest.mark.parametrize('mode', ['nearest', 'bilinear'])
def test_uniform_features_do_not_create_subpixel_phase_pattern(mode):
    stage = ResizeConvStage(1, 1, mode)
    with torch.no_grad():
        stage.projection.weight.fill_(1)
        stage.projection.bias.zero_()
        for conv in (stage.conv1, stage.conv2):
            conv.weight.fill_(1 / 9)
            conv.bias.zero_()
    result = stage(torch.ones(1, 1, 8, 8))
    torch.testing.assert_close(result[:, :, 2:-2, 2:-2], torch.ones(1, 1, 12, 12))


def test_decoder_shapes_and_localization_gradient():
    model = ResizeConvDecoder()
    assert not any(isinstance(m, nn.ConvTranspose2d) for m in model.modules())
    local = torch.rand(1, 1024, 14, 14, requires_grad=True)
    output = model(local)
    assert output.features.shape == (1, 64, 224, 224)
    assert output.reconstruction.shape == (1, 3, 224, 224)
    assert output.mask_logits.shape == (1, 2, 224, 224)
    masks = torch.zeros(1, 224, 224, dtype=torch.long)
    masks[:, 40:90, 40:90] = 1
    loss = weighted_mask_loss(output.mask_logits, masks, torch.tensor([1., 4.]))
    gradients = torch.autograd.grad(loss, [local, model.trunk.stages[0].projection.weight, model.mask_head.weight])
    assert all(torch.isfinite(g).all() and g.count_nonzero() > 0 for g in gradients)
    # A0 construction/state layout remains the transposed-convolution baseline.
    baseline = VisionDecoder(skip_policy='bottleneck_only')
    assert sum(isinstance(m, nn.ConvTranspose2d) for m in baseline.modules()) == 4
    with pytest.raises(RuntimeError):
        model.load_state_dict(baseline.state_dict(), strict=True)


def test_foreground_weight_and_loss_weight_have_distinct_effects():
    logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
    target = torch.tensor([[[1, 0], [0, 0]]])
    loss = weighted_mask_loss(logits, target, torch.tensor([1., 4.]))
    assert float(loss.detach()) == pytest.approx(math.log(2))
    grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    assert float(grad[0, 1, 0, 0].abs() / grad[0, 1, 0, 1].abs()) == pytest.approx(4.)
    twice = torch.autograd.grad(weighted_mask_loss(logits, target, torch.tensor([1., 4.]), 2.), logits)[0]
    torch.testing.assert_close(twice, 2 * grad)
    real_loss = weighted_mask_loss(logits, torch.zeros_like(target), torch.tensor([1., 4.]))
    assert float(real_loss.detach()) == pytest.approx(math.log(2))


def test_unit_weights_match_original_localization():
    logits = torch.randn(1, 2, 224, 224)
    masks = torch.randint(2, (1, 224, 224))
    torch.testing.assert_close(weighted_mask_loss(logits, masks, torch.ones(2)),
                               forgery_localization_loss(logits, masks))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_weighted_mask_deterministic_matches_cpu_reference(device):
    generator = torch.Generator().manual_seed(71)
    source = torch.randn(2, 2, 224, 224, generator=generator)
    targets = torch.randint(2, (2, 224, 224), generator=generator)
    weights = torch.tensor([1., 4.])
    cpu_logits = source.clone().requires_grad_()
    reference = torch.nn.functional.cross_entropy(cpu_logits, targets, weight=weights)
    expected_gradient = torch.autograd.grad(reference, cpu_logits)[0]
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        previous = None
        for _ in range(2):
            logits = source.to(device).clone().requires_grad_()
            actual = weighted_mask_loss(logits, targets.to(device), weights.to(device))
            gradient = torch.autograd.grad(actual, logits)[0]
            torch.testing.assert_close(actual.cpu(), reference.detach(), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(gradient.cpu(), expected_gradient, rtol=1e-5, atol=1e-7)
            if previous is not None:
                assert torch.equal(actual.detach(), previous[0])
                assert torch.equal(gradient, previous[1])
            previous = (actual.detach(), gradient)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_stop_gradient_preserves_values_and_language_parameter_updates():
    visual = torch.randn(2, 1, 512, requires_grad=True)
    layer = nn.Linear(512, 4)
    stop = language_visual_feature(visual, True)
    torch.testing.assert_close(stop, visual)
    language_loss = layer(stop).square().mean()
    g_visual, g_language = torch.autograd.grad(language_loss, [visual, layer.weight], allow_unused=True)
    assert g_visual is None
    assert g_language is not None and g_language.count_nonzero() > 0
    coupled = layer(language_visual_feature(visual, False)).square().mean()
    assert torch.autograd.grad(coupled, visual)[0].count_nonzero() > 0


def test_variant_settings_fail_closed():
    assert variant_settings({}) is None
    assert variant_settings({'model': SETTINGS}) == SETTINGS
    for key, value in [('variant', 'typo'), ('resize_mode', 'bicubic'),
                       ('language_visual_gradients', 'detach_decoder'), ('mask_foreground_weight', 0),
                       ('mask_background_weight', float('nan')), ('mask_loss_weight', -1)]:
        settings = copy.deepcopy(SETTINGS)
        settings[key] = value
        with pytest.raises(ValueError):
            variant_settings({'model': settings})
