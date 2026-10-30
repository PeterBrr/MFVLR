import copy

import pytest
import torch

from mfvlr.integration import SuppliedTargets
from mfvlr.objectives import forgery_detection_loss, appearance_reconstruction_loss
from mfvlr_research.pilot_skip import (SkipResizeDecoder, validate_loss_weights, weighted_objective_total)
from mfvlr_research.pilot_variant import (visual_model, training_model, variant_settings,
                                         language_enabled, weighted_mask_loss)


def config(detach=True):
    return {'model': {'variant': 'skip_resize_conv_v2', 'resize_mode': 'nearest',
            'language_visual_gradients': 'stop_all',
            'residual_gradient': 'detach_reconstruction' if detach else 'coupled',
            'skip_connections': 'concat',
            'mask_background_weight': 1., 'mask_foreground_weight': 4., 'mask_loss_weight': 1.},
            'loss_weights': dict(fd=1., lr=0., cmc=0., fl=.25, ar=1., kl=0.)}


def test_all_four_skips_carry_localization_gradients_and_affect_output():
    decoder = SkipResizeDecoder('nearest')
    local = torch.randn(1, 1024, 14, 14, requires_grad=True)
    skips = [torch.randn(1, c, s, s, requires_grad=True) for c, s in
             ((64, 224), (128, 112), (256, 56), (512, 28))]
    out = decoder(local, skips=skips)
    assert out.reconstruction.shape == (1, 3, 224, 224)
    assert out.mask_logits.shape == (1, 2, 224, 224)
    target = torch.zeros(1, 224, 224, dtype=torch.long)
    target[:, 50:100, 60:120] = 1
    loss = weighted_mask_loss(out.mask_logits, target, torch.tensor([1., 4.]))
    gradients = torch.autograd.grad(loss, [local, *skips])
    assert all(torch.isfinite(g).all() and g.count_nonzero() > 0 for g in gradients)
    with torch.no_grad():
        changed = decoder(local, skips=[torch.zeros_like(s) for s in skips])
        assert not torch.equal(changed.mask_logits, out.mask_logits)
    with pytest.raises(ValueError, match='four encoder skips'):
        decoder(local, skips=skips[:-1])
    bad = [*skips]
    bad[0] = torch.zeros(1, 63, 224, 224)
    with pytest.raises(ValueError, match='Skip must match'):
        decoder(local, skips=bad)


@pytest.mark.parametrize('detach', [False, True])
@pytest.mark.parametrize('activation', ['identity', 'sigmoid'])
def test_detach_stops_fd_at_decoder_but_preserves_ar_fl_and_encoder(detach, activation):
    cfg = config(detach)
    cfg['model']['reconstruction_activation'] = activation
    model = visual_model(cfg).eval()
    images = torch.rand(1, 3, 224, 224, generator=torch.Generator().manual_seed(4))
    output = model(images)
    params = [model.decoder.appearance_head.weight, model.decoder.trunk.stages[0].projection.weight,
              model.mve.shared_encoder.cnn.stages[0][0].weight]
    fd = forgery_detection_loss(output.detection_logits, torch.ones(1, dtype=torch.long))
    gradients = torch.autograd.grad(fd, params, allow_unused=True, retain_graph=True)
    if detach:
        assert gradients[0] is None and gradients[1] is None
    else:
        assert all(g is not None and g.count_nonzero() > 0 for g in gradients[:2])
    assert gradients[2] is not None and gradients[2].count_nonzero() > 0
    ar = appearance_reconstruction_loss(images, output.decoded.reconstruction)
    assert all(g.count_nonzero() > 0 for g in torch.autograd.grad(ar, params, retain_graph=True))
    target = torch.zeros(1, 224, 224, dtype=torch.long)
    target[:, 60:120, 60:120] = 1
    fl = weighted_mask_loss(output.decoded.mask_logits, target, torch.tensor([1., 4.]))
    fl_grads = torch.autograd.grad(fl, [model.decoder.mask_head.weight, *params], allow_unused=True)
    assert fl_grads[1] is None  # mask supervision never passes through the AD head
    assert all(fl_grads[i] is not None and fl_grads[i].count_nonzero() > 0 for i in (0, 2, 3))


def test_detach_preserves_forward_values_exactly():
    model = visual_model(config(False)).eval()
    images = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        coupled = model(images)
        model.detach_reconstruction = True
        detached = model(images)
    assert torch.equal(coupled.decoded.reconstruction, detached.decoded.reconstruction)
    assert torch.equal(coupled.detection_logits, detached.detection_logits)
    assert torch.equal(coupled.decoded.mask_logits, detached.decoded.mask_logits)


def test_manual_weights_scale_gradients_and_skip_disabled_objectives():
    weights = dict(fd=2., lr=0., cmc=0., fl=.25, ar=1., kl=0.)
    values = {name: torch.tensor(float('nan') if weight == 0 else 3., requires_grad=True)
              for name, weight in weights.items()}
    total = weighted_objective_total(values, weights)
    assert float(total.detach()) == pytest.approx(9.75)
    total.backward()
    for name, weight in weights.items():
        assert values[name].grad is None if weight == 0 else values[name].grad.item() == pytest.approx(weight)


@pytest.mark.parametrize('activation', ['identity', 'sigmoid'])
def test_visual_only_training_omits_flt_and_matches_deployment_weights(activation):
    cfg = config()
    cfg['model']['reconstruction_activation'] = activation
    assert not language_enabled(cfg)
    model = training_model(cfg).eval()
    assert model.flt is None
    images = torch.rand(1, 3, 224, 224)
    tokens = torch.zeros(1, 308, dtype=torch.long)
    target = torch.zeros(1, 224, 224, dtype=torch.long)
    output = model.with_language(images, tokens, sample_ids=('real',))
    assert output.language is None and output.predicted_language is None
    loss = model.compute_objectives(output, SuppliedTargets(('real',), tokens,
        torch.zeros(1, dtype=torch.long), target, ('real',)))
    assert loss.lr.item() == loss.cmc.item() == loss.kl.item() == 0
    torch.testing.assert_close(loss.total, loss.fd + loss.ar + .25 * loss.fl)
    deployment = visual_model(cfg).eval()
    weights = {key: value for key, value in model.state_dict().items() if key.startswith(('mve.', 'decoder.'))}
    weights.update({key.removeprefix('heads.'): value for key, value in model.state_dict().items()
                    if key.startswith('heads.detector.')})
    deployment.load_state_dict(weights, strict=True)
    with torch.no_grad():
        actual = deployment(images)
    assert torch.equal(actual.decoded.reconstruction, output.predictions.decoded.reconstruction)
    assert torch.equal(actual.detection_logits, output.predictions.detection_logits)


@pytest.mark.parametrize('skips', ['concat', 'none'])
def test_sigmoid_bounds_rgb_preserves_mask_and_backpropagates_ar(skips):
    from mfvlr_research.pilot_variant import ResizeConvDecoder
    decoder_class = SkipResizeDecoder if skips == 'concat' else ResizeConvDecoder
    decoder = decoder_class('nearest')
    local = torch.randn(1, 1024, 14, 14)
    options = {} if skips == 'none' else {'skips': [torch.randn(1, c, s, s) for c, s in
        ((64, 224), (128, 112), (256, 56), (512, 28))]}
    with torch.no_grad():
        linear = decoder(local, **options)
    decoder.reconstruction_activation = 'sigmoid'
    bounded = decoder(local, **options)
    torch.testing.assert_close(bounded.reconstruction, linear.reconstruction.sigmoid(), rtol=0, atol=0)
    assert torch.equal(bounded.mask_logits, linear.mask_logits)
    assert bounded.reconstruction.min() >= 0 and bounded.reconstruction.max() <= 1
    ar = appearance_reconstruction_loss(torch.zeros_like(bounded.reconstruction), bounded.reconstruction)
    grads = torch.autograd.grad(ar, [decoder.appearance_head.weight, decoder.trunk.stages[0].projection.weight])
    assert all(torch.isfinite(g).all() and g.count_nonzero() > 0 for g in grads)


def test_invalid_reconstruction_activation_is_rejected():
    cfg = config()
    cfg['model']['reconstruction_activation'] = 'clamp'
    with pytest.raises(ValueError, match='reconstruction_activation'):
        variant_settings(cfg)


@pytest.mark.parametrize('bad', [-1., float('nan'), True])
def test_invalid_manual_weights_rejected(bad):
    cfg = config()
    cfg['loss_weights']['ar'] = bad
    with pytest.raises(ValueError):
        variant_settings(cfg)


def test_empty_or_incomplete_objectives_and_unknown_gradient_policy_rejected():
    with pytest.raises(ValueError):
        validate_loss_weights(dict(fd=1.))
    with pytest.raises(ValueError):
        validate_loss_weights({name: 0. for name in config()['loss_weights']})
    cfg = copy.deepcopy(config())
    cfg['model']['residual_gradient'] = 'detach_everything'
    with pytest.raises(ValueError):
        variant_settings(cfg)


def test_no_skip_control_matches_legacy_visual_values():
    cfg = config(False)
    cfg['model']['skip_connections'] = 'none'
    model = visual_model(cfg).eval()
    legacy_cfg = copy.deepcopy(cfg)
    legacy_cfg['model'].pop('skip_connections')
    legacy_cfg['model'].pop('residual_gradient')
    legacy_cfg['model']['variant'] = 'resize_conv_v1'
    legacy = visual_model(legacy_cfg).eval()
    for name, value in legacy.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    with torch.no_grad():
        image = torch.rand(1, 3, 224, 224)
        old, new = legacy(image), model(image)
    assert torch.equal(old.decoded.reconstruction, new.decoded.reconstruction)
    assert torch.equal(old.decoded.mask_logits, new.decoded.mask_logits)
    assert torch.equal(old.detection_logits, new.detection_logits)


@pytest.mark.parametrize('objective', ['cmc', 'kl'])
def test_encoder_only_language_objectives_do_not_execute_ld(monkeypatch, objective):
    from types import SimpleNamespace
    from torch import nn
    import mfvlr_research.pilot_language

    class EncoderOnlyStub(nn.Module):
        # Routing test double: real CMC/KL and real visual graph remain in use.
        def __init__(self, seed):
            super().__init__()
            self.projection = nn.Linear(512, 512)
            self.calls = 0

        def encode(self, tokens, visual):
            self.calls += 1
            return SimpleNamespace(global_language=self.projection(visual))

        def forward(self, *args, **kwargs):
            raise AssertionError('LD/vocabulary must not run while LR is disabled')

    monkeypatch.setattr(mfvlr_research.pilot_language, 'EosFineGrainedLanguageTransformer', EncoderOnlyStub)
    cfg = config()
    cfg['loss_weights'] = {name: float(name == objective) for name in cfg['loss_weights']}
    model = training_model(cfg)
    images = torch.rand(2, 3, 224, 224)
    tokens = torch.zeros(2, 308, dtype=torch.long)
    output = model.with_language(images, tokens, sample_ids=('a', 'b'))
    losses = model.compute_objectives(output, SuppliedTargets(('a', 'b'), tokens,
        torch.zeros(2, dtype=torch.long), torch.zeros(2, 224, 224, dtype=torch.long), ('real', 'real')))
    assert model.flt.calls == 1 and losses.lr.item() == 0
    grads = torch.autograd.grad(losses.total, [model.flt.projection.weight,
        model.mve.shared_encoder.cnn.stages[0][0].weight], allow_unused=True)
    assert grads[0] is not None and grads[0].count_nonzero() > 0
    assert grads[1] is None  # stop_all applies to both CMC and KL visual paths
