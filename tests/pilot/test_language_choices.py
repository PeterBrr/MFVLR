from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mfvlr.flt import FineGrainedLanguageTransformer
from mfvlr.language_encoder import LanguageEncoding
from mfvlr_research.pilot_language import (EOS_ID, PAD_ID, last_eos_pool,
    EosFineGrainedLanguageTransformer, nonpad_language_reconstruction_loss,
    NormalizedCrossModalContrastiveLoss)


def test_last_actual_eos_for_four_and_two_prompts_and_its_gradient():
    ids = torch.zeros(2, 308, dtype=torch.long)
    ids[0, [12, 90, 168, 244]] = EOS_ID
    ids[1, [10, 87]] = EOS_ID  # Two prompts, with padded L3/L4 slots
    hidden = torch.randn(2, 308, 512, requires_grad=True)
    pooled = last_eos_pool(hidden, ids)
    assert pooled.shape == (2, 1, 512)
    assert torch.equal(pooled[0, 0], hidden[0, 244])
    assert torch.equal(pooled[1, 0], hidden[1, 87])
    pooled.sum().backward()
    expected = torch.zeros_like(hidden)
    expected[0, 244] = expected[1, 87] = 1
    assert torch.equal(hidden.grad, expected)
    ids[0].zero_()
    with pytest.raises(ValueError, match='actual EOS'):
        last_eos_pool(hidden, ids)


def test_eos_subclass_changes_global_readout_for_encode_and_full_forward(monkeypatch):
    class IdentityEncoder(nn.Module):
        def forward(self, hidden, visual):
            return hidden

    # Keep real encode and full FLT.forward, replace expensive sublayer internals.
    model = EosFineGrainedLanguageTransformer.__new__(EosFineGrainedLanguageTransformer)
    nn.Module.__init__(model)
    model.vocabulary = nn.Embedding(49408, 512)
    model.encoder_position = nn.Parameter(torch.randn(1, 308, 512))
    model.encoder = IdentityEncoder()
    ids = torch.ones(1, 308, dtype=torch.long)
    ids[0, [9, 82]] = EOS_ID
    visual = torch.randn(1, 1, 512)
    encoded = model.encode(ids, visual)
    assert isinstance(encoded, LanguageEncoding)
    assert torch.equal(encoded.global_language, encoded.encoder_hidden[:, 82:83])
    monkeypatch.setattr(model, 'decode', lambda low, hidden, visual: hidden)
    output = model(ids, visual, return_logits=False)
    assert torch.equal(output.global_language, output.encoder_hidden[:, 82:83])
    assert not torch.equal(output.global_language, output.encoder_hidden[:, -1:])


def test_nonpad_lr_retains_sum_then_batch_mean_and_zero_pad_target_gradient():
    ids = torch.zeros(2, 308, dtype=torch.long)
    ids[0, :3] = torch.tensor([49406, 7, EOS_ID])
    ids[1, :4] = torch.tensor([49406, 8, 9, EOS_ID])
    logits = torch.zeros(2, 308, 49408, requires_grad=True)
    loss = nonpad_language_reconstruction_loss(logits, ids)
    # Seven supervised positions total, batch mean, not division by seven.
    assert loss.item() == pytest.approx(3.5 * torch.log(torch.tensor(49408.)).item(), rel=1e-6)
    loss.backward()
    assert logits.grad[ids == PAD_ID].count_nonzero() == 0
    assert logits.grad[0, 0].count_nonzero() > 0  # BOS retained
    assert logits.grad[0, 2].count_nonzero() > 0  # EOS retained
    assert logits.grad[0, 1, 7].item() == pytest.approx((1 / 49408 - 1) / 2)


def test_all_pad_lr_is_connected_zero():
    logits = torch.zeros(1, 308, 49408, requires_grad=True)
    value = nonpad_language_reconstruction_loss(logits, torch.zeros(1, 308, dtype=torch.long))
    assert value.item() == 0
    value.backward()
    assert logits.grad is not None and logits.grad.count_nonzero() == 0


def test_normalized_cmc_scale_invariance_temperature_and_unchanged_inputs():
    module = NormalizedCrossModalContrastiveLoss()
    visual = torch.randn(3, 1, 512, requires_grad=True)
    language = torch.randn(3, 1, 512, requires_grad=True)
    original_v, original_t = visual.detach().clone(), language.detach().clone()
    scores = module.scores(visual, language)
    expected = F.normalize(visual[:, 0], dim=-1) @ F.normalize(language[:, 0], dim=-1).T / module.temperature
    torch.testing.assert_close(scores, expected)
    torch.testing.assert_close(module.scores(visual * 7, language * 13), scores)
    assert torch.equal(visual.detach(), original_v) and torch.equal(language.detach(), original_t)
    grads = torch.autograd.grad(module(visual, language), [visual, language, module.log_temperature])
    assert all(torch.isfinite(g).all() and g.count_nonzero() > 0 for g in grads)
    assert torch.isfinite(module(torch.zeros_like(visual), torch.zeros_like(language)))


def test_new_language_choices_are_wired_into_manual_objectives(monkeypatch):
    from mfvlr_research.pilot_skip import ManualObjectives
    settings = dict(mask_background_weight=1., mask_foreground_weight=4.)
    weights = dict(fd=0., lr=1., cmc=1., fl=0., ar=0., kl=0.)
    objectives = ManualObjectives(settings, weights)
    assert isinstance(objectives.cmc, NormalizedCrossModalContrastiveLoss)
    ids = torch.zeros(2, 308, dtype=torch.long)
    ids[:, :2] = torch.tensor([49406, EOS_ID])
    logits = torch.zeros(2, 308, 49408)
    v, t = torch.randn(2, 1, 512), torch.randn(2, 1, 512)
    output = SimpleNamespace(image=torch.zeros(2, 3, 224, 224),
        predictions=SimpleNamespace(visual=SimpleNamespace(fused_global=v)),
        language=SimpleNamespace(logits=logits, global_language=t), stop_language_visual=False)
    targets = SimpleNamespace(labels=torch.zeros(2, dtype=torch.long),
        masks=torch.zeros(2, 224, 224, dtype=torch.long), sample_kinds=('real', 'real'), token_ids=ids)
    actual = objectives(output, targets)
    torch.testing.assert_close(actual.lr, nonpad_language_reconstruction_loss(logits, ids))
    torch.testing.assert_close(actual.cmc, objectives.cmc(v, t))
    torch.testing.assert_close(actual.total, actual.lr + actual.cmc)
