import copy
from fractions import Fraction

import pytest
import torch
from torch import nn

from mfvlr.language_encoder import LanguageAttention, initialize_attention
from mfvlr_research.decoder_a11 import NormalizedDecoderBlock
from mfvlr_research.a11_evidence import gradient_check
from tools.review_phase2d2_n3_local import explicit_attention, explicit_gradients
from tools.n3_gradient_tools import selftest
from .oracles import norm_forward, norm_vjp, functional_block

FP64 = {"atol": 1e-10, "rtol": 1e-8}


@pytest.mark.parametrize("fixture", ["constant", "near_constant", "ordinary"])
def test_a1_t02_ln_forward_and_analytic_vjp(fixture, evidence):
    gen = torch.Generator().manual_seed(6951)
    noise = torch.randn(2, 3, 512, generator=gen).double()
    x = {"constant": torch.full_like(noise, .25), "near_constant": .25 + 1e-4 * noise,
         "ordinary": noise}[fixture].requires_grad_()
    layer = nn.LayerNorm(512, eps=1e-5, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.linspace(.5, 1.5, 512, dtype=torch.float64))
        layer.bias.copy_(torch.linspace(-.1, .1, 512, dtype=torch.float64))
    c = torch.randn(x.shape, generator=torch.Generator().manual_seed(319)).double()
    actual = layer(x)
    expected = norm_forward(x.detach(), layer.weight.detach(), layer.bias.detach())
    torch.testing.assert_close(actual, expected, **FP64)
    gradients = torch.autograd.grad(actual, (x, layer.weight, layer.bias), c)
    with torch.no_grad(): oracle = norm_vjp(x, layer.weight, c)
    for g, ref in zip(gradients, oracle): torch.testing.assert_close(g, ref, **FP64)
    evidence("T02_" + fixture, {"status": "PASS", "forward_max_error": (actual-expected).abs().max().item(),
             "vjp_max_errors": [(g-r).abs().max().item() for g, r in zip(gradients, oracle)], "tolerance": FP64})


@pytest.mark.parametrize("variant", ["A0", "P3", "Q3"])
def test_a1_t03_real_arithmetic_order(accepted_flt, variant):
    base = accepted_flt.decoder.blocks[0]
    block = copy.deepcopy(base) if variant == "A0" else NormalizedDecoderBlock.from_reference(base, variant)
    block.double()
    x = (torch.randn(1, 4, 512, generator=torch.Generator().manual_seed(41)) * .2).double()
    memory = (torch.randn(1, 5, 512, generator=torch.Generator().manual_seed(42)) * .3).double()
    visual = (torch.randn(1, 1, 512, generator=torch.Generator().manual_seed(43)) * .1).double()
    before = memory.clone()
    with torch.no_grad():
        actual = block(x, memory, visual)
        weights = dict(block.named_parameters())
        expected = functional_block(x, memory, visual, weights, variant)
        torch.testing.assert_close(actual, expected, **FP64)
        if variant != "A0":
            wrong = functional_block(x, memory, visual, weights, "Q3" if variant == "P3" else "P3")
            assert not torch.allclose(actual, wrong, **FP64), "fixture must distinguish ordering"
    assert torch.equal(memory, before)
    # Independent composed VJP exercises both ordinary attention roles and each
    # added norm, plus state/memory/visual connectivity. Singleton Q/K structural
    # zeros are checked separately rather than treating unused oracle maps as zero.
    x.requires_grad_(); memory.requires_grad_(); visual.requires_grad_()
    names = [n for n in weights if ".q.weight" in n or ".k.weight" in n or n.startswith("ln")]
    names = [n for n in names if not n.startswith("vim.")]
    targets = [weights[n] for n in names] + [x, memory, visual]
    cot = torch.randn(x.shape, generator=torch.Generator().manual_seed(44)).double()
    actual_grad = torch.autograd.grad(block(x, memory, visual), targets, cot)
    reference_grad = torch.autograd.grad(functional_block(x, memory, visual, weights, variant), targets, cot)
    for got, ref in zip(actual_grad, reference_grad):
        torch.testing.assert_close(got, ref, **FP64)
        assert got.isfinite().all() and got.norm() > 0


@pytest.mark.parametrize("variant", ["P3", "Q3"])
def test_a1_t03_noncommuting_sentinels(accepted_flt, variant):
    block = NormalizedDecoderBlock.from_reference(accepted_flt.decoder.blocks[0], variant).double()
    events = []
    class Self(nn.Module):
        def forward(self, x, *, causal):
            assert causal is True; events.append("S"); return x.roll(1, -1) * .25 + .1
    class Memory(nn.Module):
        def forward(self, x, m): events.append("C"); return .125*x.square() + m[:, :x.shape[1]]
    class FF(nn.Module):
        def forward(self, x): events.append("F"); return .0625*x.square() + .03
    class Delta(nn.Module):
        def forward(self, x, v): events.append("V"); return v.expand_as(x)
    block.self_attention, block.memory_attention, block.ff, block.vim = Self(), Memory(), FF(), Delta()
    for i in (1, 2, 3):
        getattr(block, f"ln{i}").register_forward_hook(lambda m, a, y, i=i: events.append(f"N{i}"))
    x = torch.linspace(-2, 3, 1024, dtype=torch.float64).reshape(1, 2, 512)
    m = x.cos(); v = x[:, :1].sin()
    def n(t): return norm_forward(t, torch.ones(512, dtype=torch.float64), torch.zeros(512, dtype=torch.float64))
    u = n(x) if variant == "P3" else x
    a = x + u.roll(1, -1)*.25 + .1
    if variant == "Q3": a = n(a)
    u = n(a) if variant == "P3" else a
    b = a + .125*u.square() + m
    if variant == "Q3": b = n(b)
    u = n(b) if variant == "P3" else b
    c = b + .0625*u.square() + .03
    if variant == "Q3": c = n(c)
    torch.testing.assert_close(block(x, m, v), c+v, **FP64)
    assert events == (["N1", "S", "N2", "C", "N3", "F", "V"] if variant == "P3"
                      else ["S", "N1", "C", "N2", "F", "N3", "V"])


def attention_fixture():
    with torch.random.fork_rng(devices=[]): model = LanguageAttention()
    initialize_attention(model, torch.Generator().manual_seed(0))
    model.double()
    x = (torch.randn(1, 308, 512, generator=torch.Generator().manual_seed(247))*.2).double().requires_grad_()
    c = torch.randn(1, 308, 512, generator=torch.Generator().manual_seed(319)).double()
    return model, x, c


def test_a1_t06_ordinary_manual_qk_and_key_bias(evidence):
    model, x, c = attention_fixture()
    names = ("q.weight", "k.weight", "q.bias", "k.bias")
    pp = dict(model.named_parameters())
    actual = torch.autograd.grad(model(x), tuple(pp[n] for n in names)+(x,), c)
    with torch.no_grad(): expected = explicit_gradients(x.detach(), {n:p.detach() for n,p in pp.items()}, c)
    for name, g in zip((*names, "input"), actual): torch.testing.assert_close(g, expected[name], **FP64)
    assert actual[0].norm() > 0 and actual[1].norm() > 0
    probabilities = (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6))
    ds = [sum(u*probabilities[i]*(int(i == j)-probabilities[j])
              for i, u in enumerate((2, -3, 5))) for j in range(3)]
    assert sum(ds) == 0
    evidence("T06_manual", {"status":"PASS", "Q_weight_l2": actual[0].norm().item(),
             "K_weight_l2": actual[1].norm().item(), "key_bias_max_floating_residual": actual[3].abs().max().item(),
             "exact_rational_key_bias": "0", "tolerance": FP64})


@pytest.mark.parametrize("name,seed", [("q.weight",1003), ("k.weight",1004)])
def test_a1_t06_ordinary_directional_derivatives(name, seed, evidence):
    model, x, c = attention_fixture(); pp = dict(model.named_parameters())
    g = torch.autograd.grad(model(x), pp[name], c)[0]
    weights = {n:p.detach() for n,p in pp.items()}
    d = torch.randn(weights[name].shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    d /= d.norm(); eps = 1e-6
    with torch.no_grad():
        plus = explicit_attention(x.detach(), {**weights, name:weights[name]+eps*d})[0]
        minus = explicit_attention(x.detach(), {**weights, name:weights[name]-eps*d})[0]
        numerical = ((plus*c).sum()-(minus*c).sum())/(2*eps)
    analytic = (g*d).sum()
    torch.testing.assert_close(numerical, analytic, atol=1e-5, rtol=1e-3)
    evidence("T06_FD_"+name, {"status":"PASS", "numerical":numerical.item(), "analytic":analytic.item(),
                             "epsilon":eps, "atol":1e-5, "rtol":1e-3})


def test_a1_t09_negative_gradient_controls(tmp_path):
    assert selftest(tmp_path)["injected_error_detection"] == "PASS"
    expected = torch.zeros(262144+9); expected[1] = 1e-8; expected[-1] = 1
    gradient_check(expected.clone(), expected)
    for index in (0, 1, 262144, 262152):
        corrupted = expected.clone(); corrupted[index] += 1e-3
        with pytest.raises(AssertionError): gradient_check(corrupted, expected)
    for invalid in (None, torch.full_like(expected, float("nan")), torch.ones(1)):
        with pytest.raises(AssertionError): gradient_check(invalid, expected)
