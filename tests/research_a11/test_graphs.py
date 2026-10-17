import copy
from pathlib import Path

import pytest
import torch

from mfvlr_research.decoder_a11 import build_variant, NormalizedDecoderBlock
from mfvlr_research.a11_evidence import (capture_flt, required_stages, summarize, provenance,
                                        seal, verify_seal, state_hash)


def inputs(batch=1):
    ids = (torch.arange(batch*308).reshape(batch, 308) % 400) + 1
    visual = .1*torch.randn(batch, 1, 512, generator=torch.Generator().manual_seed(43))
    return ids, visual


@pytest.mark.parametrize("variant", ["A0", "P3", "Q3"])
def test_a1_t04_causal_and_full_memory_access(accepted_flt, variant):
    base = accepted_flt.decoder.blocks[0]
    block = copy.deepcopy(base) if variant == "A0" else NormalizedDecoderBlock.from_reference(base, variant)
    x = (.2*torch.randn(1, 8, 512, generator=torch.Generator().manual_seed(51))).requires_grad_()
    m = (.2*torch.randn(1, 9, 512, generator=torch.Generator().manual_seed(52))).requires_grad_()
    v = (.1*torch.randn(1, 1, 512, generator=torch.Generator().manual_seed(53))).requires_grad_()
    y = block(x, m, v)
    cotangent = torch.linspace(-1, 1, 512).reshape(1, 512)
    gx, gm, gv = torch.autograd.grad(y[:, 0], (x, m, v), cotangent)
    assert gx[:, 1:].count_nonzero() == 0 and gx[:, 0].norm() > 0
    assert all(gm[:, i].norm() > 0 for i in range(9)) and gv.norm() > 0
    changed = x.detach().clone(); changed[:, 1:] += 3
    with torch.no_grad(): assert torch.equal(block(changed, m, v)[:, 0], y[:, 0])


@pytest.mark.parametrize("variant", ["A0", "P3", "Q3"])
def test_a1_t04_stack_shift_fixed_batch_independence(accepted_flt, variant, evidence):
    model = build_variant(accepted_flt, variant)
    ids, v = inputs(2)
    # Real lookup values become differentiable leaves. No integer-gradient surrogate.
    low = model.vocabulary(ids).detach().requires_grad_()
    memory = (.2*torch.randn(2, 308, 512, generator=torch.Generator().manual_seed(54))).requires_grad_()
    v.requires_grad_()
    expected_ids = torch.cat((torch.full((2,1),49406,dtype=torch.int64), ids[:, :-1]), 1)
    with torch.no_grad(): assert torch.equal(model.shift_low(low), model.vocabulary(expected_ids))
    y = model.decode(low, memory, v)
    with torch.no_grad():
        changed = [t.detach().clone() for t in (low, memory, v)]
        for t in changed: t[1] = t[1] + .7
        independent = model.decode(*changed)
        assert torch.equal(y[0], independent[0])
        assert not torch.equal(y[1], independent[1])
        permuted = model.decode(low.flip(0), memory.flip(0), v.flip(0))
        assert torch.equal(y.flip(0), permuted)
    cot = torch.randn(y[0].shape, generator=torch.Generator().manual_seed(55))/y[0].numel()
    rows = []
    for own in (0, 1):
        grads = torch.autograd.grad(y[own], (low, memory, v), cot, retain_graph=own == 0)
        for g in grads:
            assert g[1-own].count_nonzero() == 0
            assert g[own].norm() > 0 and g.isfinite().all()
        rows.append({"sample":own, "own_gradient_norms":[g[own].norm().item() for g in grads],
                     "cross_sample_nonzeros":[g[1-own].count_nonzero().item() for g in grads]})
    evidence("T04_"+variant, {"status":"PASS", "fixed_batch_exact":True, "permutation_exact":True,
                             "directions":rows, "full_decoder_shape":list(y.shape)})


@pytest.mark.parametrize("variant", ["A0", "P3", "Q3"])
def test_a1_t05_nineteen_vim_roles_full_language_connectivity(accepted_flt, variant, evidence):
    model = build_variant(accepted_flt, variant)
    ids, visual = inputs(); visual.requires_grad_()
    before = state_hash(model)
    result = model(ids, visual, return_logits=False)
    named = list(model.named_parameters())
    cot = torch.randn(result.decoder_hidden.shape, generator=torch.Generator().manual_seed(56)) / result.decoder_hidden.numel()
    gradients = torch.autograd.grad(result.decoder_hidden, [p for n,p in named]+[visual], cot)
    records = {}
    for (name, parameter), g in zip(named, gradients[:-1]):
        assert g is not None and g.shape == parameter.shape and g.isfinite().all()
        records[name] = {"allocated":True, "finite":True, "l2":g.double().norm().item(),
                         "nonzero":g.count_nonzero().item()}
    qk = {n:g for n,g in records.items() if ".vim.q.weight" in n or ".vim.k.weight" in n}
    assert len(qk) == 38 and all(g["nonzero"] == 0 for g in qk.values())
    vo = {n:g for n,g in records.items() if ".vim.v.weight" in n or ".vim.out.weight" in n}
    assert len(vo) == 38 and all(g["l2"] > 0 for g in vo.values())
    for n,g in records.items():
        if any(f".ln{i}." in n for i in (1,2,3)): assert g["l2"] > 0
    assert gradients[-1].isfinite().all() and gradients[-1].norm() > 0
    assert state_hash(model) == before
    evidence("T05_"+variant, {"status":"PASS", "parameters":records, "all_38_QK_allocated_exact_zero":True,
             "all_38_VO_nonzero":True, "visual_gradient_l2":gradients[-1].double().norm().item(),
             "state_unchanged":True, "output_shape":list(result.decoder_hidden.shape),
             "scope":"single controlled full-language module VJP, no paper loss or ablation sweep"})


@pytest.mark.parametrize("variant", ["A0", "P3", "Q3"])
def test_a1_t10_complete_live_capture_and_provenance(accepted_flt, variant, tmp_path, evidence):
    model = build_variant(accepted_flt, variant)
    ids, visual = inputs()
    config = Path("configs/research_a11")/(variant+".yaml")
    sources = sorted(Path("src/mfvlr").glob("*.py")) + sorted(Path("src/mfvlr_research").glob("*.py"))
    identity = provenance(model, {"ids":ids, "visual":visual}, sources, config, "A11-T10-"+variant)
    stages = capture_flt(model, ids, visual, logits=True)
    assert stages["logits"].shape == (1,308,49408)
    assert torch.equal(stages["global_language"], stages["LE.11"][:,307:308])
    assert all(v.grad_fn is None and not v.requires_grad for v in stages.values())
    parameters = {n:{"shape":list(p.shape), "elements":p.numel()} for n,p in model.named_parameters()}
    record = {"provenance":identity, "stages":summarize(stages), "parameters":parameters}
    envelope = seal(record)
    assert verify_seal(envelope, identity, required_stages(), parameters) is record
    # Fresh expected provenance must be supplied from current model/inputs/files.
    for field in ("sources", "config", "inputs", "state", "backend", "experiment_id"):
        stale = copy.deepcopy(identity); stale[field] = "stale"
        with pytest.raises(ValueError, match="stale"): verify_seal(envelope, stale, required_stages(), parameters)
    # Recompute identities from actual changed files/tensors, not just JSON edits.
    temporary_source = tmp_path / "source.py"; temporary_source.write_text("old source")
    temporary_config = tmp_path / "config.yaml"; temporary_config.write_text("old config")
    tracked = provenance(model, {"ids":ids, "visual":visual}, [temporary_source], temporary_config, "stale-probe")
    tracked_record = seal({**record, "provenance":tracked})
    for path in (temporary_source, temporary_config):
        original = path.read_bytes(); path.write_text("changed")
        fresh = provenance(model, {"ids":ids, "visual":visual}, [temporary_source], temporary_config, "stale-probe")
        with pytest.raises(ValueError, match="stale"): verify_seal(tracked_record, fresh, required_stages(), parameters)
        path.write_bytes(original)
    changed_ids = ids.clone(); changed_ids[0, 0] += 1
    fresh = provenance(model, {"ids":changed_ids, "visual":visual}, [temporary_source], temporary_config, "stale-probe")
    with pytest.raises(ValueError, match="stale"): verify_seal(tracked_record, fresh, required_stages(), parameters)
    original_weight = model.decoder.blocks[0].ln1.weight if variant != "A0" else model.decoder.blocks[0].self_attention.q.bias
    saved = original_weight.detach().clone()
    with torch.no_grad(): original_weight.add_(1)
    fresh = provenance(model, {"ids":ids, "visual":visual}, [temporary_source], temporary_config, "stale-probe")
    with pytest.raises(ValueError, match="stale"): verify_seal(tracked_record, fresh, required_stages(), parameters)
    with torch.no_grad(): original_weight.copy_(saved)
    damaged = copy.deepcopy(envelope); damaged["record"]["stages"]["LD.6.output"]["rms"] += 1
    with pytest.raises(ValueError, match="integrity"): verify_seal(damaged, identity, required_stages(), parameters)
    for kind, missing in (("stages", "LE.11"), ("stages", "LD.6.vim"), ("parameters", "vocabulary.weight")):
        incomplete = copy.deepcopy(record); del incomplete[kind][missing]
        with pytest.raises(ValueError, match="inventory"):
            verify_seal(seal(incomplete), identity, required_stages(), parameters)
    bad = copy.deepcopy(record); bad["stages"]["LD.0.self"]["finite"] = False
    with pytest.raises(FloatingPointError): verify_seal(seal(bad), identity, required_stages(), parameters)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    evidence("T10_"+variant, envelope)
