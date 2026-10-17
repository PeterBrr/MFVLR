"""Reproduce the CPU FP32 N1 investigation without changing MVE or training it."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import nn

from mfvlr.config import load_config
from mfvlr.mve import MVE, parameter_report


def run(output_dir: Path) -> None:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = MVE.from_config(load_config(ROOT / "configs/debug.yaml")).eval()
    original_ids = tuple(id(p) for p in model.parameters())
    image = torch.rand(1, 3, 224, 224, generator=torch.Generator().manual_seed(17))
    batch = torch.cat((image, 1 - image))
    traces = {}
    block_inputs = {}

    def capture(value, label):
        counts = {}
        observed = {}
        handles = []

        def hook(name):
            def record(module, args, output):
                counts[name] = counts.get(name, 0) + 1
                key = f"{name}:role{counts[name]}"
                observed[key] = output[:1].detach().clone()
                if name == "shared_encoder.blocks.0":
                    block_inputs[(label, counts[name])] = args[0].detach().clone()
            return record

        for name, module in model.named_modules():
            if name.startswith("shared_encoder.blocks.0") or isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(hook(name)))
        try:
            result = model(value, 0.75 * value + 0.1).fused_global.detach()
        finally:
            for handle in handles:
                handle.remove()
        traces[label] = observed
        return result

    with torch.no_grad():
        alone = capture(image, "b1")
        together = capture(batch, "b2")
        approved_old_passes = torch.allclose(alone, together[:1], atol=1e-6, rtol=1e-5)
        amended_passes = torch.allclose(alone, together[:1], atol=5e-6, rtol=1e-5)
        assert amended_passes
        peers = {
            "zeros": torch.zeros_like(image),
            "ones": torch.ones_like(image),
            "independent_random": torch.rand(image.shape, generator=torch.Generator().manual_seed(99)),
        }
        fixed_batch = {}
        for name, peer in peers.items():
            changed_batch = torch.cat((image, peer))
            changed = model(changed_batch, 0.75 * changed_batch + 0.1).fused_global
            difference = (together[:1] - changed[:1]).abs().max().item()
            assert torch.equal(together[:1], changed[:1])
            fixed_batch[name] = {"exactly_equal": True, "max_abs": difference}

        # Change the OTHER sample's reconstruction independently of its image.
        changed_reconstruction = 0.75 * batch + 0.1
        changed_reconstruction[1].fill_(2.0)
        changed = model(batch, changed_reconstruction).fused_global
        assert torch.equal(together[:1], changed[:1])
        fixed_batch["other_reconstruction_only"] = {"exactly_equal": True, "max_abs": 0.0}

        # Keep a full batch and permute its rows; no tolerance amendment here.
        swapped = model(batch.flip(0), (0.75 * batch + 0.1).flip(0)).fused_global
        assert torch.equal(together, swapped.flip(0))

        layer_differences = [
            {"module_role": name, "max_abs": (tensor - traces["b2"][name]).abs().max().item()}
            for name, tensor in traces["b1"].items()
        ]
        # Replay ONLY an isolated block in FP64; the actual MVE stays FP32.
        # Both runs start with exactly the same sample-0 inputs and weights.
        block = model.shared_encoder.blocks[0]
        block64 = copy.deepcopy(block).double()
        replay = []
        for role in (1, 2):
            one = block_inputs[("b1", role)]
            two = block_inputs[("b2", role)]
            assert torch.equal(one, two[:1])
            delta32 = (block(one) - block(two)[:1]).abs().max().item()
            delta64 = (block64(one.double()) - block64(two.double())[:1]).abs().max().item()
            assert delta32 > 0 and delta64 < delta32
            replay.append({"role": role, "inputs_exactly_equal": True,
                           "fp32_max_abs": delta32, "fp64_max_abs": delta64})
        del block64

    # Cross-sample Jacobian probe uses independent image and reconstruction leaves.
    # Shared PARAMETERS do not imply a dependency between different batch rows.
    batch_leaf = batch.clone().requires_grad_(True)
    reconstruction_leaf = (0.75 * batch + 0.1).detach().requires_grad_(True)
    result = model(batch_leaf, reconstruction_leaf)
    grads = torch.autograd.grad(result.fused_global[0].square().mean(),
                                (batch_leaf, reconstruction_leaf))
    jacobian = {}
    for name, grad in zip(("image", "reconstruction"), grads):
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad[1]).item() == 0
        assert grad[0].norm().item() > 0
        jacobian[name] = {"sample_0_gradient_norm": grad[0].norm().item(),
                          "sample_1_nonzero_gradient_elements": 0,
                          "sample_1_max_abs_gradient": grad[1].abs().max().item()}

    assert original_ids == tuple(id(p) for p in model.parameters())
    counts = parameter_report(model)
    assert counts["total_unique"] == 32078912
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "threads": torch.get_num_threads(), "device": "cpu", "dtype": "float32",
                        "platform": platform.platform()},
        "mve_source_sha256": hashlib.sha256((ROOT / "src/mfvlr/mve.py").read_bytes()).hexdigest(),
        "original_threshold_passes": approved_old_passes,
        "n1_threshold_passes": amended_passes,
        "cross_batch_max_abs": (alone - together[:1]).abs().max().item(),
        "fixed_batch_perturbations": fixed_batch,
        "batch_permutation_exact": True,
        "layer_comparisons": layer_differences,
        "isolated_block_precision_replay": replay,
        "cross_sample_jacobian_probe": jacobian,
        "parameter_identities_unchanged": True,
        "parameters": counts,
        "optimizer_updates": 0,
        "conclusion": "Batch-shape-dependent floating-point rounding supported by identical block inputs, "
                      "precision replay, exact fixed-batch invariance, zero cross-sample probe gradients, "
                      "and source operations that do not mix the batch axis.",
        "limits": "Finite fixtures and one CPU backend; not a proof for every input, device or software version.",
    }
    (output_dir / "numerical_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    (output_dir / "parameter_report.json").write_text(json.dumps({
        "source": "actual unchanged MVE construction during N1 review", "counts": counts,
        "expected_unique": 32078912, "matches_approved_contract": True,
        "establishes_author_fidelity": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in evidence.items() if key != "layer_comparisons"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/phase2a/n1_review")
    run(parser.parse_args().output_dir)
