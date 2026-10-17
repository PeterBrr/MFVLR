"""Research-only evidence: complete captures and fail-closed provenance checks."""
from pathlib import Path
import hashlib
import json
import platform

import torch
from .decoder_a11 import tensor_digest


def backend():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "mkldnn": torch.backends.mkldnn.enabled, "cuda_available": torch.cuda.is_available()}


def file_hash(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(bytes.fromhex(tensor_digest(value)))
    return digest.hexdigest()


def provenance(module, inputs, source_paths, config_path, experiment_id):
    if not source_paths or len(inputs) < 1 or not experiment_id:
        raise ValueError("complete source/input/experiment provenance required")
    return {"experiment_id": experiment_id, "backend": backend(),
            "sources": {str(p): file_hash(p) for p in sorted(map(Path, source_paths))},
            "config": {str(config_path): file_hash(config_path)}, "state": state_hash(module),
            "inputs": {name: {"sha256": tensor_digest(value), "shape": list(value.shape),
                               "dtype": str(value.dtype)} for name, value in inputs.items()}}


def seal(record):
    body = json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return {"record": record, "sha256": hashlib.sha256(body.encode()).hexdigest()}


def verify_seal(envelope, expected_provenance, required_stages, required_parameters):
    if set(envelope) != {"record", "sha256"} or seal(envelope["record"])["sha256"] != envelope["sha256"]:
        raise ValueError("record integrity failure")
    record = envelope["record"]
    if record["provenance"] != expected_provenance:
        raise ValueError("stale source/config/input/state/backend/experiment provenance")
    if set(record["stages"]) != set(required_stages) or set(record["parameters"]) != set(required_parameters):
        raise ValueError("incomplete or unexpected stage/parameter inventory")
    if isinstance(required_parameters, dict) and record["parameters"] != required_parameters:
        raise ValueError("parameter inventory metadata mismatch")
    if not all(v["finite"] for v in record["stages"].values()):
        raise FloatingPointError("nonfinite captured stage")
    return record


def required_stages(logits=True):
    return ({"low", "LE.input", "global_language", "LD.input", "visual"}
            | {f"LE.{i}" for i in range(12)}
            | {f"LD.{i}.{stage}" for i in range(7)
               for stage in ("self", "memory", "ff", "vim", "output")}
            | ({"logits"} if logits else set()))


def capture_flt(model, ids, visual, *, logits=True):
    """No-grad numerical capture, no retained autograd graphs or hidden detach in model.

    Hooks observe the real forward. Residual sums for A0/P3 are reconstructed
    solely for capture; Q3 records actual post-norm outputs. The model is called
    once and never replaced by a capture-specific forward implementation.
    """
    stages = {"visual": visual.detach().clone()}
    handles = []
    def save(name, value): stages[name] = value.detach().clone()
    handles.append(model.vocabulary.register_forward_hook(lambda m, a, y: save("low", y)))
    handles.append(model.encoder.register_forward_pre_hook(lambda m, a: save("LE.input", a[0])))
    handles.append(model.decoder.register_forward_pre_hook(lambda m, a: save("LD.input", a[0])))
    for i, block in enumerate(model.encoder.blocks):
        handles.append(block.register_forward_hook(lambda m, a, y, i=i: save(f"LE.{i}", y)))
    for i, block in enumerate(model.decoder.blocks):
        intermediate = {}
        for name in ("self_attention", "memory_attention", "ff", "vim"):
            handles.append(getattr(block, name).register_forward_hook(
                lambda m, a, y, name=name, d=intermediate: d.__setitem__(name, y.detach().clone())))
        if getattr(block, "variant", "A0") == "Q3":
            for name in ("ln1", "ln2", "ln3"):
                handles.append(getattr(block, name).register_forward_hook(
                    lambda m, a, y, name=name, d=intermediate: d.__setitem__(name, y.detach().clone())))
        def finish(m, args, output, i=i, d=intermediate):
            if getattr(m, "variant", "A0") == "Q3":
                a, b, c = (d[n] for n in ("ln1", "ln2", "ln3"))
            else:
                a = args[0] + d["self_attention"]
                b = a + d["memory_attention"]
                c = b + d["ff"]
            for name, value in (("self", a), ("memory", b), ("ff", c),
                                ("vim", c + d["vim"]), ("output", output)):
                save(f"LD.{i}.{name}", value)
            if not torch.equal(stages[f"LD.{i}.vim"], output):
                raise AssertionError("captured residual does not reproduce actual output")
            d.clear()
        handles.append(block.register_forward_hook(finish))
    try:
        with torch.no_grad():
            result = model(ids, visual, return_logits=logits)
            save("global_language", result.global_language)
            if logits: save("logits", result.logits)
    finally:
        for handle in handles: handle.remove()
    if set(stages) != required_stages(logits):
        raise AssertionError("missing or extra capture stage")
    if any(not t.isfinite().all() or t.requires_grad or t.grad_fn is not None for t in stages.values()):
        raise AssertionError("nonfinite or graph-retaining diagnostic")
    return stages


def summarize(stages):
    result = {}
    for name, value in stages.items():
        v = value.double()
        result[name] = {"shape": list(value.shape), "finite": bool(value.isfinite().all()),
                        "rms": v.square().mean().sqrt().item(), "l2": v.norm().item(),
                        "max_abs": v.abs().max().item(), "sha256": tensor_digest(value)}
    return result


def gradient_check(actual, expected, *, atol=1e-6, rtol=1e-5):
    """Exact presence/shape/finiteness; unchanged FP32 numerical thresholds."""
    if actual is None or expected is None:
        raise AssertionError("required gradient is missing")
    if actual.shape != expected.shape or not actual.isfinite().all() or not expected.isfinite().all():
        raise AssertionError("invalid gradient shape or finiteness")
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
