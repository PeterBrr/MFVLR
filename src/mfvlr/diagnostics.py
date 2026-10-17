"""Synthetic MVE forward/backward evidence only; no training or paper losses."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import time

import torch
from torch import Tensor, nn
import yaml

from .config import load_config
from .mve import MVE, parameter_report


def _shapes(value):
    if isinstance(value, Tensor):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [_shapes(item) for item in value]
    return None


def run(config_path: str | Path, *, smoke: bool, output_dir: str | Path) -> dict:
    config = load_config(config_path)
    config.require_mve_approval()
    settings = config.proposal
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    # Tests run on CPU in this phase; no implicit CUDA/AMP substitution.
    model = MVE.from_config(config).cpu()
    model.train(not smoke)
    report = parameter_report(model)
    if report["total_unique"] != settings["verification"]["expected_unique_mve_parameters"]:
        raise RuntimeError(f"Parameter count mismatch: {report}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    events = []
    call_counts = defaultdict(int)
    handles = []

    def forward_hook(name):
        def hook(module, inputs, output):
            call_counts[name] += 1
            events.append({"direction": "forward", "module": name,
                           "call": call_counts[name], "input": _shapes(inputs), "output": _shapes(output)})
        return hook

    def backward_hook(name):
        def hook(module, grad_input, grad_output):
            events.append({"direction": "backward", "module": name,
                           "grad_input": _shapes(grad_input), "grad_output": _shapes(grad_output)})
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.Linear, nn.LayerNorm, nn.GELU)):
            handles.append(module.register_forward_hook(forward_hook(name)))
            if not smoke:
                handles.append(module.register_full_backward_hook(backward_hook(name)))
    generator = torch.Generator(device="cpu").manual_seed(settings["diagnostics"]["input_rng_seed"])
    batch = 1 if smoke else settings["diagnostics"]["batch_size"]
    iterations = 1 if smoke else settings["diagnostics"]["iterations"]
    fixture = settings["diagnostics"]["reconstruction_fixture"]
    checks = []
    started = time.perf_counter()
    try:
        for index in range(iterations):
            model.zero_grad(set_to_none=True)
            image = torch.rand(batch, 3, 224, 224, generator=generator, requires_grad=not smoke)
            with torch.set_grad_enabled(not smoke):
                appearance = model.encode_appearance(image)
                reconstruction = fixture["scale"] * image + fixture["offset"]
                result = model(image, reconstruction, appearance)
                observed = {
                    "image": list(image.shape), "local": list(appearance.local.shape),
                    "skips": [list(skip.shape) for skip in appearance.skips],
                    "appearance_tokens": list(appearance.tokens.shape),
                    "appearance_global": list(appearance.global_token.shape),
                    "reconstruction": list(reconstruction.shape),
                    "residual": list(result.residual_image.shape),
                    "residual_tokens": list(result.residual_tokens.shape),
                    "residual_global": list(result.residual_global.shape),
                    "fused_global": list(result.fused_global.shape),
                }
                diagnostic = result.fused_global.square().mean()
                check = {"iteration": index + 1, "batch": batch, "shapes": observed,
                         "diagnostic_scalar": diagnostic.item(), "is_paper_loss": False}
                if not torch.isfinite(diagnostic).item():
                    raise RuntimeError("Nonfinite diagnostic scalar")
                if not smoke:
                    watched = {"local": appearance.local, "appearance_tokens": appearance.tokens,
                               "reconstruction": reconstruction, "residual_image": result.residual_image,
                               "residual_tokens": result.residual_tokens, "fused_global": result.fused_global}
                    for tensor in watched.values():
                        tensor.retain_grad()
                    diagnostic.backward()
                    check["gradient_shapes"] = {
                        name: None if tensor.grad is None else list(tensor.grad.shape)
                        for name, tensor in watched.items()
                    }
                    check["input_gradient_shape"] = list(image.grad.shape)
                    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                    if not gradients or not all(torch.isfinite(grad).all().item() for grad in gradients):
                        raise RuntimeError("Missing or nonfinite parameter gradients")
                    check["parameter_tensors_with_gradients"] = len(gradients)
                    check["first_conv_gradient_norm"] = model.shared_encoder.cnn.stages[0][0].weight.grad.norm().item()
                checks.append(check)
                print(f"check {index + 1}/{iterations}: batch={batch}, Iv={observed['fused_global']}, "
                      f"diagnostic={diagnostic.item():.8f}, backward={not smoke}", flush=True)
            # Do not retain three complete computation graphs in the evidence object.
            del result, appearance, reconstruction, diagnostic, image
    finally:
        for handle in handles:
            handle.remove()
    summary = {
        "mode": "cpu_forward_smoke" if smoke else "cpu_batch2_debug",
        "config": str(config.path), "components": ["mve"], "optimizer_updates": 0,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "pyyaml": yaml.__version__, "platform": platform.platform(),
                        "threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available()},
        "elapsed_seconds": time.perf_counter() - started,
        "parameters": report, "checks": checks,
        "approval_bundles": settings and config.review_chain[-1]["mve_approval"]["bundles"],
        "architecture_is_approved_engineering_assumption": True,
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_path / "tensor_trace.json").write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")
    print(f"unique MVE parameters: {report['total_unique']:,}; evidence: {output_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--mode", choices=("smoke", "debug"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(args.config, smoke=args.mode == "smoke", output_dir=args.output_dir)


if __name__ == "__main__":
    main()
