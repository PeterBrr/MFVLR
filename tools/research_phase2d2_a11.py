"""Fail-closed preregistered experiment launcher: prepare only under A1-D01.

The expensive execution entry point is deliberately locked, with no CLI/config
bypass. A1-D02 approval and a reviewed enablement change are required. Preparing
a manifest neither creates model weights nor evaluates a fixture.
"""
from pathlib import Path
import argparse
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from mfvlr_research.a11_config import load_config
from mfvlr_research.a11_evidence import file_hash, backend, seal

PREREG = ROOT/"docs/phase2d2_a1_ablation_preregistration_v1_1.md"
# Canonical checksum is compared to the immutable version record below.
EXECUTION_APPROVAL = None
BACKENDS = ("cpu-fp32-threads4-mkldnn", "cpu-fp32-threads4-native",
            "cpu-fp32-threads1-mkldnn", "cpu-fp64-threads4-mkldnn")
MULTIPLIERS = ((.02,2,1),(.2,2,1),(2,2,1),(.2,.2,1),(.2,20,1),(.2,2,.1),(.2,2,10),(2,20,10))
G2 = ("real", "efs", "local", "real_efs", "real_local", "local_duplicate")


def fixture_spec(name):
    if name in {f"NLD-{j}" for j in range(3)}:
        j = int(name[-1])
        return {"family":"NLD", "j":j, "FLT_seed":6101+j, "batch":[1,2],
                "master_FP32_seeds":{"state":6201+j,"memory":6301+j,"visual":6401+j},
                "B1":"first B2 master sample cloned contiguous", "multipliers":MULTIPLIERS,
                "cuts":[1,4,7], "cotangent_seeds":[6501+j,6601+j],
                "cotangent_divisor":"sqrt(B*308*512)", "cotangent_scales":[0,1e-4,1,1024],
                "direction_seeds":[6701,6702,6703,6704], "capture":"all seven blocks and four residual stages",
                "directions":"Rademacher, SHA256(name) first four bytes little-endian salt; divide sqrt(numel)",
                "comparisons":["same-forward VJP additivity", "FP32 versus promoted-weight FP64"]}
    if name in {f"NATT-{j}" for j in range(3)}:
        j = int(name[-1])
        return {"family":"NATT", "j":j, "initialization_seed":6101+j, "input_seed":6801+j,
                "batch":[1,2], "shape":[308,512], "input_multipliers":[.02,.2,2],
                "cotangent_seed":6901+j, "scales":[0,1e-4,1,1024], "FD_scale":1,
                "direction_seeds":[6701,6702], "LN_constant":.25, "LN_near_constant_seed":6951+j,
                "LN_noise_scale":1e-4, "oracle":"independent attention/LN Jacobians"}
    if name in {f"NG2-{j}" for j in range(3)}:
        j = int(name[-1])
        return {"family":"NG2", "j":j, "batch":2, "FLT_seed":6101+j, "visual_head_seed":0,
                "image_seeds":[7101+j,7201+j], "kinds":[["real","efs"],["real","local"],["local","local"]][j],
                "golden_indices":[[0,1],[0,5],[5,5]][j], "base_image":".1+.8*rand(3,224,224)",
                "local_rectangle":[64,160,80,176], "local_operation":"1-I", "masks":"supplied synthetic oracle",
                "objectives":["fd","lr","cmc","fl","ar","kl","total"], "precision":"FP32 only"}
    if name in {f"N2-F{j}" for j in range(3)}:
        return {"family":"N2", "index":int(name[-1]), "source":"tools/review_phase2c2_n2.py:fixture",
                "weights":"same FP32 initialized state, deep-copied and promoted for FP64",
                "batches":[1,2], "comparison":"same variant cross-batch diagnostic and unchanged FP64 gate"}
    if name.startswith("G2-") and name[3:] in G2:
        return {"family":"G2", "case":name[3:], "source":"tools/phase2d2_fixtures.py:fixture",
                "worker":"tools/phase2d2_diagnostics.py:run_g2", "full_streaming_extension":"tools/n31_full_gradients.py",
                "objectives":["fd","lr","cmc","fl","ar","kl","total"], "precision":"FP32 only"}
    if name in ("N3-same-forward", "N3-CMC-chain", "N31-locked-local"):
        return {"family":name, "source":"artifacts/phase2d2_n31/heldout_preregistration.json",
                "source_sha256":"7328fa21235112ac1dd081062efefc1ebaa8c96ff9444ebdbedf7e703c7e72"}
    raise ValueError("unknown preregistered fixture identity")


def prepare(experiment_id, variant, fixture, selected_backend, output_root):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", experiment_id):
        raise ValueError("explicit safe experiment identity required")
    if selected_backend not in BACKENDS:
        raise ValueError("explicit registered CPU backend required; B8/GPU are not authorized")
    config_path = ROOT/"configs/research_a11"/(variant+".yaml")
    # Validate variant before constructing or reading any untrusted path.
    from mfvlr_research.decoder_a11 import require_variant
    require_variant(variant)
    config = load_config(config_path)
    spec = fixture_spec(fixture)
    if "fp64" in selected_backend and spec["family"] in ("G2", "NG2"):
        raise ValueError("public visual FP64 interfaces are unsupported")
    destination_root = Path(output_root).resolve()
    allowed = (ROOT/"artifacts/phase2d2_a11/prepared").resolve()
    if not destination_root.is_relative_to(allowed):
        raise ValueError("output must be in the isolated A1.1 prepared tree")
    versions = json.loads((ROOT/"artifacts/phase2d2_a11/preregistration_versions.json").read_text())
    digest = file_hash(PREREG)
    if digest != versions["A1.1"]["sha256"]:
        raise ValueError("stale preregistration")
    files = list((ROOT/"src/mfvlr").glob("*.py")) + list((ROOT/"src/mfvlr_research").glob("*.py"))
    files += [Path(__file__).resolve(), PREREG, config_path,
              ROOT/"tools/review_phase2c2_n2.py", ROOT/"tools/phase2d2_fixtures.py",
              ROOT/"tools/n31_full_gradients.py", ROOT/"tools/phase2d2_diagnostics.py"]
    record = {"status":"PREPARED_NOT_RUN_A1_D02_BLOCKED", "experiment_id":experiment_id,
              "variant":variant, "fixture":fixture, "fixture_spec":spec, "requested_backend":selected_backend,
              "preparation_host":backend(), "config":config, "preregistration_sha256":digest,
              "sources":{p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted(files)},
              "criteria":{"FP32":{"atol":1e-6,"rtol":1e-5}, "FP64":{"atol":1e-10,"rtol":1e-8},
                          "FD":{"eps":1e-6,"atol":1e-5,"rtol":1e-3}},
              "runtime_protocol":{"warmup_no_grad":1,"measured_reloads":3,"include_cold_start":True,
                                  "optimizer_updates":0,"concurrent_full_models":False},
              "execution_enabled":False, "model_instantiated":False, "fixture_generated":False,
              "required_execution_evidence":["fresh source/config/input/state hashes before forward",
                  "all stages and parameters", "all failures and worsening cells", "RSS/runtime", "raw comparisons"]}
    destination = destination_root/experiment_id
    destination.mkdir(parents=True, exist_ok=False)
    with (destination/"request.json").open("x", encoding="utf-8") as handle:
        json.dump(seal(record), handle, indent=2, allow_nan=False)
    return destination


def execute(*args, **kwargs):
    # Intentionally no user-controlled enable flag or YAML override. Future review
    # must authorize and audit the numerical execution dispatch before enabling it.
    raise PermissionError("A1-D02 unapproved: expensive execution disabled; prepare a sealed request only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute"))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--variant", required=True, choices=("A0","P3","Q3"))
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "execute": execute()
    path = prepare(args.experiment_id, args.variant, args.fixture, args.backend, args.output_root)
    print(path)


if __name__ == "__main__": main()
