"""Write actual standalone foundation evidence; no optimizer or FLT blocks."""
from pathlib import Path
import json
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/".phase2c1_deps"),str(ROOT/"src")]

import psutil
import torch
import yaml
from mfvlr.foundation_config import load_foundation_config
from mfvlr.tokenizer import verify_assets
from mfvlr.vim import VisionInjectionDelta, parameter_report


def summarize(tensor):
    return {"shape":list(tensor.shape),"dtype":str(tensor.dtype),"finite":bool(tensor.isfinite().all()),
            "norm":tensor.detach().norm().item(),"max_abs":tensor.detach().abs().max().item()}


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    out = ROOT/"artifacts/phase2c1"
    (out/"resolved_configs").mkdir(parents=True,exist_ok=True)
    for profile in ("reference","debug"):
        config = load_foundation_config(ROOT/f"configs/foundation_{profile}.yaml")
        config.require_approval()
        (out/f"resolved_configs/foundation_{profile}.yaml").write_text(yaml.safe_dump(config.data,sort_keys=False))
    parameter_counts = parameter_report(VisionInjectionDelta())
    parameter_counts.update({"accepted_mve":32078912,"accepted_vd":9055365,"accepted_visual":41134277,
                             "visual_plus_one_standalone_vim":41134277+1048576,
                             "full_flt_instantiated":False})
    (out/"parameter_report.json").write_text(json.dumps(parameter_counts,indent=2)+"\n")
    cases = []
    for dtype in (torch.float32,torch.float64):
        for batch in (1,2):
            model = VisionInjectionDelta().to(dtype=dtype)
            g = torch.Generator().manual_seed(303)
            x = torch.randn(batch,308,512,generator=g,dtype=dtype,requires_grad=True)
            v = torch.randn(batch,1,512,generator=g,dtype=dtype,requires_grad=True)
            before = psutil.Process().memory_info().rss
            start = time.perf_counter()
            trace = model.forward_with_trace(x,v)
            scalar = trace.delta.square().mean()
            scalar.backward()
            elapsed = time.perf_counter()-start
            upstream = (2*trace.delta.detach()/trace.delta.numel()).sum(dim=1,keepdim=True)
            analytical_visual = upstream @ model.out.weight.detach() @ model.v.weight.detach()
            ref_delta = (v.detach() @ model.v.weight.detach().T @ model.out.weight.detach().T).expand_as(x)
            atol,rtol = (1e-6,1e-5) if dtype==torch.float32 else (1e-10,1e-8)
            torch.testing.assert_close(trace.delta,ref_delta,atol=atol,rtol=rtol)
            torch.testing.assert_close(v.grad,analytical_visual,atol=atol,rtol=rtol)
            assert model.q.weight.grad.count_nonzero()==model.k.weight.grad.count_nonzero()==0
            assert v.grad.norm()>0 and model.v.weight.grad.norm()>0 and model.out.weight.grad.norm()>0
            cases.append({"batch":batch,"sequence_length":308,"dtype":str(dtype),
                          "forward_backward_seconds":elapsed,"rss_before_bytes":before,
                          "rss_after_bytes":psutil.Process().memory_info().rss,
                          "rss_is_peak_measurement":False,
                          "trace":{name:summarize(value) for name,value in vars(trace).items()},
                          "gradients":{name:summarize(p.grad) for name,p in model.named_parameters()},
                          "visual_gradient":summarize(v.grad),"language_gradient":summarize(x.grad),
                          "delta_reference_max_abs_error":(trace.delta-ref_delta).abs().max().item(),
                          "visual_gradient_reference_max_abs_error":(v.grad-analytical_visual).abs().max().item(),
                          "probability_all_exact_one":bool(torch.equal(trace.probabilities,torch.ones_like(trace.probabilities))),
                          "status":"PASS"})
            del model,x,v,trace,scalar
    result = {"python":platform.python_version(),"torch":torch.__version__,"platform":platform.platform(),
              "threads":torch.get_num_threads(),"cuda_available":torch.cuda.is_available(),
              "gpu_execution":"NOT RUN: CPU-only build" if not torch.cuda.is_available() else "NOT REQUESTED",
              "asset_manifest":verify_assets(),"cases":cases}
    (out/"vim_diagnostics.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"parameters":parameter_counts,"cases":[{k:c[k] for k in
                      ("batch","dtype","status","delta_reference_max_abs_error","visual_gradient_reference_max_abs_error")}
                     for c in cases]},indent=2))


if __name__ == "__main__":
    main()
