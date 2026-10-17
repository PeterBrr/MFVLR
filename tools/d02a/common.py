from pathlib import Path
from functools import lru_cache
import hashlib
import json
import math
import threading
import time

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/phase2d2_d02a"
TOLS = {"float32": {"atol":1e-6,"rtol":1e-5}, "float64":{"atol":1e-10,"rtol":1e-8}}
SCALES = (0.,1e-4,1.,1024.)
MULTIPLIERS = ((.02,2,1),(.2,2,1),(2,2,1),(.2,.2,1),(.2,20,1),(.2,2,.1),(.2,2,10),(2,20,10))


def write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as handle:json.dump(data,handle,indent=2,allow_nan=False)


def sha(path):
    with Path(path).open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()


def array(t):
    return t.detach().cpu().contiguous().numpy() if isinstance(t,torch.Tensor) else np.asarray(t)


def tensor_hash(t):return hashlib.sha256(array(t).tobytes()).hexdigest()


def stats(t):
    a=array(t);v=a.astype(np.float64,copy=False)
    if not np.isfinite(v).all():raise FloatingPointError("nonfinite tensor")
    return {"shape":list(a.shape),"elements":a.size,"dtype":str(a.dtype),"finite":True,
            "rms":float(np.sqrt(np.mean(v*v))),"l2":float(np.sqrt(np.sum(v*v))),
            "max_abs":float(np.max(np.abs(v))),"min":float(v.min()),"max":float(v.max()),
            "nonzero":int(np.count_nonzero(v)),"sha256":tensor_hash(a)}


@lru_cache(maxsize=160)
def directions(name, size):
    salt=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],"little")
    return tuple(np.random.default_rng(seed+salt).integers(0,2,size=size,dtype=np.int8)*2-1
                 for seed in (6701,6702,6703,6704))


def compare(actual, reference, precision, name="", directional=False):
    if array(actual).shape!=array(reference).shape:raise AssertionError("shape mismatch")
    a=array(actual).reshape(-1).astype(np.float64,copy=False)
    b=array(reference).reshape(-1).astype(np.float64,copy=False)
    if a.shape!=b.shape:raise AssertionError("shape mismatch")
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise FloatingPointError("nonfinite comparison")
    d=a-b;ad=np.abs(d); nz=b!=0; tol=TOLS[precision]
    bad=ad > tol["atol"]+tol["rtol"]*np.abs(b)
    aa=float(np.sum(a*a));bb=float(np.sum(b*b));dd=float(np.sum(d*d));dot=float(np.sum(a*b))
    r={"status":"FAIL" if bad.any() else "PASS","elements":a.size,"atol":tol["atol"],"rtol":tol["rtol"],
       "outside_count":int(bad.sum()),"outside_proportion":float(bad.mean()),"max_abs_error":float(ad.max()),
       "max_relative_nonzero_reference":float(np.max(ad[nz]/np.abs(b[nz]))) if nz.any() else None,
       "error_l2":math.sqrt(dd),"actual_l2":math.sqrt(aa),"reference_l2":math.sqrt(bb),
       "relative_l2":math.sqrt(dd/bb) if bb else None,"relative_l2_zero_reference":"undefined" if not bb else None,
       "cosine":dot/math.sqrt(aa*bb) if aa and bb else None,"reference_zero_count":int((~nz).sum()),
       "zero_reference_nonzero_actual":int(((~nz)&(a!=0)).sum()),"sign_disagreements":int((a*b<0).sum())}
    if directional:
        dirs=directions(name,a.size);normal=math.sqrt(a.size)
        r["direction_actual"]=[float(np.sum(a*v))/normal for v in dirs]
        r["direction_reference"]=[float(np.sum(b*v))/normal for v in dirs]
        r["direction_abs_error"]=[abs(x-y) for x,y in zip(r["direction_actual"],r["direction_reference"])]
    return r


def configure(threads, native):
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest");torch.backends.mkldnn.enabled=not native
    # No optimizer may be used in any pilot worker.
    for name in dir(torch.optim):
        cls=getattr(torch.optim,name)
        if isinstance(cls,type) and issubclass(cls,torch.optim.Optimizer):
            def deny(*args,**kwargs):raise AssertionError("D02A forbids optimizer steps")
            cls.step=deny


def verify_frozen():
    records=json.loads((OUT/"preservation_before.json").read_text())
    # Every worker checks executable sources/configs/spec; the controller also
    # checks all historical archives before and after the schedule.
    chosen={p:r for p,r in records.items() if p.startswith(("src/","configs/","tests/","tools/","docs/"))}
    for p,r in chosen.items():
        if sha(ROOT/p)!=r["sha256"]:raise AssertionError("frozen file drift: "+p)
    return {p:r["sha256"] for p,r in chosen.items()}


class Resources:
    def __enter__(self):
        self.started=time.perf_counter();self.samples=[];self.available=psutil.virtual_memory().available
        self.stop=threading.Event();process=psutil.Process()
        def monitor():
            while not self.stop.wait(.05):self.samples.append(process.memory_info().rss)
        self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join()
        self.record={"elapsed_seconds":time.perf_counter()-self.started,
                     "sampled_peak_rss_bytes":max(self.samples,default=psutil.Process().memory_info().rss),
                     "samples":len(self.samples),"interval_seconds":.05,
                     "available_RAM_start":self.available,"available_RAM_end":psutil.virtual_memory().available}


class Rows:
    def __init__(self,path):self.file=Path(path).open("x",encoding="utf-8");self.count=0;self.failures=0
    def add(self,kind,metadata,result):
        self.file.write(json.dumps({"kind":kind,**metadata,**result},allow_nan=False)+"\n")
        self.count+=1;self.failures+=result.get("status")=="FAIL"
    def flush(self):self.file.flush()
    def close(self):self.file.close()
