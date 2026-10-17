"""Record installed numerical-library metadata without changing the environment."""
from pathlib import Path
import contextlib,importlib.metadata,io,json,os,platform,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
import psutil


def main():
    destination=ROOT/'artifacts/phase2d2_n31/environment.json'
    assert not destination.exists()
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture):np.show_config()
    versions={n:importlib.metadata.version(n) for n in ('torch','numpy','pytest','psutil','PyYAML','ftfy','regex','wcwidth')}
    value={'versions':versions,'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'numpy_build':capture.getvalue(),'torch_build':torch.__config__.show(),
        'environment':{n:os.environ.get(n) for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_CBWR')},
        'CUDA_available':torch.cuda.is_available(),'CUDA_build':torch.version.cuda,
        'available_RAM_at_observation':psutil.virtual_memory().available,
        'note':'Separate metadata process; per-worker Torch thread/backend settings are recorded in worker outputs. No dependencies installed or substituted.'}
    destination.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'versions':versions,'CUDA':value['CUDA_available']}))


if __name__=='__main__':main()
