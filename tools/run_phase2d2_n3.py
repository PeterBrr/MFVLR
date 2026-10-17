"""Reproduce the bounded N3 diagnostic matrix, never the acceptance amendment."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT)]


def main():
    from tools import run_phase2d2 as runner
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'artifacts/phase2d2_n3')
    parser.add_argument('--continue-after-real-pilot',action='store_true');args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);runner.OUT=out
    paths=('tools/n3_gradient_tools.py','tools/review_phase2d2_n3.py','tools/review_phase2d2_n3_local.py','tools/retry_phase2d2.py')
    (out/'diagnostic_source_hashes.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    if not args.continue_after_real_pilot:
        code=runner.command('same_real_mkldnn',[sys.executable,str(ROOT/'tools/review_phase2d2_n3.py'),'--worker','real','--output',str(out/'same_forward')])
        if code:raise RuntimeError('real pilot failed; halt diagnostic matrix')
    else:
        old=json.loads((out/'same_forward/real_mkldnn/summary.json').read_text())
        assert old['status']=='DIAGNOSTIC_COMPLETE_PHASE_BLOCKED'
    code=runner.command('local_analytical',[sys.executable,str(ROOT/'tools/review_phase2d2_n3_local.py'),'--output',str(out/'local_analytical.json')])
    if code:raise RuntimeError('independent local analytical control failed')
    for dtype in ('float32','float64'):
        code=runner.command('cmc_'+dtype,[sys.executable,'-c',
            'from pathlib import Path; import sys; from tools import retry_phase2d2 as r; r.OUT=Path(sys.argv[1]); raise SystemExit(r.chain(sys.argv[2]))',str(out),dtype])
        if code and dtype=='float64':raise RuntimeError('promoted FP64 CMC control failed')
    for case,mode in (('real_efs','mkldnn'),('local_duplicate','mkldnn'),('real_efs','native')):
        code=runner.command('same_'+case+'_'+mode,[sys.executable,str(ROOT/'tools/review_phase2d2_n3.py'),'--worker',case,'--mode',mode,'--output',str(out/'same_forward')])
        if code:raise RuntimeError('new nonfinite/identity/lifecycle/diagnostic failure; stop')
    print('N3 diagnosis completed; no policy applied; Phase 2D.2 BLOCKED',flush=True)


if __name__=='__main__':main()
