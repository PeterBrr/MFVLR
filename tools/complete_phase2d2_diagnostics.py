"""Characterize all G2 fixtures and rerun unchanged N2 after diagnosed failure.

Continues ONLY finite, explicitly diagnosed additivity-policy failures. No
acceptance claim, threshold amendment, model/gradient/optimizer update.
"""
from pathlib import Path
import json
import sys
from tools.run_phase2d2 import ROOT,OUT,command


def main():
    review=json.loads((OUT/'gradient_review/review.json').read_text())
    assert review['status']=='DIAGNOSED_POLICY_FAILURE_NOT_ACCEPTED'
    results=[]
    for case in ('real','efs','local','real_efs','real_local','local_duplicate'):
        label='final_'+case
        code=command('g2_'+label,[sys.executable,str(ROOT/'tools/run_phase2d2.py'),'--worker',case,'--run-label',label])
        path=OUT/'g2'/label/'summary.json'
        if not path.exists():raise RuntimeError('nonfinite/pathway/lifecycle failure; halt remaining diagnostics')
        result=json.loads(path.read_text())
        assert result['all_valid_values_and_gradients_finite'] and result['graph_and_hook_lifecycle']=='PASS'
        if result['status'] not in ('PASS','FAIL_GRADIENT_ADDITIVITY'):raise RuntimeError('new failure requires diagnosis')
        results.append({'case':case,'label':label,'exit_code':code,'status':result['status']})
        (OUT/'g2/index.json').write_text(json.dumps({'acceptance_gate':'BLOCKED','characterization':results},indent=2)+'\n')
    dest=OUT/'n2';dest.mkdir(parents=True,exist_ok=True)
    for index in range(3):
        for dtype in ('float32','float64'):
            name=f'fixture{index}_{dtype}_threads4'
            code=command('n2_'+name,[sys.executable,str(ROOT/'tools/review_phase2c2_n2.py'),'--worker',str(index),
                         '--dtype',dtype,'--threads','4','--output',str(dest/f'{name}.json')])
            if code:raise RuntimeError('original N2 diagnostic failed; stop')
    print('All finite G2 characterizations and N2 workers completed; strict acceptance still blocked',flush=True)


if __name__=='__main__':main()
