"""Validate raw record bookkeeping and describe zero/nonzero reference magnitudes."""
from collections import Counter
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02a'


def main():
    jobs=json.loads((OUT/'pilot_protocol.json').read_text())['jobs']
    count=Counter();groups={};issues=[]
    for job in jobs:
        with (OUT/'jobs'/job['id']/'comparisons.jsonl').open() as handle:
            for line in handle:
                row=json.loads(line,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
                count['rows']+=1;count[row.get('status','MEASUREMENT')]+=1
                if 'outside_count' in row:
                    n=row['elements'];bad=row['outside_count']
                    assert 0<=bad<=n and row['outside_proportion']==bad/n
                    assert row['status']==('FAIL' if bad else 'PASS')
                    fp32=('FP32_vs_FP64' in row['kind'] or row['kind']=='cotangent_sum_rounding' or row.get('precision')=='float32')
                    assert (row['atol'],row['rtol'])==((1e-6,1e-5) if fp32 else (1e-10,1e-8))
                    if row['reference_l2']==0 and row['reference_zero_count']!=n:
                        issues.append({'job':job['id'],'kind':'REFERENCE_NORM_UNDERFLOW',**row})
                    if row['reference_l2']==0:assert row['relative_l2'] is None
                    if row.get('singleton_QK'):
                        count['singleton_zero_checks']+=1
                        assert row['allocated'] and row['exact_structural_zero'] and row['actual_l2']==row['reference_l2']==0 and bad==0
                    name=row.get('parameter','')
                    category=('VIM_QK' if '.vim.q.' in name or '.vim.k.' in name else 'ordinary_key_bias' if name.endswith('k.bias') else 'ordinary_QK_weight' if name.endswith(('q.weight','k.weight')) else 'other')
                    if name:
                        key=(job['family'],job['variant'],row['kind'],row.get('precision','comparison'),row.get('cut'),row.get('scale'),category)
                        g=groups.setdefault(key,{'records':0,'zero_reference':0,'nonzero_reference':0,'smallest_nonzero_reference_l2':None,'largest_reference_l2':0,'FAIL':0,'max_relative_l2':None,'maximum_error_l2':0})
                        g['records']+=1;g['FAIL']+=row['status']=='FAIL'
                        ref=row['reference_l2'];g['largest_reference_l2']=max(g['largest_reference_l2'],ref)
                        g['maximum_error_l2']=max(g['maximum_error_l2'],row['error_l2'])
                        if ref==0:g['zero_reference']+=1
                        else:
                            g['nonzero_reference']+=1
                            g['smallest_nonzero_reference_l2']=min(g['smallest_nonzero_reference_l2'] or ref,ref)
                            g['max_relative_l2']=max(g['max_relative_l2'] or 0,row['relative_l2'])
                if row['kind']=='directional_FD':
                    assert (row['epsilon'],row['atol'],row['rtol'])==(1e-6,1e-5,1e-3)
                    assert row['status']==('PASS' if row['absolute_error']<=1e-5+1e-3*abs(row['analytical']) else 'FAIL')
    result={'status':'PASS','scope':'Record/threshold/status integrity, not numerical acceptance','counts':dict(count),
        'norm_underflow_records':issues,'weak_gradient_description':'Exact-zero and nonzero reference norms reported separately; minimum positive norms are descriptive, with no invented near-zero tolerance or denominator floor',
        'groups':[dict(zip(('family','variant','kind','precision','cut','scale','parameter_category'),key))|value for key,value in groups.items()]}
    dest=OUT/'verification_final/raw_record_audit.json'
    with dest.open('x',encoding='utf-8') as h:json.dump(result,h,indent=2,allow_nan=False)
    print(json.dumps({'status':'PASS','counts':dict(count),'norm_underflow_records':len(issues)}))


if __name__=='__main__':main()
