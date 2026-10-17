"""Compare fresh scientific fields with preserved evidence; never rewrite it."""
from pathlib import Path
import json
from tools.audit_phase2d2_n3 import scientific
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/phase2d2_n31'
def read(p):return json.loads(Path(p).read_text())


def main():
    records={}
    for run in ('real_mkldnn','real_efs_mkldnn','local_duplicate_mkldnn','real_efs_native'):
        old=ROOT/'artifacts/phase2d2_n3/same_forward'/run;new=OUT/'same_forward'/run
        a,b=read(old/'same_graph.json'),read(new/'same_graph.json')
        r={'all_selected_A_and_B_gradient_statistics_exact':all(a[k]==b[k] for k in ('A','B_components','B_sum','A_vs_B')),
            'source_state_hash_exact':read(old/'summary.json')['initial_state_sha256']==read(new/'summary.json')['initial_state_sha256'],
            'metadata_repaired_seven_unique_loss_wrapper_ids':len(set(b['graph']['loss_grad_fn_python_ids'].values()))==7,
            'all_summary_gradient_comparisons_exact':read(old/'summary.json')['comparisons']==read(new/'summary.json')['comparisons']}
        records[run]=r
    cmc={d:scientific(read(OUT/'chain'/f'{d}.json'))==scientific(read(ROOT/'artifacts/phase2d2_n3/chain'/f'{d}.json')) for d in ('float32','float64')}
    g2={}
    for case in ('real','efs','local','real_efs','real_local','local_duplicate'):
        new=OUT/'g2'/('final_'+case);old=ROOT/'artifacts/phase2d2_retry/g2'/('final_'+case)
        if not (new/'summary.json').exists():continue
        g2[case]={n:scientific(read(new/f'{n}.json'))==scientific(read(old/f'{n}.json')) for n in ('fd','lr','cmc','fl','ar','kl','total','gradient_additivity')}
    n2={}
    for p in (OUT/'n2').glob('*.json'):
        n2[p.name]=scientific(read(p))==scientific(read(ROOT/'artifacts/phase2c2_n2_acceptance/diagnostics'/p.name))
    result={'scope':'Gradient scientific fields compared exactly, process identities/timing excluded as specified; no inference of harmlessness',
        'same_forward':records,'CMC':cmc,'G2':g2,'N2':n2,
        'completion':{'G2_cases_available':len(g2),'N2_cases_available':len(n2)},'acceptance':'BLOCKED'}
    (OUT/'historical_scientific_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
