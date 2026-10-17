"""Aggregate preserved and new numerical evidence without accepting a policy."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
ORDER=('fd','lr','cmc','fl','ar','kl','total')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_write(p,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def group(n):
    return next((g for p,g in (
        ('mve.shared_encoder.cnn','MVE.local'),('mve.','MVE.global'),('decoder.trunk','VD.trunk'),
        ('decoder.appearance_head','VD.AD'),('decoder.mask_head','VD.MD'),('heads.detector','detector'),
        ('heads.adapter','adapter'),('objectives.','CMC.theta'),('flt.vocabulary','FLT.Wvoc'),
        ('flt.encoder_position','FLT.Pe'),('flt.decoder_position','FLT.Pd'),('flt.encoder','FLT.LE'),('flt.decoder','FLT.LD')) if n.startswith(p)),'intermediate')


def kind(n):
    if '.vim.q.weight' in n or '.vim.k.weight' in n:return 'singleton_structural_zero_when_connected'
    if any(s in n for s in ('.attention.k.bias','.self_attention.k.bias','.memory_attention.k.bias')):return 'ordinary_key_bias_analytic_zero'
    return 'generally_nonzero_or_fixture_dependent'


def scientific(x):
    if isinstance(x,dict):
        result={k:scientific(v) for k,v in x.items() if k not in ('elapsed_seconds','seconds')}
        if 'selected_loss_gradient_pair_cosines' in result:
            result['selected_loss_gradient_pair_cosines']=sorted(result['selected_loss_gradient_pair_cosines'],key=lambda r:(r['parameter'],tuple(r['losses'])))
        return result
    if isinstance(x,list):return [scientific(v) for v in x]
    return x


def historical(out):
    inputs={};parameters=[];module_rows=[];loss_rows=[];summary=[]
    for base in (ROOT/'artifacts/phase2d2/g2',ROOT/'artifacts/phase2d2_retry/g2'):
        for path in sorted(base.glob('*/gradient_additivity.json')):
            label=str(path.parent.relative_to(ROOT)).replace('\\','/');g=read(path);groups={}
            inputs[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
            total=read(path.parent/'total.json')
            for name,r in g['parameters'].items():
                stat=total['gradients']['parameters'][name]
                parameters.append({'source':label,'parameter':name,'module':group(name),'mathematical_class':kind(name),
                    **r,'outside_proportion':r['outside_tolerance_count']/r['elements'],
                    'joint_gradient_l2':stat['l2'],'joint_gradient_max_abs':stat['max_abs']})
                m=groups.setdefault(group(name),{'elements':0,'outside_tolerance_count':0,'parameter_tensors':0,
                    'max_abs_difference':0.,'maximum_parameter_relative_l2':None})
                m['elements']+=r['elements'];m['outside_tolerance_count']+=r['outside_tolerance_count'];m['parameter_tensors']+=1
                m['max_abs_difference']=max(m['max_abs_difference'],r['max_abs_difference'])
                if r['relative_l2_difference'] is not None:
                    m['maximum_parameter_relative_l2']=max(m['maximum_parameter_relative_l2'] or 0,r['relative_l2_difference'])
            for name,m in groups.items():
                module_rows.append({'source':label,'module':name,**m,
                    'outside_proportion':m['outside_tolerance_count']/m['elements'],
                    'joint_gradient_l2':total['gradients']['modules'][name]['l2'],
                    'module_error_relative_l2':'NOT RECOVERABLE from archived sufficient statistics'})
            for loss in ORDER:
                p=path.parent/(loss+'.json');r=read(p);inputs[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
                assert all(v.get('finite',True) for v in r['gradients']['parameters'].values())
                assert len(r['activations'])==19 and all(v['finite'] for v in r['activations'].values())
                for name,m in r['gradients']['modules'].items():loss_rows.append({'source':label,'loss':loss,'module':name,**m})
            summary.append({'source':label,'parameters':len(g['parameters']),
                'outside':sum(r['outside_tolerance_count'] for r in g['parameters'].values()),
                'relative_l2':g['relative_l2_difference'],'joint_l2':g['joint_l2'],
                'component_gradient_l2':g['component_gradient_l2'],
                'cancellation_ratio':g.get('cancellation_ratio_joint_to_sum_component_norms'),
                'all_19_stages_and_parameter_gradients_finite':True,
                'LE_final_max':total['activations']['LE11']['max_abs'],'LD_final_max':total['activations']['LD6']['max_abs']})
    same=[]
    for case in ('real','efs','local','real_efs','real_local','local_duplicate'):
        a=ROOT/'artifacts/phase2d2/g2'/('final_'+case);b=ROOT/'artifacts/phase2d2_retry/g2'/('final_'+case)
        checks={name:scientific(read(a/(name+'.json')))==scientific(read(b/(name+'.json'))) for name in (*ORDER,'gradient_additivity')}
        assert all(checks.values());same.append({'case':case,'exact_scientific_comparison':checks})
    csv_write(out/'historical_parameter_discrepancies.csv',parameters)
    csv_write(out/'historical_module_discrepancies.csv',module_rows)
    csv_write(out/'historical_module_loss_norms.csv',loss_rows)
    write(out/'historical_audit.json',{'scope':'all 14 preserved on-disk G2 directories, including original pilot and duplicate aliases',
        'sources':inputs,'records':summary,'original_vs_retry':same,'per_parameter_rows':len(parameters),
        'limitation':'Archived parameter reports omit error/reference squared norms and directions; full-module error norm/cosine cannot be reconstructed. New selected-vector metrics are separate.'})


def current(out):
    summaries=[];paramrows=[];modrows=[];vjprows=[]
    for run in ('real_mkldnn','real_efs_mkldnn','local_duplicate_mkldnn','real_efs_native'):
        p=out/'same_forward'/run;s=read(p/'same_graph.json');c=read(p/'separate_graphs.json');r=read(p/'summary.json')
        def exact(data):return all(v.get('exact_values_equal',v['presence_equal']) for v in data.values())
        isolated={n:exact(c[n]['C_i_vs_B_i']) for n in ORDER[:-1]}
        fields=['outside_tolerance_count','outside_tolerance_proportion','max_abs_difference','max_relative_nonzero_reference',
            'relative_l2','error_l2','actual_l2','reference_l2','cosine','sign_disagreements']
        for comparison,data in {'A_vs_B':s['A_vs_B'],**r['comparisons']}.items():
            for name,v in data.items():
                paramrows.append({'run':run,'comparison':comparison,'tensor':name,'class':kind(name),
                    'presence_equal':v['presence_equal'],**{k:v.get(k) for k in fields}})
        for comparison,data in {'A_vs_B':s['selected_module_A_vs_B'],**r['selected_module_comparisons']}.items():
            for name,v in data.items():modrows.append({'run':run,'comparison':comparison,'module':name,'scope':'selected parameters only',**v})
        if 'status' not in s['VJP']:
            for cut,data in s['VJP'].items():
                for name,v in data['comparison'].items():vjprows.append({'run':run,'cut':cut,'target':name,**{k:v.get(k) for k in fields}})
        summaries.append({'run':run,'backend':r['backend'],'resources':read(p/'resources.json'),
            'initial_state_sha256':r['initial_state_sha256'],'parameter_count':r['ownership']['total_unique'],
            'selected_parameters':len(s['graph']['parameters']),'selected_tensors_and_loss_nodes':len(s['A_vs_B']),
            'isolated_B_i_vs_C_i_exact':isolated,
            'B_sum_vs_C_sum_exact':exact(r['comparisons']['B_sum_vs_C_sum']),
            'A_vs_C_joint_exact':exact(r['comparisons']['A_vs_C_joint']),
            'historical_norm_statistic_match_count':{n:sum(v['exact_statistic_match'] for v in c[n]['historical_gradient_comparison'].values()) for n in ORDER},
            'unchanged_parameters':r['parameters_and_identities_unchanged'],
            'full_gradients_finite':all(v['all_full_parameter_gradients_finite'] for v in c.values()),
            'same_graph_released':s['graph_released_before_C']})
    csv_write(out/'selected_parameter_tensor_discrepancies.csv',paramrows)
    csv_write(out/'selected_module_discrepancies.csv',modrows);csv_write(out/'fixed_cotangent_vjp.csv',vjprows)
    chains={dtype:{'scientifically_exact_to_retry':scientific(read(out/'chain'/f'{dtype}.json'))==scientific(read(ROOT/'artifacts/phase2d2_retry/chain'/f'{dtype}.json')),
                   'status':read(out/'chain'/f'{dtype}.json')['status']} for dtype in ('float32','float64')}
    write(out/'diagnostic_summary.json',{'status':'BLOCKED','same_forward_matrix':summaries,'CMC':chains,
        'no_policy_applied':True,'new_pytest_run':'NOT RUN; diagnosis only; previous actual 313/321 preserved',
        'GPU':'NOT RUN; CPU-only Torch','CPU_B8':'NOT RUN; no capacity or acceptance claim',
        'directional_metrics':'Four seeded directions per tensor in raw JSON, no invented acceptance threshold'})


def extensions(out):
    def scan(value,path=''):
        result=[]
        if isinstance(value,dict):
            if 'allclose' in value and not value['allclose']:result.append({'path':path,'comparison':value})
            for k,v in value.items():result.extend(scan(v,path+'/'+k))
        elif isinstance(value,list):
            for i,v in enumerate(value):result.extend(scan(v,path+'/'+str(i)))
        return result
    files=('local_analytical.json','actual_LE08.json','decoder_vjp.json','chain/float32.json','chain/float64.json')
    report={p:scan(read(out/p)) for p in files}
    write(out/'reported_nonpassing_comparisons.json',{
        'scope':'All false allclose comparisons in local/CMC precision diagnostics. Cross-precision and different-cotangent comparisons are exploratory, not asserted identities. Not a pytest result.',
        'unchanged_thresholds':True,'Phase2D2':'BLOCKED','records':report})
    mismatch=[]
    for name in ('diagnostic_source_hashes.json','extension_source_hashes.json'):
        for p,h in read(out/name).items():
            if sha(ROOT/p)!=h:mismatch.append(p)
    assert not mismatch,mismatch
    write(out/'executed_tool_hash_verification.json',{'status':'PASS','mismatches':mismatch,
        'records':['diagnostic_source_hashes.json','extension_source_hashes.json'],
        'meaning':'Current numerical diagnostic sources match hashes captured before their execution.'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'artifacts/phase2d2_n3');parser.add_argument('--historical-only',action='store_true');args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);historical(out)
    if not args.historical_only:
        current(out)
        if (out/'decoder_vjp.json').exists():extensions(out)
    print('Historical and requested N3 evidence audit written; no acceptance policy applied')


if __name__=='__main__':main()
