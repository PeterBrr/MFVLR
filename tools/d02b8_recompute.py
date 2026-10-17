"""Independent disk-only analysis. No production stream, torch, or autograd."""
import hashlib,itertools,json,math,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/phase2d2_d02b8'
A=ROOT/'artifacts/phase2d2_d02b7_r2/attempt_01'
ORDER=('fd','lr','cmc','fl','ar','kl');PASSES=ORDER+('total',)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(name,x):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(n):
    return 'MVE' if n.startswith('mve.') else 'FLT' if n.startswith('flt.') else 'VD' if n.startswith('decoder.') else 'heads' if n.startswith('heads.') else 'objectives'
def main():
    started=time.perf_counter();frozen=read(A/'additivity.json')['parameters']
    shards=read(A.parent/'excluded_binary_inventory.json')['files'];integrity=[]
    for rel,v in shards.items():
        p=ROOT/rel;actual=digest(p);ok=actual==v['sha256'] and p.stat().st_size==v['bytes']
        integrity.append(dict(path=rel,sha256=actual,bytes=p.stat().st_size,match=ok))
    write('shard_integrity.json',dict(files=integrity,count=len(integrity),bytes=sum(v['bytes'] for v in integrity)))
    assert len(integrity)==2392 and sum(v['bytes'] for v in integrity)==3790126768 and all(v['match'] for v in integrity)
    nodes=ET.parse(A.parent/'live_additivity.junit.xml').getroot().findall('testcase')
    junit={n.attrib['name']:n.find('failure') is not None for n in nodes};assert len(nodes)==len(junit)==508
    results={};reconciliation=[];mods={};lossnorm={k:0. for k in PASSES}
    for index,(name,old) in enumerate(frozen.items()):
        count=old['elements'];arrays=[]
        for loss in PASSES:
            v=old['passes'][loss]
            arrays.append(np.memmap(A/'gradient_stream'/v['path'],mode='r',dtype='<f4',shape=(count,)) if v['allocated'] else None)
        saved=np.memmap(A/'gradient_stream'/old['sum_path'],mode='r',dtype='<f8',shape=(count,))
        sum_exact=True;outside=sign=zero=zero_diff=near=near_fail=0;maxabs=0.;actual2=ref2=err2=dot=0.
        norms=np.zeros(7);pairs=np.zeros((6,6));da=np.zeros(4);dr=np.zeros(4)
        salt=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
        rng=[np.random.default_rng(seed+salt) for seed in (5101,5102,5103,5104)]
        for start in range(0,count,262144):
            size=min(262144,count-start)
            x=[np.zeros(size,dtype=np.float64) if a is None else np.asarray(a[start:start+size],dtype=np.float64) for a in arrays]
            reference=np.zeros(size,dtype=np.float64)
            for value in x[:6]:np.add(reference,value,out=reference)
            sum_exact &= bool(np.array_equal(reference,saved[start:start+size]))
            actual=x[6];error=actual-reference
            rounded=reference.astype(np.float32);joint=actual.astype(np.float32)
            fail=np.abs(joint-rounded)>np.float32(1e-6)+np.float32(1e-5)*np.abs(rounded)
            outside+=int(fail.sum());sign+=int((actual*reference<0).sum());z=reference==0
            zero+=int(z.sum());zero_diff+=int((z & (actual!=0)).sum())
            # Reporting bin only, not a new acceptance threshold.
            nz=(np.abs(reference)<=1e-6)&~z;near+=int(nz.sum());near_fail+=int((nz&fail).sum())
            maxabs=max(maxabs,float(np.max(np.abs(error))))
            actual2+=float(actual@actual);ref2+=float(reference@reference);err2+=float(error@error);dot+=float(actual@reference)
            for i,value in enumerate(x):norms[i]+=float(value@value)
            for i,j in itertools.combinations(range(6),2):pairs[i,j]+=float(x[i]@x[j])
            for i,g in enumerate(rng):
                direction=(g.integers(0,2,size=size,dtype=np.int8)*2-1).astype(np.float64)
                da[i]+=float(actual@direction);dr[i]+=float(reference@direction)
        norm=np.sqrt(norms);rn=math.sqrt(ref2);an=math.sqrt(actual2)
        pair_report={ORDER[i]+':'+ORDER[j]:dict(dot=float(pairs[i,j]),cosine=float(pairs[i,j]/(norm[i]*norm[j])) if norm[i]*norm[j] else None) for i,j in itertools.combinations(range(6),2)}
        r=dict(module=module(name),category=old['mathematical_class'],elements=count,outside_tolerance_count=outside,outside_proportion=outside/count,
            error_l2=math.sqrt(err2),relative_l2=math.sqrt(err2)/rn if rn else None,max_abs_error=maxabs,reference_l2=rn,gradient_l2=an,
            cosine=dot/(rn*an) if rn*an else None,cancellation_ratio=float(norm[:6].sum()/rn) if rn else None,
            reference_zero_coordinates=zero,reference_zero_actual_nonzero=zero_diff,sign_disagreements=sign,
            near_nonzero_reference_le_1e_6=near,near_nonzero_failed=near_fail,
            loss_norms={k:float(norm[i]) for i,k in enumerate(PASSES)},presence={k:old['passes'][k]['allocated'] for k in PASSES},
            loss_pairs=pair_report,direction_actual=(da/math.sqrt(count)).tolist(),direction_reference=(dr/math.sqrt(count)).tolist(),ordered_sum_shard_exact=sum_exact)
        mismatches={}
        for k in ('outside_tolerance_count','reference_zero_coordinates','reference_zero_actual_nonzero','sign_disagreements','max_abs_error','error_l2','relative_l2','reference_l2','gradient_l2','cosine','cancellation_ratio','direction_actual','direction_reference'):
            if r[k]!=old[k]:mismatches[k]=dict(original=old[k],recomputed=r[k])
        for k,v in pair_report.items():
            for metric in ('dot','cosine'):
                if v[metric]!=old['loss_pairs'][k][metric]:mismatches[f'{k}.{metric}']=dict(original=old['loss_pairs'][k][metric],recomputed=v[metric])
        assert sum_exact and (outside>0)==junit[name]
        if '.vim.q.' in name or '.vim.k.' in name:assert all(v==0 for v in norm)
        reconciliation.append(dict(parameter=name,ordered_sum_exact=sum_exact,junit_failure_matches=(outside>0)==junit[name],metrics_exact=not mismatches,mismatches=mismatches))
        results[name]=r
        m=mods.setdefault(r['module'],dict(parameters=0,failed=0,elements=0,outside=0,error_squared=0.,reference_squared=0.,loss_squared={k:0. for k in PASSES}))
        m['parameters']+=1;m['failed']+=outside>0;m['elements']+=count;m['outside']+=outside;m['error_squared']+=err2;m['reference_squared']+=ref2
        for i,k in enumerate(PASSES):m['loss_squared'][k]+=float(norms[i]);lossnorm[k]+=float(norms[i])
        del saved,arrays
        if index%50==0:print(f'{index+1}/508 raw parameters reconciled',flush=True)
    write('parameter_forensics.json',results);write('reconciliation.json',reconciliation)
    write('module_rankings.json',sorted([dict(module=k,**v,error_l2=math.sqrt(v['error_squared']),relative_l2=math.sqrt(v['error_squared']/v['reference_squared']) if v['reference_squared'] else None) for k,v in mods.items()],key=lambda v:v['outside'],reverse=True))
    write('loss_rankings.json',sorted([dict(loss=k,global_gradient_l2=math.sqrt(v)) for k,v in lossnorm.items()],key=lambda v:v['global_gradient_l2'],reverse=True))
    write('failed_parameter_rankings.json',[dict(parameter=k,**v) for k,v in sorted(results.items(),key=lambda kv:kv[1]['outside_tolerance_count'],reverse=True) if v['outside_tolerance_count']])
    write('summary.json',dict(parameters=len(results),failed_parameters=sum(v['outside_tolerance_count']>0 for v in results.values()),outside=sum(v['outside_tolerance_count'] for v in results.values()),sum_shards_exact=all(v['ordered_sum_shard_exact'] for v in results.values()),metrics_exact=sum(r['metrics_exact'] for r in reconciliation),seconds=time.perf_counter()-started,numpy=np.__version__,no_model_execution=True))
if __name__=='__main__':main()
