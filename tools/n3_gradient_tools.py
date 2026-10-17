"""N3 diagnostic storage/comparison only; no production or policy changes."""
from pathlib import Path
import hashlib
import math
import numpy as np
import torch

CHUNK=262144
LEGACY={'atol':1e-6,'rtol':1e-5}


def sha_tensor(x):
    return hashlib.sha256(memoryview(x.detach().contiguous().numpy()).cast('B')).hexdigest()


def identity(x):
    return {'python_id':id(x),'data_ptr':x.data_ptr(),'storage_ptr':x.untyped_storage().data_ptr(),
            'storage_offset':x.storage_offset(),'shape':list(x.shape),'stride':list(x.stride()),
            'dtype':str(x.dtype),'requires_grad':x.requires_grad,'is_leaf':x.is_leaf,
            'grad_fn_type':None if x.grad_fn is None else type(x.grad_fn).__name__}


class Store:
    """Disk-only selected gradients. At most a few chunks are FP64 in memory."""
    def __init__(self,path):self.path=Path(path);self.path.mkdir();self.records={}
    def save(self,key,value):
        if value is None:self.records[key]={'allocated':False};return
        x=value.detach().contiguous().view(-1)
        if not bool(x.isfinite().all()):raise FloatingPointError(key)
        file=self.path/(hashlib.sha256(key.encode()).hexdigest()+'.bin')
        h=hashlib.sha256();sq=0.;maximum=0.;nz=0
        with file.open('wb') as output:
            for start in range(0,x.numel(),CHUNK):
                chunk=x[start:start+CHUNK];raw=chunk.numpy().tobytes();output.write(raw);h.update(raw)
                sq+=chunk.double().square().sum().item();maximum=max(maximum,chunk.abs().max().item());nz+=int(chunk.count_nonzero())
        self.records[key]={'allocated':True,'shape':list(value.shape),'elements':x.numel(),
            'dtype':str(value.dtype),'finite':True,'l2':math.sqrt(sq),'l2_squared':sq,'max_abs':maximum,
            'nonzero':nz,'sha256':h.hexdigest(),'file':str(file)}
    def public(self):return {k:{n:v for n,v in r.items() if n!='file'} for k,r in self.records.items()}
    def sum(self,stores,keys):
        for key in keys:
            active=[s.records[key] for s in stores if s.records.get(key,{}).get('allocated')]
            if not active:self.save(key,None);continue
            readers=[open(r['file'],'rb') for r in active]
            file=self.path/(hashlib.sha256(key.encode()).hexdigest()+'.bin')
            sq=0.;maxabs=0.;nz=0;digest=hashlib.sha256()
            try:
                with file.open('wb') as target:
                    for start in range(0,active[0]['elements'],CHUNK):
                        count=min(CHUNK,active[0]['elements']-start);acc=np.zeros(count,dtype=np.float64)
                        for r,f in zip(active,readers):
                            dtype=np.float64 if r['dtype']=='torch.float64' else np.float32
                            acc+=np.frombuffer(f.read(count*np.dtype(dtype).itemsize),dtype=dtype)
                        raw=acc.tobytes();target.write(raw);digest.update(raw)
                        sq+=float(acc@acc);maxabs=max(maxabs,float(np.abs(acc).max()));nz+=int(np.count_nonzero(acc))
            finally:
                for f in readers:f.close()
            self.records[key]={'allocated':True,'shape':active[0]['shape'],'elements':active[0]['elements'],
                'dtype':'torch.float64','finite':True,'l2':math.sqrt(sq),'l2_squared':sq,'max_abs':maxabs,
                'nonzero':nz,'sha256':digest.hexdigest(),'file':str(file)}


def compare_stores(actual,reference,keys):
    result={}
    for key in keys:
        a=actual.records.get(key,{'allocated':False});b=reference.records.get(key,{'allocated':False})
        if not a['allocated'] or not b['allocated']:
            result[key]={'presence_equal':a['allocated']==b['allocated'],'actual_allocated':a['allocated'],'reference_allocated':b['allocated']};continue
        assert a['shape']==b['shape']
        n=a['elements'];error=dot=aa=bb=0.;maximum=relative=0.;outside=zeros=signs=0;exact=True
        dirs_a=np.zeros(4);dirs_b=np.zeros(4)
        generators=[np.random.default_rng(s) for s in (801,802,803,804)]
        with open(a['file'],'rb') as fa,open(b['file'],'rb') as fb:
            for start in range(0,n,CHUNK):
                count=min(CHUNK,n-start)
                def get(f,r):
                    dtype=np.float64 if r['dtype']=='torch.float64' else np.float32
                    return np.frombuffer(f.read(count*np.dtype(dtype).itemsize),dtype=dtype)
                ar=get(fa,a);br=get(fb,b);x=ar.astype(np.float64);y=br.astype(np.float64);d=x-y
                # Match the old FP32 assertion: one cast of accumulated reference.
                outside+=int((~torch.isclose(torch.from_numpy(ar.astype(np.float32)),torch.from_numpy(br.astype(np.float32)),**LEGACY)).sum())
                error+=float(d@d);dot+=float(x@y);aa+=float(x@x);bb+=float(y@y)
                maximum=max(maximum,float(np.abs(d).max()));mask=y!=0
                if mask.any():relative=max(relative,float((np.abs(d[mask])/np.abs(y[mask])).max()))
                zeros+=int(np.count_nonzero((~mask)&(d!=0)));signs+=int(np.count_nonzero(x*y<0));exact=exact and np.array_equal(x,y)
                for i,g in enumerate(generators):
                    direction=(g.integers(0,2,size=count,dtype=np.int8)*2-1).astype(np.float64)/math.sqrt(n)
                    dirs_a[i]+=float(x@direction);dirs_b[i]+=float(y@direction)
        cosine=dot/math.sqrt(aa*bb) if aa and bb else None
        result[key]={'presence_equal':True,'elements':n,'finite':True,'exact_values_equal':exact,
            'legacy_atol':LEGACY['atol'],'legacy_rtol':LEGACY['rtol'],'outside_tolerance_count':outside,
            'outside_tolerance_proportion':outside/n,'max_abs_difference':maximum,
            'max_relative_nonzero_reference':relative,'zero_reference_nonzero_difference_count':zeros,
            'error_l2':math.sqrt(error),'actual_l2':math.sqrt(aa),'reference_l2':math.sqrt(bb),
            'relative_l2':math.sqrt(error/bb) if bb else None,'cosine':cosine,
            'sign_disagreements':signs,'direction_seeds':[801,802,803,804],
            'directional_derivatives_actual':dirs_a.tolist(),'directional_derivatives_reference':dirs_b.tolist(),
            'directional_absolute_errors':np.abs(dirs_a-dirs_b).tolist(),
            'squared_error':error,'squared_actual':aa,'squared_reference':bb,'dot_product':dot}
    return result


def classification(key):
    if '.vim.q.weight' in key or '.vim.k.weight' in key:return 'singleton_structural_zero_when_connected'
    if '.attention.k.bias' in key or '.self_attention.k.bias' in key or '.memory_attention.k.bias' in key:return 'ordinary_key_bias_analytic_zero'
    return 'generally_nonzero_or_fixture_dependent'


def modules(comparisons):
    groups={}
    for key,r in comparisons.items():
        if 'squared_error' not in r:continue
        if key.startswith('parameter.'):
            name=key.removeprefix('parameter.')
            group=next((g for p,g in [('mve.shared_encoder.cnn','MVE.local'),('mve.','MVE.global'),('decoder.trunk','VD.trunk'),('decoder.appearance','VD.AD'),('decoder.mask','VD.MD'),('heads.detector','detector'),('heads.adapter','adapter'),('flt.vocabulary','FLT.Wvoc'),('flt.encoder','FLT.LE'),('flt.decoder','FLT.LD'),('objectives.','CMC')] if name.startswith(p)),'other')
        else:group='intermediates_and_loss_nodes'
        g=groups.setdefault(group,{'selected_tensors':0,'elements':0,'outside_tolerance_count':0,'squared_error':0.,'squared_reference':0.,'squared_actual':0.,'dot_product':0.,'max_abs_difference':0.})
        g['selected_tensors']+=1
        for k in ('elements','outside_tolerance_count','squared_error','squared_reference','squared_actual','dot_product'):g[k]+=r[k]
        g['max_abs_difference']=max(g['max_abs_difference'],r['max_abs_difference'])
    for g in groups.values():
        g['relative_l2']=math.sqrt(g['squared_error']/g['squared_reference']) if g['squared_reference'] else None
        g['cosine']=g['dot_product']/math.sqrt(g['squared_actual']*g['squared_reference']) if g['squared_actual']*g['squared_reference'] else None
    return groups


def selftest(directory):
    # Independent exact vectors cross a chunk boundary and include None/zero.
    s=[Store(Path(directory)/str(i)) for i in range(4)]
    x=torch.arange(CHUNK+9,dtype=torch.float32)%17
    for store,m in zip(s[:2],(1.,2.)):
        store.save('x',x*m);store.save('unused',None);store.save('zero',torch.zeros(512))
    s[2].sum(s[:2],('x','unused','zero'));s[3].save('x',x*3);s[3].save('unused',None);s[3].save('zero',torch.zeros(512))
    r=compare_stores(s[3],s[2],('x','unused','zero'))
    assert r['x']['outside_tolerance_count']==0 and r['x']['error_l2']==0
    assert r['zero']['cosine'] is None and r['unused']['presence_equal']
    s[3].save('x',x*3+1)
    r2=compare_stores(s[3],s[2],('x',))['x']
    assert r2['outside_tolerance_count']==x.numel() and r2['max_abs_difference']==1
    return {'exact_sum':'PASS','chunk_boundary':'PASS','None_and_zero':'PASS','injected_error_detection':'PASS','error_l2':r2['error_l2']}
