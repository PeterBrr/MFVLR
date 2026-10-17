"""Complete streamed gradient diagnostics; legacy acceptance result untouched."""
from pathlib import Path
import hashlib
import math
import numpy as np
import torch
from tools.phase2d2_diagnostics import GradientStream,FP32,write
from tools.audit_phase2d2_n3 import group,kind


def derived(r):
    r['gradient_l2']=math.sqrt(r['actual_squared']);r['reference_l2']=math.sqrt(r['reference_squared'])
    r['error_l2']=math.sqrt(r['error_squared'])
    r['relative_l2']=r['error_l2']/r['reference_l2'] if r['reference_l2'] else None
    r['cosine']=r['dot']/math.sqrt(r['actual_squared']*r['reference_squared']) if r['actual_squared']*r['reference_squared'] else None
    r['outside_proportion']=r['outside_tolerance_count']/r['elements']
    for key in ('direction_actual_raw','direction_reference_raw'):
        r[key.removesuffix('_raw')]=[v/math.sqrt(r['elements']) for v in r[key]]
    r['direction_absolute_error']=[abs(a-b) for a,b in zip(r['direction_actual'],r['direction_reference'])]
    return r


def empty():
    return {'elements':0,'outside_tolerance_count':0,'max_abs_error':0.,'max_relative_nonzero_reference':0.,
        'actual_squared':0.,'reference_squared':0.,'error_squared':0.,'dot':0.,'reference_zero_coordinates':0,
        'reference_zero_actual_nonzero':0,'sign_disagreements':0,'direction_actual_raw':[0.]*4,'direction_reference_raw':[0.]*4}


def measure(actual,read_reference,name):
    r=empty();r['elements']=actual.numel();flat=actual.detach().reshape(-1)
    salt=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
    gens=[np.random.default_rng(seed+salt) for seed in (5101,5102,5103,5104)]
    for start in range(0,flat.numel(),262144):
        count=min(262144,flat.numel()-start);a=flat[start:start+count].double().numpy();b=read_reference(start,count);d=a-b
        r['outside_tolerance_count']+=int((~torch.isclose(torch.from_numpy(a.astype(np.float32)),torch.from_numpy(b.astype(np.float32)),**FP32)).sum())
        r['max_abs_error']=max(r['max_abs_error'],float(abs(d).max()));nz=b!=0
        if nz.any():r['max_relative_nonzero_reference']=max(r['max_relative_nonzero_reference'],float((abs(d[nz])/abs(b[nz])).max()))
        for key,value in (('actual_squared',a@a),('reference_squared',b@b),('error_squared',d@d),('dot',a@b)):r[key]+=float(value)
        r['reference_zero_coordinates']+=int((~nz).sum());r['reference_zero_actual_nonzero']+=int(((~nz)&(a!=0)).sum())
        r['sign_disagreements']+=int((a*b<0).sum())
        for i,g in enumerate(gens):
            direction=(g.integers(0,2,size=count,dtype=np.int8)*2-1).astype(np.float64)
            r['direction_actual_raw'][i]+=float(a@direction);r['direction_reference_raw'][i]+=float(b@direction)
    return derived(r)


class CompleteGradientStream(GradientStream):
    def __init__(self,root,directory):
        super().__init__(root,directory);self.directory=Path(directory)
    def compare_joint(self):
        original=super().compare_joint();parameters={};groups={}
        for name,p,offset in self.entries:
            if p.grad is None:
                parameters[name]={'allocated':False};continue
            def reference(start,count):
                self.file.seek(offset+start*8)
                return np.frombuffer(self.file.read(count*8),dtype=np.float64).copy()
            r=measure(p.grad,reference,name)
            assert r['outside_tolerance_count']==original['parameters'][name]['outside_tolerance_count']
            r.update(allocated=True,mathematical_class=kind(name),group=group(name),finite=True)
            parameters[name]=r;g=groups.setdefault(group(name),empty())
            for key in ('elements','outside_tolerance_count','actual_squared','reference_squared','error_squared','dot','reference_zero_coordinates','reference_zero_actual_nonzero','sign_disagreements'):g[key]+=r[key]
            for key in ('max_abs_error','max_relative_nonzero_reference'):g[key]=max(g[key],r[key])
            for key in ('direction_actual_raw','direction_reference_raw'):g[key]=[a+b for a,b in zip(g[key],r[key])]
        for g in groups.values():derived(g)
        write(self.directory/'complete_discrepancies.json',{'scope':'all root parameters, not selected subset','parameters':parameters,'groups':groups,
            'parameter_tensors':len(parameters),'unique_elements':sum(p.numel() for _,p,_ in self.entries),
            'direction_seeds':[5101,5102,5103,5104],'direction_salt':'little-endian first four bytes SHA256(parameter name)',
            'directions':'Per-parameter normalized Rademacher; module concatenation normalized by module element count',
            'elementwise_policy':FP32,'legacy_gate_unchanged':True,'acceptance':'BLOCKED'})
        return original


def selftest():
    # Independent known derivative pattern; verify meaningful errors, not just JSON loading.
    x=torch.arange(262153,dtype=torch.float32)%11
    def ref(start,count):return (3*x[start:start+count]).double().numpy()
    clean=measure(3*x,ref,'synthetic');bad=measure(3*x+1,ref,'synthetic')
    assert clean['error_l2']==0 and clean['outside_tolerance_count']==0
    assert bad['outside_tolerance_count']==x.numel() and bad['max_abs_error']==1
    assert abs(bad['error_l2']-math.sqrt(x.numel()))<1e-10
    zero=measure(torch.zeros(512),lambda s,n:np.zeros(n),'zero')
    assert zero['reference_zero_coordinates']==512 and zero['relative_l2'] is None
    return {'exact_reference':'PASS','injected_plus_one':'DETECTED','zero_reference':'PASS','chunk_boundary':'PASS'}
