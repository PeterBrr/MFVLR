"""HEAD-T01--HEAD-T10: standalone approval, no full integration root."""
import copy
import io
import json
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from mfvlr.heads import DetectionHeads,head_parameter_report
from mfvlr.heads_losses_config import load_heads_losses_config
from mfvlr.config import ConfigError
from .oracles import FP32,FP64,FD,central_difference

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('batch',[1,2,8])
def test_head_t01_shapes(batch):
    h=DetectionHeads();v=torch.randn(batch,1,512,generator=torch.Generator().manual_seed(29))
    result=h(v)
    assert result.detection_logits.shape==(batch,2)
    assert result.predicted_language.shape==(batch,1,512)
    assert all(x.isfinite().all() for x in (result.detection_logits,result.predicted_language))


def test_head_t02_independent_affine():
    h=DetectionHeads().double();v=torch.linspace(-1,1,1024,dtype=torch.float64).view(2,1,512)
    result=h(v)
    for layer,actual in ((h.detector,result.detection_logits),(h.adapter,result.predicted_language[:,0])):
        # Independent coordinate-wise multiplication/sum, not a second linear call.
        expected=(v[:,0,None,:]*layer.weight[None,:,:]).sum(-1)+layer.bias
        torch.testing.assert_close(actual,expected,**FP64)
    assert list(h.children())==[h.detector,h.adapter]
    assert all(type(m) is nn.Linear for m in h.children())
    with torch.no_grad():h.detector.weight.zero_();h.detector.bias.copy_(torch.tensor([-2.,3.]))
    assert torch.equal(h.detect(v),torch.tensor([[-2.,3.],[-2.,3.]],dtype=torch.float64))


def test_head_t03_class_and_probability_boundary():
    z=torch.tensor([[0.,0.],[-2.,2.]])
    p=DetectionHeads.probabilities(z)
    torch.testing.assert_close(p.sum(1),torch.ones(2),**FP32)
    assert torch.equal(DetectionHeads.fake_probability(z),p[:,1])
    assert torch.equal(DetectionHeads.predicted_class(z),torch.tensor([0,1]))
    assert DetectionHeads.class_order==('Real','Fake')
    assert torch.equal(z,torch.tensor([[0.,0.],[-2.,2.]]))


def test_head_t04_counts_and_ownership():
    h=DetectionHeads();r=head_parameter_report(h)
    assert r['detector']==1026 and r['adapter']==262656
    assert r['total_unique']==r['unique_storage_elements']==263682
    assert r['parameter_tensors']==4
    assert h.detector.weight.shape==(2,512) and h.adapter.weight.shape==(512,512)
    p=list(h.parameters());assert len({id(x) for x in p})==len({x.data_ptr() for x in p})==4
    assert list(h.state_dict())==['detector.weight','detector.bias','adapter.weight','adapter.bias']


def test_head_t05_initialization_rng_preservation():
    before=torch.random.get_rng_state().clone();h=DetectionHeads(seed=0)
    assert torch.equal(before,torch.random.get_rng_state())
    g=torch.Generator().manual_seed(0)
    for layer in (h.detector,h.adapter):
        expected=torch.empty_like(layer.weight)
        nn.init.xavier_uniform_(expected,gain=1.,generator=g)
        assert torch.equal(expected,layer.weight) and layer.bias.count_nonzero()==0
    repeat=DetectionHeads(seed=0);different=DetectionHeads(seed=1)
    assert all(torch.equal(a,b) for a,b in zip(h.parameters(),repeat.parameters()))
    assert not torch.equal(h.adapter.weight,different.adapter.weight)


@pytest.mark.parametrize('role',['detector','adapter'])
def test_head_t06_gradients_and_finite_difference(role):
    h=DetectionHeads().double();v=torch.linspace(-.2,.3,1024,dtype=torch.float64).view(2,1,512).requires_grad_()
    layer=getattr(h,role)
    call=h.detect if role=='detector' else h.align_language
    output=call(v);upstream=torch.linspace(.1,.4,output.numel(),dtype=torch.float64).reshape_as(output)
    probe=(output*upstream).sum();probe.backward()
    u=upstream.reshape(2,-1)
    torch.testing.assert_close(layer.weight.grad,u.T@v.detach()[:,0],**FP64)
    torch.testing.assert_close(layer.bias.grad,u.sum(0),**FP64)
    torch.testing.assert_close(v.grad[:,0],u@layer.weight.detach(),**FP64)
    other=h.adapter if role=='detector' else h.detector
    assert all(p.grad is None for p in other.parameters())
    f=lambda:(call(v)*upstream).sum()
    estimate=central_difference(f,layer.weight,(0,0))
    torch.testing.assert_close(layer.weight.grad[0,0],torch.tensor(estimate,dtype=torch.float64),**FD)


def test_head_t07_exact_fixed_batch_isolation():
    h=DetectionHeads();v=torch.randn(2,1,512,generator=torch.Generator().manual_seed(91)).requires_grad_()
    base=h(v);altered=v.detach().clone();altered[1]+=1
    changed=h(altered);perm=torch.tensor([1,0]);swapped=h(v[perm])
    for name in ('detection_logits','predicted_language'):
        assert torch.equal(getattr(base,name)[0],getattr(changed,name)[0])
        assert torch.equal(getattr(base,name),getattr(swapped,name)[perm])
    for sample in (0,1):
        probe=base.detection_logits[sample].square().sum()+base.predicted_language[sample].square().mean()
        grad=torch.autograd.grad(probe,v,retain_graph=sample==0)[0]
        assert grad[1-sample].count_nonzero()==0 and grad[sample].norm()>0


def test_head_t08_serialization_standalone_preservation():
    h=DetectionHeads();v=torch.ones(2,1,512)
    ids=[id(p) for p in h.parameters()];stream=io.BytesIO();torch.save(h.state_dict(),stream);stream.seek(0)
    restored=DetectionHeads(seed=1);restored.load_state_dict(torch.load(stream,weights_only=True))
    assert ids==[id(p) for p in h.parameters()]
    for a,b in zip(h.parameters(),restored.parameters()):assert torch.equal(a,b) and a is not b
    torch.testing.assert_close(h(v).detection_logits,restored(v).detection_logits,**FP32)
    torch.testing.assert_close(h(v).predicted_language,restored(v).predicted_language,**FP32)
    # Full-root identity checks remain in the accepted regressions / future 2D.2.
    assert not any(name.startswith(('mve','vd','flt')) for name in h.state_dict())


@pytest.mark.parametrize('seed',[True,-1,2**63,0.5,'0'])
def test_head_t09_invalid_seed(seed):
    with pytest.raises(ValueError):DetectionHeads(seed=seed)


@pytest.mark.parametrize('kind',['empty','tokens','width','integer','double','nan','inf','device'])
def test_head_t09_invalid_input(kind):
    h=DetectionHeads();v=torch.zeros(1,1,512)
    if kind=='empty':v=v[:0]
    if kind=='tokens':v=v.expand(1,2,512)
    if kind=='width':v=v[:,:,:511]
    if kind=='integer':v=v.long()
    if kind=='double':v=v.double()
    if kind=='nan':v[0,0,0]=float('nan')
    if kind=='inf':v[0,0,0]=float('inf')
    if kind=='device':v=torch.empty(1,1,512,device='meta')
    with pytest.raises((ValueError,TypeError,FloatingPointError)):h(v)


def test_head_t10_config_consistency(tmp_path):
    ref=load_heads_losses_config(ROOT/'configs/heads_losses_reference.yaml')
    dbg=load_heads_losses_config(ROOT/'configs/heads_losses_debug.yaml')
    for key in ('approval','architecture','initialization','objectives','runtime','tolerances'):assert ref.data[key]==dbg.data[key]
    assert dbg.data['diagnostics']['batch_size']==2
    assert head_parameter_report(DetectionHeads.from_config(ref))['total_unique']==263682
    data=ref.data;data['architecture']['embed_dim']=1;assert ref.data['architecture']['embed_dim']==512
    # Duplicate keys, non-maps, and closed-schema changes fail before construction.
    mutations=[]
    for path,value in [(['architecture','embed_dim'],True),(['architecture','class_order'],['Fake','Real']),
                       (['approval','decisions','P2D15'],'APPROVED'),(['approval','optimizer_authorized'],True),
                       (['objectives','weights','lr'],1/308),(['tolerances','fp32_atol'],5e-6),
                       (['objectives','ar','target_range'],[False,True])]:
        raw=copy.deepcopy(ref.data);node=raw
        for name in path[:-1]:node=node[name]
        node[path[-1]]=value;mutations.append(raw)
    raw=ref.data;raw['unknown']=1;mutations.append(raw)
    raw=ref.data;del raw['initialization'];mutations.append(raw)
    for i,raw in enumerate(mutations):
        path=tmp_path/f'bad{i}.yaml';path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ConfigError):load_heads_losses_config(path)
    duplicate=tmp_path/'duplicate.yaml';duplicate.write_text('phase: 2D.1\nphase: 2D.1\n')
    with pytest.raises(ConfigError):load_heads_losses_config(duplicate)
    # Relative base, exact full replacement diagnostics and one-level semantics.
    (tmp_path/'heads_losses_reference.yaml').write_text(yaml.safe_dump(ref.data))
    overlay=yaml.safe_load((ROOT/'configs/heads_losses_debug.yaml').read_text())
    target=tmp_path/'debug.yaml';target.write_text(yaml.safe_dump(overlay))
    assert load_heads_losses_config(target).data==dbg.data
    for mutation in ('architecture','partial','null','base','approval','nested'):
        raw=copy.deepcopy(overlay)
        if mutation=='architecture':raw['overrides']['architecture']={'embed_dim':1}
        if mutation=='partial':del raw['overrides']['diagnostics']['batch_size']
        if mutation=='null':raw['overrides']['diagnostics']=None
        if mutation=='base':raw['base']='../heads_losses_reference.yaml'
        if mutation=='approval':raw['approval']['scope']='full_training'
        if mutation=='nested':
            (tmp_path/'heads_losses_reference.yaml').write_text(yaml.safe_dump(overlay))
        target.write_text(yaml.safe_dump(raw))
        with pytest.raises(ConfigError):load_heads_losses_config(target)
