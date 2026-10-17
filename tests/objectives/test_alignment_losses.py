"""LOSS-CMC01--07 and LOSS-KL01--06: independent FP64 algebra/derivatives."""
import copy
import io
import math

import pytest
import torch

from mfvlr.objectives import CrossModalContrastiveLoss,knowledge_distillation_loss as kl
from .oracles import FIXTURES,FP64,FP32,FD,cmc_fixture,cmc_reference,kl_fixture,kl_reference,central_difference


def test_loss_cmc01_nonsymmetric_reference():
    module=CrossModalContrastiveLoss().double();v,l=cmc_fixture(scale=1.)
    expected,scores,*_=cmc_reference(v.detach()[:,0].tolist(),l.detach()[:,0].tolist(),module.log_temperature.item())
    actual=module(v,l)
    torch.testing.assert_close(actual,torch.tensor(expected,dtype=torch.float64),**FP64)
    torch.testing.assert_close(module.scores(v,l)*module.temperature,
                               torch.tensor(FIXTURES['cmc']['raw_dot'],dtype=torch.float64),**FP64)
    assert actual.ndim==0 and actual.isfinite()


def test_loss_cmc02_analytical_and_finite_difference():
    module=CrossModalContrastiveLoss().double();v,l=cmc_fixture(scale=.1)
    expected,scores,gv,gl,gt=cmc_reference(v.detach()[:,0].tolist(),l.detach()[:,0].tolist(),module.log_temperature.item())
    module(v,l).backward()
    torch.testing.assert_close(v.grad[:,0],torch.tensor(gv,dtype=torch.float64),**FP64)
    torch.testing.assert_close(l.grad[:,0],torch.tensor(gl,dtype=torch.float64),**FP64)
    torch.testing.assert_close(module.log_temperature.grad,torch.tensor(gt,dtype=torch.float64),**FP64)
    assert v.grad.norm()>0 and l.grad.norm()>0 and module.log_temperature.grad.abs()>0
    numerical=central_difference(lambda:module(v,l),module.log_temperature,())
    torch.testing.assert_close(module.log_temperature.grad,torch.tensor(numerical,dtype=torch.float64),**FD)
    for tensor,index in ((v,(0,0,0)),(l,(1,0,1))):
        numerical=central_difference(lambda:module(v,l),tensor,index)
        torch.testing.assert_close(tensor.grad[index],torch.tensor(numerical,dtype=torch.float64),**FD)


def test_loss_cmc03_temperature_count_scale_serialization():
    module=CrossModalContrastiveLoss();assert len(list(module.parameters()))==1
    assert module.log_temperature.ndim==0 and module.log_temperature.numel()==1
    expected=torch.tensor(math.log(.07),dtype=torch.float32)
    assert torch.equal(module.log_temperature,expected)
    torch.testing.assert_close(module.temperature,torch.tensor(.07),**FP32)
    reference=copy.deepcopy(module).double()
    assert torch.equal(reference.log_temperature.float(),module.log_temperature)
    torch.testing.assert_close(reference.temperature,torch.tensor(math.exp(expected.item()),dtype=torch.float64),**FP64)
    v,l=cmc_fixture(dtype=torch.float32)
    torch.testing.assert_close(module.scores(v*2,l),module.scores(v,l)*2,**FP32)
    stream=io.BytesIO();torch.save(module.state_dict(),stream);stream.seek(0)
    restored=CrossModalContrastiveLoss();restored.load_state_dict(torch.load(stream,weights_only=True))
    assert torch.equal(module(v,l),restored(v,l))


def test_loss_cmc04_singleton_and_duplicates():
    module=CrossModalContrastiveLoss().double();v,l=cmc_fixture(batch=1)
    value=module(v,l);assert value.item()==0 and value.requires_grad
    value.backward()
    for tensor in (v,l,module.log_temperature):
        assert tensor.grad is not None and tensor.grad.count_nonzero()==0
    # Equal duplicate rows still have off-diagonal negatives, without deduplication.
    v=torch.ones(2,1,512,dtype=torch.float64,requires_grad=True)
    l=torch.ones_like(v,requires_grad=True)
    torch.testing.assert_close(module(v,l),torch.tensor(math.log(2),dtype=torch.float64),**FP64)


def test_loss_cmc05_permutation_and_intentional_batch_coupling():
    module=CrossModalContrastiveLoss().double();v,l=cmc_fixture()
    original=module(v,l);perm=torch.tensor([1,0])
    torch.testing.assert_close(module(v[perm],l[perm]),original,**FP64)
    grad=torch.autograd.grad(original,v)[0]
    changed=l.detach().clone();changed[1,0,0]+=.3
    altered=module(v,changed);changed_grad=torch.autograd.grad(altered,v)[0]
    assert not torch.equal(grad[0],changed_grad[0])


@pytest.mark.parametrize('kind',['finite_extreme','tau_underflow','tau_overflow','theta_nan','theta_inf','raw_overflow','score_overflow','batch','width','dtype','device'])
def test_loss_cmc06_stability_and_rejections(kind):
    module=CrossModalContrastiveLoss();v,l=cmc_fixture(dtype=torch.float32)
    if kind=='finite_extreme':
        with torch.no_grad():v.mul_(100);l.mul_(100)
        value=module(v,l);value.backward()
        assert value.isfinite() and v.grad.isfinite().all() and l.grad.isfinite().all();return
    if kind in ('tau_underflow','tau_overflow','theta_nan','theta_inf','score_overflow'):
        values={'tau_underflow':-1000.,'tau_overflow':1000.,'theta_nan':float('nan'),'theta_inf':float('inf'),'score_overflow':-90.}
        with torch.no_grad():module.log_temperature.fill_(values[kind])
        if kind=='score_overflow':v=torch.ones_like(v);l=torch.ones_like(l)
    if kind=='raw_overflow':v=torch.full_like(v,1e30);l=torch.full_like(l,1e30)
    if kind=='batch':l=l[:1]
    if kind=='width':v=v[:,:,:511]
    if kind=='dtype':l=l.double()
    if kind=='device':l=l.to('meta')
    with pytest.raises((FloatingPointError,ValueError,TypeError)):module(v,l)


def test_loss_cmc07_batch_negatives_not_microbatch_equivalent():
    module=CrossModalContrastiveLoss().double();v,l=cmc_fixture()
    batch=module(v,l);singles=sum(module(v[i:i+1],l[i:i+1]) for i in range(2))/2
    assert singles.item()==0 and batch.item()>0
    assert list(module.state_dict())==['log_temperature']


@pytest.mark.parametrize('batch',[1,2])
def test_loss_kl01_probability_oracle(batch):
    text,adapted=kl_fixture(batch)
    expected,*_=kl_reference(text.detach()[:,0].tolist(),adapted.detach()[:,0].tolist())
    hand=.8*math.log(.8/.3)+.2*math.log(.2/.7)
    torch.testing.assert_close(kl(text,adapted),torch.tensor(expected,dtype=torch.float64),**FP64)
    torch.testing.assert_close(kl(text,adapted),torch.tensor(hand,dtype=torch.float64),**FP64)


def test_loss_kl02_direction_identity_and_reduction():
    text,adapted=kl_fixture();equal=kl(text,text)
    assert equal.item()==0
    grad=torch.autograd.grad(equal,text)[0]
    torch.testing.assert_close(grad,torch.zeros_like(grad),**FP64)
    actual=kl(text,adapted)
    assert not torch.isclose(actual,kl(adapted,text),**FP64)
    assert not torch.isclose(actual,actual/512,**FP64)
    assert not torch.isclose(actual,actual*.25,**FP64)


def test_loss_kl03_both_gradients_and_finite_difference():
    text,adapted=kl_fixture(batch=2)
    expected,gt,gs,*_=kl_reference(text.detach()[:,0].tolist(),adapted.detach()[:,0].tolist())
    kl(text,adapted).backward()
    torch.testing.assert_close(text.grad[:,0],torch.tensor(gt,dtype=torch.float64),**FP64)
    torch.testing.assert_close(adapted.grad[:,0],torch.tensor(gs,dtype=torch.float64),**FP64)
    assert text.grad.norm()>0 and adapted.grad.norm()>0
    for tensor,index in ((text,(0,0,0)),(adapted,(1,0,1))):
        estimate=central_difference(lambda:kl(text,adapted),tensor,index)
        torch.testing.assert_close(tensor.grad[index],torch.tensor(estimate,dtype=torch.float64),**FD)
    detached=text.detach();student=adapted.detach().clone().requires_grad_()
    kl(detached,student).backward();assert student.grad.norm()>0 and detached.grad is None


def test_loss_kl04_axis_and_temperature():
    text,adapted=kl_fixture()
    expected,*_=kl_reference(text.detach()[:,0].tolist(),adapted.detach()[:,0].tolist())
    actual=kl(text,adapted)
    assert actual>0 # singleton-axis softmax would have given zero
    _,_,_,p,q=kl_reference(text.detach()[:,0].tolist(),adapted.detach()[:,0].tolist())
    assert len(p[0])==len(q[0])==512
    assert math.isclose(math.fsum(p[0]),1,abs_tol=1e-12)
    # Scaling features by .25 reproduces multiply-by-.5 rather than divide-by-.5.
    assert not torch.isclose(actual,kl(text*.25,adapted*.25),**FP64)


def test_loss_kl05_extreme_finite():
    text=torch.zeros(1,1,512);text[0,0,0]=5000
    adapted=torch.zeros_like(text);adapted[0,0,1]=4000
    text.requires_grad_();adapted.requires_grad_()
    value=kl(text,adapted);assert value.item()==8000
    value.backward();assert text.grad.isfinite().all() and adapted.grad.isfinite().all()
    assert adapted.grad.count_nonzero()==2


@pytest.mark.parametrize('kind',['batch','dim','dtype','nan','inf','device','integer','tokens'])
def test_loss_kl06_invalid(kind):
    text,adapted=kl_fixture(batch=2)
    if kind=='batch':adapted=adapted[:1]
    if kind=='dim':adapted=adapted[:,:,:511]
    if kind=='dtype':adapted=adapted.float()
    if kind in ('nan','inf'):
        adapted=adapted.detach().clone();adapted[0,0,0]=float(kind)
    if kind=='device':adapted=adapted.to('meta')
    if kind=='integer':text=text.long()
    if kind=='tokens':adapted=adapted.expand(2,2,512)
    with pytest.raises((ValueError,TypeError,FloatingPointError)):kl(text,adapted)
