"""LOSS-FD01--04, FL01--05, AR01--04: independently controlled objectives."""
import math

import pytest
import torch

from mfvlr.objectives import forgery_detection_loss as fd
from mfvlr.objectives import forgery_localization_loss as fl
from mfvlr.objectives import appearance_reconstruction_loss as ar
from mfvlr.objectives import validate_mask_semantics
from .oracles import FIXTURES,FP64,FP32,FD,categorical,fd_gradient,central_difference


@pytest.mark.parametrize('batch',[1,2])
def test_loss_fd01_reference_uniform(batch):
    z=torch.tensor(FIXTURES['fd']['logit_rows'][:batch],dtype=torch.float64)
    y=torch.tensor(FIXTURES['fd']['labels'][:batch])
    expected=categorical(z.tolist(),y.tolist())
    torch.testing.assert_close(fd(z,y),torch.tensor(expected,dtype=torch.float64),**FP64)
    torch.testing.assert_close(fd(torch.zeros_like(z),y),torch.tensor(FIXTURES['uniform']['fd'],dtype=torch.float64),**FP64)
    assert fd(z,y).ndim==0


def test_loss_fd02_analytical_and_finite_difference():
    z=torch.tensor(FIXTURES['fd']['logit_rows'],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([0,1]);value=fd(z,y);value.backward()
    expected=torch.tensor(fd_gradient(z.detach().tolist(),y.tolist()),dtype=torch.float64)
    torch.testing.assert_close(z.grad,expected,**FP64)
    torch.testing.assert_close(z.grad.sum(1),torch.zeros(2,dtype=torch.float64),**FP64)
    numerical=central_difference(lambda:fd(z,y),z,(1,0))
    torch.testing.assert_close(z.grad[1,0],torch.tensor(numerical,dtype=torch.float64),**FD)


def test_loss_fd03_offsets_and_extremes():
    z=torch.tensor([[-2.,1.],[3.,-4.]],dtype=torch.float64);y=torch.tensor([0,1])
    torch.testing.assert_close(fd(z+17,y),fd(z,y),**FP64)
    extreme=torch.tensor([[10000.,-10000.]],requires_grad=True)
    value=fd(extreme,torch.tensor([1]));assert value.item()==20000
    value.backward();assert torch.equal(extreme.grad,torch.tensor([[1.,-1.]]))


@pytest.mark.parametrize('kind',['float','negative','large','length','rank','nan','inf','empty','device'])
def test_loss_fd04_rejections(kind):
    z=torch.zeros(2,2);y=torch.tensor([0,1])
    if kind=='float':y=y.float()
    if kind=='negative':y[0]=-1
    if kind=='large':y[0]=2
    if kind=='length':y=y[:1]
    if kind=='rank':z=z[:,:,None]
    if kind=='nan':z[0,0]=float('nan')
    if kind=='inf':z[0,0]=float('inf')
    if kind=='empty':z=z[:0];y=y[:0]
    if kind=='device':y=y.to('meta')
    with pytest.raises((TypeError,ValueError,FloatingPointError)):fd(z,y)


@pytest.mark.parametrize('target',[0,1])
def test_loss_fl01_constant_masks(target):
    z=torch.zeros(1,2,224,224,dtype=torch.float64);z[:,0]=math.log(2)
    m=torch.full((1,224,224),target,dtype=torch.int64)
    expected=-math.log((2 if target==0 else 1)/3)
    torch.testing.assert_close(fl(z,m),torch.tensor(expected,dtype=torch.float64),**FP64)
    # Both foreground mistakes on real pixels and background mistakes on EFS penalized.
    assert fl(z,m)>0


def test_loss_fl02_mixed_onehot_reference():
    m=torch.zeros(2,224,224,dtype=torch.int64)
    m[0,64:160,80:176]=1
    yy,xx=torch.meshgrid(torch.arange(224),torch.arange(224),indexing='ij');m[1]=(yy+xx)%2
    z=torch.zeros(2,2,224,224,dtype=torch.float64);z[:,0]=math.log(2)
    n_foreground=int(m.sum());n=m.numel()
    expected=((n-n_foreground)*math.log(1.5)+n_foreground*math.log(3))/n
    torch.testing.assert_close(fl(z,m),torch.tensor(expected,dtype=torch.float64),**FP64)
    torch.testing.assert_close(fl(torch.zeros_like(z),m),torch.tensor(math.log(2),dtype=torch.float64),**FP64)


def test_loss_fl03_gradients_and_spatial_permutation():
    z=torch.zeros(1,2,224,224,dtype=torch.float64,requires_grad=True)
    m=torch.zeros(1,224,224,dtype=torch.int64);m[:,20:180,40:140]=1
    value=fl(z,m);value.backward()
    expected=torch.empty_like(z);expected[:,0]=(m.double()-.5)/m.numel();expected[:,1]=-expected[:,0]
    torch.testing.assert_close(z.grad,expected,**FP64)
    assert torch.equal(z.grad.sum(1),torch.zeros_like(z.grad[:,0]))
    torch.testing.assert_close(fl(z.detach().flip(-1),m.flip(-1)),value.detach(),**FP64)
    numerical=central_difference(lambda:fl(z,m),z,(0,0,0,0))
    torch.testing.assert_close(z.grad[0,0,0,0],torch.tensor(numerical,dtype=torch.float64),**FD)


@pytest.mark.parametrize('kind',['float','fractional','ignore','byte255','channel','resolution','nan','inf','device','missing'])
def test_loss_fl04_rejections(kind):
    z=torch.zeros(1,2,224,224);m=torch.zeros(1,224,224,dtype=torch.int64)
    if kind=='float':m=m.float()
    if kind=='fractional':m=m.float()+.5
    if kind=='ignore':m[0,0,0]=-100
    if kind=='byte255':m[0,0,0]=255
    if kind=='channel':z=z[:,:1]
    if kind=='resolution':m=m[:,:,:223]
    if kind=='nan':z[0,0,0,0]=float('nan')
    if kind=='inf':z[0,0,0,0]=float('inf')
    if kind=='device':m=m.to('meta')
    if kind=='missing':m=None
    with pytest.raises((TypeError,ValueError,FloatingPointError)):fl(z,m)


def test_loss_fl05_supplied_mask_semantics():
    y=torch.tensor([0,1,1,1]);m=torch.zeros(4,224,224,dtype=torch.int64)
    m[1]=1;m[3,10:20,10:20]=1
    result=validate_mask_semantics(y,m,['real','efs','local','local'])
    assert result['empty_local_mask_indices']==[2] and y[2]==1
    m[2]=1;assert not validate_mask_semantics(y,m,['real','efs','local','local'])['empty_local_mask_indices']
    for labels,masks,kinds in [(y,m,['real','local','unknown','local']),
                              (y,m,['efs','efs','local','local']),
                              (y,m,'real'),(y,m,['real']),
                              (y,m.clone().zero_(),['real','efs','local','local']),
                              (y,None,['real','efs','local','local'])]:
        with pytest.raises((ValueError,TypeError)):validate_mask_semantics(labels,masks,kinds)
    bad=m.clone();bad[0,0,0]=1
    with pytest.raises(ValueError):validate_mask_semantics(y,bad)


@pytest.mark.parametrize('batch',[1,2])
def test_loss_ar01_mean_and_identity(batch):
    image=torch.full((batch,3,224,224),.25,dtype=torch.float64)
    assert ar(image,image).item()==0
    torch.testing.assert_close(ar(image,image+.5),torch.tensor(.25,dtype=torch.float64),**FP64)


def test_loss_ar02_both_gradients_finite_difference():
    image=torch.full((1,3,224,224),.4,dtype=torch.float64,requires_grad=True)
    rec=torch.full_like(image,.7,requires_grad=True)
    value=ar(image,rec);value.backward()
    expected=2*(rec.detach()-image.detach())/image.numel()
    torch.testing.assert_close(rec.grad,expected,**FP64)
    assert torch.equal(image.grad,-rec.grad)
    numerical=central_difference(lambda:ar(image,rec),rec,(0,0,0,0))
    torch.testing.assert_close(rec.grad[0,0,0,0],torch.tensor(numerical,dtype=torch.float64),**FD)


@pytest.mark.parametrize('target,reconstruction',[(0.,2.),(1.,-1.)])
def test_loss_ar03_unrestricted_reconstruction(target,reconstruction):
    image=torch.full((1,3,224,224),target)
    rec=torch.full_like(image,reconstruction,requires_grad=True)
    value=ar(image,rec);assert value.item()==4.;value.backward()
    assert rec.grad.isfinite().all() and rec.grad.count_nonzero()==rec.numel()
    assert torch.equal(rec.detach(),torch.full_like(rec,reconstruction))


def test_loss_ar04_population_and_rejections():
    image=torch.zeros(2,3,224,224,dtype=torch.float64)
    rec=torch.ones_like(image);rec[1]=2
    torch.testing.assert_close(ar(image,rec),torch.tensor(2.5,dtype=torch.float64),**FP64)
    for x,r in [(image[:1],rec),(image+2,rec),(image-.1,rec),(image.long(),rec),
                (image,rec.float()),(image,torch.full_like(rec,float('inf'))),
                (image,rec.to('meta'))]:
        with pytest.raises((ValueError,TypeError,FloatingPointError)):ar(x,r)
    with pytest.raises(FloatingPointError):ar(torch.zeros(1,3,224,224),torch.full((1,3,224,224),1e30))
