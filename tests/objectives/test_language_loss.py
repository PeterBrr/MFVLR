"""LOSS-LR01--06: independent all-token mathematics, no full-model graph."""
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mfvlr.flt import FineGrainedLanguageTransformer
from mfvlr.objectives import language_reconstruction_loss as lr,language_token_nll
from .oracles import FIXTURES,FP64,FD,categorical,central_difference

VOCAB=49408


def ids(batch=1):
    value=torch.zeros(batch,308,dtype=torch.int64)
    for start in (0,77,154,231):
        value[:,start]=49406;value[:,start+1]=12;value[:,start+2]=49407
    return value


@pytest.mark.parametrize('batch',[1,2])
def test_loss_lr01_uniform_and_independent_toy(batch):
    assert math.isclose(categorical([[0.,0.,0.],[0.,math.log(2),0.]],[0,1]),
                        (math.log(3)+math.log(2))/2,rel_tol=1e-14)
    logits=torch.zeros(batch,308,VOCAB,dtype=torch.float64)
    loss=lr(logits,ids(batch))
    torch.testing.assert_close(loss,torch.tensor(308*math.log(VOCAB),dtype=torch.float64),**FP64)
    assert loss.ndim==0 and loss.isfinite()


def test_loss_lr02_all_token_category_contributions():
    targets=ids();logits=torch.zeros(1,308,VOCAB,dtype=torch.float64)
    # Each category has its own independently defined target score.
    scores={0:-.5,49406:1.,49407:2.,12:.25}
    expected=[]
    for position,target in enumerate(targets[0].tolist()):
        a=scores[target];logits[0,position,target]=a
        expected.append(math.log(VOCAB-1+math.exp(a))-a)
    nll=language_token_nll(logits,targets)
    torch.testing.assert_close(nll[0],torch.tensor(expected,dtype=torch.float64),**FP64)
    partitions=[targets==0,(targets==49406)|(targets==49407),targets==12]
    assert sum(int(mask.sum()) for mask in partitions)==308
    actual=[nll[mask].sum() for mask in partitions]
    oracle=[math.fsum(x for x,keep in zip(expected,mask[0].tolist()) if keep) for mask in partitions]
    torch.testing.assert_close(torch.stack(actual),torch.tensor(oracle,dtype=torch.float64),**FP64)
    torch.testing.assert_close(lr(logits,targets),torch.tensor(math.fsum(oracle),dtype=torch.float64),**FP64)


def test_loss_lr03_accepted_global_shift_unshifted_targets():
    # Exercise the existing public shift method without building LE/LD or G2.
    owner=SimpleNamespace(vocabulary=nn.Embedding(VOCAB,512,dtype=torch.float64))
    with torch.no_grad():
        owner.vocabulary.weight.copy_(torch.arange(VOCAB,dtype=torch.float64)[:,None].expand(-1,512))
    targets=ids();low=owner.vocabulary(targets)
    shifted=FineGrainedLanguageTransformer.shift_low(owner,low)
    assert torch.equal(shifted[:,0],owner.vocabulary.weight[49406].view(1,512))
    assert torch.equal(shifted[:,1:],low[:,:-1])
    for position in (76,77,153,154,230,231,307):
        assert torch.equal(shifted[:,position],low[:,position-1])
    logits=torch.zeros(1,308,VOCAB,dtype=torch.float64)
    for position,target in enumerate(targets[0].tolist()):logits[0,position,target]=1.
    expected=308*(math.log(VOCAB-1+math.e)-1)
    torch.testing.assert_close(lr(logits,targets),torch.tensor(expected,dtype=torch.float64),**FP64)
    wrong=targets.clone();wrong[:,1:]=targets[:,:-1];wrong[:,0]=49406
    assert lr(logits,wrong)>lr(logits,targets)


def test_loss_lr04_gradient_scale_batch_mean_and_finite_difference():
    logits=torch.zeros(2,308,VOCAB,dtype=torch.float64,requires_grad=True);targets=ids(2)
    actual=lr(logits,targets);actual.backward()
    # A full-shape analytic oracle: (1/V - onehot)/B, never /308.
    expected=torch.full_like(logits,.5/VOCAB)
    expected.scatter_add_(-1,targets.unsqueeze(-1),torch.full((2,308,1),-.5,dtype=torch.float64))
    torch.testing.assert_close(logits.grad,expected,**FP64)
    assert torch.equal(logits.grad[0],logits.grad[1])
    torch.testing.assert_close(actual,lr(logits.detach()[:1],targets[:1]),**FP64)
    for coordinate in ((0,0,49406),(1,76,0),(0,307,1)):
        derivative=central_difference(lambda:lr(logits,targets),logits,coordinate)
        torch.testing.assert_close(logits.grad[coordinate],torch.tensor(derivative,dtype=torch.float64),**FD)


def test_loss_lr05_independent_three_path_vocabulary_oracle():
    # Toy algebra is explicitly NOT a reduced production FLT or G2 test.
    initial=torch.arange(21,dtype=torch.float64).view(7,3)/20-.3
    token_ids=torch.tensor([[1,2,0,3]]);bos=6
    def probe(prompt=True,direct_bos=True,projection=True):
        weight=nn.Parameter(initial.clone());lookup=weight[token_ids]
        if not prompt:lookup=lookup.detach()
        direct=weight[bos].view(1,1,3)
        if not direct_bos:direct=direct.detach()
        shifted=torch.cat([direct,lookup[:,:-1]],1)
        out_weight=weight if projection else weight.detach()
        logits=shifted@out_weight.T
        # Independent arbitrary coefficient probe avoids calling production LR.
        coefficients=torch.arange(28,dtype=torch.float64).view(1,4,7)/31
        (logits*coefficients).sum().backward()
        assert weight.grad is not None
        return weight.grad
    full=probe();prompt=probe(True,False,False);bos_only=probe(False,True,False);output=probe(False,False,True)
    torch.testing.assert_close(full,prompt+bos_only+output,**FP64)
    assert prompt.norm()>0 and bos_only.norm()>0 and output.norm()>0
    # Detaching lookup alone still leaves the direct BOS and tied output paths.
    torch.testing.assert_close(probe(False,True,True),bos_only+output,**FP64)


@pytest.mark.parametrize('case',['vocabulary','length','rank','ids_float','ids_negative','ids_large','nan','inf','device','integer_logits'])
def test_loss_lr06_rejections(case):
    # Expanded zero views keep rejection probes inexpensive; public shapes fixed.
    logits=torch.zeros(1,1,1,dtype=torch.float64).expand(1,308,VOCAB);targets=ids()
    if case=='vocabulary':logits=logits[:,:,:-1]
    elif case=='length':logits=logits[:,:-1]
    elif case=='rank':logits=logits[0]
    elif case=='ids_float':targets=targets.double()
    elif case=='ids_negative':targets[0,0]=-1
    elif case=='ids_large':targets[0,0]=VOCAB
    elif case in ('nan','inf'):logits=torch.full((1,1,1),float(case),dtype=torch.float64).expand(1,308,VOCAB)
    elif case=='device':targets=targets.to('meta')
    elif case=='integer_logits':logits=logits.long()
    with pytest.raises((ValueError,TypeError,FloatingPointError)):lr(logits,targets)


def test_loss_lr06_extreme_finite_wrong_target():
    logits=torch.zeros(1,308,VOCAB,dtype=torch.float64,requires_grad=True);targets=ids()
    with torch.no_grad():logits[0,0,49406]=-10000.;logits[0,0,1]=10000.
    actual=lr(logits,targets)
    torch.testing.assert_close(actual,torch.tensor(20000+307*math.log(VOCAB),dtype=torch.float64),**FP64)
    actual.backward();assert logits.grad.isfinite().all()
    assert logits.grad[0,0,49406]==-1 and logits.grad[0,0,1]==1
