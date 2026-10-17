"""Scoped P2D16 arithmetic and ownership units, not JOINT/G2 execution."""
import io
import math

import pytest
import torch

from mfvlr.heads import DetectionHeads,head_parameter_report
from mfvlr.heads_losses_config import load_heads_losses_config
from mfvlr.objectives import SixObjectives,unit_weight_sum
from .oracles import FP64


def test_unit_weight_sum_exact_order_and_connected_gradients():
    values=[torch.tensor(x,requires_grad=True) for x in (1e8,-1e8,3.,4.,5.,6.)]
    def call(sequence):return unit_weight_sum(**dict(zip(('fd','lr','cmc','fl','ar','kl'),sequence)))
    total=call(values)
    assert total.item()==18. # Detect reassociation around the cancellation.
    expected=values[0]+values[1]+values[2]+values[3]+values[4]+values[5]
    assert torch.equal(total,expected)
    total.backward();assert all(x.grad.item()==1 for x in values)
    for bad in (torch.tensor([1.]),torch.tensor(float('inf')),torch.tensor(1),1.):
        with pytest.raises((TypeError,ValueError,FloatingPointError)):call([bad,*values[1:]])
    with pytest.raises((TypeError,ValueError)):call([values[0].double(),*values[1:]])
    with pytest.raises(FloatingPointError):call([torch.tensor(3e38)]*6)


@pytest.mark.parametrize('batch',[1,2])
def test_independent_six_inputs_unit_values_and_backward(batch):
    module=SixObjectives().double()
    detection=torch.zeros(batch,2,dtype=torch.float64,requires_grad=True)
    labels=torch.zeros(batch,dtype=torch.int64)
    logits=torch.zeros(batch,308,49408,dtype=torch.float64,requires_grad=True)
    ids=torch.zeros(batch,308,dtype=torch.int64)
    visual=torch.zeros(batch,1,512,dtype=torch.float64,requires_grad=True)
    language=torch.zeros_like(visual,requires_grad=True)
    adapted=torch.zeros_like(visual,requires_grad=True)
    masks=torch.zeros(batch,224,224,dtype=torch.int64)
    mask_logits=torch.zeros(batch,2,224,224,dtype=torch.float64,requires_grad=True)
    image=torch.zeros(batch,3,224,224,dtype=torch.float64,requires_grad=True)
    reconstruction=torch.full_like(image,.5,requires_grad=True)
    losses=module(detection_logits=detection,labels=labels,language_logits=logits,token_ids=ids,
                  visual=visual,global_language=language,mask_logits=mask_logits,masks=masks,
                  image=image,reconstruction=reconstruction,predicted_language=adapted,
                  sample_kinds=['real']*batch)
    expected=[math.log(2),308*math.log(49408),math.log(batch),math.log(2),.25,0.]
    names=('fd','lr','cmc','fl','ar','kl')
    for name,value in zip(names,expected):
        actual=getattr(losses,name);assert actual.ndim==0 and actual.isfinite()
        torch.testing.assert_close(actual,torch.tensor(value,dtype=torch.float64),**FP64)
    assert list(losses.as_dict())==[*names,'total']
    torch.testing.assert_close(losses.total,torch.tensor(sum(expected),dtype=torch.float64),**FP64)
    losses.total.backward()
    for value in (detection,logits,visual,language,adapted,mask_logits,image,reconstruction,module.cmc.log_temperature):
        assert value.grad is not None and value.grad.isfinite().all()
    assert visual.grad.count_nonzero()==0 and language.grad.count_nonzero()==0
    assert detection.grad.norm()>0 and logits.grad.norm()>0 and reconstruction.grad.norm()>0


def test_new_subtotal_config_ownership_and_roundtrip():
    # Accepted owners are not built, copied or registered inside these modules.
    heads=DetectionHeads();module=SixObjectives()
    assert head_parameter_report(heads)['total_unique']==263682
    params=list(heads.parameters())+list(module.parameters())
    assert len(params)==5 and len({id(p) for p in params})==5
    assert len({p.data_ptr() for p in params})==5
    assert sum(p.numel() for p in params)==263683
    assert list(module.state_dict())==['cmc.log_temperature']
    buf=io.BytesIO();torch.save(module.state_dict(),buf);buf.seek(0)
    restored=SixObjectives();restored.load_state_dict(torch.load(buf,weights_only=True))
    assert torch.equal(restored.cmc.log_temperature,module.cmc.log_temperature)
    for path in ('configs/heads_losses_reference.yaml','configs/heads_losses_debug.yaml'):
        config=load_heads_losses_config(path)
        assert sum(p.numel() for p in SixObjectives.from_config(config).parameters())==1
        assert sum(p.numel() for p in DetectionHeads.from_config(config).parameters())==263682


def test_bundle_rejects_cross_loss_batch_mismatch():
    z=lambda *shape:torch.zeros(*shape)
    with pytest.raises(ValueError):
        SixObjectives()(detection_logits=z(1,2),labels=torch.zeros(1,dtype=torch.int64),
                        language_logits=z(1,1,1).expand(1,308,49408),token_ids=torch.zeros(1,308,dtype=torch.int64),
                        visual=z(2,1,512),global_language=z(1,1,512),predicted_language=z(1,1,512),
                        mask_logits=z(1,2,224,224),masks=torch.zeros(1,224,224,dtype=torch.int64),
                        image=z(1,3,224,224),reconstruction=z(1,3,224,224))
