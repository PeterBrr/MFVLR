"""VD-I01--VD-I08 with the accepted, unchanged MVE; probes are not paper losses."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from mfvlr.config import load_config
from mfvlr.mve import MVE, parameter_report
from mfvlr.vision_decoder import VisionDecoder, decoder_parameter_report


def close(first,second):
    torch.testing.assert_close(first,second,atol=1e-6,rtol=1e-5)


def pair(model,decoder,image):
    appearance=model.encode_appearance(image)
    decoded=decoder.decode_appearance(appearance)
    encoded=model(image,decoded.reconstruction,appearance_encoding=appearance)
    return appearance,decoded,encoded


@pytest.mark.shapes
@pytest.mark.parametrize('batch',[1,2])
def test_vd_i01_forward_dag(model,decoder,image,batch):
    value=image.expand(batch,-1,-1,-1).contiguous()
    calls={}
    targets={'cnn':model.shared_encoder.cnn,'block0':model.shared_encoder.blocks[0],
             'projection':model.shared_encoder.projection,'trunk':decoder.trunk,
             'ad':decoder.appearance_head,'md':decoder.mask_head}
    handles=[module.register_forward_hook(lambda m,args,out,key=key:calls.__setitem__(key,calls.get(key,0)+1)) for key,module in targets.items()]
    try:
        with torch.no_grad():appearance,decoded,encoded=pair(model,decoder,value)
    finally:
        for handle in handles:handle.remove()
    assert calls=={'cnn':2,'block0':2,'projection':2,'trunk':1,'ad':1,'md':1}
    assert encoded.appearance is appearance
    assert len(appearance.skips)==4
    assert appearance.local.shape==(batch,1024,14,14)
    assert decoded.reconstruction.shape==encoded.residual_image.shape==(batch,3,224,224)
    assert decoded.mask_logits.shape==(batch,2,224,224)
    assert encoded.fused_global.shape==(batch,1,512)
    assert torch.equal(encoded.residual_image,(decoded.reconstruction-value).abs())
    assert torch.equal(encoded.fused_global,appearance.global_token+encoded.residual_global)
    print(f'VD-I01 b={batch}: CNN/Transformer roles=2; trunk/heads=1; DAG and exact residual/fusion verified')


@pytest.mark.shapes
def test_vd_i02_cached_vs_recomputed(model,decoder,image):
    with torch.no_grad():
        appearance,decoded,first=pair(model,decoder,image)
        second=model(image,decoded.reconstruction)
    for key in ('residual_image','residual_tokens','residual_global','fused_global'):
        close(getattr(first,key),getattr(second,key))
    # Repeat with gradients enabled to check accepted appearance record structure.
    appearance=model.encode_appearance(image)
    assert len(appearance.skips)==4 and all(skip.grad_fn is not None for skip in appearance.skips)
    assert appearance.local.grad_fn is not None


@pytest.mark.backward
@pytest.mark.parametrize('batch',[1,2])
def test_vd_i03_isolated_reconstruction_residual_gradient(model,decoder,image,batch):
    value=image.expand(batch,-1,-1,-1).contiguous().clone().requires_grad_()
    appearance,decoded,encoded=pair(model,decoder,value)
    local=appearance.local; reconstruction=decoded.reconstruction
    selected=(reconstruction,local,decoder.appearance_head.weight,decoder.trunk.stages[0].up.weight,
              model.shared_encoder.cnn.stages[0][0].weight,decoder.mask_head.weight)
    gradients=torch.autograd.grad(encoded.residual_global.square().mean(),selected,allow_unused=True)
    for grad in gradients[:-1]:
        assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0
    assert gradients[-1] is None
    print(f'VD-I03 b={batch}: reconstruction={tuple(gradients[0].shape)}, local={tuple(gradients[1].shape)}, '
          f'norms={[float(g.norm()) for g in gradients[:-1]]}')
    # Free the connected graph before the independent detached control.
    del appearance,decoded,encoded,gradients,local,reconstruction,selected
    model.zero_grad(set_to_none=True);decoder.zero_grad(set_to_none=True)
    appearance=model.encode_appearance(value)
    decoded=decoder.decode_appearance(appearance)
    detached=model(value,decoded.reconstruction.detach(),appearance_encoding=appearance)
    control=torch.autograd.grad(detached.residual_global.square().mean(),
              (appearance.local,decoder.appearance_head.weight,decoder.trunk.stages[0].up.weight,
               model.shared_encoder.cnn.stages[0][0].weight,value),allow_unused=True)
    assert control[0] is None and control[1] is None and control[2] is None
    assert all(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in control[3:])
    print(f'VD-I03 b={batch}: detached decoder/local gradients=None; RE/direct-I gradients remain nonzero')


@pytest.mark.backward
def test_vd_i04_mask_dependency_isolation(model,decoder,image):
    appearance=model.encode_appearance(image)
    decoded=decoder.decode_appearance(appearance)
    weights=torch.tensor([.3,-.7])[None,:,None,None]
    probe=(decoded.mask_logits*weights).mean()
    grads=torch.autograd.grad(probe,(decoder.mask_head.weight,decoder.trunk.stages[0].up.weight,
             model.shared_encoder.cnn.stages[0][0].weight,appearance.local,
             decoder.appearance_head.weight,model.shared_encoder.blocks[0].attention.q.weight),allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in grads[:4])
    assert grads[4] is None and grads[5] is None


@pytest.mark.backward
def test_vd_i05_appearance_dependency_isolation(model,decoder,image):
    appearance=model.encode_appearance(image)
    decoded=decoder.decode_appearance(appearance)
    grads=torch.autograd.grad(decoded.reconstruction.square().mean(),
         (decoder.appearance_head.weight,decoder.trunk.stages[0].up.weight,
          model.shared_encoder.cnn.stages[0][0].weight,appearance.local,
          decoder.mask_head.weight,model.shared_encoder.blocks[0].attention.q.weight),allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in grads[:4])
    assert grads[4] is None and grads[5] is None


@pytest.mark.backward
def test_vd_i06_real_decoder_range_and_abs_derivative(model,decoder,image):
    saved=copy.deepcopy(decoder.appearance_head.state_dict())
    try:
        with torch.no_grad():
            decoder.appearance_head.weight.zero_()
            decoder.appearance_head.bias.copy_(torch.tensor([-2.,2.,.5]))
        appearance,decoded,encoded=pair(model,decoder,image)
        assert encoded.residual_image.max()>1
        assert decoded.reconstruction.min()<0 and decoded.reconstruction.max()>1
        derivative=torch.autograd.grad(encoded.residual_image.sum(),decoded.reconstruction)[0]
        assert torch.equal(derivative,(decoded.reconstruction-image).sign())
        # Actual MVE operator at equality, with reconstruction as independent leaf.
        exact=image.clone().requires_grad_()
        output=model(image,exact,appearance_encoding=appearance)
        derivative=torch.autograd.grad(output.residual_image.sum(),exact)[0]
        assert torch.count_nonzero(derivative)==0
    finally:
        decoder.appearance_head.load_state_dict(saved)


@pytest.mark.serialization
def test_vd_i07_combined_ownership_and_roundtrip(model,decoder,image,tmp_path,root):
    assert parameter_report(model)['total_unique']==32078912
    assert decoder_parameter_report(decoder)['total_unique']==9055365
    all_params=list(model.parameters())+list(decoder.parameters())
    assert len(all_params)==len({id(p) for p in all_params})==116
    assert sum(p.numel() for p in all_params)==41134277
    optimizer=torch.optim.SGD(all_params,lr=.1)
    assert len({id(p) for g in optimizer.param_groups for p in g['params']})==116
    path=tmp_path/'mve_vd.pt'
    torch.save({'mve':model.state_dict(),'vd':decoder.state_dict(),'skip_policy':decoder.skip_policy},path)
    state=torch.load(path,weights_only=True)
    restored_mve=MVE.from_config(load_config(root/'configs/paper_reference.yaml'))
    restored_vd=VisionDecoder(skip_policy=state['skip_policy'],seed=1)
    restored_mve.load_state_dict(state['mve']);restored_vd.load_state_dict(state['vd'])
    with torch.no_grad():
        _,decoded,encoded=pair(model,decoder,image)
        _,other_decoded,other_encoded=pair(restored_mve,restored_vd,image)
    assert torch.equal(encoded.fused_global,other_encoded.fused_global)
    assert torch.equal(decoded.reconstruction,other_decoded.reconstruction)
    assert torch.equal(decoded.mask_logits,other_decoded.mask_logits)


@pytest.mark.serialization
def test_vd_i08_accepted_source_config_preservation(root):
    manifest=json.loads((root/'artifacts/phase2a/n1_review/manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files_sha256'].items():
        if name.startswith(('src/','tests/','configs/')):
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    source=(root/'tests/test_mve.py').read_text(encoding='utf-8')
    assert all(f'def test_mve_t{i:02}_' in source for i in range(1,15))
    settings=load_config(root/'configs/debug.yaml').proposal['verification']
    assert settings['fp32_atol']==1e-6 and settings['fp32_rtol']==1e-5
    assert settings['mve_t12_cross_batch']['atol']==5e-6
    assert settings['mve_t12_cross_batch']['rtol']==1e-5
    print('VD-I08: all accepted MVE source/tests/config hashes intact; regression executed as complete suite separately')
