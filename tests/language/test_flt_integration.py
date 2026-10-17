"""FLT-I01--I08 external accepted-visual integration and path isolation."""
import gc
import hashlib
import json
from pathlib import Path

import pytest
import torch

from mfvlr.decoder_config import load_decoder_config
from mfvlr.flt import parameter_report
from mfvlr.visual_language import VisualLanguageOrchestrator
from mfvlr.vision_decoder import VisionDecoder

FP32={'atol':1e-6,'rtol':1e-5}


def record_gradient(test_id, values):
    """Persist measurements from the actual asserted integration execution."""
    directory=Path(__file__).resolve().parents[2]/'artifacts/phase2c2/gradients'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/f'{test_id}.json').write_text(json.dumps(values,indent=2)+'\n')


@pytest.fixture(scope='module')
def vision_decoder(root):
    return VisionDecoder.from_config(load_decoder_config(root/'configs/decoder_debug.yaml'))


@pytest.fixture(autouse=True)
def reset_visual_decoder(vision_decoder):
    vision_decoder.zero_grad(set_to_none=True)
    yield
    vision_decoder.zero_grad(set_to_none=True)


def visual_forward(model,decoder,batch=1):
    image=torch.rand(batch,3,224,224,generator=torch.Generator().manual_seed(412))
    appearance=model.encode_appearance(image)
    decoded=decoder.decode_appearance(appearance)
    result=model(image,decoded.reconstruction,appearance_encoding=appearance)
    return image,decoded,result


@pytest.mark.parametrize('batch',[1,2])
def test_flt_i01_forward_dag(model,vision_decoder,flt,language_inputs,batch):
    ids,_=language_inputs
    wrapper=VisualLanguageOrchestrator(model,vision_decoder,flt)
    image=torch.rand(batch,3,224,224,generator=torch.Generator().manual_seed(412))
    observed=[]
    hooks=[b.vim.register_forward_pre_hook(lambda m,args:observed.append(args[1])) for b in
           [*flt.encoder.blocks,*flt.decoder.blocks]]
    try:
        with torch.no_grad():output=wrapper.with_language(image,ids[:batch])
    finally:
        for h in hooks:h.remove()
    assert output.visual.appearance.local.shape==(batch,1024,14,14)
    assert output.decoded.reconstruction.shape==(batch,3,224,224)
    assert output.decoded.mask_logits.shape==(batch,2,224,224)
    assert output.visual.fused_global.shape==(batch,1,512)
    assert output.language.logits.shape==(batch,308,49408)
    assert torch.equal(output.visual.fused_global,output.visual.appearance.global_token+output.visual.residual_global)
    assert len(observed)==19 and all(t is output.visual.fused_global for t in observed)


def test_flt_i02_full_language_gradient_paths(model,vision_decoder,flt,language_inputs):
    ids,_=language_inputs
    _,decoded,visual=visual_forward(model,vision_decoder)
    for t in (decoded.reconstruction,visual.residual_image,visual.fused_global):t.retain_grad()
    language=flt(ids[:1],visual.fused_global)
    language.logits.square().mean().backward()
    for t in (decoded.reconstruction,visual.residual_image,visual.fused_global):
        assert t.grad is not None and t.grad.isfinite().all() and t.grad.norm()>0
    for p in (vision_decoder.appearance_head.weight,vision_decoder.trunk.stages[0].up.weight,
              model.shared_encoder.cnn.stages[0][0].weight,
              flt.encoder.blocks[0].vim.v.weight,flt.decoder.blocks[-1].vim.out.weight):
        assert p.grad is not None and p.grad.isfinite().all() and p.grad.norm()>0
    assert all(p.grad is None for p in vision_decoder.mask_head.parameters())
    record_gradient('FLT-I02', {
        'reconstruction_grad_norm':decoded.reconstruction.grad.norm().item(),
        'residual_grad_norm':visual.residual_image.grad.norm().item(),
        'fused_visual_grad_norm':visual.fused_global.grad.norm().item(),
        'appearance_head_grad_norm':vision_decoder.appearance_head.weight.grad.norm().item(),
        'decoder_trunk_grad_norm':vision_decoder.trunk.stages[0].up.weight.grad.norm().item(),
        'shared_mve_conv_grad_norm':model.shared_encoder.cnn.stages[0][0].weight.grad.norm().item(),
        'mask_head_grads_none':True,'all_checked_gradients_finite':True})


def test_flt_i03_reconstruction_isolated_and_detached(model,vision_decoder,flt,language_inputs):
    ids,_=language_inputs
    image,decoded,visual=visual_forward(model,vision_decoder)
    decoded.reconstruction.retain_grad();visual.residual_image.retain_grad()
    isolated=visual.appearance.global_token.detach()+visual.residual_global
    result=flt(ids[:1],isolated,return_logits=False)
    result.decoder_hidden.square().mean().backward()
    assert decoded.reconstruction.grad.isfinite().all() and decoded.reconstruction.grad.norm()>0
    expected=visual.residual_image.grad*torch.sign(decoded.reconstruction.detach()-image)
    assert torch.equal(decoded.reconstruction.grad,expected)
    assert torch.count_nonzero(decoded.reconstruction.detach()-image)>0
    connected_norm=decoded.reconstruction.grad.norm().item()
    reconstructed=decoded.reconstruction.detach()
    del result,isolated,visual,decoded
    flt.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True);vision_decoder.zero_grad(set_to_none=True)
    appearance=model.encode_appearance(image)
    control=model(image,reconstructed,appearance_encoding=appearance)
    isolated=appearance.global_token.detach()+control.residual_global
    flt(ids[:1],isolated,return_logits=False).decoder_hidden.square().mean().backward()
    assert all(p.grad is None for p in vision_decoder.parameters())
    assert model.shared_encoder.cnn.stages[0][0].weight.grad.norm()>0
    record_gradient('FLT-I03', {'isolated_reconstruction_grad_norm':connected_norm,
        'absolute_residual_chain_rule_exact':True,'detached_reconstruction_all_vd_grads_none':True,
        'detached_control_residual_encoder_grad_norm':model.shared_encoder.cnn.stages[0][0].weight.grad.norm().item()})


def test_flt_i04_visual_detach_and_zero_qk(model,vision_decoder,flt,language_inputs):
    ids,_=language_inputs
    _,decoded,visual=visual_forward(model,vision_decoder)
    flt(ids[:1],visual.fused_global.detach()).logits.square().mean().backward()
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in vision_decoder.parameters())
    assert flt.decoder.blocks[-1].vim.v.weight.grad.norm()>0
    # Connected Q/K path with detached values/output does not transmit singleton task gradient.
    primitive=flt.decoder.blocks[-1].vim
    primitive.zero_grad(set_to_none=True)
    language=torch.ones(1,3,512,requires_grad=True)
    v=visual.fused_global.detach().requires_grad_()
    trace=primitive.forward_with_trace(language,v)
    reference=(trace.probabilities@trace.value.detach()).transpose(1,2).reshape(1,3,512)
    delta=reference@primitive.out.weight.detach().T
    delta.square().mean().backward()
    assert primitive.q.weight.grad is not None and primitive.q.weight.grad.count_nonzero()==0
    assert primitive.k.weight.grad is not None and primitive.k.weight.grad.count_nonzero()==0
    assert v.grad is not None and v.grad.count_nonzero()==0
    record_gradient('FLT-I04', {'detached_visual_all_mve_vd_grads_none':True,
        'connected_q_gradient_exact_zero':True,'connected_k_gradient_exact_zero':True,
        'qk_only_visual_gradient_exact_zero':True})


def test_flt_i05_memory_and_direct_vim_controls(flt,language_inputs):
    ids,v=language_inputs
    v=v[:1].clone().requires_grad_()
    encoded=flt.encode(ids[:1],v)
    encoded.encoder_hidden.retain_grad()
    output=flt.decode(encoded.low.detach(),encoded.encoder_hidden,v.detach())
    output.square().mean().backward()
    assert encoded.encoder_hidden.grad.norm()>0 and v.grad.norm()>0
    assert flt.encoder.blocks[0].attention.v.weight.grad.norm()>0
    memory_norm=encoded.encoder_hidden.grad.norm().item();memory_visual_norm=v.grad.norm().item()
    low=encoded.low.detach();memory=encoded.encoder_hidden.detach()
    del output,encoded
    flt.zero_grad(set_to_none=True);v.grad=None
    flt.decode(low,memory,v.detach()).square().mean().backward()
    assert all(p.grad is None for p in flt.encoder.parameters())
    assert flt.encoder_position.grad is None and v.grad is None
    flt.zero_grad(set_to_none=True)
    flt.decode(low,memory,v).square().mean().backward()
    assert v.grad.isfinite().all() and v.grad.norm()>0
    assert all(p.grad is None for p in flt.encoder.parameters())
    record_gradient('FLT-I05', {'memory_grad_norm':memory_norm,
        'memory_only_visual_grad_norm':memory_visual_norm,
        'fully_detached_encoder_grads_none':True,'fully_detached_visual_grad_none':True,
        'direct_decoder_vim_visual_grad_norm':v.grad.norm().item(),
        'direct_decoder_vim_encoder_grads_none':True})


def test_flt_i06_visual_only_no_language_dependency(model,vision_decoder,flt,language_inputs):
    wrapper=VisualLanguageOrchestrator(model,vision_decoder,flt)
    image=torch.rand(1,3,224,224,generator=torch.Generator().manual_seed(412))
    with torch.no_grad():baseline=wrapper.visual_only(image)
    class Sentinel:
        def __call__(self,*args,**kwargs):raise AssertionError('text accessed at visual inference')
    wrapper.flt=Sentinel()
    with torch.no_grad():after=wrapper.visual_only(image)
    assert after.language is None
    assert torch.equal(baseline.visual.fused_global,after.visual.fused_global)
    assert torch.equal(baseline.decoded.mask_logits,after.decoded.mask_logits)
    assert torch.equal(baseline.decoded.reconstruction,after.decoded.reconstruction)
    # Inference may omit the language model entirely, including construction.
    image_only=VisualLanguageOrchestrator(model,vision_decoder)
    with torch.no_grad():without_flt=image_only.visual_only(image)
    assert torch.equal(without_flt.visual.fused_global,baseline.visual.fused_global)
    assert without_flt.language is None and image_only.flt is None
    with pytest.raises(RuntimeError,match='explicitly supplied FLT'):
        image_only.with_language(image,language_inputs[0][:1])
    ids,v=language_inputs
    with torch.no_grad():
        before=flt(ids[:1],v[:1],return_logits=False).decoder_hidden
        evaluated=flt.eval()(ids[:1],v[:1],return_logits=False).decoder_hidden
    assert torch.equal(before,evaluated)


def test_flt_i07_counts_ownership_preservation(model,vision_decoder,flt,root):
    groups=[list(m.parameters()) for m in (model,vision_decoder,flt)]
    assert sum(p.numel() for g in groups for p in g)==153892549
    assert len({id(p) for g in groups for p in g})==sum(map(len,groups))
    assert parameter_report(flt)['total_unique']==112758272
    manifest=json.loads((root/'artifacts/phase2c1/manifest.json').read_text())
    for name,expected in manifest['files_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name


def test_flt_i08_resource_records_and_combined_roundtrip(model,vision_decoder,flt,root,tmp_path):
    # Fresh-process measurements are produced by the diagnostic tool, outside this pytest process.
    resource_path=root/'artifacts/phase2c2/resources.json'
    assert resource_path.is_file(),'Run tools/diagnose_phase2c2.py before the integration suite'
    resources=json.loads(resource_path.read_text())
    for batch in (1,2):
        matches=[c for c in resources['cases'] if c.get('route')=='integrated' and c.get('batch')==batch]
        assert len(matches)==1 and matches[0]['status']=='PASS'
        assert matches[0]['forward_backward']['iterations']==3
        assert matches[0]['logits_shape']==[batch,308,49408]
    assert resources['gpu']['status'] in ('PASS','FAIL/OOM','NOT RUN')
    assert resources['batch8']['status'] in ('PASS','FAIL/OOM','NOT RUN')
    wrapper=VisualLanguageOrchestrator(model,vision_decoder,flt)
    image=torch.rand(1,3,224,224,generator=torch.Generator().manual_seed(412))
    with torch.no_grad():before=wrapper.visual_only(image).visual.fused_global.clone()
    identities=[id(p) for m in (model,vision_decoder,flt) for p in m.parameters()]
    for name,module in (('mve',model),('vd',vision_decoder),('flt',flt)):
        path=tmp_path/f'{name}.pt';torch.save(module.state_dict(),path)
        state=torch.load(path,weights_only=True,mmap=True)
        module.load_state_dict(state,strict=True)
        del state
    with torch.no_grad():after=wrapper.visual_only(image).visual.fused_global
    assert torch.equal(before,after)
    assert identities==[id(p) for m in (model,vision_decoder,flt) for p in m.parameters()]
    assert flt.vocabulary_output_weight is flt.vocabulary.weight
