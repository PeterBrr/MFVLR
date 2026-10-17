"""Stage-1 gates execute before any LD implementation exists."""
import gc

import pytest
import torch
from torch import nn

from mfvlr.language_encoder import LanguageAttention, LanguageEncoderBlock, LanguageEncoderStage
from mfvlr.vim import VisionInjectionDelta

FP64 = {"atol":1e-10,"rtol":1e-8}


@pytest.mark.parametrize("batch",[1,2])
def test_le_stage_full_shapes_backward(encoder_stage,language_inputs,batch):
    module = encoder_stage
    module.zero_grad(set_to_none=True)
    ids,visual = language_inputs
    visual = visual[:batch].clone().requires_grad_()
    observed = []
    hooks = [block.register_forward_hook(lambda m,args,out: observed.append((out.shape,bool(out.isfinite().all()))))
             for block in module.encoder.blocks]
    try:
        result = module(ids[:batch],visual)
    finally:
        for h in hooks: h.remove()
    assert observed == [(torch.Size([batch,308,512]),True)]*12
    assert result.low.shape == result.encoder_hidden.shape == (batch,308,512)
    assert result.global_language.shape == (batch,1,512)
    assert torch.equal(result.global_language,result.encoder_hidden[:,307:308])
    (result.encoder_hidden.square().mean()+result.global_language.mean()).backward()
    assert visual.grad.isfinite().all() and visual.grad.norm()>0
    for name,p in module.named_parameters():
        if name == "decoder_position":
            assert p.grad is None  # Reserved draw prefix; no LD in stage 1.
        else:
            assert p.grad is not None and p.grad.isfinite().all(),name
        if '.vim.q.' in name or '.vim.k.' in name:
            assert p.grad.count_nonzero()==0,name
    assert module.vocabulary.weight.grad[0].norm()>0
    assert sum(p.numel() for p in module.parameters()) == 76011520
    module.zero_grad(set_to_none=True)


def test_flt_t07_encoder_order_and_identity():
    block = LanguageEncoderBlock()
    state = torch.ones(1,3,512)*0.2
    visual = torch.ones(1,1,512)*0.1
    calls=[]
    class Norm(nn.Module):
        def forward(self,x): calls.append('LN');return x+0.3
    class Attention(nn.Module):
        def forward(self,x): calls.append('MHA');return 2*x+1
    class VIM(nn.Module):
        def forward(self,x,v): calls.append('VIM');return x.square()*0.2+v
    class FF(nn.Module):
        def forward(self,x): calls.append('FF');return x*0.1+2
    block.ln,block.attention,block.vim,block.ff=Norm(),Attention(),VIM(),FF()
    out=block(state,visual)
    s=state+2*(state+0.3)+1
    a=s+s.square()*0.2+visual
    assert calls==['LN','MHA','VIM','FF']
    torch.testing.assert_close(out,a+a*0.1+2,atol=1e-6,rtol=1e-5)
    real=LanguageEncoderBlock()
    assert real.ln.eps==1e-5 and real.ln.normalized_shape==(512,)
    assert real.ff.fc1.out_features==2048 and real.ff.gelu.approximate=='none'
    assert len([m for m in real.modules() if isinstance(m,nn.LayerNorm)])==1
    assert not any(isinstance(m,nn.Dropout) for m in real.modules())
    with torch.no_grad():
        for layer in (real.attention.q,real.attention.k,real.attention.v,real.attention.out,
                      real.vim.q,real.vim.k,real.vim.v,real.vim.out,real.ff.fc1,real.ff.fc2):
            layer.weight.zero_()
            if layer.bias is not None: layer.bias.zero_()
    assert torch.equal(real(state,visual),state)


def test_flt_t08_fp64_attention_reference():
    module=LanguageAttention().double()
    g=torch.Generator().manual_seed(42)
    q=torch.randn(2,3,512,generator=g,dtype=torch.float64)
    memory=torch.randn(2,4,512,generator=g,dtype=torch.float64)
    output,trace=module.forward_with_trace(q,memory)
    expected=torch.empty_like(output)
    # Independent per-sample/per-head row operations, no reused implementation helper.
    for b in range(2):
        queries=q[b]@module.q.weight.T+module.q.bias
        keys=memory[b]@module.k.weight.T+module.k.bias
        values=memory[b]@module.v.weight.T+module.v.bias
        heads=[]
        for h in range(8):
            sl=slice(h*64,(h+1)*64)
            probabilities=((queries[:,sl]@keys[:,sl].T)/8).softmax(-1)
            heads.append(probabilities@values[:,sl])
        expected[b]=torch.cat(heads,-1)@module.out.weight.T+module.out.bias
    torch.testing.assert_close(output,expected,**FP64)
    assert trace['probabilities'].shape==(2,8,3,4)


def test_flt_t09_bidirectional_last_physical_slot(encoder_stage,language_inputs):
    module=encoder_stage
    ids,visual=language_inputs
    changed=ids[:1].clone()
    assert changed[0,307]==0
    changed[0,250]=1234
    with torch.no_grad():
        before=module(ids[:1],visual[:1])
        after=module(changed,visual[:1])
    assert not torch.equal(before.encoder_hidden[:,0],after.encoder_hidden[:,0])
    assert torch.equal(before.global_language,before.encoder_hidden[:,307:308])
    assert before.global_language.data_ptr()==before.encoder_hidden[:,307:308].data_ptr()
    block=module.encoder.blocks[0]
    with torch.no_grad():
        state=module.vocabulary(ids[:1])+module.encoder_position
        _,trace=block.attention.forward_with_trace(block.ln(state))
    assert trace['probabilities'].shape==(1,8,308,308)
    assert (trace['probabilities'][...,307]>0).all()


def test_le_stage_ownership_and_initializer(encoder_stage):
    module=encoder_stage
    assert len(module.encoder.blocks)==12
    assert len({id(b) for b in module.encoder.blocks})==12
    assert len({b.vim.q.weight.data_ptr() for b in module.encoder.blocks})==12
    assert len([m for m in module.modules() if isinstance(m,VisionInjectionDelta)])==12
    assert module.vocabulary.padding_idx is None
    assert module.encoder_position is not module.decoder_position
    generator=torch.Generator().manual_seed(0)
    for p in (module.vocabulary.weight,module.encoder_position,module.decoder_position):
        expected=torch.empty_like(p)
        nn.init.normal_(expected,0,0.02,generator=generator)
        assert torch.equal(p,expected)
        del expected
    for block in module.encoder.blocks:
        assert torch.equal(block.ln.weight,torch.ones_like(block.ln.weight))
        assert torch.count_nonzero(block.ln.bias)==0
        for layer in (block.attention.q,block.attention.k,block.attention.v,block.attention.out,
                      block.vim.q,block.vim.k,block.vim.v,block.vim.out,block.ff.fc1,block.ff.fc2):
            expected=torch.empty_like(layer.weight)
            nn.init.xavier_uniform_(expected,generator=generator)
            assert torch.equal(layer.weight,expected)
            if layer.bias is not None: assert layer.bias.count_nonzero()==0


def test_le_stage_invalid_inputs_and_rng(encoder_stage,language_inputs):
    ids,v=language_inputs
    invalid=[(ids[:0],v[:0]),(ids[:,:307],v),(ids.int(),v),(ids-50000,v),(ids+50000,v),
             (ids,v.squeeze(1)),(ids,v.expand(2,197,512)),(ids,v.double()),
             (ids,v*float('nan')),(ids[:1],v),([],v)]
    for a,b in invalid:
        with pytest.raises((ValueError,TypeError)): encoder_stage(a,b)
    state=torch.random.get_rng_state().clone()
    duplicate=LanguageEncoderStage()
    assert torch.equal(state,torch.random.get_rng_state())
    for a,b in zip(duplicate.parameters(),encoder_stage.parameters()): assert torch.equal(a,b)
    del duplicate
    gc.collect()
