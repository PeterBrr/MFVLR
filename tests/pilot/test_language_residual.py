from types import SimpleNamespace
import torch
import pytest
from mfvlr.language_encoder import LanguageEncoderBlock
from mfvlr.language_decoder import LanguageDecoderBlock
from mfvlr_research.pilot_language import configure_vim_residuals, ResidualVisionInjection
from mfvlr_research.pilot_language import configure_language_normalization
from torch import nn


def test_double_residual_preserves_weights_and_is_exactly_twice_not_three():
    le, ld = LanguageEncoderBlock(), LanguageDecoderBlock()
    owner=SimpleNamespace(encoder=SimpleNamespace(blocks=[le]),decoder=SimpleNamespace(blocks=[ld]))
    before=[{k:v.clone() for k,v in b.state_dict().items()} for b in (le,ld)]
    configure_vim_residuals(owner,'double')
    configure_vim_residuals(owner,'double')  # Idempotent, never add a third residual.
    x=torch.randn(2,3,512,requires_grad=True);v=torch.randn(2,1,512,requires_grad=True)
    for block,original in zip((le,ld),before):
        assert isinstance(block.vim,ResidualVisionInjection)
        assert original.keys()==block.state_dict().keys()
        assert all(torch.equal(original[k],p) for k,p in block.state_dict().items())
        delta=block.vim.forward_with_trace(x,v).delta
        actual=x+block.vim(x,v)
        torch.testing.assert_close(actual,2*x+delta)
        gx,gv=torch.autograd.grad(actual.sum(),(x,v),retain_graph=True)
        torch.testing.assert_close(gx,torch.full_like(x,2.))
        assert torch.isfinite(gv).all() and gv.count_nonzero()
    configure_vim_residuals(owner,'single')
    torch.testing.assert_close(x+le.vim(x,v),x+le.vim.forward_with_trace(x,v).delta)
    with pytest.raises(ValueError):configure_vim_residuals(owner,'triple')


def test_double_residual_is_applied_at_actual_le_ld_call_sites():
    le,ld=LanguageEncoderBlock(),LanguageDecoderBlock()
    owner=SimpleNamespace(encoder=SimpleNamespace(blocks=[le]),decoder=SimpleNamespace(blocks=[ld]))
    configure_vim_residuals(owner,'double')
    x=torch.randn(1,3,512);v=torch.randn(1,1,512);memory=torch.randn(1,3,512)
    a=x+le.attention(le.ln(x));b=2*a+le.vim.forward_with_trace(a,v).delta
    torch.testing.assert_close(le(x,v),b+le.ff(b))
    a=x+ld.self_attention(x,causal=True);b=a+ld.memory_attention(a,memory);c=b+ld.ff(b)
    torch.testing.assert_close(ld(x,memory,v),2*c+ld.vim.forward_with_trace(c,v).delta)


def test_pre_norm_retains_base_weights_and_double_residual_semantics():
    owner=nn.Module();owner.encoder=nn.Module();owner.decoder=nn.Module()
    owner.encoder.blocks=nn.ModuleList([LanguageEncoderBlock()])
    owner.decoder.blocks=nn.ModuleList([LanguageDecoderBlock()])
    before={k:v.clone() for k,v in owner.state_dict().items()}
    configure_vim_residuals(owner,'double')
    configure_language_normalization(owner,'pre_norm','double')
    assert all(torch.equal(owner.state_dict()[k],v) for k,v in before.items())
    assert owner.encoder.blocks[0].residual_factor==owner.decoder.blocks[0].residual_factor==2
    x=torch.randn(1,3,512,requires_grad=True);v=torch.randn(1,1,512,requires_grad=True)
    le=owner.encoder.blocks[0];a=x+le.attention(le.ln(x))
    b=2*a+le.vim.forward_with_trace(le.norm_vim(a),v).delta
    expected=b+le.ff(le.norm_ff(b));torch.testing.assert_close(le(x,v),expected)
    memory=owner.encoder_final_norm(expected)
    result=owner.decoder_final_norm(owner.decoder.blocks[0](x,memory,v))
    torch.testing.assert_close(result.mean(-1),torch.zeros(1,3),atol=1e-6,rtol=0)
    result.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
