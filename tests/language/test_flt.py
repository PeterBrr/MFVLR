"""Full FLT tests; T02--T05 and T07--T09 are reused from earlier test files."""
import copy
import gc
import json
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from mfvlr.config import ConfigError
from mfvlr.flt import FineGrainedLanguageTransformer,parameter_report
from mfvlr.flt_config import load_flt_config
from mfvlr.language_decoder import LanguageDecoderBlock
from mfvlr.language_encoder import LanguageAttention
from mfvlr.vim import VisionInjectionDelta

FP32={'atol':1e-6,'rtol':1e-5}


@pytest.mark.parametrize('batch',[1,2])
def test_flt_t01_full_shapes_finiteness(flt,language_inputs,batch):
    ids,v=language_inputs
    shapes=[]
    hooks=[]
    for name,module in flt.named_modules():
        if isinstance(module,(nn.Linear,nn.LayerNorm,nn.GELU,VisionInjectionDelta)):
            hooks.append(module.register_forward_hook(lambda m,args,out,n=name:shapes.append((n,tuple(out.shape),bool(out.isfinite().all())))))
    try:
        with torch.no_grad(): result=flt(ids[:batch],v[:batch])
    finally:
        for hook in hooks:hook.remove()
    assert result.encoder_hidden.shape==result.decoder_hidden.shape==(batch,308,512)
    assert result.global_language.shape==(batch,1,512)
    assert result.logits.shape==(batch,308,49408)
    assert all(finite for name,shape,finite in shapes)
    assert len([s for n,s,f in shapes if n.endswith('ff.fc1')])==19
    assert all(s==(batch,308,2048) for n,s,f in shapes if n.endswith('ff.fc1'))
    assert len([s for n,s,f in shapes if n.endswith('.vim')])==19
    with torch.no_grad(): features=flt(ids[:batch],v[:batch],return_logits=False)
    assert features.logits is None
    assert torch.equal(result.decoder_hidden,features.decoder_hidden)


def test_flt_t06_global_shift(flt,language_inputs):
    ids,v=language_inputs
    with torch.no_grad():
        low=flt.vocabulary(ids)
        shifted=flt.shift_low(low)
        expected_ids=torch.cat((torch.full((2,1),49406,dtype=torch.int64),ids[:,:-1]),1)
        expected=flt.vocabulary(expected_ids)
    assert torch.equal(shifted,expected)
    for index in (0,1,76,77,78,153,154,155,230,231,232,307):
        target=49406 if index==0 else ids[:,index-1]
        assert torch.equal(shifted[:,index],flt.vocabulary.weight[target]) if index else torch.equal(
            shifted[:,index],flt.vocabulary.weight[49406].expand(2,512))
    assert torch.equal(expected_ids[:,77],ids[:,76])
    assert not torch.equal(low+flt.encoder_position,low)
    assert torch.equal(ids,language_inputs[0])


def test_flt_t10_decoder_order_and_identity():
    block=LanguageDecoderBlock()
    x=torch.ones(1,3,512)*0.2
    memory=torch.ones(1,3,512)*0.4
    v=torch.ones(1,1,512)*0.1
    calls=[]
    class Self(nn.Module):
        def forward(self,x,*,causal=False): assert causal;calls.append('self');return x*2+1
    class Cross(nn.Module):
        def forward(self,x,m): calls.append('memory');return x.square()*0.1+m
    class FF(nn.Module):
        def forward(self,x): calls.append('FF');return x*0.3+2
    class VIM(nn.Module):
        def forward(self,x,v): calls.append('VIM');return x.square()*0.02+v
    block.self_attention,block.memory_attention,block.ff,block.vim=Self(),Cross(),FF(),VIM()
    output=block(x,memory,v)
    u=x+2*x+1
    w=u+u.square()*0.1+memory
    f=w+w*0.3+2
    torch.testing.assert_close(output,f+f.square()*0.02+v,**FP32)
    assert calls==['self','memory','FF','VIM']
    real=LanguageDecoderBlock()
    assert not any(isinstance(m,nn.LayerNorm) for m in real.modules())
    assert real.ff.gelu.approximate=='none'
    with torch.no_grad():
        for p in real.parameters():p.zero_()
    assert torch.equal(real(x,memory,v),x)


def test_flt_t11_causal_self_and_full_memory(flt,language_inputs):
    attention=LanguageAttention().double()
    x=torch.randn(1,5,512,generator=torch.Generator().manual_seed(55),dtype=torch.float64)
    _,trace=attention.forward_with_trace(x,causal=True)
    probabilities=trace['probabilities']
    assert torch.count_nonzero(probabilities.triu(1))==0
    assert (probabilities.diagonal(dim1=-2,dim2=-1)>0).all()
    torch.testing.assert_close(probabilities.sum(-1),torch.ones_like(probabilities.sum(-1)),atol=1e-10,rtol=1e-8)
    block=LanguageDecoderBlock()
    g=torch.Generator().manual_seed(67)
    state=torch.randn(1,5,512,generator=g)*0.02
    memory=torch.randn(1,5,512,generator=g)*0.02
    v=torch.randn(1,1,512,generator=g)*0.02
    future=state.clone();future[:,4]+=0.5
    past=state.clone();past[:,0]+=0.5
    with torch.no_grad():
        out=block(state,memory,v)
        changed=block(future,memory,v)
        previous=block(past,memory,v)
    assert torch.equal(out[:,:4],changed[:,:4])
    assert not torch.equal(out[:,4],previous[:,4])
    ids,visual=language_inputs
    late=ids[:1].clone();late[:,250]=1234
    with torch.no_grad():
        before=flt(ids[:1],visual[:1],return_logits=False)
        after=flt(late,visual[:1],return_logits=False)
    assert not torch.equal(before.decoder_hidden[:,0],after.decoder_hidden[:,0])


def test_flt_t12_memory_query_and_gradients():
    block=LanguageDecoderBlock()
    g=torch.Generator().manual_seed(90)
    state=(torch.randn(1,6,512,generator=g)*0.03).requires_grad_()
    memory=(torch.randn(1,6,512,generator=g)*0.03).requires_grad_()
    visual=torch.zeros(1,1,512)
    record={}
    h1=block.self_attention.register_forward_hook(lambda m,args,out:record.update(post_self=args[0]+out))
    h2=block.memory_attention.register_forward_pre_hook(lambda m,args:record.update(query=args[0],memory=args[1]))
    result=block(state,memory,visual)
    h1.remove();h2.remove()
    assert torch.equal(record['query'],record['post_self'])
    assert record['memory'] is memory
    result.square().mean().backward()
    assert memory.grad.isfinite().all() and memory.grad.norm()>0
    with torch.no_grad():
        changed=block(state,memory+0.2,visual)
    assert not torch.equal(result,changed)


def test_flt_t13_vocabulary_tying_and_grad_sums(flt,language_inputs):
    assert flt.vocabulary_output_weight is flt.vocabulary.weight
    assert flt.vocabulary.padding_idx is None
    assert [n for n,p in flt.named_parameters() if p is flt.vocabulary.weight]==['vocabulary.weight']
    hidden=torch.randn(1,308,512,generator=torch.Generator().manual_seed(99))*0.03
    with torch.no_grad():
        actual=flt.project_vocabulary(hidden)
        expected=hidden@flt.vocabulary.weight.T
    assert torch.equal(actual,expected)
    del actual,expected
    # Same actual weight, two graph roles; integer/selected-output probe avoids one-hot loss.
    ids=torch.tensor([[0,49406,49407,17]])
    w=flt.vocabulary.weight
    lookup=flt.vocabulary(ids).mean()
    output=(hidden[:,:1]@w[[0,17,49406]].T).mean()
    both=torch.autograd.grad(lookup+output,w,retain_graph=True)[0]
    input_grad=torch.autograd.grad(lookup,w,retain_graph=True)[0]
    output_grad=torch.autograd.grad(output,w)[0]
    torch.testing.assert_close(both,input_grad+output_grad,**FP32)
    assert input_grad[0].norm()>0 and output_grad[0].norm()>0


def test_flt_t14_unique_counts_independence(flt,model):
    report=parameter_report(flt)
    assert report['total_unique']==report['unique_storage_elements']==112758272
    assert report['encoder_blocks']==[4199936]*12
    assert report['decoder_blocks']==[5249536]*7
    assert report['vim_instances']==19 and report['vim_subtotal']==19922944
    assert report['vim_qk_subtotal']==9961472
    vims=[m for m in flt.modules() if isinstance(m,VisionInjectionDelta)]
    assert len(vims)==len({id(m) for m in vims})==19
    assert len({m.q.weight.data_ptr() for m in vims})==19
    assert not {id(p) for p in flt.parameters()} & {id(p) for p in model.parameters()}
    assert len([m for m in flt.modules() if isinstance(m,nn.LayerNorm)])==12
    assert flt.encoder_position is not flt.decoder_position


def test_flt_t15_full_initialization_replay_and_rng(flt):
    generator=torch.Generator().manual_seed(0)
    def normal(p):
        expected=torch.empty_like(p);nn.init.normal_(expected,0,0.02,generator=generator)
        assert torch.equal(p,expected)
    def linear(layer):
        expected=torch.empty_like(layer.weight);nn.init.xavier_uniform_(expected,generator=generator)
        assert torch.equal(layer.weight,expected)
        if layer.bias is not None:assert layer.bias.count_nonzero()==0
    for p in (flt.vocabulary.weight,flt.encoder_position,flt.decoder_position):normal(p)
    for b in flt.encoder.blocks:
        assert torch.equal(b.ln.weight,torch.ones_like(b.ln.weight)) and b.ln.bias.count_nonzero()==0
        for m in (b.attention,b.vim):
            for layer in (m.q,m.k,m.v,m.out):linear(layer)
        linear(b.ff.fc1);linear(b.ff.fc2)
    for b in flt.decoder.blocks:
        for m in (b.self_attention,b.memory_attention):
            for layer in (m.q,m.k,m.v,m.out):linear(layer)
        linear(b.ff.fc1);linear(b.ff.fc2)
        for layer in (b.vim.q,b.vim.k,b.vim.v,b.vim.out):linear(layer)
    state=torch.random.get_rng_state().clone()
    duplicate=FineGrainedLanguageTransformer()
    assert torch.equal(state,torch.random.get_rng_state())
    for a,b in zip(duplicate.parameters(),flt.parameters()):assert torch.equal(a,b)
    del duplicate;gc.collect()


@pytest.mark.parametrize('batch',[1,2])
def test_flt_t16_full_backward(flt,language_inputs,batch):
    ids,v=language_inputs
    v=v[:batch].clone().requires_grad_()
    result=flt(ids[:batch],v)
    probe=result.logits.square().mean()+result.decoder_hidden.square().mean()+result.global_language.square().mean()
    probe.backward()
    assert v.grad.isfinite().all() and v.grad.norm()>0
    for name,p in flt.named_parameters():
        assert p.grad is not None and p.grad.isfinite().all(),name
        if '.vim.q.' in name or '.vim.k.' in name:assert p.grad.count_nonzero()==0,name
    for p in (flt.vocabulary.weight,flt.encoder_position,flt.decoder_position,
              flt.encoder.blocks[0].attention.v.weight,flt.decoder.blocks[0].memory_attention.v.weight,
              flt.encoder.blocks[0].ff.fc1.weight,flt.decoder.blocks[-1].ff.fc2.weight):
        assert p.grad.norm()>0


def test_flt_t17_fixed_batch_independence(flt,language_inputs):
    ids,v=language_inputs
    changed_ids=ids.clone();changed_ids[1,250]=1234
    changed_v=v.clone();changed_v[1]+=0.5
    with torch.no_grad():
        before=flt(ids,v)
        after=flt(changed_ids,changed_v)
    for name in ('encoder_hidden','global_language','decoder_hidden','logits'):
        assert torch.equal(getattr(before,name)[0],getattr(after,name)[0]),name
    del before,after
    v=v.clone().requires_grad_()
    result=flt(ids,v,return_logits=False)
    result.decoder_hidden[0].square().mean().backward()
    assert v.grad[1].count_nonzero()==0 and v.grad[0].norm()>0


def test_flt_t17_cross_batch_numerical_diagnostic(flt,language_inputs,root):
    # N2 PROPOSED: apply only after explicit approval. FP32 constants unchanged.
    import tempfile
    from tools.review_phase2c2_n2 import forward_capture,compare,backend,write
    ids,v=language_inputs
    before=[id(p) for p in flt.parameters()]
    batch=forward_capture(flt,ids,v)
    expected={'vocabulary','encoder.input','decoder.input'}
    expected.update(f'encoder.blocks.{i}' for i in range(12))
    expected.update(f'decoder.blocks.{i}' for i in range(7))
    expected.update('output.'+name for name in
                    ('low','encoder_hidden','global_language','decoder_hidden','logits'))
    assert set(batch)==expected
    assert batch['output.logits'].shape==(2,308,49408)
    records=[]
    for sample in (0,1):
        single=forward_capture(flt,ids[sample:sample+1],v[sample:sample+1])
        assert set(single)==expected
        # compare raises on nonfinite inputs; an outside-tolerance count is evidence.
        records.append({'sample':sample,'stages':{name:compare(value[sample:sample+1],
                       single[name],**FP32) for name,value in batch.items()}})
        del single
    directory=root/'artifacts/phase2c2_n2_acceptance'
    directory.mkdir(parents=True,exist_ok=True)
    destination=Path(tempfile.mkdtemp(prefix='t17_',dir=directory))/'cross_batch.json'
    evidence={'policy':'N2','purpose':'mandatory FP32 diagnostic, not equality assertion',
              'backend':backend(),'atol':FP32['atol'],'rtol':FP32['rtol'],
              'N1_applied':False,'cross_batch':records}
    write(destination,evidence)
    assert json.loads(destination.read_text())==evidence
    assert [id(p) for p in flt.parameters()]==before
    assert flt.vocabulary_output_weight is flt.vocabulary.weight
    print('N2_DIAGNOSTIC:',destination)


def test_flt_t17_fixed_batch_permutation(flt,language_inputs):
    from tools.review_phase2c2_n2 import forward_capture
    ids,v=language_inputs
    permutation=torch.tensor([1,0])
    baseline=forward_capture(flt,ids,v)
    reordered=forward_capture(flt,ids[permutation],v[permutation])
    assert set(baseline)==set(reordered)
    for name,value in baseline.items():
        assert torch.equal(value,reordered[name][permutation]),name


def test_flt_t17_bidirectional_cross_sample_gradients(flt,language_inputs):
    from tools.review_phase2c2_n2 import cross_sample_gradients
    ids,v=language_inputs
    records=cross_sample_gradients(flt,ids,v)
    assert len(records)==2
    for record in records:
        assert record['other_visual_gradient_exact_zero']
        assert record['other_lookup_gradient_exact_zero']
        assert record['own_visual_gradient']['finite']
        assert record['own_lookup_gradient']['finite']
        assert record['own_visual_gradient']['l2_norm']>0
        assert record['own_lookup_gradient']['l2_norm']>0


def test_flt_t17_promoted_weight_fp64_cross_batch(flt,language_inputs):
    from tools.review_phase2c2_n2 import forward_capture
    ids,v=language_inputs
    # Deep copy preserves source state/parameter identity; no new initialization.
    reference=copy.deepcopy(flt).double()
    v64=v.double()
    for actual,original in zip(reference.parameters(),flt.parameters()):
        assert torch.equal(actual.float(),original)
    batch=forward_capture(reference,ids,v64)
    for sample in (0,1):
        single=forward_capture(reference,ids[sample:sample+1],v64[sample:sample+1])
        assert set(single)==set(batch)
        for name,value in batch.items():
            torch.testing.assert_close(value[sample:sample+1],single[name],atol=1e-10,rtol=1e-8)
        del single
    del batch,reference
    gc.collect()


def test_flt_t18_serialization(flt,language_inputs,tmp_path):
    ids,v=language_inputs
    with torch.no_grad():before=flt(ids[:1],v[:1],return_logits=False).decoder_hidden.clone()
    path=tmp_path/'flt.pt'
    torch.save(flt.state_dict(),path)
    loaded=FineGrainedLanguageTransformer(seed=9)
    state=torch.load(path,weights_only=True,mmap=True)
    loaded.load_state_dict(state)
    del state
    with torch.no_grad():after=loaded(ids[:1],v[:1],return_logits=False).decoder_hidden
    assert torch.equal(before,after)
    assert loaded.vocabulary_output_weight is loaded.vocabulary.weight
    assert len(loaded.state_dict())==387
    assert not any('cache' in name or 'trace' in name for name in vars(loaded))
    bad=dict(loaded.state_dict());bad['unapproved.bias']=torch.zeros(1)
    with pytest.raises(RuntimeError): loaded.load_state_dict(bad,strict=True)
    del loaded,bad;gc.collect()


@pytest.mark.parametrize('field,value',[('heads',4),('ff_hidden',1024),('encoder_blocks',6),
    ('decoder_blocks',6),('extra_norms',True),('padding_masks',True),('global_slot',306),
    ('vim_scale','sqrt(d/r)'),('vocabulary_output','untied'),('sequence_length',77)])
def test_flt_t19_configuration_changes_rejected(root,tmp_path,field,value):
    config=load_flt_config(root/'configs/flt_reference.yaml')
    data=config.data;data['architecture'][field]=value
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError):load_flt_config(path)


def test_flt_t19_strict_overlay_and_scope(root,tmp_path):
    ref=load_flt_config(root/'configs/flt_reference.yaml')
    debug=load_flt_config(root/'configs/flt_debug.yaml')
    a,b=ref.data,debug.data
    for key in ('profile','diagnostics'):a.pop(key);b.pop(key)
    assert a==b and not a['approval']['training_authorized']
    original=yaml.safe_load((root/'configs/flt_debug.yaml').read_text())
    (tmp_path/'flt_reference.yaml').write_text((root/'configs/flt_reference.yaml').read_text())
    variants=[]
    for section,field,value in [('approval','training_authorized',True),('overrides','architecture',{}),
                                ('overrides','tolerances',{'fp32_atol':5e-6})]:
        data=copy.deepcopy(original);data[section][field]=value;variants.append(data)
    for base in ('../flt_reference.yaml','C:/flt_reference.yaml'):
        data=copy.deepcopy(original);data['base']=base;variants.append(data)
    data=copy.deepcopy(original);data['overrides']['diagnostics']['batch_size']=True;variants.append(data)
    for data in variants:
        path=tmp_path/'debug.yaml';path.write_text(yaml.safe_dump(data))
        with pytest.raises(ConfigError):load_flt_config(path)
    path=tmp_path/'bad.yaml'
    for data in ({**ref.data,'schema_version':True},{k:v for k,v in ref.data.items() if k!='architecture'},
                 {**ref.data,'unknown':1}):
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ConfigError):load_flt_config(path)
    path.write_text((root/'configs/flt_reference.yaml').read_text()+'profile: reference\n')
    with pytest.raises(ConfigError):load_flt_config(path)


def test_flt_t20_boundaries(flt,language_inputs):
    ids,v=language_inputs
    for a,b in [(ids[:0],v[:0]),(ids[:,:307],v),(ids.int(),v),(ids+50000,v),(ids-50000,v),
                (ids,v.squeeze(1)),(ids,v.expand(2,197,512)),(ids,v.double()),
                (ids,v*float('nan')),(ids[:1],v),([],v)]:
        with pytest.raises((TypeError,ValueError)):flt(a,b)
    low=flt.vocabulary(ids)
    for memory,visual in ((low[:,:307],v),(low,v[:1]),(low.double(),v),(low,v*float('inf'))):
        with pytest.raises((TypeError,ValueError)):flt.decode(low,memory,visual)
    with pytest.raises(TypeError):flt(ids,v,return_logits=1)
    with pytest.raises(ValueError):flt.project_vocabulary(low[:,:307])
