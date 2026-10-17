"""VD-T01--VD-T14, approved Option B; no N1 tolerance used here."""
import copy
from dataclasses import fields, replace
import math

import pytest
import torch
from torch import nn
import yaml

from mfvlr.config import ConfigError, merge_overlay
from mfvlr.decoder_config import load_decoder_config
from mfvlr.mve import AppearanceEncoding
from mfvlr.vision_decoder import VisionDecoder, VisionDecoderOutput, decoder_parameter_report


def close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.shapes
@pytest.mark.parametrize("batch", [1, 2])
def test_vd_t01_layer_shapes(decoder, batch):
    shapes = {}
    handles = []
    for name, module in decoder.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.ReLU)):
            handles.append(module.register_forward_hook(lambda m, args, out, name=name: shapes.__setitem__(name, tuple(out.shape))))
    try:
        with torch.no_grad():
            result = decoder(torch.ones(batch, 1024, 14, 14))
    finally:
        for handle in handles:
            handle.remove()
    for i, (width, size) in enumerate(((512, 28), (256, 56), (128, 112), (64, 224))):
        for key in ("up", "conv1", "relu1", "conv2", "relu2"):
            assert shapes[f"trunk.stages.{i}.{key}"] == (batch, width, size, size)
    assert shapes["appearance_head"] == (batch, 3, 224, 224)
    assert shapes["mask_head"] == (batch, 2, 224, 224)
    assert result.features.shape == (batch, 64, 224, 224)
    print(f"VD-T01 b={batch}: all 22 operator output shapes verified")


@pytest.mark.backward
def test_vd_t02_independent_transpose_calculation(decoder):
    operation = nn.ConvTranspose2d(2, 3, 2, stride=2, bias=True).double()
    with torch.no_grad():
        operation.weight.copy_(decoder.trunk.stages[0].up.weight[:2, :3].double())
        operation.bias.copy_(torch.tensor([-0.3, 0.2, 0.7], dtype=torch.float64))
    image = torch.tensor([[[[1., -2., 3.], [4., 5., -6.]], [[0.5, 1., 2.], [-3., 0., 7.]]]],
                         dtype=torch.float64, requires_grad=True)
    actual = operation(image)
    manual = torch.stack([
        torch.stack([
            torch.stack([operation.bias[c] + sum(image[0, i, row // 2, col // 2] *
                          operation.weight[i, c, row % 2, col % 2] for i in range(2))
                         for col in range(6)]) for row in range(4)]) for c in range(3)
    ]).unsqueeze(0)
    torch.testing.assert_close(actual, manual, atol=1e-10, rtol=1e-8)
    selected = (image, operation.weight, operation.bias)
    grad_actual = torch.autograd.grad(actual.square().sum(), selected, retain_graph=True)
    grad_manual = torch.autograd.grad(manual.square().sum(), selected)
    for first, second in zip(grad_actual, grad_manual):
        torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-8)


@pytest.mark.shapes
def test_vd_t03_bottleneck_dependency(decoder, local):
    record = AppearanceEncoding(local, (), torch.empty(0), torch.empty(0))
    with torch.no_grad():
        first = decoder.decode_appearance(record)
        poisoned = replace(record, skips=(torch.full((1,), float("nan")),),
                           tokens=torch.full((1,), float("nan")), global_token=torch.full((1,), float("inf")))
        second = decoder.decode_appearance(poisoned)
    for key in ("features", "reconstruction", "mask_logits"):
        assert torch.equal(getattr(first, key), getattr(second, key))
    for skips in ((), [], (torch.empty(0),)):
        with pytest.raises(ValueError, match="rejects"):
            decoder(local, skips=skips)
    with pytest.raises(ValueError, match="Only approved"):
        VisionDecoder(skip_policy="concatenate_corresponding_appearance_skip")
    with pytest.raises(TypeError):
        VisionDecoder()


@pytest.mark.shapes
def test_vd_t04_ownership_call_counts(decoder, model, local):
    calls = {}
    handles = [module.register_forward_hook(lambda m, args, output, name=name: calls.__setitem__(name, calls.get(name, 0)+1))
               for name, module in (("trunk", decoder.trunk), ("ad", decoder.appearance_head), ("md", decoder.mask_head))]
    try:
        with torch.no_grad():
            decoder(local)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == {"trunk": 1, "ad": 1, "md": 1}
    assert list(decoder._modules) == ["trunk", "appearance_head", "mask_head"]
    params = list(decoder.parameters())
    assert len(params) == len({id(p) for p in params}) == 28
    assert {id(p) for p in params}.isdisjoint(id(p) for p in model.parameters())
    optimizer = torch.optim.SGD(params, lr=0.1)
    assert len({id(p) for group in optimizer.param_groups for p in group["params"]}) == 28
    assert decoder.appearance_head.weight is not decoder.mask_head.weight
    assert all(key.startswith(("trunk.", "appearance_head.", "mask_head.")) for key in decoder.state_dict())


@pytest.mark.shapes
def test_vd_t05_exact_counts(decoder, model):
    report = decoder_parameter_report(decoder)
    assert report == dict(U1=6817280, U2=1704704, U3=426368, U4=106688,
                          trunk=9055040, appearance_head=195, mask_head=130, total_unique=9055365)
    assert sum(p.numel() for p in model.parameters()) == 32078912
    assert report["total_unique"] + sum(p.numel() for p in model.parameters()) == 41134277
    print("VD-T05 actual count:", report)


@pytest.mark.shapes
def test_vd_t06_affine_heads_and_range(decoder, local):
    sample = torch.randn(1, 64, 2, 3, generator=torch.Generator().manual_seed(21))
    for head in (decoder.appearance_head, decoder.mask_head):
        actual = head(sample)
        manual = torch.einsum("oc,bchw->bohw", head.weight[:, :, 0, 0], sample) + head.bias[None, :, None, None]
        close(actual, manual)
    original = copy.deepcopy({"ad": decoder.appearance_head.state_dict(), "md": decoder.mask_head.state_dict()})
    try:
        with torch.no_grad():
            decoder.appearance_head.weight.zero_()
            decoder.appearance_head.bias.copy_(torch.tensor([-2., 2., 0.5]))
            decoder.mask_head.weight.zero_()
            decoder.mask_head.bias.copy_(torch.tensor([-3., 4.]))
            result = decoder(local)
        assert torch.equal(result.reconstruction, torch.tensor([-2., 2., 0.5])[None,:,None,None].expand_as(result.reconstruction))
        assert torch.equal(result.mask_logits, torch.tensor([-3., 4.])[None,:,None,None].expand_as(result.mask_logits))
    finally:
        decoder.appearance_head.load_state_dict(original["ad"])
        decoder.mask_head.load_state_dict(original["md"])


@pytest.mark.shapes
def test_vd_t07_exact_topology(decoder):
    assert sum(isinstance(m, nn.ConvTranspose2d) for m in decoder.modules()) == 4
    assert sum(isinstance(m, nn.Conv2d) for m in decoder.modules()) == 10
    assert sum(isinstance(m, nn.ReLU) for m in decoder.modules()) == 8
    assert not any(isinstance(m, (nn.Dropout, nn.BatchNorm2d, nn.GroupNorm, nn.Upsample)) for m in decoder.modules())
    for stage, (cin, cout) in zip(decoder.trunk.stages, ((1024,512),(512,256),(256,128),(128,64))):
        up = stage.up
        assert (up.in_channels, up.out_channels) == (cin,cout)
        assert up.kernel_size == up.stride == (2,2)
        assert up.padding == up.output_padding == (0,0)
        assert up.dilation == (1,1) and up.groups == 1 and up.bias is not None
        assert list(stage._modules) == ["up","conv1","relu1","conv2","relu2"]
        for conv in (stage.conv1,stage.conv2):
            assert conv.in_channels == conv.out_channels == cout
            assert conv.kernel_size == (3,3) and conv.stride == conv.padding == conv.dilation == (1,1)
            assert conv.groups == 1 and conv.bias is not None
        assert not stage.relu1.inplace and not stage.relu2.inplace
    for head, channels in ((decoder.appearance_head,3),(decoder.mask_head,2)):
        assert head.in_channels == 64 and head.out_channels == channels
        assert head.kernel_size == head.stride == (1,1) and head.padding == (0,0)


@pytest.mark.serialization
def test_vd_t08_initialization_replay(decoder, model):
    global_rng = torch.get_rng_state().clone()
    version_counters = {name:p._version for name,p in model.named_parameters()}
    replica = VisionDecoder(skip_policy="bottleneck_only", seed=0)
    assert torch.equal(global_rng, torch.get_rng_state())
    assert version_counters == {name:p._version for name,p in model.named_parameters()}
    for key, tensor in decoder.state_dict().items():
        assert torch.equal(tensor, replica.state_dict()[key])
    rng = torch.Generator().manual_seed(0)
    for stage in decoder.trunk.stages:
        for index, layer in enumerate((stage.up,stage.conv1,stage.conv2)):
            expected = torch.empty_like(layer.weight)
            if index == 0:
                bound = math.sqrt(6/(4*(layer.in_channels+layer.out_channels)))
                expected.uniform_(-bound,bound,generator=rng)
                assert layer.weight.abs().max() <= bound
            else:
                standard_deviation = math.sqrt(2/(9*layer.in_channels))
                expected.normal_(0,standard_deviation,generator=rng)
                assert abs(layer.weight.std().item()/standard_deviation-1) < .02
            assert torch.equal(expected,layer.weight)
            assert torch.count_nonzero(layer.bias) == 0
    for head in (decoder.appearance_head,decoder.mask_head):
        bound = math.sqrt(6/(64+head.out_channels))
        expected=torch.empty_like(head.weight).uniform_(-bound,bound,generator=rng)
        assert torch.equal(expected,head.weight)
        assert torch.count_nonzero(head.bias) == 0


@pytest.mark.shapes
def test_vd_t09_boundary_errors(decoder, local):
    for invalid in (None, [], 7):
        with pytest.raises(TypeError):
            decoder(invalid)
    for shape in ((1024,14,14),(0,1024,14,14),(1,512,14,14),(1,1024,13,14)):
        with pytest.raises(ValueError):
            decoder(torch.zeros(shape))
    with pytest.raises(TypeError):
        decoder(local.double())
    with pytest.raises(ValueError,match="device"):
        decoder(torch.empty(local.shape,device="meta"))
    for value in (float("nan"),float("inf")):
        with pytest.raises(ValueError,match="finite"):
            decoder(torch.full_like(local,value))
    with pytest.raises(TypeError):
        decoder.decode_appearance(local)
    for seed in (-1,True,0.1):
        with pytest.raises(ValueError):
            VisionDecoder(skip_policy="bottleneck_only",seed=seed)


@pytest.mark.shapes
def test_vd_t10_modes_and_fixed_batch_independence(decoder, local):
    with torch.no_grad():
        decoder.train(); train=decoder(local)
        decoder.eval(); evaluation=decoder(local)
        batch=torch.cat((local,1-local))
        together=decoder(batch)
        changed=decoder(torch.cat((local,torch.zeros_like(local))))
        permuted=decoder(batch.flip(0))
    for field in fields(VisionDecoderOutput):
        name=field.name
        close(getattr(train,name),getattr(evaluation,name))
        torch.testing.assert_close(getattr(together,name)[:1],getattr(changed,name)[:1],atol=0,rtol=0)
        torch.testing.assert_close(getattr(together,name),getattr(permuted,name).flip(0),atol=0,rtol=0)


@pytest.mark.serialization
def test_vd_t11_serialization(decoder, local, tmp_path):
    path=tmp_path/'decoder.pt'
    torch.save({'skip_policy':decoder.skip_policy,'state_dict':decoder.state_dict()},path)
    checkpoint=torch.load(path,weights_only=True)
    restored=VisionDecoder(skip_policy=checkpoint['skip_policy'],seed=1)
    restored.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():
        first=decoder(local); second=restored(local)
    for field in fields(VisionDecoderOutput):
        assert torch.equal(getattr(first,field.name),getattr(second,field.name))
    damaged=dict(checkpoint['state_dict']); damaged['appearance_head.weight']=torch.empty(4,64,1,1)
    with pytest.raises(RuntimeError):
        restored.load_state_dict(damaged)
    assert len(list(restored.parameters())) == 28 and not list(restored.buffers())


@pytest.mark.backward
def test_vd_t12_head_gradient_sum_and_unique_update(decoder, local):
    params=tuple(decoder.parameters())
    result=decoder(local)
    ad=result.reconstruction.square().mean(); md=result.mask_logits.square().mean()
    first=torch.autograd.grad(ad,params,allow_unused=True,retain_graph=True)
    second=torch.autograd.grad(md,params,allow_unused=True,retain_graph=True)
    combined=torch.autograd.grad(ad+md,params,allow_unused=True)
    for parameter,a,b,total in zip(params,first,second,combined):
        expected=(torch.zeros_like(parameter) if a is None else a)+(torch.zeros_like(parameter) if b is None else b)
        close(total,expected)
    assert first[-2] is None and first[-1] is None
    assert second[-4] is None and second[-3] is None
    chosen=params[0]; original=chosen.detach().clone()
    try:
        chosen.grad=combined[0]
        optimizer=torch.optim.SGD([chosen],lr=.1)
        optimizer.step()
        close(chosen,original-.1*combined[0])
    finally:
        with torch.no_grad(): chosen.copy_(original)
        chosen.grad=None
    print('VD-T12: AD/MD trunk gradient sum verified; one SGD unit update restored')


@pytest.mark.serialization
def test_vd_t13_strict_configuration(root,tmp_path):
    reference=load_decoder_config(root/'configs/decoder_reference.yaml')
    debug=load_decoder_config(root/'configs/decoder_debug.yaml')
    reference.require_approval(); debug.require_approval()
    for key in ('decoder','initialization','integration','verification'):
        assert reference.proposal[key] == debug.proposal[key]
    assert debug.proposal['diagnostics']['note'] is None
    assert debug.proposal['diagnostics']['batch_size']==2
    assert debug.proposal['verification']['n1_applies_to_decoder'] is False
    assert debug.proposal['verification']['fp32_atol']==1e-6
    value={'list':[1,2],'scalar':7,'nested':{'clear':'x','keep':1}}
    assert merge_overlay(value,{'list':[3],'scalar':None,'nested':{'clear':None}})=={'list':[3],'scalar':None,'nested':{'clear':None,'keep':1}}
    assert value['list']==[1,2]
    raw=yaml.safe_load((root/'configs/decoder_reference.yaml').read_text(encoding='utf-8'))
    overlay=yaml.safe_load((root/'configs/decoder_debug.yaml').read_text(encoding='utf-8'))
    raw['mve_config']=str(reference.mve_path)
    base_path=tmp_path/'base.yaml'; overlay_path=tmp_path/'overlay.yaml'; overlay['base']='base.yaml'
    def write(value,path): path.write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
    write(raw,base_path);write(overlay,overlay_path)
    assert load_decoder_config(overlay_path).proposal==debug.proposal
    invalid=[]
    for path,value in (
        (('schema_version',),2),(('unknown',),1),
        (('proposal','decoder','skip_policy'),'skip_connected'),
        (('proposal','decoder','channels'),[[1024,512]]),
        (('proposal','verification','fp32_atol'),5e-6),
        (('proposal','initialization','seed'),True),
        (('proposal','integration','detach_reconstruction'),True),
        (('review','training_authorized'),True),(('review','selected_option'),'A'),
        (('review','option_a_status'),'APPROVED'),(('review','scopes'),['vision_decoder','training']),
        (('proposal','diagnostics','optimizer_updates'),True)):
        variant=copy.deepcopy(raw);target=variant
        for key in path[:-1]:target=target[key]
        target[path[-1]]=value; invalid.append(variant)
    variant=copy.deepcopy(raw);del variant['proposal']['decoder']['dropout'];invalid.append(variant)
    for variant in invalid:
        write(variant,base_path)
        with pytest.raises(ConfigError):load_decoder_config(base_path)
    write(raw,base_path)
    bad_overlays=[]
    variant=copy.deepcopy(overlay);variant['base']='overlay.yaml';bad_overlays.append(variant)
    variant=copy.deepcopy(overlay);variant['overrides']['decoder']={'bogus':1};bad_overlays.append(variant)
    variant=copy.deepcopy(overlay);variant['merge_policy']['propagate_approval']=True;bad_overlays.append(variant)
    variant=copy.deepcopy(overlay);variant['overrides']['runtime']={'amp':'false'};bad_overlays.append(variant)
    for variant in bad_overlays:
        write(variant,overlay_path)
        with pytest.raises(ConfigError):load_decoder_config(overlay_path)
    for unapproved_path in (base_path,overlay_path):
        write(raw,base_path);write(overlay,overlay_path)
        value=copy.deepcopy(raw if unapproved_path==base_path else overlay)
        value['review']['status']='PROPOSED';value['review']['decisions']['D01']='PROPOSED'
        write(value,unapproved_path)
        with pytest.raises(ConfigError,match='approval'):
            VisionDecoder.from_config(load_decoder_config(overlay_path))
    overlay_path.write_text('schema_version: 1\nschema_version: 1\n',encoding='utf-8')
    with pytest.raises(ConfigError,match='Duplicate'):load_decoder_config(overlay_path)


@pytest.mark.backward
def test_vd_t14_record_graph_and_lifetime(decoder,local):
    assert [field.name for field in fields(VisionDecoderOutput)]==['features','reconstruction','mask_logits']
    leaf=local.clone().requires_grad_()
    result=decoder(leaf)
    assert isinstance(result,VisionDecoderOutput)
    assert all(getattr(result,key).grad_fn is not None for key in ('features','reconstruction','mask_logits'))
    gradient=torch.autograd.grad(result.reconstruction.square().mean(),leaf)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0
    assert set(decoder._modules)=={'trunk','appearance_head','mask_head'}
    assert not any(isinstance(value,VisionDecoderOutput) for value in vars(decoder).values())
    with torch.no_grad():
        result=decoder(local)
    assert result.reconstruction.grad_fn is None
