"""JOINT-I01--I12, approved contracts; full size, fixed original tolerances."""
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import tempfile

import pytest
import torch

from mfvlr.integration import ImageOnlyMFVLR,ownership_report
from mfvlr.objectives import unit_weight_sum,knowledge_distillation_loss as kl
from mfvlr.vim import VisionInjectionDelta
from tools.phase2d2_fixtures import fixture
from tools.phase2d2_diagnostics import controlled_forward,loss_only,FP32,FP64,stats,write
from tests.objectives.oracles import cmc_reference

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('MFVLR_PHASE2D2_EVIDENCE_ROOT', str(ROOT/'artifacts/phase2d2')))/'joint'


@pytest.mark.parametrize('batch',[1,2])
def test_joint_i01_single_owner_forward_dag(graph,batch):
    f=fixture('real' if batch==1 else 'real_efs');counts={};input_refs={}
    modules={'cnn':graph.mve.shared_encoder.cnn,'trunk':graph.decoder.trunk,'AD':graph.decoder.appearance_head,
             'MD':graph.decoder.mask_head,'FLT':graph.flt,'adapter':graph.heads.adapter,'detector':graph.heads.detector}
    handles=[]
    for name,module in modules.items():
        def hook(m,args,out,name=name):
            counts[name]=counts.get(name,0)+1
            if name in ('FLT','adapter'):input_refs[name]=args[-1 if name=='FLT' else 0]
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():o=graph.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
    finally:
        for h in handles:h.remove()
    assert counts=={'cnn':2,'trunk':1,'AD':1,'MD':1,'FLT':1,'adapter':1,'detector':1}
    v=o.predictions.visual
    assert input_refs['FLT'] is v.fused_global and input_refs['adapter'] is v.fused_global
    assert torch.equal(v.residual_image,(o.predictions.decoded.reconstruction-f.image).abs())
    assert torch.equal(v.fused_global,v.appearance.global_token+v.residual_global)
    assert o.language.logits.shape==(batch,308,49408)
    assert o.predictions.decoded.mask_logits.shape==(batch,2,224,224)
    assert len(v.appearance.skips)==4 and graph.decoder.skip_policy=='bottleneck_only'


def test_joint_i02_six_inputs_and_literal_sum(graph):
    f=fixture('real_efs')
    with torch.no_grad():
        o=graph.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
        losses=graph.compute_objectives(o,f.targets)
        components={name:loss_only(graph,o,f.targets,name) for name in ('fd','lr','cmc','fl','ar','kl')}
        for name,expected in components.items():assert torch.equal(getattr(losses,name),expected)
        assert torch.equal(losses.total,unit_weight_sum(**components))
    for bad in (replace(f.targets,sample_ids=tuple(reversed(f.targets.sample_ids))),
                replace(f.targets,token_ids=f.targets.token_ids.flip(0))):
        with pytest.raises(ValueError):graph.compute_objectives(o,bad)
    for sample_ids in (['same','same'],['one']):
        with pytest.raises(ValueError):graph.with_language(f.image,f.targets.token_ids,sample_ids=sample_ids)
    write(OUT/'I02.json',{n:v.item() for n,v in losses.as_dict().items()})


@pytest.mark.parametrize('name',['fd','lr','cmc','fl','ar','kl'])
def test_joint_i03_gradient_presence_matrix(graph,name):
    f=fixture('real_efs');o=graph.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
    params={'cnn':graph.mve.shared_encoder.cnn.stages[0][0].weight,
            'global':graph.mve.shared_encoder.blocks[0].fc1.weight,'trunk':graph.decoder.trunk.stages[0].conv1.weight,
            'AD':graph.decoder.appearance_head.weight,'MD':graph.decoder.mask_head.weight,
            'detector':graph.heads.detector.weight,'adapter':graph.heads.adapter.weight,
            'theta':graph.objectives.cmc.log_temperature,'LE':graph.flt.encoder.blocks[0].ff.fc1.weight,
            'LD':graph.flt.decoder.blocks[0].ff.fc1.weight,'Wvoc':graph.flt.vocabulary.weight,
            'Pe':graph.flt.encoder_position,'Pd':graph.flt.decoder_position}
    expected={'fd':{'cnn','global','trunk','AD','detector'},'lr':{'cnn','global','trunk','AD','LE','LD','Wvoc','Pe','Pd'},
              'cmc':{'cnn','global','trunk','AD','LE','Wvoc','Pe','theta'},'fl':{'cnn','trunk','MD'},
              'ar':{'cnn','trunk','AD'},'kl':{'cnn','global','trunk','AD','adapter','LE','Wvoc','Pe'}}[name]
    grads=torch.autograd.grad(loss_only(graph,o,f.targets,name),tuple(params.values()),allow_unused=True)
    records={}
    for (key,p),g in zip(params.items(),grads):
        assert (g is not None)==(key in expected),key
        records[key]={'allocated':g is not None,**(stats(g) if g is not None else {})}
    assert any(r.get('nonzero',0)>0 for r in records.values()),'saturated fixture cannot prove a nonzero path'
    write(OUT/f'I03_{name}.json',records)


@pytest.mark.parametrize('name',['fd','lr','cmc','kl'])
def test_joint_i04_reconstruction_mediated_detached_control(graph,name):
    f=fixture('real_efs');records={}
    for detached in (False,True):
        o=controlled_forward(graph,f,detach_reconstruction=detached)
        selected=(o.predictions.decoded.reconstruction,graph.decoder.appearance_head.weight,
                  graph.mve.shared_encoder.cnn.stages[0][0].weight)
        grads=torch.autograd.grad(loss_only(graph,o,f.targets,name),selected,allow_unused=True)
        if detached:assert grads[0] is None and grads[1] is None
        else:assert all(g is not None and g.isfinite().all() and g.norm()>0 for g in grads)
        assert grads[2] is not None and grads[2].norm()>0
        records[str(detached)]=[None if g is None else stats(g) for g in grads]
        del o,selected,grads;gc.collect()
    write(OUT/f'I04_{name}.json',records)


def test_joint_i05_ar_plus_residual_and_abs_subgradient(graph):
    from mfvlr.objectives import appearance_reconstruction_loss as ar,forgery_detection_loss as fd
    f=fixture('real');o=graph.visual_only(f.image);rec=o.decoded.reconstruction
    direct=ar(f.image,rec);indirect=fd(o.detection_logits,f.targets.labels)
    ga=torch.autograd.grad(direct,rec,retain_graph=True)[0]
    gf=torch.autograd.grad(indirect,rec,retain_graph=True)[0]
    joint=torch.autograd.grad(direct+indirect,rec)[0]
    torch.testing.assert_close(joint,ga+gf,**FP32)
    torch.testing.assert_close(ga,2*(rec-f.image)/rec.numel(),**FP32)
    x=torch.tensor([-2.,0.,3.],requires_grad=True);x.abs().sum().backward()
    assert torch.equal(x.grad,torch.tensor([-1.,0.,1.]))


def test_joint_i06_shared_trunk_and_local_loss_paths(graph):
    from mfvlr.objectives import appearance_reconstruction_loss as ar,forgery_localization_loss as fl
    f=fixture('real_local');o=graph.visual_only(f.image)
    selected=(graph.decoder.trunk.stages[0].conv1.weight,graph.mve.shared_encoder.cnn.stages[0][0].weight,
              graph.decoder.appearance_head.weight,graph.decoder.mask_head.weight,
              graph.mve.shared_encoder.blocks[0].fc1.weight,graph.flt.encoder.blocks[0].ff.fc1.weight)
    la=ar(f.image,o.decoded.reconstruction);lm=fl(o.decoded.mask_logits,f.targets.masks)
    ga=torch.autograd.grad(la,selected,retain_graph=True,allow_unused=True)
    gm=torch.autograd.grad(lm,selected,retain_graph=True,allow_unused=True)
    gj=torch.autograd.grad(la+lm,selected,allow_unused=True)
    assert ga[2] is not None and ga[3] is None and gm[2] is None and gm[3] is not None
    assert all(g is None for g in (*ga[4:],*gm[4:],*gj[4:]))
    for i in (0,1):
        assert ga[i].norm()>0 and gm[i].norm()>0
        torch.testing.assert_close(gj[i],ga[i]+gm[i],**FP32)


def test_joint_i07_kl_teacher_student_and_visual_conditioning(graph):
    f=fixture('real');o=graph.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
    iv=o.predictions.visual.fused_global;l=o.language.global_language;a=o.predicted_language
    teacher=torch.autograd.grad(kl(l,a.detach()),(iv,graph.heads.adapter.weight,graph.flt.encoder.blocks[0].ff.fc1.weight),retain_graph=True,allow_unused=True)
    student=torch.autograd.grad(kl(l.detach(),a),(iv,graph.heads.adapter.weight,graph.flt.encoder.blocks[0].ff.fc1.weight),retain_graph=True,allow_unused=True)
    joint=torch.autograd.grad(kl(l,a),(iv,graph.flt.decoder.blocks[0].ff.fc1.weight,graph.flt.decoder_position),allow_unused=True)
    assert teacher[0].norm()>0 and teacher[1] is None and teacher[2].norm()>0
    assert student[0].norm()>0 and student[1].norm()>0 and student[2] is None
    assert joint[1] is None and joint[2] is None
    torch.testing.assert_close(joint[0],teacher[0]+student[0],**FP32)
    expected_student=student[0].detach().clone()
    del o,iv,l,a,teacher,student,joint;gc.collect()
    control=controlled_forward(graph,f,detach_visual=True)
    iv=control.predictions.visual.fused_global
    g=torch.autograd.grad(kl(control.language.global_language,control.predicted_language),iv)[0]
    assert g.norm()>0 # adapter direct path remains
    torch.testing.assert_close(g,expected_student,**FP32)


def test_joint_i08_cmc_chain_rule_and_batch_coupling(graph):
    # Controlled nonsaturated partial derivatives at full 512 feature width.
    from tests.objectives.oracles import cmc_fixture
    v,l=cmc_fixture(dtype=torch.float64)
    from mfvlr.objectives import CrossModalContrastiveLoss
    cmc=CrossModalContrastiveLoss().double();value=cmc(v,l)
    oracle,_,gv,gl,_=cmc_reference(v[:,0].tolist(),l[:,0].tolist(),cmc.log_temperature.item())
    grads=torch.autograd.grad(value,(v,l))
    torch.testing.assert_close(grads[0][:,0],torch.tensor(gv,dtype=torch.float64),**FP64)
    torch.testing.assert_close(grads[1][:,0],torch.tensor(gl,dtype=torch.float64),**FP64)
    f=fixture('real_efs');o=graph.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
    iv=o.predictions.visual.fused_global;text=o.language.global_language
    assert torch.equal(text,o.language.encoder_hidden[:,307:308])
    full=graph.objectives.cmc(iv,text)
    direct=torch.autograd.grad(graph.objectives.cmc(iv,text.detach()),iv,retain_graph=True)[0]
    via=torch.autograd.grad(graph.objectives.cmc(iv.detach(),text),iv,retain_graph=True)[0]
    both=torch.autograd.grad(full,iv)[0]
    assert direct.norm()>0 and via.norm()>0
    v2=v.detach().clone().requires_grad_();l2=l.detach().clone();l2[1,0,0]+=1
    changed=torch.autograd.grad(cmc(v2,l2),v2)[0]
    assert not torch.equal(changed[0],grads[0][0])
    from tools.review_phase2c2_n2 import compare
    write(OUT/'I08.json',{'full_FP32_chain_rule':compare(both,direct+via,**FP32),
                        'independent_FP64_derivatives':'PASS','cross_sample_coupling':'PASS',
                        'direct_path_gradient':stats(direct),'language_conditioning_gradient':stats(via)})
    torch.testing.assert_close(both,direct+via,**FP32)


def test_joint_i09_all_singleton_vim_zero_qk(graph):
    f=fixture('real');o=graph.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
    graph.compute_objectives(o,f.targets).total.backward()
    vims=[m for m in graph.flt.modules() if isinstance(m,VisionInjectionDelta)]
    assert len(vims)==19
    for module in vims:
        assert module.q.weight.grad is not None and module.k.weight.grad is not None
        assert module.q.weight.grad.count_nonzero()==module.k.weight.grad.count_nonzero()==0
        assert module.v.weight.grad.isfinite().all() and module.out.weight.grad.isfinite().all()
        assert module.v.weight.grad.norm()>0 and module.out.weight.grad.norm()>0


def test_joint_i10_fp64_oracle_and_full_fp32_gradient_additivity():
    # Independent FP64 chain: J(sum_i L_i) == sum_i J(L_i).
    x=torch.tensor([.2,-.3,.7],dtype=torch.float64,requires_grad=True)
    losses=[x.square().sum(),(x.exp()).sum(),(x@torch.tensor([.3,.7,-.2],dtype=torch.float64))]
    components=[torch.autograd.grad(l,x,retain_graph=True)[0] for l in losses]
    actual=torch.autograd.grad(sum(losses),x)[0]
    torch.testing.assert_close(actual,sum(components),**FP64)
    # This mandatory full-parameter assertion is deliberately NOT exempted by N2.
    evidence=json.loads((OUT.parent/'g2/real/gradient_additivity.json').read_text())
    assert all(p['atol']==1e-6 and p['rtol']==1e-5 for p in evidence['parameters'].values())
    assert evidence['no_tolerance_changes']
    assert evidence['allclose'],f"full FP32 additivity failed; relative L2={evidence['relative_l2_difference']}; see preserved G2 evidence"


def test_joint_i11_visual_only_has_no_text_dependencies(graph,monkeypatch):
    f=fixture('real')
    def fail(*a,**kw):raise AssertionError('language/adapter/objective called during inference')
    with monkeypatch.context() as patch:
        for module in (graph.flt,graph.heads.adapter,graph.objectives):patch.setattr(module,'forward',fail)
        with torch.no_grad():o=graph(f.image)
    assert o.detection_logits.shape==(1,2) and o.decoded.mask_logits.shape==(1,2,224,224)
    with torch.no_grad():
        first=graph.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
        second=graph.with_language(f.image,fixture('efs').targets.token_ids,sample_ids=f.targets.sample_ids)
    assert torch.equal(first.predictions.detection_logits,second.predictions.detection_logits)
    assert torch.equal(first.predictions.decoded.mask_logits,second.predictions.decoded.mask_logits)


def test_joint_i12_ownership_and_full_serialization(graph):
    report=ownership_report(graph);assert report['total_unique']==154156232 and report['vocabulary_tied']
    deployment=ImageOnlyMFVLR();deploy=ownership_report(deployment)
    assert deploy['total_unique']==41135303
    assert set(dict(deployment.named_children()))=={'mve','decoder','detector'}
    assert torch.equal(deployment.detector.weight,graph.heads.detector.weight)
    del deployment;gc.collect()
    f=fixture('real')
    with torch.no_grad():before=graph.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
    expected=before.language.logits.detach().clone();del before;gc.collect()
    with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/phase2d2') as directory:
        assert Path(directory).resolve().is_relative_to((ROOT/'artifacts/phase2d2').resolve())
        path=Path(directory)/'state.pt';torch.save(graph.state_dict(),path)
        saved=torch.load(path,weights_only=True,mmap=True)
        assert all(torch.equal(value,saved[name]) for name,value in graph.state_dict().items())
        original_ids=[id(p) for p in graph.parameters()]
        graph.load_state_dict(saved)
        assert original_ids==[id(p) for p in graph.parameters()]
        assert graph.flt.vocabulary_output_weight is graph.flt.vocabulary.weight
        del saved
    with torch.no_grad():after=graph.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
    assert torch.equal(after.language.logits,expected)
    write(OUT/'I12.json',{'full':report,'deployment':deploy,'serialization_exact_output_and_parameter_identity':True})
