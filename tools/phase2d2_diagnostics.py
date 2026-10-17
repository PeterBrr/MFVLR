"""Test-only G2 instrumentation. Fresh graphs, streamed gradient evidence.

No model mutation/update. Test controls here never enter production APIs.
"""
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import tempfile
import time
import weakref

import numpy as np
import torch
from mfvlr.integration import MFVLRIntegration,LanguagePredictions,VisualPredictions
from mfvlr.objectives import (forgery_detection_loss as fd,language_reconstruction_loss as lr,
    forgery_localization_loss as fl,appearance_reconstruction_loss as ar,knowledge_distillation_loss as kl)
from mfvlr.vim import VisionInjectionDelta

ORDER=('fd','lr','cmc','fl','ar','kl')
FP32={'atol':1e-6,'rtol':1e-5}
FP64={'atol':1e-10,'rtol':1e-8}


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def stats(tensor):
    x=tensor.detach()
    if not bool(x.isfinite().all()):raise FloatingPointError('Nonfinite diagnostic tensor')
    norm=float(torch.linalg.vector_norm(x,dtype=torch.float64))
    return {'shape':list(x.shape),'finite':True,'l2':norm,'rms':norm/math.sqrt(x.numel()),
            'max_abs':x.abs().max().item(),'min':x.min().item(),'max':x.max().item(),
            'nonzero':int(x.count_nonzero())}


def state_hash(module):
    h=hashlib.sha256()
    for name,p in module.named_parameters():
        h.update(name.encode());h.update(memoryview(p.detach().contiguous().numpy()).cast('B'))
    return h.hexdigest()


class Capture:
    """Stats/weakrefs only; owns removable handles, never output tensors."""
    def __init__(self,root):
        self.root=root;self.handles=[];self.values={};self.refs=[]
    def __enter__(self):
        for name,block in [('LE'+str(i),b) for i,b in enumerate(self.root.flt.encoder.blocks)]+[('LD'+str(i),b) for i,b in enumerate(self.root.flt.decoder.blocks)]:
            def hook(module,args,out,name=name):
                self.values[name]=stats(out);self.refs.append(weakref.ref(out))
            self.handles.append(block.register_forward_hook(hook))
        return self
    def __exit__(self,*unused):
        for h in self.handles:h.remove()
        self.handles.clear()


def controlled_forward(root,f,*,detach_reconstruction=False,detach_visual=False):
    """Test reference composition, with explicitly requested detached controls."""
    image=f.image;appearance=root.mve.encode_appearance(image)
    decoded=root.decoder.decode_appearance(appearance)
    reconstruction=decoded.reconstruction.detach() if detach_reconstruction else decoded.reconstruction
    visual=root.mve(image,reconstruction,appearance_encoding=appearance)
    logits=root.heads.detect(visual.fused_global)
    language=root.flt(f.targets.token_ids,visual.fused_global.detach() if detach_visual else visual.fused_global)
    adapted=root.heads.align_language(visual.fused_global)
    return LanguagePredictions(image,f.targets.sample_ids,f.targets.token_ids,
         VisualPredictions(visual,decoded,logits),language,adapted)


def loss_only(root,output,targets,name):
    # Isolation computes only the selected loss, not a sum multiplied by zero.
    p=output.predictions
    if name=='fd':return fd(p.detection_logits,targets.labels)
    if name=='lr':return lr(output.language.logits,targets.token_ids)
    if name=='cmc':return root.objectives.cmc(p.visual.fused_global,output.language.global_language)
    if name=='fl':return fl(p.decoded.mask_logits,targets.masks)
    if name=='ar':return ar(output.image,p.decoded.reconstruction)
    if name=='kl':return kl(output.language.global_language,output.predicted_language)
    raise ValueError(name)


def gradient_report(root):
    parameters={};groups={}
    def group(name):
        if name.startswith('mve.shared_encoder.cnn'):return 'MVE.local'
        if name.startswith('mve.'):return 'MVE.global'
        if name.startswith('decoder.trunk'):return 'VD.trunk'
        if name.startswith('decoder.appearance_head'):return 'VD.AD'
        if name.startswith('decoder.mask_head'):return 'VD.MD'
        if name.startswith('heads.detector'):return 'detector'
        if name.startswith('heads.adapter'):return 'adapter'
        if name.startswith('objectives.'):return 'CMC.theta'
        if name=='flt.vocabulary.weight':return 'FLT.Wvoc'
        if name=='flt.encoder_position':return 'FLT.Pe'
        if name=='flt.decoder_position':return 'FLT.Pd'
        if name.startswith('flt.encoder'):return 'FLT.LE'
        return 'FLT.LD'
    for name,p in root.named_parameters():
        record={'allocated':p.grad is not None}
        if p.grad is not None:record.update(stats(p.grad))
        parameters[name]=record
        category=groups.setdefault(group(name),{'allocated_tensors':0,'nonzero_elements':0,'l2_squared':0.,'max_abs':0.,'finite':True})
        if p.grad is not None:
            category['allocated_tensors']+=1;category['nonzero_elements']+=record['nonzero']
            category['l2_squared']+=record['l2']**2;category['max_abs']=max(category['max_abs'],record['max_abs'])
    for value in groups.values():value['l2']=math.sqrt(value.pop('l2_squared'))
    return {'parameters':parameters,'modules':groups}


def output_report(root,o,f):
    p=o.predictions;language=o.language
    values={'Iv':p.visual.fused_global,'Tl':language.global_language,'adapted':o.predicted_language,
            'logits_fd':p.detection_logits,'logits_fl':p.decoded.mask_logits,'logits_lr':language.logits,
            'Ipre':p.decoded.reconstruction,'residual':p.visual.residual_image,
            'LE':language.encoder_hidden,'LD':language.decoder_hidden}
    with torch.no_grad():
        targets=f.targets.token_ids
        nll=-language.logits.log_softmax(-1).gather(-1,targets[...,None])[...,0]
        masks={'padding':targets==0,'BOS':targets==49406,'EOS':targets==49407,
               'content':(targets!=0)&(targets!=49406)&(targets!=49407)}
        contributions={name:{'count':int(mask.sum()),'batch_mean_nll':float(nll[mask].sum()/len(targets))} for name,mask in masks.items()}
        scores=root.objectives.cmc.scores(p.visual.fused_global,language.global_language)
        def entropy(logp):return (-(logp.exp().double()*logp.double()).sum(-1)).tolist()
        text=(language.global_language/.5).log_softmax(-1);student=(o.predicted_language/.5).log_softmax(-1)
        diff=p.decoded.reconstruction-f.image
        return {'tensors':{k:stats(v) for k,v in values.items()},'LR_categories':contributions,
                'CMC':{'raw_scores':(p.visual.fused_global[:,0]@language.global_language[:,0].T).tolist(),
                       'scores':scores.tolist(),'tau':root.objectives.cmc.temperature.item(),
                       'theta':root.objectives.cmc.log_temperature.item(),'row_entropy':entropy(scores.log_softmax(1)),
                       'column_entropy':entropy(scores.T.log_softmax(1)),'prompt_unique_count':f.provenance['prompt_unique_count']},
                'KL':{'teacher_entropy':entropy(text),'student_entropy':entropy(student),
                      'teacher_max_probability':text.exp().max(-1).values.tolist(),
                      'student_max_probability':student.exp().max(-1).values.tolist()},
                'reconstruction':{'outside_01_count':int(((p.decoded.reconstruction<0)|(p.decoded.reconstruction>1)).sum()),
                                  'elements':diff.numel(),'residual_positive_signs':int((diff>0).sum()),
                                  'residual_negative_signs':int((diff<0).sum()),'residual_exact_zeros':int((diff==0).sum())}}


class GradientStream:
    """Bounded-memory FP64 accumulator file, no six gradient tensors retained."""
    def __init__(self,root,directory):
        self.entries=[];offset=0
        for name,p in root.named_parameters():
            self.entries.append((name,p,offset));offset+=p.numel()*8
        self.file=tempfile.TemporaryFile(dir=directory);self.file.truncate(offset)
        self.component_norms={};self.bytes=offset;self.presence={};self.selected={}
    def add(self,name):
        self.presence[name]=[];norm_sq=0.;self.selected[name]={}
        for key,p,offset in self.entries:
            if p.grad is None:continue
            self.presence[name].append(key);flat=p.grad.detach().reshape(-1)
            if key in ('mve.shared_encoder.cnn.stages.0.0.weight','decoder.appearance_head.weight',
                       'flt.encoder.blocks.7.attention.q.weight'):
                self.selected[name][key]=flat.double().clone()
            norm_sq+=float(torch.linalg.vector_norm(flat,dtype=torch.float64))**2
            for start in range(0,flat.numel(),262144):
                count=min(262144,flat.numel()-start);self.file.seek(offset+start*8)
                acc=np.frombuffer(self.file.read(count*8),dtype=np.float64).copy()
                acc+=flat[start:start+count].double().numpy()
                self.file.seek(offset+start*8);self.file.write(acc.tobytes())
        self.component_norms[name]=math.sqrt(norm_sq)
    def compare_joint(self):
        reports={};allclose=True;sum_norm_sq=joint_norm_sq=error_sq=0.
        for key,p,offset in self.entries:
            participated=any(key in keys for keys in self.presence.values())
            if (p.grad is not None)!=participated:raise AssertionError(f'gradient presence sum mismatch: {key}')
            if p.grad is None:continue
            flat=p.grad.detach().reshape(-1);outside=0;max_abs=0.;err=ref_sq=joint_sq=0.
            for start in range(0,flat.numel(),262144):
                count=min(262144,flat.numel()-start);self.file.seek(offset+start*8)
                reference=torch.from_numpy(np.frombuffer(self.file.read(count*8),dtype=np.float64).copy())
                actual=flat[start:start+count];delta=actual.double()-reference
                outside+=int((~torch.isclose(actual,reference.float(),**FP32)).sum())
                max_abs=max(max_abs,delta.abs().max().item());err+=float(delta.square().sum())
                ref_sq+=float(reference.square().sum());joint_sq+=float(actual.double().square().sum())
            reports[key]={'outside_tolerance_count':outside,'elements':p.numel(),'max_abs_difference':max_abs,
                          'relative_l2_difference':math.sqrt(err/ref_sq) if ref_sq else None,
                          'allclose':outside==0,'atol':FP32['atol'],'rtol':FP32['rtol']}
            allclose=allclose and outside==0;sum_norm_sq+=ref_sq;joint_norm_sq+=joint_sq;error_sq+=err
        sum_component=sum(self.component_norms.values())
        pair_cosines=[]
        loss_names=list(self.component_norms)
        for i,first in enumerate(loss_names):
            for second in loss_names[i+1:]:
                for key in set(self.selected[first])|set(self.selected[second]):
                    a=self.selected[first].get(key);b=self.selected[second].get(key)
                    if a is None or b is None or a.norm()==0 or b.norm()==0:cosine=None
                    else:cosine=float((a@b)/(a.norm()*b.norm()))
                    pair_cosines.append({'parameter':key,'losses':[first,second],'cosine':cosine,
                                         'undefined_if_unused_or_zero':cosine is None})
        return {'allclose':allclose,'parameters':reports,'component_gradient_l2':self.component_norms,
                'joint_l2':math.sqrt(joint_norm_sq),'sum_isolated_l2':math.sqrt(sum_norm_sq),
                'relative_l2_difference':math.sqrt(error_sq/sum_norm_sq) if sum_norm_sq else None,
                'cancellation_ratio_joint_to_sum_component_norms':math.sqrt(joint_norm_sq)/sum_component if sum_component else None,
                'relative_component_gradient_norms':{n:v/sum_component if sum_component else None for n,v in self.component_norms.items()},
                'selected_loss_gradient_pair_cosines':pair_cosines,
                'FP64_accumulator_file_bytes':self.bytes,'chunk_elements':262144,'no_tolerance_changes':True}
    def close(self):self.file.close();self.entries.clear();self.selected.clear()


def run_g2(root,f,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    initial=state_hash(root);identities=[id(p) for p in root.parameters()];records={}
    stream=GradientStream(root,directory)
    try:
        for name in (*ORDER,'total'):
            root.zero_grad(set_to_none=True);started=time.perf_counter()
            with Capture(root) as capture:
                output=root.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
            assert len(capture.values)==19
            predictions_refs=[weakref.ref(output.language.logits),weakref.ref(output.predictions.visual.fused_global)]
            outputs=output_report(root,output,f)
            if name=='total':
                losses=root.compute_objectives(output,f.targets);loss=losses.total
                scalars={key:value.item() for key,value in losses.as_dict().items()}
            else:loss=loss_only(root,output,f.targets,name);scalars={name:loss.item()}
            if not bool(loss.isfinite()):raise FloatingPointError(f'{name} nonfinite')
            output.predictions.decoded.reconstruction.retain_grad();output.predictions.visual.fused_global.retain_grad()
            output.language.global_language.retain_grad();output.predicted_language.retain_grad()
            loss.backward()
            report=gradient_report(root)
            for n,p in root.named_parameters():
                if '.vim.q.weight' in n or '.vim.k.weight' in n:
                    expected_connected=name in ('lr','total') or ('.encoder.' in n and name in ('cmc','kl'))
                    if expected_connected:assert p.grad is not None and torch.count_nonzero(p.grad)==0,n
                    else:assert p.grad is None,n
            intermediates={key:({'allocated':False} if tensor.grad is None else {'allocated':True,**stats(tensor.grad)})
                for key,tensor in [('Ipre',output.predictions.decoded.reconstruction),('Iv',output.predictions.visual.fused_global),
                                   ('Tl',output.language.global_language),('adapted',output.predicted_language)]}
            if name!='total':stream.add(name)
            else:comparison=stream.compare_joint();write(directory/'gradient_additivity.json',comparison)
            records[name]={'scalars':scalars,'activations':capture.values,'outputs':outputs,
                           'gradients':report,'intermediate_gradients':intermediates,'elapsed_seconds':time.perf_counter()-started}
            write(directory/f'{name}.json',records[name])
            del output,loss,report,outputs,capture,intermediates
            if name=='total':del losses
            root.zero_grad(set_to_none=True);gc.collect()
            assert all(ref() is None for ref in predictions_refs),'diagnostic retained autograd output'
            assert all(not module._forward_hooks for module in root.modules()),'hook leak'
        assert state_hash(root)==initial and [id(p) for p in root.parameters()]==identities
        result={'status':'PASS' if comparison['allclose'] else 'FAIL_GRADIENT_ADDITIVITY',
                'fixture':f.provenance,'losses':{n:records['total']['scalars'][n] for n in (*ORDER,'total')},
                'joint_comparison':{k:v for k,v in comparison.items() if k!='parameters'},
                'parameter_bytes_unchanged':True,'parameter_identities_unchanged':True,
                'initialized_state_sha256':initial,
                'all_valid_values_and_gradients_finite':True,'all_19_blocks_captured':True,
                'zero_optimizer_updates':True,'graph_and_hook_lifecycle':'PASS'}
        write(directory/'summary.json',result)
        return result
    finally:stream.close();root.zero_grad(set_to_none=True);gc.collect()
