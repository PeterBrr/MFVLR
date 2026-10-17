"""N3 diagnosis only. One live full graph; selected gradients streamed to disk."""
from pathlib import Path
import argparse
import gc
import json
import os
import sys
import tempfile
import threading
import time
import weakref

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)]
OUT=ROOT/'artifacts/phase2d2_n3'

PARAMETERS=(
 'mve.shared_encoder.cnn.stages.0.0.weight',
 'mve.shared_encoder.blocks.1.attention.q.weight',
 'mve.shared_encoder.blocks.1.attention.k.weight',
 'mve.shared_encoder.blocks.1.attention.k.bias',
 'decoder.trunk.stages.0.conv1.weight','decoder.appearance_head.weight','decoder.mask_head.weight',
 'heads.detector.weight','heads.adapter.weight','objectives.cmc.log_temperature',
 'flt.vocabulary.weight',
 'flt.encoder.blocks.0.attention.q.weight','flt.encoder.blocks.0.attention.k.weight',
 'flt.encoder.blocks.2.attention.k.bias','flt.encoder.blocks.7.attention.q.weight',
 'flt.encoder.blocks.11.attention.q.weight',
 'flt.decoder.blocks.0.self_attention.q.weight','flt.decoder.blocks.0.self_attention.k.bias',
 'flt.decoder.blocks.6.self_attention.q.weight','flt.decoder.blocks.6.memory_attention.k.bias',
 'flt.encoder.blocks.0.vim.q.weight','flt.encoder.blocks.0.vim.k.weight',
 'flt.decoder.blocks.6.vim.q.weight','flt.decoder.blocks.6.vim.k.weight')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def forward(root,f,graph_label):
    from tools.n3_gradient_tools import identity,sha_tensor
    captured={};hooks=[]
    for prefix,blocks in (('LE',root.flt.encoder.blocks),('LD',root.flt.decoder.blocks)):
        for index,block in enumerate(blocks):
            if index not in (0,len(blocks)-1):continue
            name=f'{prefix}{index+1:02d}'
            hooks.append(block.register_forward_hook(lambda m,a,o,name=name:captured.__setitem__(name,o)))
    try:o=root.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
    finally:
        for hook in hooks:hook.remove()
    tensors={'fd_logits':o.predictions.detection_logits,'mask_logits':o.predictions.decoded.mask_logits,
        'adapter':o.predicted_language,'lr_logits':o.language.logits,'Iv':o.predictions.visual.fused_global,
        'Ipre':o.predictions.decoded.reconstruction,'Tl':o.language.global_language,**captured}
    values={'parameter.'+n:dict(root.named_parameters())[n] for n in PARAMETERS}
    values.update({'tensor.'+n:t for n,t in tensors.items()})
    record={'graph_label':graph_label,'output_python_id':id(o),'parameters':{n:identity(t) for n,t in values.items() if n.startswith('parameter.')},
            'tensors':{n:{**identity(t),'value_sha256':sha_tensor(t)} for n,t in tensors.items()}}
    # Hold Python grad_fn wrappers for identity stability within this graph only.
    anchors=[t.grad_fn for t in tensors.values()]
    record['grad_fn_python_ids']={n:id(fn) for n,fn in zip(tensors,anchors)}
    return o,values,record,anchors


def save_grad_tuple(store,keys,gradients):
    for key,gradient in zip(keys,gradients):store.save(key,gradient)


def vjp_controls(root,values,directory):
    import torch
    from tools.n3_gradient_tools import Store,compare_stores,sha_tensor,identity
    results={}
    for cut in ('tensor.Iv','tensor.LD07'):
        output=values[cut]
        selected={k:v for k,v in values.items() if k in (
            'tensor.Ipre','tensor.Iv','tensor.LE01','tensor.LE12',
            'parameter.mve.shared_encoder.cnn.stages.0.0.weight',
            'parameter.decoder.appearance_head.weight','parameter.flt.encoder.blocks.0.attention.q.weight',
            'parameter.flt.decoder.blocks.0.self_attention.q.weight') and v is not output}
        u=torch.randn(output.shape,generator=torch.Generator().manual_seed(901))/output.numel()**.5
        v=torch.randn(output.shape,generator=torch.Generator().manual_seed(902))/output.numel()**.5
        for scale in (1.,1024.):
            stores=[]
            for name,cotangent in (('u',u*scale),('v',v*scale),('u_plus_v',(u*scale)+(v*scale))):
                s=Store(directory/f'vjp_{cut}_{scale}_{name}')
                g=torch.autograd.grad(output,tuple(selected.values()),grad_outputs=cotangent,retain_graph=True,allow_unused=True)
                save_grad_tuple(s,selected,g);stores.append(s);del g
            reference=Store(directory/f'vjp_{cut}_{scale}_sum');reference.sum(stores[:2],selected)
            results[f'{cut}_scale{scale}']={'output_identity':identity(output),
                'target_identities':{k:identity(t) for k,t in selected.items()},
                'cotangent_seeds':[901,902],'cotangent_scale':scale,
                'unscaled_cotangent_hashes':[sha_tensor(u),sha_tensor(v)],
                'same_forward_graph':True,'comparison':compare_stores(stores[2],reference,selected)}
    return results


def same_graph(root,f,directory,run_vjp):
    import torch
    from tools.n3_gradient_tools import Store,compare_stores,identity,classification,modules
    from tools.phase2d2_diagnostics import ORDER
    o,values,graph_record,anchors=forward(root,f,'G0_same_forward_A_B')
    losses=root.compute_objectives(o,f.targets);components=losses.as_dict()
    for name in ORDER:values['loss.'+name]=components[name]
    graph_record['loss_nodes']={n:identity(components[n]) for n in (*ORDER,'total')}
    loss_anchors={n:t.grad_fn for n,t in components.items()}
    graph_record['loss_grad_fn_python_ids']={n:id(fn) for n,fn in loss_anchors.items()}
    anchors.extend(loss_anchors.values())
    del loss_anchors
    keys=list(values);references=[weakref.ref(v) for k,v in values.items() if not k.startswith('parameter.')]
    vjp=vjp_controls(root,values,directory) if run_vjp else {'status':'NOT RUN for this fixture; primary mixed fixture covers both cuts/scales'}
    actual=Store(directory/'A_joint')
    grad=torch.autograd.grad(losses.total,tuple(values.values()),retain_graph=True,allow_unused=True)
    save_grad_tuple(actual,keys,grad);del grad
    isolated=[]
    for i,name in enumerate(ORDER):
        s=Store(directory/f'B_{name}')
        grad=torch.autograd.grad(components[name],tuple(values.values()),retain_graph=i<len(ORDER)-1,allow_unused=True)
        save_grad_tuple(s,keys,grad);isolated.append(s);del grad
    summed=Store(directory/'B_sum');summed.sum(isolated,keys)
    comparison=compare_stores(actual,summed,keys)
    for key,r in comparison.items():r['mathematical_class']=classification(key)
    record={'graph':graph_record,'method':'A and B use autograd.grad on exactly the same selected input tuple and the same six scalar nodes',
        'losses':{n:float(t.detach()) for n,t in components.items()},'A':actual.public(),
        'B_components':{n:s.public() for n,s in zip(ORDER,isolated)},'B_sum':summed.public(),
        'A_vs_B':comparison,'selected_module_A_vs_B':modules(comparison),'VJP':vjp,
        'parameter_grad_fields_none':all(p.grad is None for p in root.parameters())}
    assert record['parameter_grad_fields_none']
    del o,values,components,losses,anchors;gc.collect()
    assert all(r() is None for r in references),'same-forward graph reference leaked'
    record['graph_released_before_C']=True
    return actual,isolated,summed,record,keys


def separate(root,f,directory,keys,B,expected_graph):
    import torch
    from tools.n3_gradient_tools import Store,compare_stores,identity
    from tools.phase2d2_diagnostics import ORDER,loss_only,stats
    stores=[];records={}
    for name in (*ORDER,'total'):
        root.zero_grad(set_to_none=True)
        o,values,graph,anchors=forward(root,f,'G_C_'+name)
        if name=='total':
            losses=root.compute_objectives(o,f.targets);loss=losses.total
            for n in ORDER:values['loss.'+n]=getattr(losses,n)
        else:
            loss=loss_only(root,o,f.targets,name);values['loss.'+name]=loss
        for key,t in values.items():
            if not key.startswith('parameter.'):t.retain_grad()
        loss.backward()
        s=Store(directory/f'C_{name}')
        for key in keys:s.save(key,values[key].grad if key in values else None)
        qk={n:{'allocated':p.grad is not None,'exact_zero':None if p.grad is None else bool(p.grad.count_nonzero()==0)}
            for n,p in root.named_parameters() if '.vim.q.weight' in n or '.vim.k.weight' in n}
        for n,r in qk.items():
            expected=name in ('lr','total') or ('.encoder.' in n and name in ('cmc','kl'))
            assert r['allocated']==expected and (not expected or r['exact_zero']),n
        same_values=all(r['value_sha256']==expected_graph['tensors'][n]['value_sha256'] for n,r in graph['tensors'].items())
        same_parameters=graph['parameters']==expected_graph['parameters']
        assert same_values and same_parameters,'forward/parameter drift requires review'
        all_finite=all(p.grad is None or bool(p.grad.isfinite().all()) for p in root.parameters());assert all_finite
        historical=ROOT/'artifacts/phase2d2_retry/g2'/('final_'+f.provenance['name'])/f'{name}.json'
        old=json.loads(historical.read_text())
        old_comparison={}
        for p in PARAMETERS:
            gradient=dict(root.named_parameters())[p].grad
            fresh={'allocated':gradient is not None}
            if gradient is not None:fresh.update(stats(gradient))
            old_comparison[p]={'new':fresh,'historical':old['gradients']['parameters'][p],
                'exact_statistic_match':fresh==old['gradients']['parameters'][p]}
        records[name]={'graph':graph,'loss_identity':identity(loss),'loss_value':loss.item(),
            'same_forward_values_as_G0':same_values,'same_parameter_identities_as_G0':same_parameters,
            'gradients':s.public(),'all_full_parameter_gradients_finite':all_finite,'all_38_VIM_QK':qk,
            'historical_gradient_comparison':old_comparison}
        if name!='total':records[name]['C_i_vs_B_i']=compare_stores(s,B[len(stores)],keys);stores.append(s)
        else:joint=s
        refs=[weakref.ref(tensor) for key,tensor in values.items() if not key.startswith('parameter.')]
        del o,values,graph,anchors,loss,t
        if name=='total':del losses
        root.zero_grad(set_to_none=True);gc.collect()
        assert all(ref() is None for ref in refs),'separate graph reference leaked'
    summed=Store(directory/'C_sum');summed.sum(stores,keys)
    return summed,joint,records


def worker(case,mode,out):
    import psutil
    import torch
    from mfvlr.integration import MFVLRIntegration,ownership_report
    from tools.phase2d2_fixtures import fixture
    from tools.phase2d2_diagnostics import state_hash
    from tools.n3_gradient_tools import selftest,compare_stores,modules
    from tools.review_phase2c2_n2 import backend
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    torch.backends.mkldnn.enabled=mode=='mkldnn'
    dest=out/f'{case}_{mode}';dest.mkdir(parents=True,exist_ok=False)
    rss=[];stop=threading.Event();process=psutil.Process();started=time.perf_counter()
    def sample():
        while not stop.is_set():rss.append(process.memory_info().rss);stop.wait(.01)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start()
    try:
        with tempfile.TemporaryDirectory(dir=dest,prefix='selected_gradients_') as temp:
            directory=Path(temp);assert directory.resolve().is_relative_to(dest.resolve())
            checkdir=directory/'selftest';checkdir.mkdir();checks=selftest(checkdir)
            root=MFVLRIntegration();f=fixture(case);initial=state_hash(root);ids=[id(p) for p in root.parameters()]
            ownership=ownership_report(root)
            A,B,Bsum,same,keys=same_graph(root,f,directory,case=='real_efs')
            write(dest/'same_graph.json',same)
            Csum,Cjoint,separate_records=separate(root,f,directory,keys,B,same['graph'])
            write(dest/'separate_graphs.json',separate_records)
            comparisons={name:compare_stores(x,y,keys) for name,x,y in (
                ('A_vs_C_sum',A,Csum),('B_sum_vs_C_sum',Bsum,Csum),('A_vs_C_joint',A,Cjoint),('C_joint_vs_C_sum',Cjoint,Csum))}
            report={'status':'DIAGNOSTIC_COMPLETE_PHASE_BLOCKED','fixture':f.provenance,'backend':backend(),
                'selftests':checks,'ownership':ownership,'initial_state_sha256':initial,
                'comparisons':comparisons,'selected_module_comparisons':{k:modules(v) for k,v in comparisons.items()},
                'selected_keys':keys,'maximum_simultaneous_full_graphs':1,
                'temporary_gradient_storage_removed_on_exit':True,'no_numerical_policy_amendment':True}
            assert state_hash(root)==initial and [id(p) for p in root.parameters()]==ids
            assert all(not m._forward_hooks for m in root.modules())
            report['parameters_and_identities_unchanged']=True;report['zero_optimizer_updates']=True
            write(dest/'summary.json',report)
    finally:
        stop.set();monitor.join();rss.append(process.memory_info().rss)
        write(dest/'resources.json',{'seconds':time.perf_counter()-started,'peak_sampled_RSS_bytes':max(rss),
            'nominal_interval_seconds':.01,'samples':len(rss),'available_RAM_bytes':psutil.virtual_memory().available})
    print(json.dumps({'case':case,'mode':mode,'status':'DIAGNOSTIC_COMPLETE_PHASE_BLOCKED'}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=('real','real_efs','local_duplicate'));parser.add_argument('--mode',choices=('mkldnn','native'),default='mkldnn');parser.add_argument('--output',type=Path,default=OUT/'same_forward');args=parser.parse_args()
    worker(args.worker,args.mode,args.output)


if __name__=='__main__':main()
