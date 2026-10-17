"""Fixed-cotangent seven-block LD reference at captured full-model inputs."""
from pathlib import Path
import argparse
import copy
import gc
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)]


def main():
    import torch
    import psutil
    from mfvlr.integration import MFVLRIntegration
    from tools.phase2d2_fixtures import fixture
    from tools.phase2d2_diagnostics import FP32,FP64,stats,state_hash,write
    from tools.n3_gradient_tools import sha_tensor,identity
    from tools.review_phase2c2_n2 import compare,backend
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'artifacts/phase2d2_n3/decoder_vjp.json');args=parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    started=time.perf_counter();root=MFVLRIntegration();root_hash=state_hash(root);f=fixture('real_efs');captured={}
    def hook(m,a,o):captured.update(inputs=tuple(t.clone() for t in a),output=o.clone())
    handle=root.flt.decoder.register_forward_hook(hook)
    try:
        with torch.no_grad():output=root.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
    finally:handle.remove()
    source=copy.deepcopy(root.flt.decoder);source_hash=state_hash(source)
    assert root_hash==state_hash(root)
    del root,output;gc.collect()
    shape=captured['output'].shape
    u=torch.randn(shape,generator=torch.Generator().manual_seed(901))/captured['output'].numel()**.5
    v=torch.randn(shape,generator=torch.Generator().manual_seed(902))/captured['output'].numel()**.5
    records=[]
    for dtype in (torch.float32,torch.float64):
        model=copy.deepcopy(source);assert state_hash(model)==source_hash
        model.to(dtype=dtype);inputs=tuple(x.to(dtype).clone().requires_grad_() for x in captured['inputs'])
        activations={};handles=[]
        for index,block in enumerate(model.blocks):
            handles.append(block.register_forward_hook(lambda m,a,o,index=index:activations.__setitem__(f'LD{index+1:02d}',stats(o))))
        try:y=model(*inputs)
        finally:
            for handle in handles:handle.remove()
        names=('state','memory','visual','blocks.0.self_attention.q.weight','blocks.0.self_attention.k.bias',
               'blocks.6.memory_attention.k.bias','blocks.6.vim.q.weight','blocks.6.vim.k.weight')
        pp=dict(model.named_parameters());targets=inputs+tuple(pp[n] for n in names[3:])
        for scale in (1.,1024.):
            # The FP32 values of u/v are promoted before the reference addition.
            uu=u.to(dtype)*scale;vv=v.to(dtype)*scale;gg={}
            for name,c in (('u',uu),('v',vv),('sum',uu+vv),('rounded_sum',(u*scale+v*scale).to(dtype))):
                gs=torch.autograd.grad(y,targets,grad_outputs=c,retain_graph=True)
                assert all(bool(g.isfinite().all()) for g in gs)
                gg[name]=tuple(g.detach().clone() for g in gs)
            comparisons={n:compare(a,(b.double()+c.double()).to(dtype),**(FP64 if dtype==torch.float64 else FP32))
                         for n,a,b,c in zip(names,gg['sum'],gg['u'],gg['v'])}
            for name in names[-2:]:
                index=names.index(name)
                assert all(int(gg[key][index].count_nonzero())==0 for key in gg),name
            records.append({'dtype':str(dtype),'scale':scale,'comparisons':comparisons,
                'rounded_FP32_cotangent_sum_effect':{n:compare(a,b,**(FP64 if dtype==torch.float64 else FP32))
                    for n,a,b in zip(names,gg['rounded_sum'],gg['sum'])},
                'cotangent_sum_FP32_vs_FP64':compare((u*scale+v*scale).double(),u.double()*scale+v.double()*scale,**FP64),
                'summed_vjp_norms':{n:stats(g) for n,g in zip(names,gg['sum'])},'activations':activations,
                'forward_vs_captured_FP32':compare(y.detach(),captured['output'].to(dtype),**(FP64 if dtype==torch.float64 else FP32)),
                'identities':{n:identity(t) for n,t in zip(names,targets)},'same_graph':True,
                'finite':True,'singleton_QK_exact_zero':True})
        del model,inputs,targets,pp,y,gs,gg;gc.collect()
    write(args.output,{'status':'DIAGNOSTIC_COMPLETE_PHASE_BLOCKED','scope':'all seven LD blocks, fixed actual state/memory/visual leaves; no LE/visual Jacobian or vocabulary projection',
        'fixture':f.provenance,'backend':backend(),'source_root_sha256':root_hash,'source_LD_sha256':source_hash,
        'input_hashes':[sha_tensor(t) for t in captured['inputs']],'cotangent_hashes':[sha_tensor(u),sha_tensor(v)],
        'cotangent_seeds':[901,902],'records':records,'zero_updates':True,'full_FP32_values_copied_before_local_FP64_promotion':True,
        'seconds':time.perf_counter()-started,'Windows_peak_working_set_bytes':getattr(psutil.Process().memory_info(),'peak_wset',None),
        'note':'Existing tolerances reported unchanged, not imposed as new universal FP64 additivity policy; no production architecture edits.'})
    print('Actual seven-block LD fixed-cotangent FP32/FP64 diagnostic completed; Phase BLOCKED',flush=True)


if __name__=='__main__':main()
