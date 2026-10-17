"""Pre-registered held-out local references. No acceptance thresholds are added."""
from pathlib import Path
import argparse,copy,gc,hashlib,json,math,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)]


def setup(out):
    import torch
    from tools.phase2d2_diagnostics import write
    reg=json.loads((out/'heldout_preregistration.json').read_text());lock=json.loads((out/'preregistration_lock.json').read_text())
    assert hashlib.sha256((out/'heldout_preregistration.json').read_bytes()).hexdigest()==lock['sha256']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    return reg


def compare_full(a,b):
    from tools.review_phase2c2_n2 import compare
    from tools.phase2d2_diagnostics import FP32,FP64
    r=compare(a,b,**(FP64 if str(a.dtype)=='torch.float64' else FP32))
    x=a.detach().double().flatten();y=b.detach().double().flatten()
    r['actual_l2']=x.norm().item();r['reference_l2']=y.norm().item();r['error_l2']=(x-y).norm().item()
    r['cosine']=(x@y/(x.norm()*y.norm())).item() if x.norm()>0 and y.norm()>0 else None
    r['reference_zero_coordinates']=int((y==0).sum())
    return r


def gamma(n,u):return n*u/(1-n*u)


def bounded_record(a,ref,bound):
    d=(a.double()-ref).abs();return {'max_absolute_error':d.max().item(),'error_l2':d.norm().item(),
        'bound_max':bound.max().item(),'bound_l2':bound.norm().item(),'outside_conditional_bound':int((d>bound).sum()),
        'max_error_to_nonzero_bound':(d[bound>0]/bound[bound>0]).max().item() if (bound>0).any() else None,
        'reference_zero_coordinates':int((ref==0).sum()),'no_overflow_or_underflow_bound_proof':False}


def attention(out,reg):
    import torch
    from mfvlr.language_encoder import LanguageAttention,initialize_attention
    from tools.review_phase2d2_n3_local import explicit_attention,explicit_gradients
    from tools.phase2d2_diagnostics import write,state_hash,stats
    from tools.n3_gradient_tools import sha_tensor
    from tools.review_phase2c2_n2 import backend
    spec=reg['attention'];s=spec['seeds'];source=LanguageAttention();initialize_attention(source,torch.Generator().manual_seed(s['weights']))
    sourcehash=state_hash(source);records=[];bounds=[];fds=[];start=time.perf_counter()
    for magnitude in spec['input_magnitudes']:
        x32=torch.randn(*spec['shape'],generator=torch.Generator().manual_seed(s['input']))*magnitude
        c32=torch.randn(*spec['shape'],generator=torch.Generator().manual_seed(s['cotangent']))
        # Conditional forward affine rounding bound at actual 512 accumulation dimension.
        with torch.no_grad():
            for n in ('q','k'):
                layer=getattr(source,n);actual=layer(x32)
                ref=x32.double()@layer.weight.double().T+layer.bias.double()
                operands=x32.double().abs()@layer.weight.double().abs().T+layer.bias.double().abs()
                bound=(gamma(513,2**-24)+gamma(513,2**-53))*operands
                bounds.append({'kind':'affine_'+n,'input_magnitude':magnitude,'n':513,'u':2**-24,
                    'operand_sum_abs_max':operands.max().item(),**bounded_record(actual,ref,bound)})
        for dtype in (torch.float32,torch.float64):
            model=copy.deepcopy(source);assert state_hash(model)==sourcehash;model.to(dtype=dtype)
            pp=dict(model.named_parameters());names=('q.weight','k.weight','k.bias')
            for scale in spec['cotangent_scales']:
                x=x32.to(dtype).clone().requires_grad_();c=c32.to(dtype)*scale
                y,trace=model.forward_with_trace(x)
                gs=torch.autograd.grad(y,tuple(pp[n] for n in names)+(x,),grad_outputs=c)
                assert all(bool(g.isfinite().all()) for g in gs)
                with torch.no_grad():
                    weights={n:p.detach().clone() for n,p in pp.items()}
                    expected=explicit_gradients(x.detach(),weights,c)
                comparisons={n:compare_full(g,expected[n]) for n,g in zip((*names,'input'),gs)}
                records.append({'input_magnitude':magnitude,'scale':scale,'dtype':str(dtype),
                    'input_FP32_sha256':sha_tensor(x32),'unscaled_cotangent_FP32_sha256':sha_tensor(c32),
                    'manual_derivative_comparisons':comparisons,'gradient_stats':{n:stats(g) for n,g in zip((*names,'input'),gs)},
                    'key_bias_vs_zero':compare_full(gs[2],torch.zeros_like(gs[2])),'finite':True})
                if dtype==torch.float32 and scale!=0:
                    # Independent backward contraction at STORED rounded operands, not a bound on total autograd error.
                    with torch.no_grad():
                        _,(q,k,v,p,context)=explicit_attention(x.detach(),weights)
                        dc=(c@weights['out.weight']).reshape(1,308,8,64).transpose(1,2)
                        dp=dc@v.transpose(-1,-2);ds=p*(dp-(dp*p).sum(-1,keepdim=True))
                        dq=(ds@k/8).transpose(1,2).reshape_as(x)
                        dk=(ds.transpose(-1,-2)@q/8).transpose(1,2).reshape_as(x)
                        for name,d in (('q.weight',dq),('k.weight',dk)):
                            a=d.reshape(-1,512).T@x.detach().reshape(-1,512)
                            d64=d.double().reshape(-1,512);x64=x.detach().double().reshape(-1,512)
                            ref=d64.T@x64;op=d64.abs().T@x64.abs()
                            bound=(gamma(308,2**-24)+gamma(308,2**-53))*op
                            bounds.append({'kind':'backward_contraction_'+name,'input_magnitude':magnitude,'scale':scale,'n':308,
                                'operand_sum_abs_max':op.max().item(),**bounded_record(a,ref,bound)})
                        ref=dk.double().sum((0,1));op=dk.double().abs().sum((0,1))
                        bounds.append({'kind':'key_bias_sum_stored_dk','input_magnitude':magnitude,'scale':scale,'n':308,
                            'operand_sum_abs_max':op.max().item(),**bounded_record(dk.sum((0,1)),ref,(gamma(308,2**-24)+gamma(308,2**-53))*op)})
                        # Exact-real stored-p identity differs from ideal softmax if rounded p sums to != 1.
                        p64=p.double();dp64=dp.double();mu=(p64*dp64).sum(-1)
                        formula=mu*(1-p64.sum(-1));manualsum=(p64*(dp64-mu[...,None])).sum(-1)
                        bounds.append({'kind':'stored_probability_softmax_Jacobian_row_sum','input_magnitude':magnitude,'scale':scale,
                            'p_sum_defect_max':(p64.sum(-1)-1).abs().max().item(),'mu_max':mu.abs().max().item(),
                            'exact_real_formula_evaluated_FP64_max':formula.abs().max().item(),'FP64_formula_rearrangement_error_max':(manualsum-formula).abs().max().item(),
                            'FP32_computed_ds_row_sum_max':ds.sum(-1).abs().max().item(),
                            'claim':'DERIVED formula for stored p; no full softmax/backprop bound'})
                if dtype==torch.float64 and scale==1:
                    for index,name in enumerate(('q.weight','k.weight')):
                        direction=torch.randn(pp[name].shape,generator=torch.Generator().manual_seed(s['directions'][index]),dtype=dtype);direction/=direction.norm()
                        with torch.no_grad():
                            plus=explicit_attention(x.detach(),{**weights,name:weights[name]+1e-6*direction})[0]
                            minus=explicit_attention(x.detach(),{**weights,name:weights[name]-1e-6*direction})[0]
                            fd=((plus*c).sum()-(minus*c).sum())/(2e-6);analytical=(gs[index]*direction).sum()
                        ok=bool(torch.isclose(fd,analytical,atol=1e-5,rtol=1e-3))
                        fds.append({'input_magnitude':magnitude,'parameter':name,'finite_difference':fd.item(),'analytical':analytical.item(),
                            'absolute_error':abs(fd.item()-analytical.item()),'epsilon':1e-6,'atol':1e-5,'rtol':1e-3,'pass':ok})
                del x,y,trace,gs,expected
            del model;gc.collect()
    write(out/'heldout_attention.json',{'status':'DIAGNOSTIC_COMPLETE_PHASE_BLOCKED','preregistration_sha256':hashlib.sha256((out/'heldout_preregistration.json').read_bytes()).hexdigest(),
        'backend':backend(),'source_FP32_weights_sha256':sourcehash,'records':records,'conditional_local_bounds':bounds,'finite_differences':fds,'seconds':time.perf_counter()-start,
        'bound_limitations':'gamma_n assumes round-to-nearest elementary operations without overflow/underflow and standard reduction; optimized backend not formally certified. FP64 reference allowance included for contraction only. Does not bound upstream computed dq/dk, exp, whole attention or network.'})


def vjp_record(output,targets,names,u,v,scale):
    import torch
    from tools.phase2d2_diagnostics import stats
    values={}
    for name,c in (('u',u*scale),('v',v*scale),('sum',u*scale+v*scale)):
        gs=torch.autograd.grad(output,targets,grad_outputs=c,retain_graph=True,allow_unused=True)
        assert all(g is None or bool(g.isfinite().all()) for g in gs)
        values[name]=gs
    result={}
    for index,name in enumerate(names):
        a,b,c=(values[k][index] for k in ('sum','u','v'))
        assert (a is None)==(b is None)==(c is None)
        if a is None:result[name]={'allocated':False};continue
        result[name]={'allocated':True,'comparison':compare_full(a,(b.double()+c.double()).to(a.dtype)),
            'joint_stats':stats(a),'single_u_stats':stats(b),'single_v_stats':stats(c)}
        if '.vim.q.' in name or '.vim.k.' in name:assert int(a.count_nonzero())==int(b.count_nonzero())==int(c.count_nonzero())==0
    return result


def decoder(out,reg):
    import torch
    from mfvlr.integration import MFVLRIntegration
    from tools.phase2d2_fixtures import fixture
    from tools.phase2d2_diagnostics import state_hash,write,stats
    from tools.n3_gradient_tools import sha_tensor
    from tools.review_phase2c2_n2 import backend
    root=MFVLRIntegration();f=fixture('real_efs');captured={};initial=state_hash(root)
    h=root.flt.decoder.register_forward_hook(lambda m,a,o:captured.update(inputs=tuple(t.clone() for t in a)))
    try:
        with torch.no_grad():o=root.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
    finally:h.remove()
    source=copy.deepcopy(root.flt.decoder);sourcehash=state_hash(source);assert initial==state_hash(root)
    del root,o;gc.collect();spec=reg['LD'];records=[];started=time.perf_counter()
    for multipliers in spec['state_memory_visual_multipliers']:
        inputs32=tuple(t*m for t,m in zip(captured['inputs'],multipliers))
        for dtype in (torch.float32,torch.float64):
            model=copy.deepcopy(source);assert state_hash(model)==sourcehash;model.to(dtype=dtype)
            inputs=tuple(t.to(dtype).clone().requires_grad_() for t in inputs32);stages={};handles=[]
            for i,b in enumerate(model.blocks):handles.append(b.register_forward_hook(lambda m,a,o,i=i:stages.__setitem__(f'LD{i+1:02}',o)))
            try:y=model(*inputs)
            finally:
                for h in handles:h.remove()
            pp=dict(model.named_parameters());names=spec['targets'];targets=inputs+tuple(pp[n] for n in names[3:])
            for cut in spec['cuts']:
                output=stages[cut]
                u=torch.randn(output.shape,generator=torch.Generator().manual_seed(spec['cotangent_seeds'][0]))/math.sqrt(output.numel())
                v=torch.randn(output.shape,generator=torch.Generator().manual_seed(spec['cotangent_seeds'][1]))/math.sqrt(output.numel())
                for scale in spec['cotangent_scales']:
                    r=vjp_record(output,targets,names,u.to(dtype),v.to(dtype),scale)
                    records.append({'multipliers':multipliers,'dtype':str(dtype),'cut':cut,'scale':scale,'parameters':r,
                        'activation':stats(output),'input_hashes_FP32':[sha_tensor(t) for t in inputs32],
                        'cotangent_hashes_FP32':[sha_tensor(u),sha_tensor(v)],'same_forward':True,'finite':True})
            del model,inputs,stages,y,pp,targets,output;gc.collect()
    write(out/'heldout_LD.json',{'status':'DIAGNOSTIC_COMPLETE_PHASE_BLOCKED','backend':backend(),'source_root_sha256':initial,
        'source_LD_FP32_sha256':sourcehash,'preregistration_sha256':hashlib.sha256((out/'heldout_preregistration.json').read_bytes()).hexdigest(),
        'records':records,'seconds':time.perf_counter()-started,'scope':'All seven LD blocks, cuts 1/4/7; independent leaf state/memory/visual; no LE or visual Jacobian FP64 claim'})


def visual(out,reg,mode):
    import torch
    from mfvlr.integration import ImageOnlyMFVLR
    from tools.phase2d2_diagnostics import state_hash,stats,write
    from tools.n3_gradient_tools import sha_tensor,identity
    from tools.review_phase2c2_n2 import backend
    torch.backends.mkldnn.enabled=mode=='mkldnn';root=ImageOnlyMFVLR();initial=state_hash(root);spec=reg['visual'];started=time.perf_counter()
    x=.1+.8*torch.rand(spec['image_shape'],generator=torch.Generator().manual_seed(spec['image_seed']));o=root(x)
    iv=o.visual.fused_global;pre=o.decoded.reconstruction
    targets=(root.mve.shared_encoder.cnn.stages[0][0].weight,root.mve.shared_encoder.blocks[1].attention.q.weight,root.decoder.appearance_head.weight,pre)
    names=spec['targets'];u=torch.randn(iv.shape,generator=torch.Generator().manual_seed(spec['cotangent_seeds'][0]))/math.sqrt(iv.numel());v=torch.randn(iv.shape,generator=torch.Generator().manual_seed(spec['cotangent_seeds'][1]))/math.sqrt(iv.numel())
    records=[{'scale':scale,'parameters':vjp_record(iv,targets,names,u,v,scale)} for scale in spec['cotangent_scales']]
    assert state_hash(root)==initial
    write(out/f'heldout_visual_{mode}.json',{'status':'DIAGNOSTIC_COMPLETE_PHASE_BLOCKED','backend':backend(),'image_sha256':sha_tensor(x),
        'source_state_sha256':initial,'source_parameter_identities':{n:identity(t) for n,t in zip(names,targets)},'records':records,
        'preregistration_sha256':hashlib.sha256((out/'heldout_preregistration.json').read_bytes()).hexdigest(),
        'Iv':stats(iv),'Ipre':stats(pre),'parameters_unchanged':True,'seconds':time.perf_counter()-started,
        'scope':'Accepted public MVE/VD FP32 visual path; same graph cotangents; no FP64 visual wrapper'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worker',choices=['attention','LD','visual_mkldnn','visual_native'],required=True);args=parser.parse_args()
    reg=setup(args.output)
    target=args.output/('heldout_'+args.worker+'.json');assert not target.exists()
    if args.worker=='attention':attention(args.output,reg)
    elif args.worker=='LD':decoder(args.output,reg)
    else:visual(args.output,reg,args.worker.removeprefix('visual_'))
    print(args.worker+' held-out diagnostics completed; policy unchanged')
