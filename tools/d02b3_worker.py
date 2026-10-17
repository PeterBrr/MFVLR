"""Live FLT-only worker, called by the exclusive six-cell coordinator."""
from pathlib import Path
import sys,os,json,copy,gc,time,traceback,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'.phase2c1_deps')]
from mfvlr_research.d02b3.policy import identity,write,hashfile,resources,check_backend,classify

def run(cell,destination,owner_file):
    fixture,variant,precision=identity(cell)
    destination=Path(destination)
    if destination.exists(): raise FileExistsError(destination)
    destination.mkdir(parents=True)
    started=time.perf_counter(); result={'cell':cell,'execution':'RUNNING','variant':variant,'precision':precision}
    try:
        import psutil,shutil
        owner=json.loads(Path(owner_file).read_text())
        if psutil.Process(owner['pid']).create_time()!=owner['created']: raise RuntimeError('Coordinator identity changed')
        if cell!=owner['active_cell']: raise RuntimeError('Wrong coordinator cell claim')
        ram=psutil.virtual_memory().available; disk=shutil.disk_usage(ROOT).free
        if not resources(ram,disk):
            write(destination/'result.json',dict(result,execution='WAIT/NOT RUN',available_ram=ram,free_disk=disk)); return 3
        import torch
        torch.set_num_threads(4); torch.set_num_interop_threads(4)
        torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision('highest'); torch.backends.mkldnn.enabled=True
        from mfvlr_research.d02b2.contract import bundle,verify_sources,load_fixture,stages,COUNTS
        from mfvlr_research.decoder_a11 import build_variant,parity_report
        from mfvlr.flt import FineGrainedLanguageTransformer
        from tools.review_phase2c2_n2 import backend,state_hash,tensor_hash,forward_capture,compare,stats
        pins=json.loads((Path(owner_file).parent/'driver_manifest.json').read_text())
        for rel,digest in pins.items():
            if hashfile(ROOT/rel)!=digest: raise RuntimeError('Driver source changed: '+rel)
        verify_sources(ROOT)
        expected_backend=bundle()['fixtures']['F0']['backend']
        actual_backend=backend(); check_backend(actual_backend,expected_backend)
        write(destination/'preflight.json',dict(backend=actual_backend,available_ram=ram,free_disk=disk,pid=os.getpid(),created=psutil.Process().create_time(),owner=owner))
        ids_np,v_np=load_fixture('F0')
        ids=torch.from_numpy(ids_np.copy()); visual32=torch.from_numpy(v_np.copy())
        reference=FineGrainedLanguageTransformer(seed=0)
        base_hash=state_hash(reference)
        if base_hash!=bundle()['fixtures']['F0']['a0_fp32_state_sha256']: raise RuntimeError('Canonical FP32 state mismatch')
        candidate=build_variant(reference,variant)
        parity=parity_report(reference,candidate)
        write(destination/'common_parity.json',parity)
        if state_hash(reference)!=base_hash: raise RuntimeError('Reference mutation')
        del reference; gc.collect()
        candidate_hash=state_hash(candidate)
        def inventory(model):
            rows=[]; objects=set(); ptrs=set()
            for name,p in model.named_parameters(remove_duplicate=False):
                if id(p) in objects or p.data_ptr() in ptrs: raise RuntimeError('Aliased owner')
                objects.add(id(p));ptrs.add(p.data_ptr())
                rows.append(dict(name=name,shape=list(p.shape),dtype=str(p.dtype),sha256=tensor_hash(p),object_id=id(p),storage_id=p.data_ptr()))
            if (len(rows),sum(p.numel() for p in model.parameters()))!=COUNTS[variant]: raise RuntimeError('Count')
            if len(model.encoder.blocks)!=12 or len(model.decoder.blocks)!=7: raise RuntimeError('Block coverage')
            if model.vocabulary_output_weight is not model.vocabulary.weight: raise RuntimeError('Tie')
            vims=[n for n,m in model.named_modules() if type(m).__name__=='VisionInjectionDelta']
            if len(vims)!=19: raise RuntimeError('VIM roles')
            norms=[dict(name=n,width=list(m.normalized_shape),eps=m.eps) for n,m in model.named_modules() if isinstance(m,torch.nn.LayerNorm)]
            if len(norms)!=(12 if variant=='A0' else 33) or any(n['width']!=[512] or n['eps']!=1e-5 for n in norms): raise RuntimeError('Norms')
            return dict(parameters=rows,vim_roles=vims,norms=norms,vocabulary_tied=True)
        write(destination/'fp32_inventory.json',inventory(candidate))
        if precision=='float64':
            model=copy.deepcopy(candidate).double(); promoted=[]
            for (name,p),(n,q) in zip(candidate.named_parameters(),model.named_parameters()):
                if name!=n or id(p)==id(q) or p.data_ptr()==q.data_ptr(): raise RuntimeError('Promotion ownership')
                a=p.detach().reshape(-1);b=q.detach().reshape(-1)
                for start in range(0,a.numel(),262144):
                    if not torch.equal(a[start:start+262144].double(),b[start:start+262144]): raise RuntimeError('Inexact FP64 promotion')
                promoted.append(dict(name=name,fp32_sha256=tensor_hash(p),fp64_sha256=tensor_hash(q),exact_cast=True))
            write(destination/'promotion.json',dict(parameters=promoted,fp32_source_unchanged=state_hash(candidate)==candidate_hash))
            if state_hash(candidate)!=candidate_hash: raise RuntimeError('Promotion mutated source')
            del candidate; gc.collect(); visual=visual32.double()
        else: model=candidate; visual=visual32
        initial_inventory=inventory(model); initial_state=state_hash(model)
        write(destination/'live_inventory.json',initial_inventory)
        forward_index=0
        def preforward(module,args):
            nonlocal forward_index
            if psutil.Process(owner['pid']).create_time()!=owner['created']: raise RuntimeError('Lost coordinator')
            check_backend(backend(),expected_backend)
            if state_hash(module)!=initial_state or inventory(module)!=initial_inventory: raise RuntimeError('Pre-forward state/ownership changed')
            for rel,digest in pins.items():
                if hashfile(ROOT/rel)!=digest: raise RuntimeError('Driver changed')
            a,b=args
            record=dict(index=forward_index,state_sha256=initial_state,fp32_candidate_sha256=candidate_hash,base_fp32_sha256=base_hash,
                        input_ids_sha256=tensor_hash(a),visual_sha256=tensor_hash(b),ids_shape=list(a.shape),visual_shape=list(b.shape),
                        backend=backend(),source_hashes=pins,available_ram=psutil.virtual_memory().available)
            write(destination/f'forward_{forward_index:02d}_seal.json',record);forward_index+=1
        handle=model.register_forward_pre_hook(preforward)
        def capture(a,b):
            values=forward_capture(model,a,b,detail=True)
            expected=stages('F0',len(a))
            if set(values)!=set(expected): raise RuntimeError('Missing capture')
            for name,x in values.items():
                if list(x.shape)!=expected[name] or x.requires_grad or x.grad_fn is not None: raise RuntimeError('Capture shape/graph')
                stats(x)
            return values
        batch=capture(ids,visual)
        write(destination/'activations.json',{n:stats(x) for n,x in batch.items()})
        atol,rtol=(1e-6,1e-5) if precision=='float32' else (1e-10,1e-8)
        comparisons=[]
        for sample in (0,1):
            single=capture(ids[sample:sample+1],visual[sample:sample+1]); record={}
            for name,x in batch.items():
                metric=compare(x[sample:sample+1],single[name],atol,rtol)
                metric['outside_proportion']=metric['outside_tolerance_count']/metric['elements']
                metric['numerical']='PASS' if metric['allclose'] else 'FAIL';record[name]=metric;comparisons.append(metric)
            write(destination/f'cross_batch_sample{sample}.json',record);del single
        result.update(classify(comparisons,precision))
        if result['mandatory_failure']: raise AssertionError('Promoted FP64 cross-batch disagreement: stop dependent cells')
        permutation=torch.tensor([1,0]); swapped=capture(ids[permutation],visual[permutation])
        perm={n:torch.equal(x,swapped[n][permutation]) for n,x in batch.items()}
        write(destination/'permutation.json',perm);del swapped
        if not all(perm.values()): raise AssertionError('Exact permutation failure')
        for sample in (0,1):
            ci=ids.clone();cv=visual.clone();ci[sample,250]=1234;cv[sample]+=.5
            changed=capture(ci,cv); checks={n:torch.equal(x[1-sample],changed[n][1-sample]) for n,x in batch.items()}
            write(destination/f'independence_changed{sample}.json',checks);del changed
            if not all(checks.values()): raise AssertionError('Fixed-batch independence')
        del batch;gc.collect()
        # Genuine lookup and visual leaves, both sample directions, each precision.
        vv=visual.clone().requires_grad_(); output=model(ids,vv)
        if output.low.grad_fn is None or 'EmbeddingBackward' not in type(output.low.grad_fn).__name__: raise RuntimeError('Surrogate lookup')
        derivatives=[]
        for sample in (0,1):
            probe=output.logits[sample].square().mean()+output.decoder_hidden[sample].square().mean()+output.global_language[sample].square().mean()
            gv,gl=torch.autograd.grad(probe,(vv,output.low),retain_graph=sample==0)
            row=dict(sample=sample,lookup_grad_fn=type(output.low.grad_fn).__name__,visual_leaf=vv.is_leaf,
                     own_visual=stats(gv[sample]),own_lookup=stats(gl[sample]),other_visual=stats(gv[1-sample]),other_lookup=stats(gl[1-sample]),
                     other_visual_nonzero=int(gv[1-sample].count_nonzero()),other_lookup_nonzero=int(gl[1-sample].count_nonzero()))
            derivatives.append(row)
            write(destination/f'cross_derivative_{sample}.json',row)
            if row['other_visual_nonzero'] or row['other_lookup_nonzero'] or row['own_visual']['l2_norm']<=0 or row['own_lookup']['l2_norm']<=0: raise AssertionError('Cross-sample derivative isolation')
        del output,probe,gv,gl,vv;gc.collect()
        if precision=='float32':
            from tools.review_phase2c2_n2 import Capture
            for size in (1,2):
                model.zero_grad(set_to_none=True);vv=visual[:size].clone().requires_grad_(); cap=Capture(model,gradient=True)
                try:
                    output=model(ids[:size],vv)
                    probe=output.logits.square().mean()+output.decoder_hidden.square().mean()+output.global_language.square().mean()
                    if not torch.isfinite(probe): raise FloatingPointError('Probe nonfinite')
                    probe.backward()
                finally: cap.close()
                grads={};zeros=[]
                for name,p in model.named_parameters():
                    if p.grad is None: raise AssertionError('Disconnected '+name)
                    rec=stats(p.grad);rec.update(nonzero=int(p.grad.count_nonzero()),allocated=True,sha256=tensor_hash(p.grad));grads[name]=rec
                    if '.vim.q.' in name or '.vim.k.' in name:
                        zeros.append(name)
                        if rec['nonzero']: raise AssertionError('VIM structural zero')
                positive=('vocabulary.weight','encoder_position','decoder_position','encoder.blocks.0.attention.v.weight','decoder.blocks.0.memory_attention.v.weight','encoder.blocks.0.ff.fc1.weight','decoder.blocks.6.ff.fc2.weight')
                write(destination/f'gradients_B{size}.json',dict(parameters=grads,vim_qk=zeros,probe=float(probe.detach()),visual=stats(vv.grad),
                     stage_gradients={n:stats(x.grad) for n,x in cap.values.items()},all_parameter_l2=sum(x['l2_norm']**2 for x in grads.values())**.5))
                if len(zeros)!=38 or any(grads[n]['l2_norm']<=0 for n in positive) or stats(vv.grad)['l2_norm']<=0: raise AssertionError('Positive controls')
                del output,probe,vv,cap;model.zero_grad(set_to_none=True);gc.collect()
        handle.remove()
        final=inventory(model)
        write(destination/'final_inventory.json',final)
        if final!=initial_inventory or state_hash(model)!=initial_state: raise RuntimeError('Final state mutation')
        result.update(execution='COMPLETE',structural='PASS',state_preserved=True,forward_calls=forward_index,
                      backend=actual_backend,fp32_base_sha256=base_hash,fp32_candidate_sha256=candidate_hash,live_state_sha256=initial_state)
        code=0
    except Exception as exc:
        result.update(execution='HARD STOP',exception=repr(exc),traceback=traceback.format_exc());code=1
    result['elapsed_seconds']=time.perf_counter()-started
    write(destination/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return code
if __name__=='__main__':
    if len(sys.argv)!=4: raise SystemExit('exact cell, exclusive output and coordinator claim required')
    raise SystemExit(run(*sys.argv[1:]))
