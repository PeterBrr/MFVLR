"""Single G01 A0 worker. Seven fresh graphs only; no timing repetitions."""
from pathlib import Path
import gc,json,os,sys,time,traceback,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'.phase2c1_deps')]
from mfvlr_research.d02b7.policy import CELL,require_cell,resource_gate,status
from mfvlr_research.d02b1.schema import write_new,file_hash,verify_files,PASSES,participation,validate_inventory

def run(cell,destination,claim_path):
    require_cell(cell);destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    result=dict(cell=cell,**status());start=time.perf_counter();phase='provenance'
    try:
        import psutil,shutil
        claim_path=Path(claim_path);claim=json.loads(claim_path.read_text())
        def owner():
            if claim['cell']!=CELL or psutil.Process(claim['pid']).create_time()!=claim['created']:raise RuntimeError('Coordinator identity')
        owner();ram=psutil.virtual_memory().available;disk=shutil.disk_usage(ROOT).free
        write_new(destination/'resources_before_import.json',dict(available_ram=ram,free_disk=disk,pid=os.getpid(),created=psutil.Process().create_time()))
        if not resource_gate(ram,disk):
            result.update(status(resource='INSUFFICIENT'));write_new(destination/'result.json',result);return 3
        result['resource']='PASS'
        pins=json.loads((claim_path.parent/'driver_manifest.json').read_text());verify_files(ROOT,pins)
        import torch
        torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest');torch.backends.mkldnn.enabled=True
        from tools.review_phase2c2_n2 import backend
        from mfvlr_research.d02b2.contract import bundle
        expected_backend=bundle()['fixtures']['F0']['backend']
        if backend()!=expected_backend:raise RuntimeError('Pinned Windows CPU C4M backend mismatch')
        from tools.phase2d2_fixtures import fixture,tensor_hash
        from tools.phase2d2_diagnostics import state_hash,stats,loss_only
        from mfvlr.integration import MFVLRIntegration,ownership_report
        from mfvlr.objectives import validate_mask_semantics
        from mfvlr_research.decoder_a11 import build_variant,parity_report
        from mfvlr_research.d02b1.gradients import GradientStore
        from mfvlr_research.d02b7.capture import Capture
        from mfvlr_research.d02b7.arithmetic import audit
        f=fixture('real');historical=json.loads((ROOT/'artifacts/phase2d2_n31/g2/final_real/summary.json').read_text())
        if f.provenance!=historical['fixture']:raise RuntimeError('Exact original G01 fixture provenance mismatch')
        def inputs():
            return dict(image=tensor_hash(f.image),tokens=tensor_hash(f.targets.token_ids),mask=tensor_hash(f.targets.masks),
                labels=tensor_hash(f.targets.labels),sample_order=list(f.targets.sample_ids),kinds=list(f.targets.sample_kinds),
                sample_order_sha256=hashlib.sha256(json.dumps(list(f.targets.sample_ids),separators=(',',':')).encode()).hexdigest())
        initial_inputs=inputs();write_new(destination/'fixture.json',dict(provenance=f.provenance,inputs=initial_inputs))
        reference=MFVLRIntegration()
        if state_hash(reference)!='1865527facba9adc8a14d443c112d492698d275ce4e8beaa784aee61a514895a':raise RuntimeError('Canonical seed-0 state mismatch')
        model=build_variant(reference,'A0');write_new(destination/'common_parity.json',parity_report(reference,model))
        del reference;gc.collect()
        roster=json.loads((ROOT/'tests/research_d02b1/fixtures/historical_rosters.json').read_text())['rosters']['A0']
        def inventory():
            rows=[dict(name=n,shape=list(p.shape),dtype='float32',sha256=tensor_hash(p),object_id=str(id(p)),storage_id=str(p.data_ptr())) for n,p in model.named_parameters(remove_duplicate=False)]
            if any(not p.requires_grad or p.dtype!=torch.float32 or p.device.type!='cpu' for p in model.parameters()):raise RuntimeError('Trainable CPU FP32 ownership')
            validate_inventory(rows,'A0',roster);ownership_report(model)
            if model.flt.vocabulary_output_weight is not model.flt.vocabulary.weight:raise RuntimeError('Vocabulary tie')
            return rows
        initial=inventory();initial_state=state_hash(model);write_new(destination/'inventory_before.json',initial)
        result['provenance']='PASS';phase='structural'
        specs={r['name']:dict(shape=r['shape'],presence=participation(r['name'])) for r in initial}
        store=GradientStore(destination/'gradient_stream',specs);losses={};forward_count=0;active=None
        def preforward(module,args,kwargs):
            nonlocal forward_count,phase
            phase='provenance'
            owner();verify_files(ROOT,pins)
            if backend()!=expected_backend or inputs()!=initial_inputs or inventory()!=initial or state_hash(model)!=initial_state:raise RuntimeError('Pre-forward seal changed')
            if active!=PASSES[forward_count]:raise RuntimeError('Unregistered or extra forward')
            if tensor_hash(args[0])!=initial_inputs['tokens'] or tuple(args[1].shape)!=(1,1,512):raise RuntimeError('FLT input identity/shape')
            write_new(destination/f'{active}_seal.json',dict(pass_name=active,state=initial_state,inputs=inputs(),fused_visual_sha256=tensor_hash(args[1]),source_hashes=pins,backend=backend(),available_ram=psutil.virtual_memory().available,free_disk=shutil.disk_usage(ROOT).free))
            forward_count+=1
            phase='structural'
        # with_language is a method, so seal before visual work explicitly as well as before FLT.
        hook=model.flt.register_forward_pre_hook(preforward,with_kwargs=True)
        try:
            for active in PASSES:
                phase='provenance'
                owner();verify_files(ROOT,pins)
                if inventory()!=initial or inputs()!=initial_inputs or backend()!=expected_backend:raise RuntimeError('Graph entry seal')
                write_new(destination/f'{active}_graph_entry.json',dict(state=state_hash(model),inputs=inputs(),backend=backend(),source_hashes=pins))
                phase='structural'
                model.zero_grad(set_to_none=True);pass_start=time.perf_counter()
                validate_mask_semantics(f.targets.labels,f.targets.masks,f.targets.sample_kinds)
                with Capture(model,stats) as cap:
                    out=model.with_language(f.image,f.targets.token_ids,sample_ids=f.targets.sample_ids)
                    activations=cap.finish(out)
                    if active=='total':
                        values=model.compute_objectives(out,f.targets).as_dict();objective=values['total']
                        arithmetic=audit(out,f.targets,model,values)
                        write_new(destination/'live_arithmetic.json',arithmetic)
                        if not arithmetic['all_exact']:raise RuntimeError('Independent FP32 objective arithmetic mismatch')
                        for name in PASSES[:-1]:
                            if float(values[name].detach())!=losses[name]:raise RuntimeError('Isolated/joint objective drift '+name)
                    else:objective=loss_only(model,out,f.targets,active)
                    if objective.ndim!=0 or not torch.isfinite(objective):raise FloatingPointError('Nonfinite/rank objective')
                    intermediates={'Ipre':out.predictions.decoded.reconstruction,'Iv':out.predictions.visual.fused_global,'Tl':out.language.global_language,'adapter':out.predicted_language}
                    for x in intermediates.values():x.retain_grad()
                    losses[active]=float(objective.detach());objective.backward()
                write_new(destination/f'{active}_activations.json',activations)
                write_new(destination/f'{active}_intermediates.json',{n:dict(allocated=x.grad is not None,statistics=stats(x.grad) if x.grad is not None else None) for n,x in intermediates.items()})
                for n,p in model.named_parameters():store.put(active,n,None if p.grad is None else p.grad.detach().contiguous().numpy())
                write_new(destination/f'{active}_pass.json',dict(loss=losses[active],seconds=time.perf_counter()-pass_start,resource=dict(rss=psutil.Process().memory_info().rss,available_ram=psutil.virtual_memory().available,free_disk=shutil.disk_usage(ROOT).free),state_unchanged=inventory()==initial))
                if inventory()!=initial:raise RuntimeError('State mutation after backward')
                if active=='total':del values,arithmetic
                del out,objective,intermediates,cap,x;model.zero_grad(set_to_none=True);gc.collect()
        finally:hook.remove()
        if forward_count!=7:raise RuntimeError('Seven fresh full graphs required')
        write_new(destination/'losses.json',losses)
        final=inventory();write_new(destination/'inventory_after.json',final)
        if final!=initial or state_hash(model)!=initial_state:raise RuntimeError('Final state/ownership mutation')
        # Free model before the full disk-backed arithmetic audit.
        del model;gc.collect();store.complete();analysis=store.analyze()
        write_new(destination/'additivity.json',analysis)
        result.update(status(completed=True,numerical=analysis['status'],structural='PASS',provenance='PASS',resource='PASS'))
        result.update(forwards=7,backwards=7,parameter_tensors=508,unique_elements=154156232,outside_coordinates=sum(p['outside_tolerance_count'] for p in analysis['parameters'].values()))
        code=0  # Successful computation is independent of numerical FAIL.
    except Exception as exc:
        result.update(status(error=True,provenance='FAIL' if phase=='provenance' else result['provenance'],structural='FAIL' if phase=='structural' else 'NOT RUN',resource=result['resource']))
        result.update(exception=repr(exc),traceback=traceback.format_exc());code=1
    result['elapsed_seconds']=time.perf_counter()-start
    write_new(destination/'result.json',result);print(json.dumps(result),flush=True);return code

if __name__=='__main__':
    if len(sys.argv)!=4:raise SystemExit('exact cell, exclusive destination, coordinator claim required')
    raise SystemExit(run(*sys.argv[1:]))
