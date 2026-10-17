"""A1-D02A only: closed local schedule, append-only evidence, no full-model route."""
from pathlib import Path
import argparse
import copy
import gc
import json
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)]
from tools.d02a.common import OUT,write,sha,verify_frozen,configure,Resources,Rows


def prepare():
    import torch
    from mfvlr.flt import FineGrainedLanguageTransformer
    from mfvlr_research.decoder_a11 import build_variant,parity_report
    from mfvlr_research.a11_evidence import state_hash
    configure(4,False);frozen=verify_frozen();jobs=[]
    # Full registered local fixture grid, plus the specified nominal one-thread
    # control. Full language N2, six-loss G2/NG2 and the old N3 matrix are outside
    # this bounded authorization and explicitly NOT RUN.
    for family,magnitudes in (("NLD",range(8)),("NATT",range(3))):
        for native in (False,True):
            for j in range(3):
                for batch in (1,2):
                    for index in magnitudes:
                        for variant in ("A0","P3","Q3"):
                            jobs.append(dict(family=family,j=j,batch=batch,index=index,variant=variant,threads=4,native=native))
    for j in range(3):
        for batch in (1,2):
            for variant in ("A0","P3","Q3"):
                jobs.append(dict(family="NLD",j=j,batch=batch,index=1,variant=variant,threads=1,native=False))
    for i,job in enumerate(jobs):job['id']=f"{i:04d}_{job['family']}_{job['variant']}_j{job['j']}_b{job['batch']}_m{job['index']}_t{job['threads']}_{'native' if job['native'] else 'mkldnn'}"
    write(OUT/'pilot_protocol.json',{'authorization':'A1-D02A only, user request 2026-10-12','phase':'BLOCKED',
        'frozen_sources':frozen,'preregistration_sha256':sha(ROOT/'docs/phase2d2_a1_ablation_preregistration_v1_1.md'),
        'jobs':jobs,'job_count':len(jobs),'both_precisions':['float32','float64'],
        'NLD_cuts':[1,4,7],'cotangent_scales':[0,1e-4,1,1024],
        'NATT_operational_definition':'Independent unmasked ordinary primitive (matching N3 manual oracle), plus A0 x+S(x), P3 x+S(N(x)), Q3 N(x+S(x)); NLD actual self-attention stays causal.',
        'master_precision':'Random tensors, normalization by sqrt(B*N*d), magnitude/cotangent multiplications in FP32; promote exact scaled operands to FP64; sum u/v after promotion in FP64.',
        'direction_generator':'NumPy default_rng PCG64 seeded 6701..6704 + SHA256(name) first4 bytes little-endian; signed Rademacher /sqrt(numel)',
        'timing':'cold forward; one no-grad warm-up; three identical-reloaded-state unit-cotangent full-depth forward/VJP repetitions per precision/cell',
        'hard_stops':'nonfinite, unexpected disconnected gradient, singleton Q/K nonzero, parameter/ownership/state/provenance violation, mandatory primitive/LN oracle failure; preserve finite stress comparison failures and continue',
        'NOT_RUN':['full N2','six-loss G2/NG2','old N3 full-model probes','I1','R2','GPU','batch8','optimizers','training','MVE-T10','VD-T12'],
        'no_threshold_changes':True,'production_promotion':False})
    (OUT/'states').mkdir()
    for j in range(3):
        reference=FineGrainedLanguageTransformer(seed=6101+j);before=state_hash(reference);rng=torch.random.get_rng_state().clone()
        for variant in ('A0','P3','Q3'):
            candidate=build_variant(reference,variant);report=parity_report(reference,candidate)
            assert torch.equal(rng,torch.random.get_rng_state()) and state_hash(reference)==before
            path=OUT/f'states/j{j}_{variant}.pt'
            with path.open('xb') as handle:torch.save(candidate.decoder.state_dict(),handle)
            write(OUT/f'states/j{j}_{variant}.json',{'report':report,'FLT_FP32_source_hash':before,
                  'LD_state_hash':state_hash(candidate.decoder),'checkpoint_sha256':sha(path),
                  'seed':6101+j,'caller_RNG_unchanged':True,'source_unchanged':True,
                  'full_model_count_derived_with_frozen_visual_heads_objectives':report['parameters']+41397960})
            del candidate;gc.collect()
        del reference;gc.collect()
    print(json.dumps({'prepared_jobs':len(jobs),'state_pairs':9}),flush=True)


def worker(index):
    from mfvlr_research.a11_evidence import backend
    from tools.d02a import nld,natt
    protocol=json.loads((OUT/'pilot_protocol.json').read_text());job=protocol['jobs'][index]
    configure(job['threads'],job['native']);verify_frozen()
    source_records={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'tools/d02a').glob('*.py')}
    approved=json.loads((OUT/'driver_manifest_v2.json').read_text())
    if source_records!=approved:raise AssertionError('pilot driver changed after registration')
    destination=OUT/'jobs'/job['id'];destination.mkdir(parents=True,exist_ok=False)
    for variant in ('A0','P3','Q3'):
        p=OUT/f"states/j{job['j']}_{variant}.pt";r=json.loads(p.with_suffix('.json').read_text())
        if sha(p)!=r['checkpoint_sha256']:raise AssertionError('state checkpoint hash mismatch')
    write(destination/'provenance.json',{'job':job,'backend':backend(),'driver_hashes':source_records,
          'protocol_sha256':sha(OUT/'pilot_protocol.json'),'variant_config_sha256':sha(ROOT/f"configs/research_a11/{job['variant']}.yaml"),
          'frozen_acceptance_manifest_sha256':sha(ROOT/'artifacts/phase2d2_a11_reconciliation/acceptance_manifest.json')})
    rows=Rows(destination/'comparisons.jsonl')
    with Resources() as resources:
        try:
            module=nld if job['family']=='NLD' else natt
            result=module.run(job['j'],job['batch'],job['index'],job['variant'],destination,rows)
            result['failed_comparisons']=rows.failures;result['records']=rows.count;code=0
        except Exception as exc:
            result={'status':'HARD_STOP','exception':repr(exc),'traceback':traceback.format_exc(),
                    'failed_comparisons':rows.failures,'records':rows.count};code=2
        finally:rows.close()
    write(destination/'summary.json',{**result,'resources':resources.record})
    print(json.dumps({'job':job['id'],**result,'resources':resources.record}),flush=True)
    return code


def controller(indices):
    jobs=json.loads((OUT/'pilot_protocol.json').read_text())['jobs']
    logs=OUT/'logs_v2';logs.mkdir(exist_ok=True)
    for i in indices:
        job=jobs[i];summary=OUT/'jobs'/job['id']/'summary.json'
        if summary.exists():
            result=json.loads(summary.read_text())
            if result['status']=='HARD_STOP':raise RuntimeError('prior hard stop requires review; no automatic retry')
            continue
        cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--index',str(i)]
        t=time.perf_counter()
        with (logs/(job['id']+'.stdout.log')).open('x') as stdout,(logs/(job['id']+'.stderr.log')).open('x') as stderr:
            completed=subprocess.run(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
        write(logs/(job['id']+'.execution.json'),{'command':cmd,'exit_code':completed.returncode,'seconds':time.perf_counter()-t})
        print(json.dumps({'index':i,'job':job['id'],'exit_code':completed.returncode,'seconds':time.perf_counter()-t}),flush=True)
        if completed.returncode:raise RuntimeError('pilot hard stop; no subsequent dependent execution')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','worker','run'))
    parser.add_argument('--index',type=int);parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='worker':raise SystemExit(worker(args.index))
    else:
        count=len(json.loads((OUT/'pilot_protocol.json').read_text())['jobs'])
        controller(range(args.start,args.stop if args.stop is not None else count))
