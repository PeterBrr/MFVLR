"""Read-only identity/provenance/input audit, with no model forward or backward."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02a'


def sha(p):
    with p.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()


def write(p,value):
    with p.open('x',encoding='utf-8') as h:json.dump(value,h,indent=2,allow_nan=False)


def main(destination,only):
    import torch
    torch.set_num_threads(1)
    destination.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((OUT/'pilot_protocol.json').read_text())
    jobs=protocol['jobs'];ids=[j['id'] for j in jobs]
    assert len(ids)==len(set(ids))==414
    assert protocol['NLD_cuts']==[1,4,7] and protocol['cotangent_scales']==[0,1e-4,1,1024]
    driver=json.loads((OUT/'driver_manifest_v2.json').read_text())
    assert all(sha(ROOT/p)==value for p,value in driver.items())
    chosen=jobs if only is None else [jobs[only]]
    expected_interop=json.loads((OUT/'jobs'/ids[0]/'provenance.json').read_text())['backend']['interop_threads']
    if only is None:assert {p.name for p in (OUT/'jobs').iterdir()}==set(ids)
    states={};rows=[];natt_common={}
    def tensor_hash(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
    def check(t,record):
        assert list(t.shape)==record['shape']
        assert tensor_hash(t)==record['sha256']
        assert record['finite']
    multipliers=((.02,2,1),(.2,2,1),(2,2,1),(.2,.2,1),(.2,20,1),(.2,2,.1),(.2,2,10),(2,20,10))
    for job in chosen:
        d=OUT/'jobs'/job['id'];j=job['j'];b=job['batch'];variant=job['variant']
        summary=json.loads((d/'summary.json').read_text());inputs=json.loads((d/'inputs.json').read_text())
        prov=json.loads((d/'provenance.json').read_text())
        assert prov['job']==job and prov['driver_hashes']==driver
        assert prov['protocol_sha256']==sha(OUT/'pilot_protocol.json')
        assert prov['variant_config_sha256']==sha(ROOT/f'configs/research_a11/{variant}.yaml')
        assert prov['frozen_acceptance_manifest_sha256']==sha(ROOT/'artifacts/phase2d2_a11_reconciliation/acceptance_manifest.json')
        backend=prov['backend']
        assert backend['threads']==job['threads'] and backend['interop_threads']==expected_interop
        assert backend['mkldnn']==(not job['native']) and backend['deterministic']
        assert backend['matmul_precision']=='highest'
        execution=json.loads((OUT/'logs_v2'/f"{job['id']}.execution.json").read_text())
        assert execution['exit_code']==0
        assert execution['command'][-3:]==['worker','--index',str(ids.index(job['id']))]
        assert summary['status']=='COMPLETED_LOCAL_DIAGNOSTIC' and summary['optimizer_steps']==0
        if job['family']=='NLD':
            assert summary['state_and_parameter_identities_unchanged']
            state_key=(j,variant)
            if state_key not in states:
                p=OUT/f'states/j{j}_{variant}.pt';metadata=json.loads(p.with_suffix('.json').read_text())
                assert sha(p)==metadata['checkpoint_sha256']
                tensors=torch.load(p,map_location='cpu',weights_only=True)
                states[state_key]=(metadata,{k:(list(t.shape),tensor_hash(t)) for k,t in tensors.items()})
                del tensors
            metadata,hashes=states[state_key]
            assert inputs['state_hash_FP32']==metadata['LD_state_hash']
            assert set(inputs['parameters'])==set(hashes)
            for name,record in inputs['parameters'].items():
                assert (record['shape'],record['sha256'])==hashes[name]
            assert inputs['cut_schedule']==[1,4,7] and inputs['scales']==[0,1e-4,1,1024]
            assert inputs['multipliers']==list(multipliers[job['index']])
            for role,seed,shape,multiplier in zip(('state','memory','visual'),(6201+j,6301+j,6401+j),((2,308,512),(2,308,512),(2,1,512)),multipliers[job['index']]):
                x=torch.randn(shape,generator=torch.Generator().manual_seed(seed),dtype=torch.float32)
                x=(x[:b].clone().contiguous()*multiplier).contiguous();check(x,inputs['leaves'][role])
            for role,seed in (('u',6501+j),('v',6601+j)):
                x=torch.randn(b,308,512,generator=torch.Generator().manual_seed(seed),dtype=torch.float32)/math.sqrt(b*308*512)
                check(x,inputs[role])
            assert all((d/f'activations_{precision}.npz').is_file() for precision in ('float32','float64'))
        else:
            assert summary['state_unchanged']
            assert inputs['initialization_seed']==6101+j
            common=(inputs['state_hash_FP32'],inputs['parameters'])
            if j in natt_common:assert common==natt_common[j]
            else:natt_common[j]=common
            for role,seed,multiplier in (('x',6801+j,(.02,.2,2)[job['index']]),('cotangent_base',6901+j,1)):
                x=torch.randn(b,308,512,generator=torch.Generator().manual_seed(seed),dtype=torch.float32)
                if role=='x':x=x*multiplier
                check(x,inputs[role])
            assert all((d/f'activation_{precision}.npy').is_file() for precision in ('float32','float64'))
        # Keep original byte hashes for every required raw output and log.
        files=[d/'inputs.json',d/'provenance.json',d/'summary.json',d/'comparisons.jsonl']
        files += [OUT/'logs_v2'/f"{job['id']}.{suffix}" for suffix in ('stdout.log','stderr.log','execution.json')]
        rows.append({**job,'validation':'PASS','worker_exit':execution['exit_code'],
                     'numerical_FAIL_comparisons':summary['failed_comparisons'],
                     'raw_output_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in files}})
    if only is None:
        events=[json.loads(line) for line in (OUT/'resource_guard_v2/events.jsonl').read_text().splitlines()]
        assert events[-1]['event']=='COORDINATOR_HELD' and events[-1]['previous_worker_index']==407
        expected=list(range(278,288))+list(range(306,408))
        observed=[e['index'] for e in events if e['event']=='NEXT_WORKER_OBSERVED']
        assert observed==expected
        pending_gate=False
        for e in events:
            assert e['event']!='GUARD_STOP_REQUIRES_INSPECTION'
            if e['event']=='RAM_GATE_PASS':
                assert e['available_RAM']>=e['minimum_RAM']==3*2**30
                pending_gate=True
            if e['event']=='COORDINATOR_RELEASED' and e['expected_next_index'] is not None:
                assert pending_gate
                pending_gate=False
        handoff=json.loads((OUT/'resource_guard_handoff_v2.json').read_text())
        assert handoff['available_RAM_before_release']>=handoff['minimum_RAM']==3*2**30
        recovery_events=[json.loads(line) for line in (OUT/'continuation_0407/events.jsonl').read_text().splitlines()]
        assert recovery_events[-1]['event']=='REGISTERED_SCHEDULE_FINISHED'
        assert [e['index'] for e in recovery_events if e['event']=='RAM_GATE_PASS']==list(range(407,414))
        assert [e['index'] for e in recovery_events if e['event']=='WORKER_COMPLETED']==list(range(407,414))
        for e in recovery_events:
            if e['event']=='RAM_GATE_PASS':assert e['available_RAM']>=e['minimum_RAM']==3*2**30
        assert any(e['event']=='REPLAY_PREFIX_PASS' and e['records']==1946 and e['original_FAILs']==426 for e in recovery_events)
        replay=json.loads((OUT/'history/poweroff_0407_20261013/replay_comparison.json').read_text())
        assert replay['status']=='PASS' and not replay['differing_records']
        write(destination/'RAM_gate_audit.json',{'status':'PASS','first_gated_worker':277,
            'subsequent_gated_workers':observed,'threshold_bytes':3*2**30,
            'post_shutdown_recovery_gated_workers':list(range(407,414)),
            'old_guard_interruption':'Windows power-off during407; old coordinator exit1073807364, no worker exit recorded',
            'recovery_events_sha256':sha(OUT/'continuation_0407/events.jsonl'),
            'historical_workers_before_277':'No per-worker automatic gate is claimed; earlier manual boundary checks only',
            'guard_v2_source_sha256':sha(ROOT/'tools/d02a_ram_guard_v2.py'),
            'events_sha256':sha(OUT/'resource_guard_v2/events.jsonl')})
    write(destination/'identity_input_provenance_audit.json',{'status':'PASS','jobs_verified':len(rows),'complete_registration':only is None,
        'protocol_sha256':sha(OUT/'pilot_protocol.json'),'driver_manifest_sha256':sha(OUT/'driver_manifest_v2.json'),
        'input_validation':'Independent seeded FP32 recipe reconstruction, exact shapes and byte hashes; no model forward/backward',
        'checkpoint_validation':'Serialized LD checkpoint SHA and each parameter byte hash against recorded worker inputs',
        'jobs':rows})
    with (destination/'coverage.csv').open('x',newline='',encoding='utf-8') as h:
        fields=[k for k in rows[0] if k!='raw_output_hashes'];writer=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps({'status':'PASS','jobs_verified':len(rows),'complete_registration':only is None}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--only',type=int)
    args=p.parse_args();main(args.out,args.only)
