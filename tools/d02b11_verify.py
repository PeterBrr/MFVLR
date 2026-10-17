"""Read-only pin/backend/resource verification. No model construction/dispatch."""
import ast
import hashlib
import json
import math
import os
import platform
import shutil
import time
import zipfile
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02b11'
PIN='80ec547ea34e6ede3af6d4c09408c82b0acbc43c002a0cc3de2a263f13e3281f'

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def write(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def main():
    pins_path=ROOT/'configs/research_d02b10/live_pins.json'
    if sha(pins_path)!=PIN:raise RuntimeError('Trusted pin file changed')
    pins=json.loads(pins_path.read_text())['files']
    if len(pins)!=4343:raise RuntimeError('Pin count changed')
    with zipfile.ZipFile(ROOT/'artifacts/phase2d2_d02b10/phase2d2_d02b10_review.zip') as z:
        omitted=sorted(set(pins)-set(z.namelist()))
    records={}
    for rel,expected in pins.items():
        p=(ROOT/rel).resolve()
        actual=sha(p) if p.is_relative_to(ROOT) and p.is_file() else None
        records[rel]=dict(expected=expected,actual=actual,status='PASS' if actual==expected else 'FAIL')
    bad=[p for p,r in records.items() if r['status']!='PASS']
    write('pins.json',dict(pin_file_sha256=PIN,total=len(pins),failed=bad,records=records))
    write('zip_omissions.json',dict(count=len(omitted),files={p:records[p] for p in omitted}))
    if bad:
        write('decision.json',dict(execution='WAIT/NOT RUN',reason='Pin prerequisite failed',failed=bad));return
    # Inspect the complete historical inventory, not a newly instantiated model.
    old=ROOT/'artifacts/phase2d2_d02b7_r2/attempt_01'
    roster=json.loads((ROOT/'tests/research_d02b1/fixtures/historical_rosters.json').read_text())['rosters']['A0']
    inv=json.loads((old/'inventory_before.json').read_text())
    same=json.loads((old/'inventory_after.json').read_text())==inv
    assert len(inv)==508 and sum(math.prod(r['shape']) for r in inv)==154156232
    assert [(r['name'],r['shape']) for r in inv]==[(r['name'],r['shape']) for r in roster]
    assert same and len({r['object_id'] for r in inv})==508 and len({r['storage_id'] for r in inv})==508
    seal=json.loads((old/'fd_graph_entry.json').read_text())
    state='1865527facba9adc8a14d443c112d492698d275ce4e8beaa784aee61a514895a'
    assert seal['state']==state
    result=json.loads((old/'result.json').read_text())
    assert result['numerical']=='FAIL' and result['outside_coordinates']==8772234
    analysis=json.loads((old/'additivity.json').read_text())
    assert sum(r['outside_tolerance_count']>0 for r in analysis['parameters'].values())==293
    fixture=json.loads((old/'fixture.json').read_text())
    write('frozen_identity.json',dict(status='HISTORICAL_PINNED_EVIDENCE_VERIFIED',parameter_tensors=508,
        elements=154156232,state_sha256=state,fixture=fixture,inventory_unchanged=same,
        new_model_constructed=False,new_live_state_measurement='NOT RUN',failed_tensors=293,outside_coordinates=8772234))
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(4)
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.mkldnn.enabled=True
    # Evaluate only the frozen backend() function AST, not its module (which imports FLT).
    source=ast.parse((ROOT/'tools/review_phase2c2_n2.py').read_text(encoding='utf-8'))
    node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='backend')
    namespace=dict(torch=torch,platform=platform,os=os)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen backend probe>','exec'),namespace)
    actual=namespace['backend']()
    expected=json.loads((ROOT/'src/mfvlr_research/d02b2/frozen_inputs.json').read_text())['fixtures']['F0']['backend']
    mismatches={k:dict(expected=expected.get(k),actual=actual.get(k)) for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k)}
    write('backend.json',dict(status='PASS' if not mismatches else 'FAIL',actual=actual,expected=expected,mismatches=mismatches))
    r=dict(time_ns=time.time_ns(),pid=os.getpid(),available_ram=psutil.virtual_memory().available,
           total_physical_ram=psutil.virtual_memory().total,free_disk=shutil.disk_usage(ROOT).free,
           inspection_process_rss=psutil.Process().memory_info().rss,minimum_ram=8*1024**3,minimum_disk=8*1024**3)
    r['status']='PASS' if r['available_ram']>=r['minimum_ram'] and r['free_disk']>=r['minimum_disk'] else 'WAIT'
    write('resources.json',r)
    write('decision.json',dict(execution='WAIT/NOT RUN',pins='PASS',backend='PASS' if not mismatches else 'FAIL',
        resources=r['status'],authorization='NOT GRANTED',live_model_state='NOT RUN',
        reason='Prerequisites insufficient and/or explicit live authorization absent',B0_D08='DISABLED',phase='BLOCKED'))
    print(json.dumps(dict(pins=len(pins),omitted_from_zip=len(omitted),backend_mismatches=mismatches,resources=r,execution='WAIT/NOT RUN')))

if __name__=='__main__':main()
