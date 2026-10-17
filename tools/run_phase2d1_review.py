"""Actual independent unit/regression runs; preserve all historical evidence."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d1'


def run(stage):
    index=1
    while (OUT/f'logs/{stage}_{index:02}.execution.json').exists():index+=1
    name=f'{stage}_{index:02}'
    logs=OUT/'logs';logs.mkdir(parents=True,exist_ok=True)
    junit=OUT/'junit'/f'{name}.xml';junit.parent.mkdir(parents=True,exist_ok=True)
    paths={
        'heads':['tests/objectives/test_heads.py'],
        'image':['tests/objectives/test_detection_localization_appearance.py'],
        'language':['tests/objectives/test_language_loss.py'],
        'alignment':['tests/objectives/test_alignment_losses.py'],
        'unit':['tests/objectives'],
        'regression':['tests','--ignore=tests/objectives'],
        'complete':['tests'],
    }[stage]
    command=[sys.executable,'-m','pytest',*paths,'-vv','-s',f'--junitxml={junit}']
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)])
    historical={p:p.read_bytes() for p in (ROOT/'artifacts/phase2c2/gradients').glob('*.json')}
    for p,data in historical.items():
        target=OUT/'history/gradients'/p.name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(data)
    started=time.perf_counter()
    try:
        with (logs/f'{name}.stdout.log').open('w',encoding='utf-8') as stdout, (logs/f'{name}.stderr.log').open('w',encoding='utf-8') as stderr:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    finally:
        if stage in ('regression','complete'):
            for p,data in historical.items():
                fresh=OUT/'regression_gradients'/name/p.name;fresh.parent.mkdir(parents=True,exist_ok=True)
                fresh.write_bytes(p.read_bytes());p.write_bytes(data)
    record={'stage':stage,'command':command,'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-started,
            'cwd':str(ROOT),'PYTHONPATH':env['PYTHONPATH'],'source_modified_by_runner':False}
    (logs/f'{name}.execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    raise SystemExit(result.returncode)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('heads','image','language','alignment','unit','regression','complete'))
    run(parser.parse_args().stage)
