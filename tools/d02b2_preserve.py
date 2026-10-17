"""File-only D02B2 preservation snapshot. Never imports model/experiment code."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02b2'

def digest(p):
    with p.open('rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
    return {'sha256':h,'bytes':p.stat().st_size}

def main(mode):
    OUT.mkdir(parents=True,exist_ok=True)
    if mode=='before':
        previous=json.loads((ROOT/'artifacts/phase2d2_d02b1/baseline_before.json').read_text())['files']
        paths=set(previous)
        for folder in ('src','configs','tests','tools','docs','artifacts/phase2d2_d02b1','artifacts/phase2d2_n31/n2'):
            paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*')
                         if p.is_file() and '__pycache__' not in p.parts and 'd02b2' not in p.as_posix())
        records={p:digest(ROOT/p) for p in sorted(paths)}
        bad=[p for p,v in previous.items() if records[p]!=v]
        if bad: raise RuntimeError(bad)
        data={'files':records,'prior_1637_file_inventory_mismatches':bad,
              'basis':'Fresh byte hashing, not independent scientific/source review of D02B1'}
    elif mode=='after':
        records=json.loads((OUT/'before.json').read_text())['files']
        bad=[p for p,v in records.items() if digest(ROOT/p)!=v]
        data={'files_checked':len(records),'mismatches':bad,'status':'FAIL' if bad else 'PASS',
              'before_sha256':digest(OUT/'before.json')['sha256']}
    else: raise ValueError(mode)
    with (OUT/f'{mode}.json').open('x') as f: json.dump(data,f,indent=2)
    print(json.dumps({k:v for k,v in data.items() if k!='files'}))
    if bad: raise SystemExit(1)

if __name__=='__main__': main(sys.argv[1])
