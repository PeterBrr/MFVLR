"""File-only D02B8 baseline preservation."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02b8'
def digest(p):
    with p.open('rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(sha256=h,bytes=p.stat().st_size)
def main(mode):
    OUT.mkdir(parents=True,exist_ok=True)
    if mode=='before':
        old=json.loads((ROOT/'artifacts/phase2d2_d02b7/before.json').read_text())['files']
        paths=set(old)
        for folder in ('src','configs','tests','tools','docs','artifacts/phase2d2_d02b7','artifacts/phase2d2_d02b7_r1','artifacts/phase2d2_d02b7_r2'):
            paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'd02b8' not in p.as_posix())
        files={p:digest(ROOT/p) for p in sorted(paths)}
        bad=[p for p,v in old.items() if files[p]!=v]
        record={'files':files,'mismatches':bad}
    elif mode=='after':
        files=json.loads((OUT/'before.json').read_text())['files']
        bad=[p for p,v in files.items() if digest(ROOT/p)!=v]
        record={'files_checked':len(files),'mismatches':bad,'status':'FAIL' if bad else 'PASS'}
    else: raise ValueError(mode)
    with (OUT/f'{mode}.json').open('x') as f: json.dump(record,f,indent=2)
    print(json.dumps({k:v for k,v in record.items() if k!='files'}))
    if bad: raise SystemExit(1)
if __name__=='__main__': main(sys.argv[1])
