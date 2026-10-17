"""Package existing D02B10 evidence only; never execute models or tests."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/phase2d2_d02b10'

def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    preservation=json.loads((OUT/'after.json').read_text(encoding='utf-8'))
    if preservation['status']!='PASS':raise RuntimeError('Preservation failed')
    verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    if verification['exit_code']!=0:raise RuntimeError('Mock verification failed')
    paths=set()
    for folder in ('src','configs','tests/research_d02b10','tools'):
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.yaml','.toml','.txt'))
    paths.update((ROOT/'docs').glob('phase2d2_d02b10*.md'))
    paths.add(ROOT/'pyproject.toml')
    paths.add(ROOT/'tests/research_d02b1/fixtures/historical_rosters.json')
    paths.add(ROOT/'tests/foundation/fixtures/golden_tokens.json')
    paths.update((ROOT/'src/mfvlr/tokenizer_assets').rglob('*.gz'))
    paths.add(ROOT/'src/mfvlr/tokenizer_assets/LICENSE')
    paths.update((ROOT/'artifacts/phase2d2_d02b7_r2/attempt_01/gradient_stream').glob('*.json'))
    paths.update(p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.xml','.log') and p.name not in ('manifest.json','package_verification.json'))
    paths.update(ROOT/p for p in ('artifacts/phase2d2_d02b7_r2/attempt_01/fixture.json',
        'artifacts/phase2d2_d02b7_r2/attempt_01/result.json',
        'artifacts/phase2d2_d02b7_r2/attempt_01/losses.json',
        'artifacts/phase2d2_d02b8/reconciliation.json'))
    files={p.relative_to(ROOT).as_posix():dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(paths)}
    manifest=dict(scope='D02B10 preparation only',files=files,real_execution='NOT RUN',
        note='Historical binary evidence remains in original archive/workspace; before.json lists preservation hashes.')
    with (OUT/'manifest.json').open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2)
    paths.add(OUT/'manifest.json')
    archive=OUT/'phase2d2_d02b10_review.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(paths):
            name=p.relative_to(ROOT).as_posix()
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
    failures=[]
    with zipfile.ZipFile(archive) as z:
        crc=z.testzip()
        for name,record in files.items():
            data=z.read(name)
            if hashlib.sha256(data).hexdigest()!=record['sha256'] or len(data)!=record['bytes']:failures.append(name)
        if z.read('artifacts/phase2d2_d02b10/manifest.json') != (OUT/'manifest.json').read_bytes():failures.append('manifest')
    record=dict(zip_sha256=digest(archive),zip_bytes=archive.stat().st_size,
        manifest_sha256=digest(OUT/'manifest.json'),payloads_checked=len(files),crc_error=crc,
        hash_failures=failures,status='PASS' if crc is None and not failures else 'FAIL')
    with (OUT/'package_verification.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
    print(json.dumps(record))
    if record['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
