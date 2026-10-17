"""Compact read-only R2 evidence packaging; original gradient shards untouched."""
import json,hashlib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/phase2d2_d02b7_r2'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,v):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
assert json.loads((OUT/'live_evidence_audit.json').read_text())['status']=='PASS'
paths={p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ('.f32','.f64','.zip') and p.name not in ('manifest.json','package_verification.json')}
old=json.loads((ROOT/'artifacts/phase2d2_d02b7/manifest.json').read_text())
paths.update(ROOT/p for p in old['files'])
paths.update((ROOT/'tools').glob('d02b7_r2_*.py'))
paths.add(ROOT/'docs/phase2d2_d02b7_r2_report.md')
files={p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths)}
write('manifest.json',dict(files=files,numerical='FAIL',execution='COMPLETE',phase2d2='BLOCKED'))
archive=OUT/'phase2d2_d02b7_r2_review.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for rel in files:z.write(ROOT/rel,rel)
    z.write(OUT/'manifest.json','manifest.json')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(files)+1 and set(z.namelist())==set(files)|{'manifest.json'}
    assert all(hashlib.sha256(z.read(p)).hexdigest()==v['sha256'] for p,v in files.items())
    assert z.read('manifest.json')==(OUT/'manifest.json').read_bytes()
write('package_verification.json',dict(status='PASS',crc='PASS',payload_hashes=len(files),archive_bytes=archive.stat().st_size,
    archive_sha256=sha(archive),manifest_sha256=sha(OUT/'manifest.json'),raw_gradient_shards='preserved externally; full inventory included'))
print(archive)
