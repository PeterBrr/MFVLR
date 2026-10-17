"""Package single attempt evidence without any model execution."""
import json,hashlib,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mfvlr_research.d02b1.schema import write_new,file_hash
OUT=ROOT/'artifacts/phase2d2_d02b7'
assert json.loads((OUT/'after.json').read_text())['status']=='PASS'
record=json.loads((OUT/'execution_record.json').read_text())
write_new(OUT/'verification_scope.json',dict(source_integrity='PASS',isolated_tests='18 PASS',
    live_model='NOT RUN' if not record['worker_started'] else 'see execution record',
    numerical=record['numerical'],provenance_pass_scope='file and registration integrity only when worker not started',
    historical_failures='unchanged',phase2d2='BLOCKED'))
paths=set()
for folder in ('src','configs','tests/research_d02b7','artifacts/phase2d2_d02b7'):
    paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.zip','.f32','.f64') and p.name not in ('manifest.json','package_verification.json'))
paths.update((ROOT/'tools').glob('d02b7_*.py'))
paths.add(ROOT/'docs/phase2d2_d02b7_g01_report.md')
pins=json.loads((OUT/'driver_manifest.json').read_text());paths.update(ROOT/p for p in pins)
paths.update((ROOT/'docs').glob('phase2d2_d02b0_*.md'))
paths.update((ROOT/'docs').glob('phase2d2_d02b6_*.md'))
files={p.relative_to(ROOT).as_posix():dict(sha256=file_hash(p),bytes=p.stat().st_size) for p in sorted(paths)}
write_new(OUT/'manifest.json',dict(files=files,execution=record['execution'],phase2d2='BLOCKED'))
archive=OUT/'phase2d2_d02b7_review.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for rel in files:z.write(ROOT/rel,rel)
    z.write(OUT/'manifest.json','manifest.json')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and set(z.namelist())==set(files)|{'manifest.json'} and len(z.namelist())==len(files)+1
    assert all(hashlib.sha256(z.read(p)).hexdigest()==v['sha256'] for p,v in files.items())
    assert z.read('manifest.json')==(OUT/'manifest.json').read_bytes()
write_new(OUT/'package_verification.json',dict(status='PASS',crc='PASS',payload_hashes=len(files),archive_bytes=archive.stat().st_size,
    archive_sha256=file_hash(archive),manifest_sha256=file_hash(OUT/'manifest.json')))
print(archive)
