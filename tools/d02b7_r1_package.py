"""Read-only preservation verification and packaging for stopped R1; no worker API."""
import json,hashlib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02b7_r1';OLD=ROOT/'artifacts/phase2d2_d02b7'
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
record=json.loads((OUT/'execution_record.json').read_text())
assert record['execution']=='WAIT/NOT RUN' and record['worker_started'] is False
preserved=json.loads((OUT/'previous_attempt_preservation.json').read_text())
assert all(digest(ROOT/p)==r['sha256'] for p,r in preserved['files'].items())
write('log_provenance.json',{'resource_check.stdout.log':'Transcribed initial tool resource observation and stop decision; not model-worker stdout',
    'resource_check.stderr.log':'Empty transcription; no resource-query error observed',
    'worker_logs':'NOT CREATED: no worker launched','new_pytest_or_junit':'NOT RUN; D02B7 logs in package are historical'})
paths={p for p in OUT.rglob('*') if p.is_file()}
old_manifest=json.loads((OLD/'manifest.json').read_text())
paths.update(ROOT/p for p in old_manifest['files'])
paths.update(p for p in OLD.rglob('*') if p.is_file() and p.suffix!='.zip')
paths.update((ROOT/'docs/phase2d2_d02b7_r1_report.md',Path(__file__)))
files={p.relative_to(ROOT).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(paths)}
write('manifest.json',{'files':files,'execution':'WAIT/NOT RUN','phase2d2':'BLOCKED'})
archive=OUT/'phase2d2_d02b7_r1_review.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for rel in files:z.write(ROOT/rel,rel)
    z.write(OUT/'manifest.json','manifest.json')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)+1 and set(z.namelist())==set(files)|{'manifest.json'}
    assert all(hashlib.sha256(z.read(p)).hexdigest()==r['sha256'] for p,r in files.items())
    assert z.read('manifest.json')==(OUT/'manifest.json').read_bytes()
write('package_verification.json',{'crc':'PASS','payload_hashes':len(files),'status':'PASS','archive_bytes':archive.stat().st_size,
    'archive_sha256':digest(archive),'manifest_sha256':digest(OUT/'manifest.json')})
print(archive)
