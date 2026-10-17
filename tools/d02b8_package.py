"""Compact forensic records; no scientific shards removed or model calls."""
import hashlib,json,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/phase2d2_d02b8'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,v):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
assert read(OUT/'after.json')['status']=='PASS' and read(OUT/'recompute_exit.json')['exit_code']==read(OUT/'mock_exit.json')['exit_code']==0
audit=read(OUT/'independent_recomputation_report.json')
assert audit['all_parameter_failure_counts_exact'] and audit['all_junit_decisions_exact']
suite=ET.Element('testsuite',name='D02B8 independent record reconciliation',tests='508',failures='0',errors='0',skipped='0')
for row in read(OUT/'reconciliation.json'):
    ET.SubElement(suite,'testcase',name=row['parameter'],classname='D02B8.evidence_reconciliation')
ET.ElementTree(suite).write(OUT/'reconciliation.junit.xml',encoding='utf-8',xml_declaration=True)
paths=set()
for folder in ('src/mfvlr_research/d02b8','tests/research_d02b8','configs/research_d02b8','artifacts/phase2d2_d02b8'):
    paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json','package_verification.json') and p.suffix!='.zip')
paths.update((ROOT/'tools').glob('d02b8_*.py'));paths.update((ROOT/'docs').glob('phase2d2_d02b8_*.md'))
prior=ROOT/'artifacts/phase2d2_d02b7_r2'
paths.update(p for p in prior.rglob('*') if p.is_file() and p.suffix not in ('.f32','.f64','.zip'))
pins=read(prior/'driver_manifest.json');paths.update(ROOT/p for p in pins)
files={p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths)}
write('manifest.json',dict(files=files,numerical='FAIL preserved',phase2d2='BLOCKED',raw_binary_shards='omitted from ZIP only; original paths/hashes/sizes in shard_integrity.json'))
archive=OUT/'phase2d2_d02b8_review.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in files:z.write(ROOT/p,p)
    z.write(OUT/'manifest.json','manifest.json')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(files)+1 and set(z.namelist())==set(files)|{'manifest.json'}
    assert all(hashlib.sha256(z.read(p)).hexdigest()==v['sha256'] for p,v in files.items())
    assert z.read('manifest.json')==(OUT/'manifest.json').read_bytes()
write('package_verification.json',dict(status='PASS',crc='PASS',payload_hashes=len(files),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),manifest_sha256=sha(OUT/'manifest.json')))
print(archive)
