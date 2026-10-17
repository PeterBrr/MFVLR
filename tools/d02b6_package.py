"""Package static/mock implementation and immutable read-only N2 evidence."""
import hashlib,json,sys,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mfvlr_research.d02b1.schema import write_new,file_hash
OUT=ROOT/'artifacts/phase2d2_d02b6'
test=json.loads((OUT/'complete_test_execution.json').read_text())
assert test['exit_code']==0 and test['status']=='PASS'
assert json.loads((OUT/'after.json').read_text())['status']=='PASS'
tree=ET.parse(OUT/'complete_mock_junit.xml');suite=tree.getroot().find('testsuite')
assert suite is not None and int(suite.attrib['failures'])==int(suite.attrib['errors'])==int(suite.attrib['skipped'])==0
paths=set()
for folder in ('src','configs','tests/research_d02b6','tests/research_d02b1','artifacts/phase2d2_d02b6'):
    paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.zip' and p.name not in ('manifest.json','package_verification.json'))
for phase in ('d02b3','d02b4','d02b5'):
    paths.update(p for p in (ROOT/'artifacts'/f'phase2d2_{phase}').rglob('*') if p.is_file() and p.suffix!='.zip')
paths.update((ROOT/'tools').glob('d02b6_*.py'))
paths.update((ROOT/'docs').glob('phase2d2_d02b6_*.md'))
for pattern in ('phase2d2_d02b0_*.md','phase2d2_d02b1_*.md','phase2d2_d02b3_*.md','phase2d2_d02b4_*.md','phase2d2_d02b5_*.md'):
    paths.update((ROOT/'docs').glob(pattern))
for rel in ('docs/phase2d_losses_contract.md','docs/phase2d2_a1_ablation_preregistration_v1_1.md','docs/phase2d2_n31_heldout_validation.md','tools/phase2d2_fixtures.py'):
    paths.add(ROOT/rel)
from mfvlr_research.d02b2.contract import bundle
paths.update(ROOT/p for p in bundle()['sources'])
anchors=json.loads((OUT/'historical_g2_anchors.json').read_text())
paths.update(ROOT/r['path'] for r in anchors.values())
files={p.relative_to(ROOT).as_posix():dict(sha256=file_hash(p),bytes=p.stat().st_size) for p in sorted(paths)}
write_new(OUT/'manifest.json',dict(files=files,phase2d2='BLOCKED',execution_scope='static/mock only'))
archive=OUT/'phase2d2_d02b6_review.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for rel in files:z.write(ROOT/rel,rel)
    z.write(OUT/'manifest.json','manifest.json')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)+1 and set(z.namelist())==set(files)|{'manifest.json'}
    assert all(hashlib.sha256(z.read(p)).hexdigest()==v['sha256'] for p,v in files.items())
    assert z.read('manifest.json')==(OUT/'manifest.json').read_bytes()
write_new(OUT/'package_verification.json',dict(status='PASS',crc='PASS',payload_hashes=len(files),archive_bytes=archive.stat().st_size,
    archive_sha256=file_hash(archive),manifest_sha256=file_hash(OUT/'manifest.json'),mock_tests=int(suite.attrib['tests']),real_g2_cells_executed=0))
print(archive)
