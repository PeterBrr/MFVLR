"""Apply byte-preserving recovery to the explicitly authorized 40-file attempt."""
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02a'
JOB='0407_NLD_Q3_j1_b2_m1_t1_mkldnn'
job=OUT/'jobs'/JOB
assert sum(p.is_file() for p in job.rglob('*'))==40
prior=json.loads((OUT/'progress_snapshots/20261014_131145/progress.json').read_text())
partial=next(p for p in prior['partial_attempts'] if p['index']==407)
assert partial['complete_partial_rows']==1946 and partial['partial_FAILs']==426
for name,record in partial['files'].items():
    p=ROOT/name
    with p.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
    assert p.stat().st_size==record['bytes'] and digest==record['sha256'],name

archive=OUT/'history/poweroff_0407_20261013'
archive.mkdir(exist_ok=False)
events=OUT/'history/poweroff_0407_20261013_inspection/system_events.json'
(archive/'system_events.json').open('xb').write(events.read_bytes())
source=Path(__file__).with_name('recover_d02a_poweroff.py').read_text()
for old,new in {
    'poweroff_20261013':'poweroff_0407_20261013',
    '0084_NLD_A0_j1_b2_m4_t4_mkldnn':JOB,
    "['jobs'][84]":"['jobs'][407]", "'index':84":"'index':407",
    'rerun index 84':'rerun index 407',
    '2026-10-13: kiem tra va tiep tuc cong viec truoc do; user also agreed to free RAM':'2026-10-14 explicit recovery authorization for all 40 partial files and original 426 FAIL comparisons',
    "'original_attempt_status':'INTERRUPTED / completion and exit code NOT RECORDED'":"'original_attempt_status':'INTERRUPTED / worker completion and exit code NOT RECORDED; old coordinator exit 1073807364 recorded separately'",
}.items():
    assert old in source,old
    source=source.replace(old,new)
exec(compile(source,str(Path(__file__)),'exec'))
record=json.loads((archive/'archive_verification.json').read_text())
assert record['files']==42  # all 40 partial files, plus original stdout and stderr
assert record['all_bytes_preserved']
