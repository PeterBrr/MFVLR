"""Exact scientific-prefix comparison; timing fields are the only exclusions."""
import json
from pathlib import Path

source=Path(__file__).with_name('compare_d02a_replay.py').read_text()
for old,new in {
    'poweroff_20261013':'poweroff_0407_20261013',
    '0084_NLD_A0_j1_b2_m4_t4_mkldnn':'0407_NLD_Q3_j1_b2_m1_t1_mkldnn',
}.items():
    assert old in source
    source=source.replace(old,new)
exec(compile(source,str(Path(__file__)),'exec'))
p=Path(__file__).resolve().parents[1]/'artifacts/phase2d2_d02a/history/poweroff_0407_20261013/replay_comparison.json'
r=json.loads(p.read_text())
assert r['compared_prefix_records']==1946 and r['old_prefix_FAILs']==426
if r['status']!='PASS':raise SystemExit('Scientific prefix differences preserved; STOP for diagnosis')
