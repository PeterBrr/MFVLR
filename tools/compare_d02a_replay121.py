"""Apply the unchanged numerical-prefix comparison to interrupted job 121."""
from pathlib import Path

source = Path(__file__).with_name('compare_d02a_replay.py').read_text()
for old, new in {
    'poweroff_20261013': 'interruption_0121_20261013',
    '0084_NLD_A0_j1_b2_m4_t4_mkldnn': '0121_NLD_P3_j2_b2_m0_t4_mkldnn',
}.items():
    assert old in source
    source = source.replace(old, new)
exec(compile(source, str(Path(__file__)), 'exec'))
