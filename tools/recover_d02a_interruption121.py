"""Use the reviewed byte-preserving recovery procedure for interrupted job 121.

This is host interruption recovery, not a scientific driver modification.
The original job-84 recovery script and all its evidence remain unchanged.
"""
from pathlib import Path

source = Path(__file__).with_name('recover_d02a_poweroff.py').read_text()
replacements = {
    'poweroff_20261013': 'interruption_0121_20261013',
    '0084_NLD_A0_j1_b2_m4_t4_mkldnn': '0121_NLD_P3_j2_b2_m0_t4_mkldnn',
    "['jobs'][84]": "['jobs'][121]",
    "'index':84": "'index':121",
    'system_events.json': 'interruption_inspection.json',
    'Capture the shutdown events before recovery.': 'Capture process/log/event inspection before recovery.',
    'Windows System events 1074/6006 record power-off immediately after the partial rows; subsequent event 6005/boot time confirms restart.':
        'No live runner, summary or exit record; empty stdout/stderr; no recorded numerical hard stop. Cause UNKNOWN: event queries did not identify a matching event. Power-off and OOM are not established.',
    'rerun index 84': 'rerun index 121',
    'cut-7 FP64 would pass': 'remaining comparisons would pass',
}
for old, new in replacements.items():
    assert old in source, old
    source = source.replace(old, new)
exec(compile(source, str(Path(__file__)), 'exec'))
