"""Freeze per-split manifests from the completed candidate decode audit."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mfvlr_research.pilot_data import freeze_candidates

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/diffae_pilot/preparation/checked_candidates.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/diffae_pilot/manifests')
    args = parser.parse_args()
    report = args.source.with_name('decode_report.json')
    audit = json.loads(report.read_text(encoding='utf-8'))
    if audit['failures']:
        parser.error('Decode audit has failures; resolve them before freezing')
    rows = sum(1 for line in args.source.read_text(encoding='utf-8').splitlines() if line.strip())
    if rows != audit['counts']['checked']:
        parser.error('Incomplete checked_candidates manifest')
    print(json.dumps(freeze_candidates(args.source, args.output), indent=2))
