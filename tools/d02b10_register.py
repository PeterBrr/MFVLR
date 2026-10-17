"""Read-only registration of existing source/evidence hashes; no model imports."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    paths=set()
    for folder in ('src','configs','tools'):
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix in ('.py','.json','.yaml','.gz','.toml'))
    paths.update((ROOT/'docs').glob('phase2d2_d02b9*.md'))
    paths.update((ROOT/'docs').glob('phase2d2_d02b0*.md'))
    paths.add(ROOT/'docs/phase2d2_d02b10_operational_contract.md')
    paths.add(ROOT/'tests/research_d02b1/fixtures/historical_rosters.json')
    paths.add(ROOT/'tests/foundation/fixtures/golden_tokens.json')
    old=ROOT/'artifacts/phase2d2_d02b7_r2/attempt_01'
    paths.update(old.glob('*.json'))
    paths.update((old/'gradient_stream').glob('*.json'))
    analysis=json.loads((old/'additivity.json').read_text())
    counts=dict(failed_tensors=sum(r['outside_tolerance_count']>0 for r in analysis['parameters'].values()),
                outside_coordinates=sum(r['outside_tolerance_count'] for r in analysis['parameters'].values()))
    assert counts==dict(failed_tensors=293,outside_coordinates=8772234)
    pins={}
    for p in sorted(paths):
        with p.open('rb') as stream:pins[p.relative_to(ROOT).as_posix()]=hashlib.file_digest(stream,'sha256').hexdigest()
    record=dict(files=pins,cell='D02B-G01-A0-SAME-FORWARD',historical_result=counts)
    target=ROOT/'configs/research_d02b10/live_pins.json';target.parent.mkdir(exist_ok=True)
    with target.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
    print(json.dumps(dict(files=len(pins),sha256=hashlib.sha256(target.read_bytes()).hexdigest())))

if __name__=='__main__':main()
