"""Transcribe existing N2 JSON inputs/metadata; no fixture/model execution."""
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'src/mfvlr_research/d02b2'

def file_hash(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def flat(x):
    if isinstance(x,list):
        for v in x: yield from flat(v)
    else: yield x

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources=set()
    sources.update((ROOT/'src/mfvlr').glob('*.py'))
    sources.update((ROOT/'configs').glob('*.yaml'))
    sources.update((ROOT/'configs/research_a11').glob('*.yaml'))
    for rel in ('tools/review_phase2c2_n2.py','tests/language/test_flt.py','tests/language/conftest.py',
                'tests/foundation/fixtures/golden_tokens.json','src/mfvlr_research/decoder_a11.py',
                'src/mfvlr_research/a11_config.py','docs/phase2d2_a1_ablation_preregistration_v1_1.md',
                'docs/phase2d2_d02b1_oracle_prerequisites.md','src/mfvlr_research/d02b1/runner.py'):
        sources.add(ROOT/rel)
    fixtures={}
    for index in range(3):
        pair=[]
        for dtype in ('float32','float64'):
            path=ROOT/f'artifacts/phase2d2_n31/n2/fixture{index}_{dtype}_threads4.json'
            sources.add(path)
            pair.append(json.loads(path.read_text()))
        a,b=pair
        if a['fixture_description']!=b['fixture_description'] or a['fp32_initialized_state_sha256']!=b['fp32_initialized_state_sha256']:
            raise RuntimeError('Historical precision pair inputs/state do not match')
        desc=a['fixture_description']
        for key,fmt,expected in (('ids','q','ids_sha256'),('fp32_visual','f','fp32_visual_sha256')):
            digest=hashlib.sha256(b''.join(struct.pack('<'+fmt,x) for x in flat(desc[key]))).hexdigest()
            if digest!=desc[expected]: raise RuntimeError('Historical literal round-trip hash mismatch')
        fixtures[f'F{index}']={'historical_source':f'artifacts/phase2d2_n31/n2/fixture{index}_float32_threads4.json',
                             'fixture_description':desc,'backend':a['backend'],
                             'a0_fp32_state_sha256':a['fp32_initialized_state_sha256']}
    rosters={}
    for variant in ('A0','P3','Q3'):
        path=ROOT/f'artifacts/phase2d2_a11/runs/research_complete_01/evidence/ownership_{variant}.json'
        sources.add(path)
        report=json.loads(path.read_text())
        roster=[{'name':n.removeprefix('flt.'),'shape':v['shape']} for n,v in report['common'].items() if n.startswith('flt.')]
        roster += [{'name':n.removeprefix('flt.'),'shape':[512]} for n in report['new_names']]
        expected=(387,112758272) if variant=='A0' else (429,112779776)
        if (len(roster),sum(math.prod(x['shape']) for x in roster))!=expected: raise RuntimeError('Historical inventory')
        rosters[variant]=roster
    data={'scope':'Existing literal inputs and ownership metadata; no new numerical run',
          'state_hash_algorithm':'N2: name UTF-8 + str(tuple(shape)) UTF-8 + contiguous native CPU little-endian parameter bytes, in named_parameters order',
          'sources':{p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted(sources)},
          'fixtures':fixtures,'rosters':rosters}
    with (OUT/'frozen_inputs.json').open('x',encoding='utf-8') as f: json.dump(data,f,indent=2,allow_nan=False)
    print('frozen_inputs_sha256='+file_hash(OUT/'frozen_inputs.json'))

if __name__=='__main__': main()
