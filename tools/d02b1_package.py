"""Compact file-only review packaging; no scientific regeneration."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02b1'

def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def write(path, value):
    with path.open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)

def main():
    verification=json.loads((OUT/'verification.json').read_text())
    preservation=json.loads((OUT/'baseline_after.json').read_text())
    if verification['status']!='PASS' or preservation['status']!='PASS':
        raise RuntimeError('Cannot package as verified')
    # Fresh test identities and raw stdout remain included, not only a PASS summary.
    paths=set()
    for folder in ('src','tests/research_d02b1','artifacts/phase2d2_d02b1'):
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix!='.zip' and p.name not in ('manifest.json','package_verification.json'))
    paths.update((ROOT/'tools').glob('d02b1_*.py'))
    paths.update((ROOT/'docs').glob('phase2d2_d02b1_*.md'))
    paths.update((ROOT/'docs').glob('phase2d2_d02b0_*.md'))
    for name in ('phase2d2_a1_ablation_preregistration_v1_1.md','phase2d2_a11_acceptance_snapshot.md',
                 'phase2d2_a11_incident_closure.md','phase2d2_d02a_numerical_pilot_report.md','phase2d2_n31_heldout_validation.md'):
        paths.add(ROOT/'docs'/name)
    for name in ('A0','P3','Q3'):
        paths.add(ROOT/f'configs/research_a11/{name}.yaml')
        paths.add(ROOT/f'artifacts/phase2d2_a11/runs/research_complete_01/evidence/ownership_{name}.json')
    for name in ('junit.xml','execution.json','stdout.log','stderr.log'):
        paths.add(ROOT/'artifacts/phase2d2_d02a/oracles'/name)
    paths.add(ROOT/'tools/research_phase2d2_a11.py')
    paths.add(ROOT/'pyproject.toml')
    files={p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(paths)}
    manifest={'scope':'D02B1 source/static/mock review; real cells NOT RUN','files':files,
              'historical_external_evidence':'Full original archives remain outside this compact ZIP; exact fingerprints in baseline_before.json',
              'self_hash_policy':'manifest excludes itself; its SHA and archive SHA are in package_verification.json'}
    write(OUT/'manifest.json',manifest)
    archive=OUT/'phase2d2_d02b1_review.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for rel in files: z.write(ROOT/rel,rel)
        z.write(OUT/'manifest.json','manifest.json')
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(set(z.namelist())): raise RuntimeError('Duplicate ZIP names')
        if set(z.namelist())!=set(files)|{'manifest.json'}: raise RuntimeError('Unexpected ZIP payload')
        bad_crc=z.testzip()
        mismatches=[rel for rel,r in files.items() if hashlib.sha256(z.read(rel)).hexdigest()!=r['sha256'] or len(z.read(rel))!=r['bytes']]
        if z.read('manifest.json')!=(OUT/'manifest.json').read_bytes(): raise RuntimeError('Manifest mismatch')
    if bad_crc or mismatches: raise RuntimeError((bad_crc,mismatches))
    record={'status':'PASS','crc':'PASS','included_sha256_checks':len(files),'mismatches':mismatches,
            'manifest_sha256':sha(OUT/'manifest.json'),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
            'archive':archive.relative_to(ROOT).as_posix(),'full_graph_execution':'NOT RUN','phase2d2':'BLOCKED'}
    write(OUT/'package_verification.json',record)
    print(json.dumps(record,indent=2))

if __name__=='__main__': main()
