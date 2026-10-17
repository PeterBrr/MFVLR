"""Render measured review tables and verify additive N2 artifacts/preservation."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2c2_n2'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cases=[json.loads((OUT/f'run_01/fixture{i}_float32_threads4.json').read_text()) for i in range(3)]
    double=json.loads((OUT/'run_01/fixture0_float64_threads4.json').read_text())
    stages=[(f'LE{i+1}',f'encoder.blocks.{i}') for i in range(12)]+[(f'LD{i+1}',f'decoder.blocks.{i}') for i in range(7)]
    differences=['| Stage, F0 sample 0 | FP32 max absolute difference | Outside original FP32 tolerance / 157,696 | FP64 max absolute difference |',
                 '|---|---:|---:|---:|']
    activations=['| Stage, B=2 | F0 RMS / max absolute | F1 RMS / max absolute | F2 RMS / max absolute |',
                 '|---|---:|---:|---:|']
    for label,name in stages:
        f=cases[0]['cross_batch'][0]['stages'][name]
        d=double['cross_batch'][0]['stages'][name]
        differences.append(f"| {label} | {f['max_absolute_difference']:.8g} | {f['outside_tolerance_count']:,} | {d['max_absolute_difference']:.8g} |")
        cells=[f"{c['activations'][name]['rms']:.5f} / {c['activations'][name]['max_abs']:.5f}" for c in cases]
        activations.append('| '+label+' | '+' | '.join(cells)+' |')
    doc=ROOT/'docs/phase2c2_numerical_review.md'
    value=doc.read_text(encoding='utf-8')
    value=value.replace('<!-- N2_STAGE_DIFFERENCE_TABLE -->','\n'.join(differences))
    value=value.replace('<!-- N2_ACTIVATION_TABLE -->','\n'.join(activations))
    doc.write_text(value,encoding='utf-8')
    summary=json.loads((OUT/'numerical_summary.json').read_text())
    for c in summary['cases']:
        assert c['parameter_count']==112758272
        for k in ('parameter_objects_unchanged','state_unchanged','vocabulary_tie','fixed_independence_exact','permutation_exact'):
            assert c[k],(c['fixture'],k)
        if c['dtype']=='float64':assert c['all_stages_cross_batch_within_existing_tolerance']
        else:
            for k in ('all_parameter_gradients_finite','all_38_vim_qk_allocated_zero','cross_sample_gradients_exact_zero'):
                assert c[k],(c['fixture'],k)
    baseline=json.loads((OUT/'baseline_preservation.json').read_text())
    for name,expected in baseline['existing_files_sha256'].items():assert sha(ROOT/name)==expected,name
    # Links to the final new manifest/checksum are not used in these documents.
    for path in (doc,ROOT/'docs/phase2c2_n2_proposal.md'):
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' not in target:assert (path.parent/target.split('#')[0]).exists(),target
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ('manifest.json','verification.json')]
    files += [doc,ROOT/'docs/phase2c2_n2_proposal.md']
    files += sorted((ROOT/'tools').glob('*phase2c2_n2*.py'))
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}
    manifest={'review':'Phase 2C.2 N2','status':'PROPOSED','implementation':'BLOCKED pending explicit user approval',
        'proposal_applied':False,'full_phase2c2_accepted':False,
        'original_complete_suite':{'cases':161,'passed':160,'failed':1},
        'new_targeted_unchanged_tests':{'cases':15,'passed':14,'failed':1,'skipped':0,'xfail':0},
        'new_complete_suite_under_N2':'NOT RUN','expected_proposed_collection':164,
        'preserved_existing_files':len(baseline['existing_files_sha256']),
        'full_language_parameter_count':112758272,'combined_visual_language_count':153892549,
        'files_sha256':hashes,'self_hash_excluded':True}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result={'all_existing_files_preserved':True,'files_preserved':len(baseline['existing_files_sha256']),
        'all_review_invariants_verified':True,'document_links_valid':True,
        'proposal_patch_applied':False,'manifest_sha256':sha(OUT/'manifest.json'),
        'artifact_payloads':len(hashes),'full_model_acceptance':'BLOCKED'}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
