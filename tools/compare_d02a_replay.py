"""Compare a resumed job with its preserved shutdown-interrupted prefix."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02a'
OLD=OUT/'history/poweroff_20261013'
JOB='0084_NLD_A0_j1_b2_m4_t4_mkldnn'


def digest(p):
    with p.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()


def main():
    old=OLD/'jobs'/JOB;new=OUT/'jobs'/JOB
    summary=json.loads((new/'summary.json').read_text())
    assert summary['status']=='COMPLETED_LOCAL_DIAGNOSTIC'
    before=json.loads((old/'inputs.json').read_text());after=json.loads((new/'inputs.json').read_text())
    assert before==after, 'Different initialized state or input recipe; stop review.'
    old_prov=json.loads((old/'provenance.json').read_text());new_prov=json.loads((new/'provenance.json').read_text())
    assert old_prov==new_prov, 'Different source/config/backend/protocol; stop review.'
    archive_manifest=json.loads((OLD/'interrupted_attempt_manifest.json').read_text())
    assert all(digest(OLD/p)==v['sha256'] for p,v in archive_manifest.items())
    differing=[];records=0;old_fail=new_fail=0;non_numeric={'VJP_triplet_seconds'}
    with (old/'comparisons.jsonl').open() as a,(new/'comparisons.jsonl').open() as b:
        for previous in a:
            earlier=json.loads(previous);later=json.loads(next(b))
            records+=1;old_fail+=earlier.get('status')=='FAIL';new_fail+=later.get('status')=='FAIL'
            # Timings legitimately change after reboot. Every mathematical value,
            # FAIL/PASS outcome, gradient hash, shape and threshold remains compared.
            skip=non_numeric|({'cold_forward_seconds','measured_seconds','measured_full_depth_unit_cotangent_seconds'} if earlier['kind']=='timing' else set())
            fields=sorted(k for k in set(earlier)|set(later) if k not in skip and earlier.get(k)!=later.get(k))
            if fields:differing.append({'row':records,'fields':fields,'original':earlier,'replay':later})
    output=OLD/'replay_comparison.json'
    with output.open('x',encoding='utf-8') as h:json.dump({
        'status':'PASS' if not differing else 'DIFFERENCES_REQUIRE_REVIEW',
        'compared_prefix_records':records,'old_prefix_FAILs':old_fail,'new_prefix_FAILs':new_fail,
        'input_state_and_provenance_exact':True,'archived_attempt_bytes_unchanged':True,
        'ignored_fields':'timings only; no numerical thresholds or outcomes omitted',
        'differing_records':differing,'resumed_job_summary':summary},h,indent=2,allow_nan=False)
    print(json.dumps({'compared':records,'differing_records':len(differing),'original_FAILs_preserved':old_fail}))


if __name__=='__main__':main()
