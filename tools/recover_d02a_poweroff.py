"""Archive an interrupted attempt, preserving bytes; no model/optimizer imports."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02a'
ARCHIVE=OUT/'history/poweroff_20261013'
JOB='0084_NLD_A0_j1_b2_m4_t4_mkldnn'


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:json.dump(value,handle,indent=2,allow_nan=False)


def main():
    for p in psutil.process_iter(['pid','cmdline']):
        if any(str(a).replace('\\','/').endswith('tools/d02a/run.py') for a in (p.info['cmdline'] or [])):
            raise RuntimeError('A numerical runner still exists; do not archive active evidence.')
    job=OUT/'jobs'/JOB
    assert job.is_dir() and not (job/'summary.json').exists()
    assert not (OUT/'logs_v2'/f'{JOB}.execution.json').exists()
    protocol=json.loads((OUT/'pilot_protocol.json').read_text())
    assert protocol['jobs'][84]['id']==JOB
    assert (ARCHIVE/'system_events.json').exists(), 'Capture the shutdown events before recovery.'
    baseline=json.loads((OUT/'preservation_before.json').read_text())
    changed=[]
    for name,record in baseline.items():
        p=ROOT/name
        if not p.is_file() or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:changed.append(name)
    write(ARCHIVE/'baseline_recheck.json',{'status':'FAIL' if changed else 'PASS',
          'files_checked':len(baseline),'changed_or_missing':changed,
          'before_manifest_sha256':sha(OUT/'preservation_before.json')})
    if changed:raise RuntimeError('Frozen baseline drift; no resume.')
    driver=json.loads((OUT/'driver_manifest_v2.json').read_text())
    assert all(sha(ROOT/name)==digest for name,digest in driver.items())
    sources=[job,OUT/'logs_v2'/f'{JOB}.stdout.log',OUT/'logs_v2'/f'{JOB}.stderr.log']
    before={}
    for source in sources:
        files=sorted(source.rglob('*')) if source.is_dir() else [source]
        for p in files:
            if p.is_file():before[p.relative_to(OUT).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
    write(ARCHIVE/'interrupted_attempt_manifest.json',before)
    records=failures=0;partial_lines=[]
    with (job/'comparisons.jsonl').open(encoding='utf-8') as handle, (ARCHIVE/'interrupted_comparisons.csv').open('x',newline='',encoding='utf-8') as target:
        fields=('kind','precision','stage','cut','scale','parameter','role','status','complete_record_json')
        writer=csv.DictWriter(target,fieldnames=fields);writer.writeheader()
        for index,line in enumerate(handle):
            try:row=json.loads(line)
            except json.JSONDecodeError:partial_lines.append(index+1);continue
            records+=1;failures+=row.get('status')=='FAIL'
            writer.writerow({**{k:row.get(k) for k in fields[:-1]},'complete_record_json':json.dumps(row,allow_nan=False)})
    # Each resolved source AND destination is checked inside the intended pilot
    # directory before an atomic same-volume rename. Nothing is deleted/overwritten.
    for source in sources:
        destination=ARCHIVE/source.relative_to(OUT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        source_abs=source.resolve();destination_abs=destination.resolve()
        assert source_abs.is_relative_to(OUT.resolve()) and destination_abs.is_relative_to(OUT.resolve())
        assert not destination_abs.exists()
        source_abs.rename(destination_abs)
    after={}
    for name,record in before.items():
        p=ARCHIVE/name;after[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
        assert after[name]==record
    write(ARCHIVE/'archive_verification.json',{'status':'PASS','files':len(before),
          'all_bytes_preserved':True,'original_to_archive_prefix':str(ARCHIVE.relative_to(ROOT)),
          'manifest_sha256':sha(ARCHIVE/'interrupted_attempt_manifest.json')})
    write(ARCHIVE/'recovery_record.json',{
        'recorded_utc':datetime.now(timezone.utc).isoformat(),'job':JOB,'index':84,
        'original_attempt_status':'INTERRUPTED / completion and exit code NOT RECORDED',
        'evidence':'Windows System events 1074/6006 record power-off immediately after the partial rows; subsequent event 6005/boot time confirms restart.',
        'numerical_hard_stop_recorded':False,'uncomputed_checks':'NOT RUN; no assumption that cut-7 FP64 would pass',
        'complete_partial_rows':records,'partial_attempt_failed_comparisons':failures,
        'malformed_or_truncated_line_numbers':partial_lines,
        'truncated_bytes_preserved_in_original_JSONL':True,
        'user_resume_instruction':'2026-10-13: kiem tra va tiep tuc cong viec truoc do; user also agreed to free RAM',
        'resume':'rerun index 84 from the same frozen state/input recipes, then continue unexecuted registered jobs',
        'changed_fixtures_operations_thresholds_or_driver':False,
        'already_completed_jobs_to_repeat':0,'optimizer_steps_authorized':False,
        'runtime_resource_gate':'wait for at least 3 GiB available RAM before launching; this is a host-resource gate, not a numerical acceptance threshold',
        'protocol_sha256':sha(OUT/'pilot_protocol.json'),'driver_manifest_sha256':sha(OUT/'driver_manifest_v2.json'),
        'baseline_recheck_sha256':sha(ARCHIVE/'baseline_recheck.json')})
    print(json.dumps({'status':'PASS','archived_files':len(before),'partial_rows':records,
                      'preserved_failed_comparisons':failures,'frozen_files_unchanged':len(baseline)}))


if __name__=='__main__':main()
