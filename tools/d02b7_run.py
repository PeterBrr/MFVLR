"""One attempt, no wait loop, retry, continuation, or other cell selection."""
import json,os,sys,time,subprocess,shutil
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mfvlr_research.d02b7.policy import CELL,resource_gate,status
from mfvlr_research.d02b1.schema import write_new,file_hash,verify_files
OUT=ROOT/'artifacts/phase2d2_d02b7'

def main():
    test=json.loads((OUT/'preflight_tests.json').read_text())
    if test['exit_code']!=0:raise RuntimeError('Isolated tests must pass')
    verify_files(ROOT,test['source_hashes'])
    baseline=json.loads((OUT/'before.json').read_text())
    if baseline['mismatches']:raise RuntimeError('Preservation mismatches')
    # Trusted existing file hashes, rather than re-pinning changed sources.
    frozen={p:v['sha256'] for p,v in baseline['files'].items() if p.startswith(('src/','configs/','tools/','tests/research_d02b1/fixtures/','tests/foundation/fixtures/'))}
    verify_files(ROOT,frozen)
    pins=dict(frozen,**test['source_hashes'])
    historical='artifacts/phase2d2_n31/g2/final_real/summary.json'
    pins[historical]=baseline['files'][historical]['sha256'];verify_files(ROOT,pins)
    write_new(OUT/'driver_manifest.json',pins)
    owner=dict(pid=os.getpid(),created=psutil.Process().create_time(),cell=CELL)
    write_new(OUT/'coordinator.lock.json',owner)
    write_new(OUT/'claim.json',owner)
    ram=psutil.virtual_memory().available;disk=shutil.disk_usage(ROOT).free
    write_new(OUT/'resource_preflight.json',dict(available_ram=ram,free_disk=disk,owner=owner,minimum_ram=5*1024**3,minimum_disk=8*1024**3))
    if not resource_gate(ram,disk):
        write_new(OUT/'execution_record.json',dict(cell=CELL,**status(resource='INSUFFICIENT',provenance='PASS'),worker_started=False))
        print('WAIT/NOT RUN: resource gate failed; single attempt stopped',flush=True);return
    command=[sys.executable,str(ROOT/'tools/d02b7_worker.py'),CELL,str(OUT/'attempt_01'),str(OUT/'claim.json')]
    started=time.perf_counter();peak=0
    with (OUT/'worker.stdout.log').open('xb') as stdout,(OUT/'worker.stderr.log').open('xb') as stderr:
        p=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr)
        while p.poll() is None:
            try:peak=max(peak,psutil.Process(p.pid).memory_info().rss)
            except psutil.Error:pass
            time.sleep(.25)
        code=p.wait()
    path=OUT/'attempt_01/result.json'
    result=json.loads(path.read_text()) if path.exists() else dict(cell=CELL,execution='INTERRUPTED',numerical='NOT RUN',dependent_work='BLOCKED',cause='UNKNOWN')
    write_new(OUT/'execution_record.json',dict(result,worker_started=True,exit_code=code,wall_seconds=time.perf_counter()-started,sampled_peak_rss_bytes=peak,command=command))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
