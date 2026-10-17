"""Exclusive sequential coordinator for the exact six user-authorized F1 cells."""
import json,os,sys,time,subprocess,shutil
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mfvlr_research.d02b4.policy import CELLS,write,hashfile,resources
OUT=ROOT/'artifacts/phase2d2_d02b4'
def main():
    if not (OUT/'preflight_tests.json').is_file() or json.loads((OUT/'preflight_tests.json').read_text())['status']!='PASS': raise RuntimeError('Independent isolated preflight tests required')
    if json.loads((OUT/'before.json').read_text())['mismatches']: raise RuntimeError('Preservation mismatch')
    owner=dict(pid=os.getpid(),created=psutil.Process().create_time())
    write(OUT/'coordinator.lock.json',owner)
    paths=[ROOT/'tools/d02b4_worker.py',ROOT/'tools/d02b4_run.py']
    paths.extend((ROOT/'src/mfvlr_research/d02b4').glob('*.py'))
    from mfvlr_research.d02b2.contract import bundle
    pins=dict(bundle()['sources'])
    for folder in ('src/mfvlr_research/d02b1','src/mfvlr_research/d02b2','src/mfvlr_research/d02b3'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    pins.update({p.relative_to(ROOT).as_posix():hashfile(p) for p in paths})
    write(OUT/'driver_manifest.json',pins)
    matrix=[]
    for i,cell in enumerate(CELLS):
        attempt=OUT/'cells'/cell/'attempt_01'
        gate=0
        while True:
            ram=psutil.virtual_memory().available;disk=shutil.disk_usage(ROOT).free
            good=resources(ram,disk)
            write(OUT/'resources'/f'{i:02d}_{gate:04d}.json',dict(cell=cell,available_ram=ram,free_disk=disk,gate='PASS' if good else 'WAIT/NOT RUN',owner=owner,time=time.time()))
            if good: break
            print(json.dumps(dict(cell=cell,status='WAIT/NOT RUN',available_ram=ram)),flush=True)
            gate+=1
            time.sleep(10)
        claim=OUT/f'claim_{i:02d}.json'; write(claim,dict(owner,active_cell=cell))
        log=OUT/'logs';log.mkdir(exist_ok=True)
        command=[sys.executable,str(ROOT/'tools/d02b4_worker.py'),cell,str(attempt),str(claim)]
        start=time.perf_counter();peak=0
        with (log/f'{cell}.stdout.log').open('xb') as stdout,(log/f'{cell}.stderr.log').open('xb') as stderr:
            p=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr)
            while p.poll() is None:
                try: peak=max(peak,psutil.Process(p.pid).memory_info().rss)
                except psutil.Error: pass
                time.sleep(.25)
            code=p.wait()
        result=json.loads((attempt/'result.json').read_text()) if (attempt/'result.json').exists() else {'execution':'INTERRUPTED','cause':'UNKNOWN'}
        row=dict(cell=cell,exit_code=code,result=result,wall_seconds=time.perf_counter()-start,sampled_peak_rss_bytes=peak,command=command)
        matrix.append(row);write(OUT/f'progress_{i:02d}.json',matrix)
        print(json.dumps(dict(cell=cell,exit_code=code,execution=result['execution'],numerical=result.get('numerical'),peak_rss=peak)),flush=True)
        if code or result['execution']!='COMPLETE':
            matrix.extend(dict(cell=c,result={'execution':'NOT RUN','reason':'dependent work stopped'}) for c in CELLS[i+1:])
            break
    write(OUT/'execution_matrix.json',matrix)
if __name__=='__main__': main()
