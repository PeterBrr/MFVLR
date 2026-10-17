"""Execute only isolated D02B1 array/mock tests; record the exact invocation."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mfvlr_research.d02b1.runner import coverage
from mfvlr_research.d02b1.schema import write_new, file_hash
from mfvlr_research.d02b1.stages import stage_shapes

def main():
    out=ROOT/'artifacts/phase2d2_d02b1'
    run=out/'final_01'
    run.mkdir(exist_ok=False)
    command=[sys.executable,'-m','pytest','--confcutdir=tests/research_d02b1','tests/research_d02b1',
             '-v','-ra',f'--junitxml={run / "junit.xml"}']
    env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    before={p.relative_to(ROOT).as_posix():file_hash(p) for folder in ('src/mfvlr_research/d02b1','tests/research_d02b1')
            for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    start=time.perf_counter()
    peak=0
    try:
        import psutil
    except ImportError:
        psutil=None
    with (run/'stdout.log').open('wb') as stdout, (run/'stderr.log').open('wb') as stderr:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
        while proc.poll() is None:
            if psutil:
                try: peak=max(peak,psutil.Process(proc.pid).memory_info().rss)
                except psutil.Error: pass
            time.sleep(.1)
        code=proc.wait()
    elapsed=time.perf_counter()-start
    tree=ET.parse(run/'junit.xml')
    cases=tree.findall('.//testcase')
    counts=dict(cases=len(cases),failures=len(tree.findall('.//failure')),errors=len(tree.findall('.//error')),skipped=len(tree.findall('.//skipped')))
    changed=[p for p,h in before.items() if file_hash(ROOT/p)!=h]
    result=dict(scope='D02B1 array/mock verification only',command=command,exit_code=code,
                elapsed_seconds=elapsed,sampled_pytest_peak_rss_bytes=peak or None,
                memory_scope='isolated pytest process, not MFVLR capacity',collection=counts,
                environment={'python':platform.python_version(),'platform':platform.platform(),
                             'numpy':importlib.metadata.version('numpy'),'pytest':importlib.metadata.version('pytest'),
                             'torch_imported_by_harness':'torch' in sys.modules,'external_plugin_autoload':False},
                tested_source_hashes=before,source_changes_during_tests=changed,
                status='PASS' if code==0 and not changed and counts['cases']>0 and not any(counts[k] for k in ('failures','errors','skipped')) else 'FAIL',
                NOT_RUN=['G01-G06','C01-C07','N01-N06','N2','N3','local sweeps','real full-graph forward/backward','MVE-T10','VD-T12','optimizer','training'],
                phase2d2='BLOCKED',B0_D03='NOT AUTHORIZED',B0_D08='DISABLED')
    write_new(out/'verification.json',result)
    write_new(out/'coverage.json',coverage())
    write_new(out/'stage_schema.json',{'B1':stage_shapes(1),'B2':stage_shapes(2),'observation_status':'NOT RUN'})
    write_new(out/'approval_record.json',{'authority':'Current user D02B1 instruction','approved':'B0-D02 implementation plus isolated mock/unit verification only',
              'execution_approved_partitions':[],'dispatch':'unconditionally disabled','finite_failure_continuation':'disabled',
              'old_MVE_T10_exception':'not applicable','VD_T12_incident':'preserved; not retroactively approved','phase2d2':'BLOCKED'})
    print(json.dumps({k:v for k,v in result.items() if k not in ('tested_source_hashes','command')},indent=2))
    if result['status']!='PASS': raise SystemExit(1)

if __name__=='__main__': main()
