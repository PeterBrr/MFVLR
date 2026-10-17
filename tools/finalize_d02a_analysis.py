"""Run read-only final audits and aggregation, recording actual subprocess logs."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02a'


def main():
    destination=OUT/'finalization_logs';destination.mkdir(exist_ok=False)
    stages=[
        ('identity_audit',['tools/verify_d02a_evidence.py','--out','artifacts/phase2d2_d02a/verification_final']),
        ('aggregation',['tools/summarize_d02a.py','--out','artifacts/phase2d2_d02a/analysis']),
        ('report',['tools/report_d02a.py','--analysis','artifacts/phase2d2_d02a/analysis','--report','docs/phase2d2_d02a_numerical_pilot_report.md']),
    ]
    for label,args in stages:
        command=[sys.executable,*args];start=time.perf_counter();utc=datetime.now(timezone.utc).isoformat()
        with (destination/f'{label}.stdout.log').open('x') as a,(destination/f'{label}.stderr.log').open('x') as b:
            result=subprocess.run(command,cwd=ROOT,stdout=a,stderr=b)
        record={'command':command,'started_utc':utc,'seconds':time.perf_counter()-start,'exit_code':result.returncode}
        with (destination/f'{label}.execution.json').open('x') as h:json.dump(record,h,indent=2)
        print(json.dumps({'stage':label,**record}),flush=True)
        if result.returncode:raise SystemExit('Read-only finalization stopped; inspect preserved evidence')


if __name__=='__main__':main()
