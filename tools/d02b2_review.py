"""Static/mock verification and packaging only. No numerical experiment dispatch."""
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
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02b2'
sys.path.insert(0,str(ROOT/'src'))
from mfvlr_research.d02b2.contract import bundle,verify_sources,schedule,file_hash

def write(path,value):
    with path.open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)

def verify():
    folder=OUT/'final_01'; folder.mkdir(exist_ok=False)
    command=[sys.executable,'-m','pytest','--confcutdir=tests/research_d02b2','tests/research_d02b2','-v',
             f'--junitxml={folder/"junit.xml"}']
    paths=[p for prefix in ('src/mfvlr_research/d02b2','tests/research_d02b2') for p in (ROOT/prefix).rglob('*')
           if p.is_file() and '__pycache__' not in p.parts]
    before={p.relative_to(ROOT).as_posix():file_hash(p) for p in paths}
    start=time.perf_counter()
    with (folder/'stdout.log').open('wb') as stdout,(folder/'stderr.log').open('wb') as stderr:
        done=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'),stdout=stdout,stderr=stderr)
    elapsed=time.perf_counter()-start
    tree=ET.parse(folder/'junit.xml'); cases=tree.findall('.//testcase')
    failures=len(tree.findall('.//failure')); errors=len(tree.findall('.//error')); skipped=len(tree.findall('.//skipped'))
    changed=[p for p,h in before.items() if file_hash(ROOT/p)!=h]
    only_allowed=all('research_d02b2' in c.attrib['classname'] for c in cases)
    data={'scope':'D02B2 static/mock verification only','command':command,'exit_code':done.returncode,
          'elapsed_seconds':elapsed,'cases':len(cases),'failures':failures,'errors':errors,'skips':skipped,
          'status':'PASS' if done.returncode==0 and cases and not (failures or errors or skipped or changed) and only_allowed else 'FAIL',
          'environment':{'python':platform.python_version(),'platform':platform.platform(),'numpy':importlib.metadata.version('numpy'),
                         'pytest':importlib.metadata.version('pytest'),'torch_imported': 'torch' in sys.modules},
          'source_hashes':before,'source_changes_during_tests':changed,'only_isolated_tests_collected':only_allowed,
          'selected_tests':[c.attrib['classname']+'::'+c.attrib['name'] for c in cases],
          'predecessor_D02B1':'conditional document review; independent source/ZIP audit outstanding',
          'N2_execution':'NOT RUN','B0_D03':'DISABLED','B0_D08':'DISABLED','phase2d2':'BLOCKED'}
    write(OUT/'verification.json',data)
    write(OUT/'source_verification.json',verify_sources(ROOT))
    write(OUT/'matrix.json',{'scope':'Prepared identities only; not an execution approval','cells':schedule()})
    write(OUT/'approval.json',{'approved':'D02B2 isolated N2 preparation and static/mock tests only','real_N2':'NOT AUTHORIZED / NOT RUN',
           'MVE_T10':'NOT COLLECTED / NOT RUN','VD_T12':'NOT COLLECTED / NOT RUN','G01_G06':'NOT RUN','NG2':'NOT RUN',
           'optimizer':'NOT RUN','training':'NOT RUN','B0_D03':'DISABLED','B0_D08':'DISABLED','phase2d2':'BLOCKED'})
    print(json.dumps({k:v for k,v in data.items() if k not in ('source_hashes','selected_tests','command')},indent=2))
    if data['status']!='PASS': raise SystemExit(1)

def package():
    if json.loads((OUT/'verification.json').read_text())['status']!='PASS' or json.loads((OUT/'after.json').read_text())['status']!='PASS':
        raise RuntimeError('Verification/preservation gate failed')
    paths=set(ROOT/p for p in bundle()['sources'])
    for prefix in ('src','tests/research_d02b2','artifacts/phase2d2_d02b2'):
        paths.update(p for p in (ROOT/prefix).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix!='.zip' and p.name not in ('manifest.json','package_verification.json'))
    paths.update((ROOT/'tools').glob('d02b2_*.py'))
    paths.update((ROOT/'docs').glob('phase2d2_d02b2_*.md'))
    for rel in ('pyproject.toml','docs/phase2c2_n2_acceptance_report.md','artifacts/phase2d2_d02b1/package_verification.json',
                'artifacts/phase2d2_d02b1/manifest.json','docs/phase2d2_d02b1_runner_implementation.md',
                'docs/phase2d2_d02b1_execution_safety.md'):
        paths.add(ROOT/rel)
    files={p.relative_to(ROOT).as_posix():{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in sorted(paths)}
    manifest={'scope':'D02B2 preparation only, no real N2 execution','files':files,
              'historical_evidence':'Pinned original sources and N2 records included; prior large archives remain external with hashes in before.json',
              'self_reference':'Manifest excludes itself; manifest and ZIP hashes recorded externally'}
    write(OUT/'manifest.json',manifest)
    archive=OUT/'phase2d2_d02b2_review.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files: z.write(ROOT/p,p)
        z.write(OUT/'manifest.json','manifest.json')
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(files)|{'manifest.json'}: raise RuntimeError('ZIP membership')
        bad=z.testzip()
        mismatches=[p for p,r in files.items() if hashlib.sha256(z.read(p)).hexdigest()!=r['sha256'] or len(z.read(p))!=r['bytes']]
        if z.read('manifest.json')!=(OUT/'manifest.json').read_bytes(): raise RuntimeError('Manifest bytes')
    if bad or mismatches: raise RuntimeError((bad,mismatches))
    result={'status':'PASS','zip_crc':'PASS','payload_hashes_verified':len(files),'mismatches':mismatches,
            'archive':archive.relative_to(ROOT).as_posix(),'archive_bytes':archive.stat().st_size,'archive_sha256':file_hash(archive),
            'manifest_sha256':file_hash(OUT/'manifest.json'),'phase2d2':'BLOCKED','real_N2':'NOT RUN'}
    write(OUT/'package_verification.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('verify','package'): raise SystemExit('Only verify (isolated mock tests) or package is allowed')
    {'verify':verify,'package':package}[sys.argv[1]]()
