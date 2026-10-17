"""Package recorded Phase 2B evidence; never fabricates or reruns tests."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import yaml
from mfvlr.decoder_config import load_decoder_config


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_json(path):return json.loads(path.read_text(encoding='utf-8'))
def write_json(path,data):path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def main():
    destination=ROOT/'artifacts/phase2b'
    prior=read_json(ROOT/'artifacts/phase2a/n1_review/manifest.json')
    preserved={name:value for name,value in prior['files_sha256'].items() if name.startswith(('src/','tests/','configs/','docs/'))}
    for name,value in preserved.items():assert digest(ROOT/name)==value,name
    for path in (destination/'logs').glob('*.log'):
        content=path.read_bytes()
        if content.startswith((b'\xff\xfe',b'\xfe\xff')):path.write_bytes(content.decode('utf-16').encode('utf-8'))
    suite=ET.parse(destination/'05_complete_suite.xml').getroot().find('testsuite')
    assert {k:int(suite.attrib[k]) for k in ('tests','failures','errors','skipped')}=={'tests':40,'failures':0,'errors':0,'skipped':0}
    cases=[]
    for case in suite.findall('testcase'):
        assert all(case.find(k) is None for k in ('failure','error','skipped'))
        cases.append({'name':case.attrib['name'],'status':'PASS','seconds':float(case.attrib['time'])})
    for prefix,limit,double in (('test_mve_t',14,{1}),('test_vd_t',14,{1}),('test_vd_i',8,{1,3})):
        for index in range(1,limit+1):
            names=[c['name'] for c in cases if c['name'].startswith(f'{prefix}{index:02}_')]
            assert len(names)==(2 if index in double else 1),(prefix,index,names)
            if index in double:assert any(n.endswith('[1]') for n in names) and any(n.endswith('[2]') for n in names)
    regression=ET.parse(destination/'04_mve_regression.xml').getroot().find('testsuite')
    assert regression.attrib['tests']=='15' and regression.attrib['failures']==regression.attrib['errors']==regression.attrib['skipped']=='0'

    diagnostics={path.parent.name:read_json(path) for path in sorted((destination/'diagnostics').glob('*/summary.json'))}
    assert len(diagnostics)==8
    for name,record in diagnostics.items():
        assert record['vd_parameters']['total_unique']==9055365 and record['optimizer_updates']==0
        if record['component']=='integrated':assert record['mve_parameters']['total_unique']==32078912
        if record['backward']:
            for check in record['checks']:
                assert check['finite_vd_parameter_gradient_tensors']==28
                if record['component']=='integrated':assert check['finite_mve_parameter_gradient_tensors']==88
    measured=diagnostics['integrated_b2_backward']
    write_json(destination/'parameter_report.json',{
        'source':'actual production construction, diagnostic summaries and VD-T05 passed final suite',
        'vd':measured['vd_parameters'],'mve':measured['mve_parameters'],
        'mve_vd_total_unique':41134277,'vd_parameter_tensors':28,'combined_parameter_tensors':116,
        'matches_approved_option_b':True,'establishes_author_fidelity':False})
    write_json(destination/'memory_runtime.json',{'method':'fresh process per case; psutil 10ms RSS polling plus Windows peak_wset',
        'units':'bytes and seconds; MiB divisor 2^20', 'cuda_measurements':'NOT RUN: installed runtime has no CUDA support',
        'records':diagnostics})
    text=(destination/'logs/05_complete_suite.log').read_text(encoding='utf-8')
    probes={}
    for batch,norms in re.findall(r'VD-I03 b=(\d): reconstruction=.*?norms=(\[[^\]]+\])',text):
        assert f'VD-I03 b={batch}: detached decoder/local gradients=None; RE/direct-I gradients remain nonzero' in text
        values=json.loads(norms);assert all(value>0 for value in values)
        probes[batch]={'norms':dict(zip(('reconstruction','appearance_local','ad_head_weight','U1_up_weight','encoder_first_conv_weight'),values)),
            'detached_decoder_local_gradients':None,'detached_RE_direct_image_gradients_nonzero':True,
            'distinct_mask_head_gradient':None,'case_status':'PASS'}
    assert set(probes)=={'1','2'}
    write_json(destination/'gradient_flow.json',{'source_log':'artifacts/phase2b/logs/05_complete_suite.log',
        'probe':'mean(residual_global ** 2), not a paper loss', 'cases':probes,
        'separate_head_dependency_tests':['VD-I04','VD-I05'],'abs_derivative_test':'VD-I06'})

    resolved_dir=destination/'resolved_configs';resolved_dir.mkdir(exist_ok=True)
    chains={}
    for profile in ('reference','debug'):
        config=load_decoder_config(ROOT/f'configs/decoder_{profile}.yaml');config.require_approval()
        data=config.data
        data['mve_config']=os.path.relpath(config.mve_path,resolved_dir).replace('\\','/')
        export=resolved_dir/f'decoder_{profile}.yaml'
        export.write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
        restored=load_decoder_config(export);restored.require_approval()
        assert restored.proposal==config.proposal
        chains[profile]={'source':f'configs/decoder_{profile}.yaml','review_chain':config.review_chain}
    write_json(resolved_dir/'review_chains.json',chains)
    environment=measured['environment']|{'pytest':importlib.metadata.version('pytest'),'pyyaml':importlib.metadata.version('PyYAML')}
    write_json(destination/'environment.json',environment)

    manifest_path=destination/'manifest.json';archive=destination/'phase2b_review_package.zip'
    files=[ROOT/'README.md',ROOT/'pyproject.toml',ROOT/'.gitignore']
    for folder in ('src/mfvlr','tests','configs','docs','tools','papers','artifacts/phase2a','artifacts/phase2b'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p!=manifest_path and p.suffix not in ('.zip','.sha256','.pyc'))
    files=sorted(set(files),key=lambda p:p.relative_to(ROOT).as_posix())
    manifest={
        'phase':'2B','client_date':'2026-10-11','approved_decisions':[f'D{i:02}' for i in range(1,9)],
        'selected_option':'B','skip_policy':'bottleneck_only','option_a_status':'PROPOSED_not_implemented',
        'classification':'ASSUMED','author_architecture_confirmed':False,'training_authorized':False,'evaluation_authorized':False,
        'acceptance_satisfied':True,'final_suite':{'cases':40,'ids':36,'passed':40,'failed':0,'errors':0,'skipped':0,'expected_failures':0,
            'junit_seconds':float(suite.attrib['time']),'host_timestamp':suite.attrib['timestamp'],'outcomes':cases},
        'independent_mve_regression':{'cases':15,'passed':15,'source':'artifacts/phase2b/04_mve_regression.xml'},
        'vd_parameters':9055365,'mve_parameters':32078912,'combined_unique_parameters':41134277,
        'accepted_phase2a_files_unchanged':preserved,'N1_used_by_decoder':False,
        'cpu_resource_cases':8,'gpu_measurements':'NOT RUN: CUDA unavailable','environment':environment,
        'files_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in files},
        'manifest_and_zip_self_hashes_excluded':True,
    }
    write_json(manifest_path,manifest)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as bundle:
        for p in sorted(files+[manifest_path],key=lambda p:p.relative_to(ROOT).as_posix()):
            info=zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            bundle.writestr(info,p.read_bytes(),compresslevel=9)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for name,value in manifest['files_sha256'].items():assert hashlib.sha256(bundle.read(name)).hexdigest()==value
    archive.with_suffix('.zip.sha256').write_text(f'{digest(archive)}  {archive.name}\n',encoding='utf-8')
    print('Verified 40 cases / 36 IDs, unchanged accepted files, VD=9,055,365, MVE+VD=41,134,277.')
    print('Measurements: 8 fresh CPU processes, CUDA NOT RUN; resolved profiles reload consistently.')
    print(f'Archive: {archive}; entries={len(files)+1}; bytes={archive.stat().st_size}; SHA-256={digest(archive)}')


if __name__=='__main__':main()
