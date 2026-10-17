"""Deterministic review ZIP preserving real failures, never an acceptance waiver."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2c2'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old=json.loads((ROOT/'artifacts/phase2c1/manifest.json').read_text())
    for name,expected in old['files_sha256'].items():
        if sha(ROOT/name)!=expected:raise RuntimeError(f'Accepted file changed: {name}')
    reports={}
    for path in sorted((OUT/'junit').glob('*.xml')):
        root=ET.parse(path).getroot()
        suite=root.find('testsuite') if root.tag=='testsuites' else root
        cases=[]
        for c in suite.findall('testcase'):
            status='FAIL' if c.find('failure') is not None else 'ERROR' if c.find('error') is not None else 'SKIP' if c.find('skipped') is not None else 'PASS'
            cases.append({'name':c.attrib['name'],'class':c.attrib.get('classname',''),'status':status})
        reports[path.stem]={**suite.attrib,'cases':cases}
    final_name=sorted(n for n in reports if n.startswith('complete_'))[-1]
    final=reports[final_name]
    statuses={s:sum(c['status']==s for c in final['cases']) for s in ('PASS','FAIL','ERROR','SKIP')}
    if int(final['tests'])!=161 or statuses['ERROR'] or statuses['SKIP']:
        raise RuntimeError('Mandatory full collection/zero-skip gate not satisfied; inspect actual logs')
    # Packaging a disclosed failure is authorized; it is not classified PASS.
    if any(c['status']=='FAIL' and c['name']!='test_flt_t17_cross_batch_original_tolerance' for c in final['cases']):
        raise RuntimeError('An additional failure needs explicit diagnosis/reporting before packaging')
    regression=reports[sorted(n for n in reports if n.startswith('regression_'))[-1]]
    assert int(regression['tests'])==119 and all(c['status']=='PASS' for c in regression['cases'])
    ids={prefix:set() for prefix in ('flt_t','flt_i','vim_t')}
    for c in final['cases']:
        for prefix in ids:
            match=re.match('test_'+prefix+r'(\d\d)_',c['name'])
            if match:ids[prefix].add(match.group(1))
    for prefix,count in (('flt_t',20),('flt_i',8),('vim_t',10)):
        assert ids[prefix]=={f'{i:02}' for i in range(1,count+1)},prefix
    summary={'phase':'2C.2','final_report':final_name,'cases':161,'passed':statuses['PASS'],
        'failed':statuses['FAIL'],'errors':0,'skipped':0,'xfail':0,'xpass':0,
        'acceptance_status':'BLOCKED' if statuses['FAIL'] else 'AWAITING REVIEW',
        'failed_cases':[c for c in final['cases'] if c['status']=='FAIL'],
        'accepted_regression_cases':119,'all_accepted_regression_passed':True,
        'flt_test_ids':20,'integration_test_ids':8,'unchanged_vim_test_ids':10,
        'new_cases':42,'new_passed':statuses['PASS']-119,'N1_extended':False,'reports':reports}
    (OUT/'test_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    paths={ROOT/name for name in old['files_sha256']}
    paths.add(ROOT/'artifacts/phase2c1/manifest.json')
    paths.add(ROOT/'docs/phase2c2_implementation_report.md')
    for name in ('language_encoder.py','language_decoder.py','flt.py','flt_config.py',
                 'flt_contract_snapshot.json','visual_language.py'):
        paths.add(ROOT/'src/mfvlr'/name)
    paths.update((ROOT/'configs').glob('flt*'))
    paths.update((ROOT/'tools').glob('*phase2c2*.py'))
    paths.update((ROOT/'tools').glob('analyze_flt*.py'))
    for directory in (ROOT/'tests/language',OUT):
        paths.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.difference_update(OUT/n for n in ('manifest.json','phase2c2_review_package.zip',
        'phase2c2_review_package.zip.sha256','package_verification.json'))
    paths=sorted(paths,key=lambda p:p.relative_to(ROOT).as_posix())
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    manifest={'phase':'2C.2','client_date':'2026-10-11','report':'docs/phase2c2_implementation_report.md',
        'approval_scope':'L01-L13 scratch FLT baseline and external integration only',
        'author_architecture_confirmed':False,'full_flt_implemented':True,'training_authorized':False,
        'acceptance_status':summary['acceptance_status'],
        'complete_suite':{k:summary[k] for k in ('cases','passed','failed','errors','skipped','xfail','xpass')},
        'failed_cases':summary['failed_cases'],'flt_unique_parameters':112758272,
        'visual_unique_parameters':41134277,'visual_flt_unique_parameters':153892549,
        'accepted_phase2c1_payloads_preserved':len(old['files_sha256']),
        'N1_extended':False,'files_sha256':hashes,'manifest_zip_self_hashes_excluded':True}
    manifest_path=OUT/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=OUT/'phase2c2_review_package.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as package:
        for path in sorted(paths+[manifest_path],key=lambda p:p.relative_to(ROOT).as_posix()):
            info=zipfile.ZipInfo(path.relative_to(ROOT).as_posix(),date_time=(2026,10,11,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3;info.external_attr=0o100644<<16
            package.writestr(info,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert len(package.namelist())==len(hashes)+1
        for name,expected in hashes.items():assert hashlib.sha256(package.read(name)).hexdigest()==expected,name
        assert package.read('artifacts/phase2c2/manifest.json')==manifest_path.read_bytes()
    checksum=sha(archive)
    (OUT/'phase2c2_review_package.zip.sha256').write_text(checksum+'  '+archive.name+'\n')
    result={'zip':archive.relative_to(ROOT).as_posix(),'sha256':checksum,'bytes':archive.stat().st_size,
        'entries':len(hashes)+1,'all_payload_hashes_verified':True,
        'accepted_files_preserved':len(old['files_sha256']),'fixed_zip_timestamps':True,
        'acceptance_status':summary['acceptance_status'],'disclosed_failed_tests':summary['failed_cases']}
    (OUT/'package_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
