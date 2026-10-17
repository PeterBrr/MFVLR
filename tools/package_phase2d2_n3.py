"""Hash-preserve the pre-N3 workspace and build the diagnostic review package."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_n3'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    baseline=read(OUT/'history/baseline_manifest.json')
    changes=[p for p,h in baseline.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    assert not changes,changes
    cases=ET.parse(OUT/'junit/proposed_controls_dryrun.xml').getroot().findall('.//testcase')
    assert len(cases)==7 and all(c.find('failure') is None and c.find('error') is None and c.find('skipped') is None for c in cases)
    assert not (ROOT/'tests/joint/test_n3_controls.py').exists()
    write(OUT/'preservation.json',{'status':'PASS','pre_existing_files':len(baseline),'all_SHA256_match':True,
        'changed_pre_existing_files':changes,'production_test_config_or_historical_artifact_changes':False,
        'accepted_276_cases_untouched':True,'full_suite_this_review':'NOT RUN; actual historical 313/321 remains BLOCKED',
        'isolated_proposed_control_dryrun':{'cases':7,'passed':7,'failures':0,'skips':0},
        'unapplied_patches':True,'no_optimizer_or_parameter_updates':True})
    for name in ('phase2d2_n3_numerical_review.md','phase2d2_n3_proposal.md'):assert (ROOT/'docs'/name).is_file()
    excluded={OUT/'manifest.json',OUT/'package_verification.json',OUT/'phase2d2_n3_review.zip',OUT/'phase2d2_n3_review.zip.sha256'}
    paths=[];omitted=[]
    for folder in ('src','tests','configs','docs','tools','artifacts','papers'):
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or any(part in ('__pycache__','.pytest_cache') for part in p.parts):continue
            if p in excluded:continue
            if p.suffix=='.zip':omitted.append(str(p.relative_to(ROOT)).replace('\\','/'));continue
            if any(part.startswith('selected_gradients_') for part in p.parts):raise RuntimeError('Temporary gradient vectors still live')
            paths.append(p)
    paths.extend(ROOT/n for n in ('README.md','pyproject.toml','.gitignore'))
    paths=sorted(set(paths))
    files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}
    manifest={'status':'N3_DIAGNOSTIC_REVIEW_PHASE_BLOCKED','files_sha256':files,
        'historical_ZIPs_preserved_on_disk_but_omitted_from_new_package':omitted,
        'scope':'Full source/tests/configs/tokenizer assets, original paper and unpacked old/new evidence. Environment and Torch wheels not bundled.',
        'self_reference_policy':'Manifest is bundled but does not hash itself. Package hash and verification are external.'}
    write(OUT/'manifest.json',manifest)
    archive=OUT/'phase2d2_n3_review.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths+[OUT/'manifest.json']:
            info=zipfile.ZipInfo(str(p.relative_to(ROOT)).replace('\\','/'),date_time=(2026,10,11,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            with p.open('rb') as source,z.open(info,'w') as destination:
                for block in iter(lambda:source.read(1024*1024),b''):destination.write(block)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p,h in files.items():
            digest=hashlib.sha256()
            with z.open(p) as f:
                for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
            assert digest.hexdigest()==h,p
        entries=len(z.namelist())
    digest=sha(archive)
    (OUT/'phase2d2_n3_review.zip.sha256').write_text(digest+'  phase2d2_n3_review.zip\n')
    write(OUT/'package_verification.json',{'status':'PASS','archive_sha256':digest,'bytes':archive.stat().st_size,
        'entries':entries,'CRC_all_passed':True,'all_payload_SHA256_match':True,'baseline_pre_existing_files_unchanged':len(baseline),
        'Phase2D2':'BLOCKED','N3_policy':'NOT APPLIED'})
    print(json.dumps(read(OUT/'package_verification.json')))


if __name__=='__main__':main()
