"""Package completed N3.1 evidence with deterministic ZIP entry metadata."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/phase2d2_n31'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')


def main():
    for n in ('numerical_budget','heldout_validation','acceptance_proposal'):assert (ROOT/f'docs/phase2d2_n31_{n}.md').is_file()
    verify=json.loads((OUT/'verification.json').read_text());assert verify['Phase2D2']=='BLOCKED'
    preservation=json.loads((OUT/'preservation.json').read_text());assert preservation['status']=='PASS'
    excluded={OUT/n for n in ('manifest.json','package_verification.json','phase2d2_n31_review.zip','phase2d2_n31_review.zip.sha256')}
    paths=[];omitted=[]
    for folder in ('src','tests','configs','docs','tools','artifacts','papers'):
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or any(part in ('__pycache__','.pytest_cache') for part in p.parts) or p in excluded:continue
            if p.suffix=='.zip':omitted.append(p.relative_to(ROOT).as_posix());continue
            if any(part.startswith('selected_gradients_') for part in p.parts):raise RuntimeError('live temporary gradients')
            paths.append(p)
    paths.extend(ROOT/n for n in ('README.md','pyproject.toml','.gitignore'));paths=sorted(set(paths))
    files={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    write(OUT/'manifest.json',{'phase':'N3.1 diagnosis/design; Phase2D2 BLOCKED','files_sha256':files,
        'omitted_nested_historical_ZIPs_preserved_on_disk':omitted,
        'integrity_scope':'All packaged actual source/tests/configs/paper/assets/evidence. Existing environment and Torch wheels not bundled.',
        'self_reference':'Manifest does not hash itself; ZIP SHA and CRC verification external'})
    archive=OUT/'phase2d2_n31_review.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths+[OUT/'manifest.json']:
            info=zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),date_time=(2026,10,11,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            with p.open('rb') as src,z.open(info,'w') as dst:
                for block in iter(lambda:src.read(1048576),b''):dst.write(block)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in files.items():
            digest=hashlib.sha256()
            with z.open(name) as f:
                for b in iter(lambda:f.read(1048576),b''):digest.update(b)
            assert digest.hexdigest()==h,name
        entries=len(z.namelist())
    digest=sha(archive);(OUT/'phase2d2_n31_review.zip.sha256').write_text(digest+'  phase2d2_n31_review.zip\n')
    write(OUT/'package_verification.json',{'status':'PASS','sha256':digest,'bytes':archive.stat().st_size,'entries':entries,
        'CRC_passed':True,'all_payload_hashes_match':True,'Phase2D2':'BLOCKED','complete_cases':verify['complete']['cases'],
        'passed':verify['complete']['passed'],'failed':verify['complete']['failed']})
    print(json.dumps(json.loads((OUT/'package_verification.json').read_text())))


if __name__=='__main__':main()
