"""Snapshot, verify and apply the two explicitly approved non-production patches."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_n31'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')


def apply_diff(original,diff):
    import re
    lines=original.splitlines(keepends=True);result=[];index=0;patch=diff.splitlines(keepends=True);i=2
    while i<len(patch):
        match=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',patch[i]);assert match
        start=int(match[1]);start=max(0,start-1);result.extend(lines[index:start]);index=start;i+=1
        while i<len(patch) and not patch[i].startswith('@@'):
            line=patch[i];i+=1
            if line[0] in ' -':assert lines[index]==line[1:];index+=1
            if line[0] in ' +':result.append(line[1:])
    result.extend(lines[index:]);return ''.join(result)


def main():
    OUT.mkdir(exist_ok=False)
    baseline={}
    for folder in ('src','tests','configs','docs','tools','artifacts','papers'):
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or p.is_relative_to(OUT) or '__pycache__' in p.parts:continue
            baseline[p.relative_to(ROOT).as_posix()]=sha(p)
    write(OUT/'history/baseline_manifest.json',baseline)
    shutil.copyfile(Path(r'C:\Users\dinhq\.codex\attachments\bcc5b6ae-18cd-4ba9-b651-0e61edae33f0\Pasted text.txt'),OUT/'history/user_request.txt')
    proposal=ROOT/'artifacts/phase2d2_n3/proposal';meta=json.loads((proposal/'proposal_metadata.json').read_text())
    issue=json.loads((proposal/'metadata_issue.json').read_text())
    for p,h in meta['existing_mandatory_tests_sha256'].items():assert sha(ROOT/p)==h,p
    assert sha(ROOT/issue['file'])==issue['expected_source_sha256']
    assert not (ROOT/meta['target_must_not_exist']).exists()
    records=[]
    for patch,target in (('proposed_tests.patch',meta['target_must_not_exist']),('proposed_metadata_repair.patch',issue['file'])):
        path=ROOT/target;old=path.read_text() if path.exists() else ''
        before=sha(path) if path.exists() else None
        if path.exists():shutil.copyfile(path,OUT/'history'/Path(target).name)
        text=apply_diff(old,(proposal/patch).read_text());ast.parse(text)
        if before is None:assert hashlib.sha256(text.encode()).hexdigest()==meta['new_test_source_sha256']
        path.write_text(text,encoding='utf-8');shutil.copyfile(proposal/patch,OUT/'history'/patch)
        records.append({'patch':patch,'patch_sha256':sha(proposal/patch),'target':target,'before_sha256':before,'after_sha256':sha(path),'exact_patch_applied':True})
    write(OUT/'applied_patches.json',{'approval':'N3D01 and N3D02 only','applied':records,'original_321_assertions_unchanged':True,
        'N3D03':'DESIGN DIRECTION ONLY','N3D06':'BOUNDED NO-UPDATE INVESTIGATION','N3D04_N3D05_N3D07_N3D08_N3D09_N3D10':'NOT APPROVED'})
    # This immutable registration precedes every N3.1 numerical experiment.
    write(OUT/'heldout_preregistration.json',{'version':1,'registered_UTC':datetime.now(timezone.utc).isoformat(),
        'status':'LOCKED BEFORE EXECUTION','no_threshold_selection_from_observed_maxima':True,
        'attention':{'seeds':{'weights':2101,'input':2102,'cotangent':2103,'directions':[2104,2105]},
            'shape':[1,308,512],'heads':8,'head_dim':64,'input_magnitudes':[0.02,0.2,2.0],
            'cotangent_scales':[0.0,0.0001,1.0,10000.0],'generation':'magnitude * seeded torch.randn FP32; promote same values for FP64; cotangent = scale * independent seeded randn',
            'targets':['q.weight','k.weight','k.bias','input'],'checks':['FP64 independent manual formula','FP64 central FD at scale1 for Q/K; eps1e-6 atol1e-5 rtol1e-3','exact rational key-bias cancellation','original FP32/FP64 coordinate diagnostics','conditional dot/reduction gamma_n bounds','zero and nonzero reference coordinates']},
        'LD':{'seed_source':0,'base':'accepted full root real_efs decoder inputs copied under no_grad',
            'B':2,'shape':[2,308,512],'state_memory_visual_multipliers':[[0.25,0.25,0.25],[1.0,1.0,1.0],[2.0,2.0,2.0]],
            'cotangent_seeds':[3101,3102],'cotangent_scales':[0.0001,1.0,1024.0],
            'cuts':['LD01','LD04','LD07'],'generation':'seeded FP32 randn/sqrt(numel), promoted before FP64 sum',
            'targets':['state','memory','visual','blocks.0.self_attention.q.weight','blocks.0.self_attention.k.weight','blocks.0.self_attention.k.bias','blocks.6.vim.q.weight','blocks.6.vim.k.weight'],
            'checks':['same-forward fixed VJP linearity','original unchanged allclose diagnostic','norm/cosine/directions','finite','exact connected VIM Q/K zero','promoted FP32 weights; local LD only']},
        'visual':{'image_seed':4101,'image_shape':[1,3,224,224],'generation':'0.1+0.8*seeded FP32 uniform',
            'cotangent_seeds':[4102,4103],'cotangent_scales':[0.0001,1.0,1024.0],'cuts':['fused_visual'],
            'targets':['first_shared_CNN_weight','shared_visual_Q_weight','AD_weight','Ipre'],
            'backends':['mkldnn','native'],'checks':['same-forward fixed VJP linearity','finite','identities/state preserved','public visual FP32 only']},
        'fresh_existing_controls':{'same_forward':['real_mkldnn','real_efs_mkldnn','local_duplicate_mkldnn','real_efs_native'],
            'CMC':['float32','float64'],'N2_fixtures':[0,1,2],'N2_dtypes':['float32','float64']},
        'complete_G2':{'cases':['real','efs','local','real_efs','real_local','local_duplicate'],
            'backend':'mkldnn','direction_seeds':[5101,5102,5103,5104],
            'coverage':'all 508 parameter tensors; original six isolated then joint fresh FP32 graphs; FP64 disk sum; all module sufficient statistics',
            'extra_backend_case':'real_efs_native'},
        'numerical_policies':{'FP32':{'atol':1e-6,'rtol':1e-5},'FP64':{'atol':1e-10,'rtol':1e-8},
            'new_acceptance_thresholds':None,'N1_N2':'unchanged original scope'},
        'stop':'Any nonfinite or unexpected identity/source/presence failure stops dependent experiments; finite tolerance failures are retained diagnostics; Phase2D2 BLOCKED'})
    write(OUT/'preregistration_lock.json',{'sha256':sha(OUT/'heldout_preregistration.json'),'file':'heldout_preregistration.json'})
    print('Exact approved patches applied; held-out protocol registered before experiments')


if __name__=='__main__':main()
