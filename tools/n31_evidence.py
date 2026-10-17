"""Fresh-workflow provenance gate. Integrity is not numerical acceptance."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,uuid
ROOT=Path(__file__).resolve().parents[1]
RUNS=('real_mkldnn','real_efs_mkldnn','local_duplicate_mkldnn','real_efs_native')
CASES=('real','efs','local','real_efs','real_local','local_duplicate')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def sources():
    paths=[]
    for folder in ('src','configs','tests'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend(p for p in (ROOT/'tools').glob('*.py') if p.name in (
        'n31_evidence.py','n31_full_gradients.py','review_phase2d2_n31_heldout.py','run_phase2d2_n31.py',
        'review_phase2d2_n3.py','review_phase2d2_n3_local.py','n3_gradient_tools.py','phase2d2_diagnostics.py','phase2d2_fixtures.py','retry_phase2d2.py','review_phase2c2_n2.py'))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}


def required():
    paths=['heldout_preregistration.json','preregistration_lock.json','local_analytical.json','stream_selftest.json',
        'heldout_attention.json','heldout_LD.json','heldout_visual_mkldnn.json','heldout_visual_native.json',
        'chain/float32.json','chain/float64.json','resource_availability.json','baseline_manifest.json']
    for run in RUNS:
        paths.extend(f'same_forward/{run}/{name}.json' for name in ('same_graph','separate_graphs','summary','resources'))
    for case in (*CASES,'real_efs_native'):
        paths.extend(f'g2/final_{case}/{name}.json' for name in ('summary','gradient_additivity','complete_discrepancies','fd','lr','cmc','fl','ar','kl','total','resources','ownership'))
    paths.append('g2/real/gradient_additivity.json')
    paths.extend(f'n2/fixture{i}_{dtype}_threads4.json' for i in range(3) for dtype in ('float32','float64'))
    return paths


def begin(out):
    assert not (out/'workflow_start.json').exists()
    reg=read(out/'heldout_preregistration.json')
    # Primary refresh began after registration, before this stage's orchestrator.
    write(out/'workflow_start.json',{'run_id':str(uuid.uuid4()),'registered_UTC':reg['registered_UTC'],
        'source_snapshot_UTC':datetime.now(timezone.utc).isoformat(),'sources':sources(),
        'preregistration_sha256':sha(out/'heldout_preregistration.json'),'root':str(ROOT),
        'required_records':required(),'purpose':'provenance and completeness only; no numerical waiver'})


def seal(out):
    start=read(out/'workflow_start.json');assert start['sources']==sources(),'source/config/test drift during workflow'
    preregtime=datetime.fromisoformat(start['registered_UTC']).timestamp()
    hashes={}
    for p in required():
        path=out/p;assert path.is_file(),p
        if p not in ('heldout_preregistration.json','preregistration_lock.json','baseline_manifest.json'):
            assert path.stat().st_mtime>=preregtime,p
        hashes[p]=sha(path)
    logs={p.relative_to(out).as_posix():sha(p) for p in (out/'logs').glob('*') if p.is_file()}
    write(out/'evidence_seal.json',{'run_id':start['run_id'],'sources':start['sources'],'records':hashes,'logs':logs,
        'sealed_UTC':datetime.now(timezone.utc).isoformat(),'purpose':'integrity/completeness, NOT numerical acceptance'})
    validate(out,start['run_id'])


def validate(out,run_id):
    start=read(out/'workflow_start.json');seal=read(out/'evidence_seal.json')
    assert run_id and run_id==start['run_id']==seal['run_id'],'stale/wrong workflow nonce'
    assert start['sources']==seal['sources']==sources(),'source/configuration/test hash mismatch'
    assert set(seal['records'])==set(required()),'incomplete evidence inventory'
    for p,h in seal['records'].items():assert (out/p).is_file() and sha(out/p)==h,'missing/altered record '+p
    for p,h in seal['logs'].items():assert sha(out/p)==h,'altered log '+p
    assert sha(out/'heldout_preregistration.json')==start['preregistration_sha256']==read(out/'preregistration_lock.json')['sha256']
    from tools.phase2d2_fixtures import fixture
    root_hash=None
    for run in RUNS:
        r=read(out/'same_forward'/run/'summary.json');s=read(out/'same_forward'/run/'same_graph.json')
        case=run.removesuffix('_mkldnn').removesuffix('_native')
        assert r['fixture']==fixture(case).provenance,'input hash mismatch'
        root_hash=root_hash or r['initial_state_sha256'];assert r['initial_state_sha256']==root_hash
        assert r['parameters_and_identities_unchanged'] and r['ownership']['total_unique']==154156232
        ids=list(s['graph']['loss_grad_fn_python_ids'].values());assert len(ids)==len(set(ids))==7,'metadata repair absent'
    for case in (*CASES,'real_efs_native'):
        directory=out/'g2'/('final_'+case);r=read(directory/'summary.json');full=read(directory/'complete_discrepancies.json')
        assert r['fixture']==fixture(case.removesuffix('_native')).provenance
        assert r['initialized_state_sha256']==root_hash and r['parameter_bytes_unchanged'] and r['parameter_identities_unchanged']
        assert r['zero_optimizer_updates'] and r['all_valid_values_and_gradients_finite']
        assert len(full['parameters'])==508 and full['unique_elements']==154156232
        assert full['legacy_gate_unchanged'] and full['elementwise_policy']=={'atol':1e-6,'rtol':1e-5}
    a,b=(read(out/'chain'/f'{d}.json') for d in ('float32','float64'))
    assert a['FP32_initialized_FLT_source_sha256']==b['FP32_initialized_FLT_source_sha256']
    assert a['FP32_visual_input_sha256']==b['FP32_visual_input_sha256']
    assert a['fixture']==b['fixture']==fixture('real_efs').provenance
    return {'provenance':'PASS','numerical_acceptance':'NOT GRANTED','run_id':run_id}


def pytest_sessionstart(session):
    out=Path(os.environ['MFVLR_N3_EVIDENCE_ROOT']);run_id=os.environ['MFVLR_N31_RUN_ID']
    result=validate(out,run_id)
    print('N31_FRESH_EVIDENCE_GATE:',json.dumps(result),flush=True)


def negative_tests(out):
    # Never mutate actual records: validator against a copied small manifest, with source hash inputs patched only in this process.
    import tempfile,shutil
    tests={};run_id=read(out/'workflow_start.json')['run_id']
    validate(out,run_id)
    try:validate(out,'old-run-id')
    except AssertionError:tests['stale_nonce']='REJECTED'
    else:raise AssertionError('stale nonce accepted')
    with tempfile.TemporaryDirectory(dir=out,prefix='negative_provenance_') as temp:
        p=Path(temp)
        for n in ('workflow_start.json','evidence_seal.json'):shutil.copyfile(out/n,p/n)
        for name in ('missing','incomplete','wrong_source','altered_record'):
            seal=read(out/'evidence_seal.json')
            if name=='incomplete':seal['records'].pop(next(iter(seal['records'])))
            if name=='wrong_source':seal['sources'][next(iter(seal['sources']))]='0'*64
            if name=='altered_record':
                first=next(iter(seal['records']));(p/first).write_text('{}')
            write(p/'evidence_seal.json',seal)
            try:validate(p,run_id)
            except (AssertionError,FileNotFoundError):tests[name]='REJECTED'
            else:raise AssertionError(name+' accepted')
    write(out/'provenance_negative_tests.json',{'status':'PASS','checks':tests,'numerical_proof':False})
