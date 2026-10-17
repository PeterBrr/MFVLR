import ast
import copy
import json
from pathlib import Path
import pytest
from mfvlr_research.d02b1 import schema as s
from mfvlr_research.d02b1 import runner as r
from mfvlr_research.d02b1 import safety as safe
from mfvlr_research.d02b1.stages import stage_shapes, validate_stages

ROOT = Path(__file__).resolve().parents[2]
ROSTERS = json.loads((Path(__file__).parent/'fixtures/historical_rosters.json').read_text())

def cell(variant='A0', row='G01'):
    identity = next(c for c in s.schedule() if c['row']==row and c['variant']==variant)
    intra, inter, mkldnn = s.BACKENDS[identity['backend']]
    ownership = [dict(x, dtype='float32', sha256='a'*64, object_id=f'o{i}', storage_id=f's{i}') for i,x in enumerate(ROSTERS['rosters'][variant])]
    return dict(identity=identity, hashes=dict(source='1'*64, config='2'*64, protocol='3'*64, driver='4'*64,
                inputs={k:'5'*64 for k in ('image','tokens','mask','labels','sample_order')}, state='6'*64, inventory=s.fingerprint(ownership)),
                backend=dict(device='cpu', dtype='float32', intra=intra, inter=inter, mkldnn=mkldnn,
                    deterministic=True, matmul='highest', attention='explicit', python='3.14.3', torch='2.12.1+cpu', platform='Windows'),
                seeds=dict(mve=0, vd=0, heads=0, flt=6101+int(identity['fixture'][-1]) if row.startswith('N') else 0),
                ownership=ownership, vocabulary_tied=True, loss_order=list(s.ORDER),
                hash_algorithms={'files':'sha256-file-bytes','inputs':'sha256-contiguous-little-endian-bytes',
                                 'state':'sha256-ordered-named-parameter-name-plus-bytes','inventory':'sha256-canonical-json'})

def validate(record, seal=None):
    return s.validate_cell(record, approved_partition=record['identity']['partition'],
                           pinned=seal or s.fingerprint(record), expected_inventory=ROSTERS['rosters'][record['identity']['variant']])

def test_schedule_exact_57_and_no_execution():
    cells = s.schedule()
    assert len(cells)==len({c['experiment_id'] for c in cells})==57
    assert [sum(c['partition']==p for c in cells) for p in s.PARTITIONS]==[18,21,18]
    assert all(c['execution']=='NOT RUN' for c in r.coverage())

@pytest.mark.parametrize('variant', s.VARIANTS)
def test_historical_full_inventory_metadata_only(variant):
    record = cell(variant)
    validate(record)
    assert sum(s.math.prod(e['shape']) for e in record['ownership']) == s.COUNTS[variant][1]
    assert sum(s.category(e['name'])=='singleton_vim_qk' for e in record['ownership']) == 38
    for path, digest in ROSTERS['sources'].items():
        assert s.file_hash(ROOT/path)==digest

@pytest.mark.parametrize('row', ('G01','G06','C03','C07','N01','N06'))
def test_registered_metadata_partitions_are_not_execution_authority(row):
    validate(cell(row=row))
    with pytest.raises(r.ExecutionNotAuthorized): r.dispatch(cell(row=row), approval=True, execute=True)

@pytest.mark.parametrize('mutation', ('unknown','missing_hash','wrong_hash','fixture','backend','seed','owner','shape','count','tie','order','dtype','input_missing','partition','unknown_nested'))
def test_invalid_records_rejected(mutation):
    x=cell(); seal=s.fingerprint(x)
    if mutation=='unknown': x['execute']=True
    elif mutation=='missing_hash': del x['hashes']['protocol']
    elif mutation=='wrong_hash': x['hashes']['source']='b'*64
    elif mutation=='fixture': x['identity']['fixture']='other'
    elif mutation=='backend': x['backend']['intra']=1
    elif mutation=='seed': x['seeds']['flt']=1
    elif mutation=='owner': x['ownership'][1]['storage_id']='s0'
    elif mutation=='shape': x['ownership'][0]['shape']=[512]
    elif mutation=='count': x['ownership'].pop()
    elif mutation=='tie': x['vocabulary_tied']=False
    elif mutation=='order': x['loss_order'].reverse()
    elif mutation=='dtype': x['ownership'][0]['dtype']='float64'
    elif mutation=='input_missing': del x['hashes']['inputs']['labels']
    elif mutation=='partition': x['identity']['partition']='UNAPPROVED'
    else: x['hashes']['inputs']['bypass']=True
    with pytest.raises((s.InvalidRecord, KeyError)): validate(x,seal)

def test_partition_and_file_provenance(tmp_path):
    x=cell()
    with pytest.raises(s.InvalidRecord):
        s.validate_cell(x, approved_partition='NG2', pinned=s.fingerprint(x), expected_inventory=ROSTERS['rosters']['A0'])
    (tmp_path/'source').write_bytes(b'original')
    digest=s.file_hash(tmp_path/'source')
    s.verify_files(tmp_path, {'source':digest})
    (tmp_path/'source').write_bytes(b'mutated')
    with pytest.raises(s.InvalidRecord): s.verify_files(tmp_path, {'source':digest})

def test_connectivity_classes():
    assert s.participation('flt.decoder.blocks.0.vim.q.weight')==dict(fd='none',lr='zero',cmc='none',fl='none',ar='none',kl='none',total='zero')
    assert s.participation('flt.encoder.blocks.0.vim.k.weight')['cmc']=='zero'
    assert s.participation('flt.decoder.blocks.0.ln1.weight')['kl']=='none'
    assert s.participation('decoder.mask_head.weight')['ar']=='none'
    assert s.participation('decoder.appearance_head.weight')['fl']=='none'
    assert s.category('flt.decoder.blocks.0.memory_attention.k.bias')=='ordinary_key_bias_cancellation'
    assert s.category('flt.decoder.blocks.0.self_attention.q.weight')=='ordinary_qk_weight'

def test_exclusive_coordinator_and_cells(tmp_path):
    owner=dict(pid=11,created=20.)
    c=safe.Coordinator(tmp_path,owner,lambda p:20.)
    c.claim()
    with pytest.raises(FileExistsError): safe.Coordinator(tmp_path,owner,lambda p:20.).claim()
    x=cell()
    p=r.prepare(x,approved_partition='G2-primary',pinned=s.fingerprint(x),expected_inventory=ROSTERS['rosters']['A0'],coordinator=c)
    assert p['execution']=='NOT RUN'
    with pytest.raises(FileExistsError): c.reserve('D02B-G01-A0',s.fingerprint(x))
    c.liveness=lambda p:21.
    with pytest.raises(s.InvalidRecord): c.reserve('D02B-G02-A0',s.fingerprint(x))

@pytest.mark.parametrize('live', (20.,21.))
def test_recovery_rejects_live_and_reused_pid(tmp_path,live):
    src=tmp_path/'attempt'; src.mkdir()
    with pytest.raises(s.InvalidRecord): safe.archive_attempt(src,tmp_path/'history',owner=dict(pid=11,created=20.),liveness=lambda p:live)

def test_interrupted_archive_and_prefix(tmp_path):
    src=tmp_path/'attempt'; src.mkdir()
    (src/'partial.f32').write_bytes(b'\0\xff\x05')
    (src/'comparisons.jsonl').write_bytes(b'{"status":"FAIL"}\n{"partial":')
    files={p.name:p.read_bytes() for p in src.iterdir()}
    result=safe.archive_attempt(src,tmp_path/'history',owner=dict(pid=11,created=20.),liveness=lambda p:None)
    assert all((src/n).read_bytes()==(tmp_path/'history'/n).read_bytes()==data for n,data in files.items())
    assert len(result['files'])==2
    original=[dict(comparison_id='a',scientific=dict(status='FAIL',error=1e-12),resources=dict(seconds=2.,rss_bytes=10))]
    replay=copy.deepcopy(original); replay[0]['resources']['seconds']=3.
    assert safe.scientific_prefix(original,replay)['status']=='PASS'
    replay[0]['scientific']['error']=2e-12
    assert safe.scientific_prefix(original,replay)['mismatch_indices']==[0]
    assert safe.scientific_prefix(original,[])['status']=='FAIL'
    with pytest.raises(FileExistsError): safe.archive_attempt(src,tmp_path/'history',owner=dict(pid=11,created=20.),liveness=lambda p:None)

@pytest.mark.parametrize('batch,ram,disk,status', [(1,5,8,'PASS'),(2,6,8,'PASS'),(1,4,8,'INSUFFICIENT'),(2,5,8,'INSUFFICIENT'),(1,6,7,'INSUFFICIENT')])
def test_resource_gates(batch,ram,disk,status):
    assert safe.Reservations().check(batch,ram*safe.GIB,disk*safe.GIB)['resource']==status

@pytest.mark.parametrize('kwargs', ({'b1':4*safe.GIB},{'b2':5*safe.GIB},{'disk':7*safe.GIB}))
def test_reservations_cannot_be_lowered(kwargs):
    with pytest.raises(s.InvalidRecord): safe.Reservations(**kwargs)

def test_missing_resources_and_failure_continuation():
    with pytest.raises(s.InvalidRecord): safe.Reservations().check(1,None,8*safe.GIB)
    with pytest.raises(r.ExecutionNotAuthorized): r.continue_after_numerical_failure(authorized=True)
    for bad in ('provenance','structure','finite','complete'):
        values=dict(provenance=True,structure=True,finite=True,complete=True,numerical='FAIL',resource='PASS'); values[bad]=False
        assert r.status_record(**values)['execution']=='HARD STOP'

@pytest.mark.parametrize('batch',(1,2))
def test_complete_stage_shapes(batch):
    shapes=stage_shapes(batch)
    assert len([n for n in shapes if n.startswith('LE.') and n.endswith('output')])==12
    assert len([n for n in shapes if n.startswith('LD.') and n.endswith('output')])==7
    records={n:dict(shape=sh,finite=True,l2=0.,rms=0.,max_abs=0.) for n,sh in shapes.items()}
    validate_stages(records,batch)
    records.pop('LD.6.memory')
    with pytest.raises(s.InvalidRecord): validate_stages(records,batch)

def test_no_execution_imports_or_legacy_collection():
    folder=ROOT/'src/mfvlr_research/d02b1'
    banned={'torch','subprocess','mfvlr','multiprocessing'}
    for p in folder.glob('*.py'):
        tree=ast.parse(p.read_text())
        for n in ast.walk(tree):
            names=[a.name.split('.')[0] for a in n.names] if isinstance(n,ast.Import) else [str(n.module).split('.')[0]] if isinstance(n,ast.ImportFrom) else []
            assert not set(names)&banned
    for nodes in (['tests/test_mve.py::test_mve_t10'],['tests/decoder/test_decoder.py::test_vd_t12'],[]):
        with pytest.raises(s.InvalidRecord): r.assert_mock_selection(nodes)
    r.assert_mock_selection(['tests/research_d02b1/test_schema_safety.py::test_schedule_exact_57_and_no_execution'])
