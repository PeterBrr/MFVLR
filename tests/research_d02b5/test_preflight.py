import ast,json
from pathlib import Path
import pytest
from mfvlr_research.d02b5.policy import CELLS,identity,resources,check_backend,classify,write
ROOT=Path(__file__).resolve().parents[2]
@pytest.mark.parametrize('cell',CELLS)
def test_exact_authorized_cell(cell):
    assert identity(cell)[0]=='F2'
@pytest.mark.parametrize('cell',('D02B5-F0-A0-float32','D02B5-F1-P3-float64','G01','NG2','D02B5-F2-I1-float32','D02B5-F2-A0-float16',None))
def test_unauthorized(cell):
    with pytest.raises(PermissionError): identity(cell)
@pytest.mark.parametrize('ram,disk,answer',((5,8,True),(4,8,False),(5,7,False),(8,8,True)))
def test_resource_gate(ram,disk,answer): assert resources(ram*1024**3,disk*1024**3)==answer
def test_backend():
    check_backend({'threads':4},{'threads':4})
    with pytest.raises(RuntimeError): check_backend({'threads':1},{'threads':4})
@pytest.mark.parametrize('precision',('float32','float64'))
def test_failure_classification(precision):
    r=classify([{'outside_tolerance_count':5,'finite':True}],precision)
    assert r['numerical']=='FAIL' and r['mandatory_failure']==(precision=='float64')
    with pytest.raises(FloatingPointError): classify([{'outside_tolerance_count':0,'finite':False}],precision)
def test_exclusive_evidence(tmp_path):
    write(tmp_path/'a.json',{'value':1})
    with pytest.raises(FileExistsError): write(tmp_path/'a.json',{'value':2})
    assert json.loads((tmp_path/'a.json').read_text())=={'value':1}
def test_worker_static_no_optimizer_or_other_fixture():
    source=(ROOT/'tools/d02b5_worker.py').read_text(); tree=ast.parse(source)
    assert 'load_fixture(\'F2\')' in source
    assert 'copy.deepcopy(candidate).double()' in source
    assert 'build_variant(reference,variant)' in source
    assert 'output.low.grad_fn' in source and 'retain_graph=sample==0' in source
    assert 'model.register_forward_pre_hook(preforward)' in source
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute): assert n.func.attr not in ('step','train','backward_step')
    for name in ('d02b1','d02b2'):
        old=ROOT/f'src/mfvlr_research/{name}'/('runner.py' if name=='d02b1' else 'adapter.py')
        assert 'raise Execution' in old.read_text()


def test_original_f2_bytes():
    from mfvlr_research.d02b2.contract import load_fixture,tensor_hash,bundle
    ids,visual=load_fixture('F2')
    assert tensor_hash(ids)=='be9475b3515453a299929a95fd160bbf08d7a997ff9ba5b8fe9a4f9bfa86a1cf'
    assert tensor_hash(visual)=='27a62aad20c5c1a3cc0abd2c537137a356a53c13460597e886afcc71f18a838e'
    assert ids.shape==(2,308) and visual.shape==(2,1,512)

def test_original_f2_capture_scope():
    from mfvlr_research.d02b2.contract import stages
    for batch in (1,2):
        assert len(stages('F2',batch))==27
        assert stages('F2',batch)['output.logits']==[batch,308,49408]
    source=(ROOT/'tools/d02b5_worker.py').read_text()
    assert 'detail=False' in source and 'detail=True' not in source


def test_no_visual_regeneration():
    source=(ROOT/'tools/d02b5_worker.py').read_text()
    tree=ast.parse(source)
    imported=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert 'mfvlr.mve' not in imported and 'mfvlr.vision_decoder' not in imported
    assert 'load_fixture(\'F2\')' in source
    from mfvlr_research.d02b2.contract import bundle
    desc=bundle()['fixtures']['F2']['fixture_description']
    golden=json.loads((ROOT/'tests/foundation/fixtures/golden_tokens.json').read_text())
    assert desc['golden_prompt_indices']==[8,9]
    assert desc['ids']==[golden['prompt_cases'][i]['ids'] for i in (8,9)]
