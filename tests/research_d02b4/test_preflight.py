import ast,json
from pathlib import Path
import pytest
from mfvlr_research.d02b4.policy import CELLS,identity,resources,check_backend,classify,write
ROOT=Path(__file__).resolve().parents[2]
@pytest.mark.parametrize('cell',CELLS)
def test_exact_authorized_cell(cell):
    assert identity(cell)[0]=='F1'
@pytest.mark.parametrize('cell',('D02B4-F0-A0-float32','D02B4-F2-P3-float64','G01','NG2','D02B4-F1-I1-float32','D02B4-F1-A0-float16',None))
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
    source=(ROOT/'tools/d02b4_worker.py').read_text(); tree=ast.parse(source)
    assert 'load_fixture(\'F1\')' in source
    assert 'copy.deepcopy(candidate).double()' in source
    assert 'build_variant(reference,variant)' in source
    assert 'output.low.grad_fn' in source and 'retain_graph=sample==0' in source
    assert 'model.register_forward_pre_hook(preforward)' in source
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute): assert n.func.attr not in ('step','train','backward_step')
    for name in ('d02b1','d02b2'):
        old=ROOT/f'src/mfvlr_research/{name}'/('runner.py' if name=='d02b1' else 'adapter.py')
        assert 'raise Execution' in old.read_text()


def test_original_f1_bytes():
    from mfvlr_research.d02b2.contract import load_fixture,tensor_hash,bundle
    ids,visual=load_fixture('F1')
    assert tensor_hash(ids)=='a860a1bd09980157ecfb7f0afd4c629238cf82348c801811adfb4621cea304d9'
    assert tensor_hash(visual)=='68cd54f1fdae7602eb57ae0e6eb821c018fadf2e5fef55c927c040293033e10d'
    assert ids.shape==(2,308) and visual.shape==(2,1,512)

def test_original_f1_capture_scope():
    from mfvlr_research.d02b2.contract import stages
    for batch in (1,2):
        assert len(stages('F1',batch))==27
        assert stages('F1',batch)['output.logits']==[batch,308,49408]
    source=(ROOT/'tools/d02b4_worker.py').read_text()
    assert 'detail=False' in source and 'detail=True' not in source
