import copy,importlib.util
from pathlib import Path
import numpy as np
import pytest
from mfvlr_research.d02b8 import contracts as c

@pytest.mark.parametrize('mode',('training','inference'))
def test_config(mode):
    x=dict(schema='D02B8-proposed-v1',mode=mode,variant='A0',execution_enabled=False,batch=1,precision='float32',device='cpu',loss_order=['fd','lr','cmc','fl','ar','kl'],loss_weights=[1]*6,optimizer=None,dataset_manifest_sha256='a'*64)
    assert c.config(x)['execution']=='DISABLED'
    for field,value in [('execution_enabled',True),('optimizer','SGD'),('variant','P3'),('loss_weights',[.1]*6),('unknown',1),('dataset_manifest_sha256','')]:
        bad=copy.deepcopy(x);bad[field]=value
        with pytest.raises(c.ContractError):c.config(bad)
def row():
    return dict(sample_id='x',split='train',subject_group='s',source_sha256='a'*64,image_shape=[3,224,224],image_dtype='float32',label=0,kind='real',mask_shape=[224,224],mask_dtype='int64',mask_sha256='b'*64,mask_semantics='all_background',prompt_metadata_sha256='c'*64,preprocessing_contract_sha256='d'*64)
def test_dataset_contract():
    x=row();assert c.dataset([x]).startswith('METADATA_VALID')
    with pytest.raises(c.ContractError):c.dataset([x,x])
    y=dict(x,sample_id='y',split='test')
    with pytest.raises(c.ContractError):c.dataset([x,y])
@pytest.mark.parametrize('field,value',[('label',1),('mask_semantics','inferred'),('mask_dtype','float32'),('preprocessing_contract_sha256',''),('extra',1)])
def test_dataset_negative(field,value):
    x=row();x[field]=value
    with pytest.raises(c.ContractError):c.sample(x)
def test_checkpoint():
    x=dict(schema='D02B8-proposed-v1',variant='A0',parameter_tensors=508,elements=154156232,initialization_seed=0,update_count=0,optimizer_state=None,numerical_status='FAIL')
    for k in ('state','source_manifest','config','tokenizer_manifest','dataset_manifest','loss_contract','backend'):x[k+'_sha256']='a'*64
    assert c.checkpoint(x).startswith('METADATA_VALID')
    for k,v in [('update_count',1),('numerical_status','PASS'),('parameter_tensors',507),('optimizer_state',{})]:
        bad=dict(x);bad[k]=v
        with pytest.raises(c.ContractError):c.checkpoint(bad)

def test_analysis_no_model_imports():
    import ast
    p=Path(__file__).resolve().parents[2]/'tools/d02b8_recompute.py';tree=ast.parse(p.read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):assert all(not x.name.startswith(('torch','mfvlr')) for x in n.names)
        if isinstance(n,ast.ImportFrom):assert not (n.module or '').startswith(('torch','mfvlr'))
def test_original_rounding_semantics_small_control():
    xs=[np.array([1e8,1e-8],dtype=np.float32),np.array([1,1e-8],dtype=np.float32),np.array([-1e8,-1e-8],dtype=np.float32)]
    ref=np.zeros(2,dtype=np.float64)
    for x in xs:ref+=x
    assert ref[0]==1 and ref[1]==np.float64(np.float32(1e-8))
    actual=np.array([0,0],dtype=np.float32);rounded=ref.astype(np.float32)
    failed=np.abs(actual-rounded)>np.float32(1e-6)+np.float32(1e-5)*np.abs(rounded)
    assert failed.tolist()==[True,False]
