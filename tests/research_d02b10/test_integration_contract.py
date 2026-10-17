import pytest
import torch
from mfvlr_research.d02b10.integration_contract import validate_batch,invoke_mock

def batch():
    return dict(image=torch.zeros(1,3,224,224),mask=torch.zeros(1,224,224,dtype=torch.int64),
        label=torch.zeros(1,dtype=torch.int64),token_ids=torch.zeros(1,308,dtype=torch.int64),
        sample_ids=('synthetic',),token_sample_ids=('synthetic',),kinds=('real',))

@pytest.mark.parametrize('case',['pair','label','mask','tokens','range','kind','unknown'])
def test_invalid_handoffs(case):
    b=batch()
    if case=='pair':b['token_sample_ids']=('other',)
    elif case=='label':b['label'][0]=1
    elif case=='mask':b['mask'][0,0,0]=1
    elif case=='tokens':b['token_ids'][0,0]=49408
    elif case=='range':b['image'][0,0,0,0]=float('nan')
    elif case=='kind':b['kinds']=('unknown',)
    else:b['resize']=True
    with pytest.raises(ValueError):validate_batch(b)

def test_mock_batch_to_six_objectives():
    class Model:
        scope='synthetic'
        def with_language(self,image,token_ids,*,sample_ids):
            assert sample_ids==('synthetic',) and token_ids.shape==(1,308)
            return 'mock-output'
        def compute_objectives(self,output,targets):
            assert output=='mock-output'
            return {n:0. for n in ('fd','lr','cmc','fl','ar','kl','total')}
    assert list(invoke_mock(Model(),batch()))==['fd','lr','cmc','fl','ar','kl','total']

def test_live_model_forbidden():
    with pytest.raises(PermissionError):invoke_mock(object(),batch())
