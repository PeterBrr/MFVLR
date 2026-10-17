import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from mfvlr_research.d02b9 import data,checkpoint
from mfvlr_research.d02b9.inference import predict_visual
from mfvlr_research.d02b9.trainer import PreparedTrainer,validate_config

def fixture():
    image=torch.full((3,224,224),.5)
    mask=torch.zeros((224,224),dtype=torch.int64)
    row=dict(sample_id='synthetic',split='train',subject_group='synthetic_subject',
      source_sha256=data.tensor_hash(image),image_shape=[3,224,224],image_dtype='float32',label=0,kind='real',
      mask_shape=[224,224],mask_dtype='int64',mask_sha256=data.tensor_hash(mask),mask_semantics='all_background',
      prompt_metadata_sha256='1'*64,preprocessing_contract_sha256='2'*64)
    return [row],{'synthetic':(image,mask)}

def test_loader_and_owned_data():
    rows,tensors=fixture()
    ds=data.InMemoryFaces(rows,tensors,'train',expected_manifest_sha256=data.manifest_hash(rows))
    tensors['synthetic'][0].zero_()
    item=ds[0];item['image'].zero_()
    batch=next(iter(data.loader(ds)))
    assert batch['image'].shape==(1,3,224,224) and batch['image'].mean()==.5
    assert batch['label'].tolist()==[0] and batch['mask'].sum()==0

@pytest.mark.parametrize('kind',['subject','source','label','mask','hash','range','unknown','manifest'])
def test_data_rejections(kind):
    rows,tensors=fixture()
    if kind in ('subject','source'):
        other=copy.deepcopy(rows[0]);other.update(sample_id='other',split='test')
        if kind=='source':other['subject_group']='another'
        rows.append(other);tensors['other']=tensors['synthetic']
    elif kind=='label':rows[0]['label']=1
    elif kind=='mask':tensors['synthetic'][1][0,0]=2
    elif kind=='hash':rows[0]['source_sha256']='0'*64
    elif kind=='range':tensors['synthetic'][0][0,0,0]=2
    elif kind=='unknown':rows[0]['resize']=True
    expected='0'*64 if kind=='manifest' else data.manifest_hash(rows)
    with pytest.raises(ValueError):data.InMemoryFaces(rows,tensors,'train',expected_manifest_sha256=expected)

class VisualMock(torch.nn.Module):
    def __init__(self):
        super().__init__();self.p=torch.nn.Parameter(torch.tensor([1.,2.]));self.calls=0
    def forward(self,image):
        self.calls+=1
        return SimpleNamespace(detection_logits=self.p.expand(image.shape[0],2),
            decoded=SimpleNamespace(mask_logits=self.p.reshape(1,2,1,1).expand(image.shape[0],2,224,224)))

@pytest.mark.parametrize('batch',[1,2])
def test_visual_only(batch):
    model=VisualMock();ids=[id(p) for p in model.parameters()]
    result=predict_visual(model,torch.zeros(batch,3,224,224))
    assert model.calls==1 and model.training and ids==[id(p) for p in model.parameters()]
    assert torch.allclose(result['class_probabilities'].sum(1),torch.ones(batch))
    assert torch.allclose(result['mask_probabilities'].sum(1),torch.ones(batch,224,224))
    assert result['class_order']==('Real','Fake') and model.p.grad is None

def test_visual_error_restores_mode():
    model=VisualMock();model.p.data.fill_(float('nan'))
    with pytest.raises(FloatingPointError):predict_visual(model,torch.zeros(1,3,224,224))
    assert model.training

def info():
    return dict(schema='D02B9-tensor-v1',variant='A0',scope='mock',update_count=0,numerical_status='NOT RUN',
                **{k:'1'*64 for k in ('source_sha256','config_sha256','dataset_sha256','tokenizer_sha256','loss_contract_sha256')})

def test_checkpoint_roundtrip(tmp_path):
    model=VisualMock();identity=id(model.p)
    pin=checkpoint.save(model,tmp_path/'state',info())
    with torch.no_grad():model.p.zero_()
    checkpoint.load(model,tmp_path/'state',trusted_manifest_sha256=pin,max_bytes=4096)
    assert torch.equal(model.p,torch.tensor([1.,2.])) and id(model.p)==identity

def test_checkpoint_tied_parameter_identity(tmp_path):
    model=VisualMock();model.alias=model.p
    pin=checkpoint.save(model,tmp_path/'state',info())
    checkpoint.load(model,tmp_path/'state',trusted_manifest_sha256=pin,max_bytes=4096)
    assert model.alias is model.p

def test_trainer_loader_provenance():
    rows,tensors=fixture();digest=data.manifest_hash(rows)
    ds=data.InMemoryFaces(rows,tensors,'train',expected_manifest_sha256=digest)
    config=json.loads(Path('configs/research_d02b9/training_mock.json').read_text())
    with pytest.raises(ValueError):PreparedTrainer(config).batches(ds)
    config['dataset_manifest_sha256']=digest
    assert next(iter(PreparedTrainer(config).batches(ds)))['sample_id']==['synthetic']

@pytest.mark.parametrize('kind',['hash','bytes','corrupt','path','shape','metadata','keys'])
def test_checkpoint_rejects_before_mutation(tmp_path,kind):
    model=VisualMock();pin=checkpoint.save(model,tmp_path/'state',info());before=model.p.detach().clone()
    path=tmp_path/'state'/'manifest.json';doc=json.loads(path.read_text())
    limit=4096
    if kind=='hash':pin='0'*64
    elif kind=='bytes':limit=1
    elif kind=='corrupt':(tmp_path/'state'/'00000.npy').write_bytes(b'bad')
    else:
        if kind=='path':doc['tensors']['p']['file']='../escape.npy'
        elif kind=='shape':doc['tensors']['p']['shape']=[1]
        elif kind=='metadata':doc['metadata']['update_count']=1
        else:doc['tensors']['extra']=doc['tensors']['p']
        path.write_text(json.dumps(doc));pin=checkpoint.digest(path)
    with pytest.raises(ValueError):checkpoint.load(model,tmp_path/'state',trusted_manifest_sha256=pin,max_bytes=limit)
    assert torch.equal(model.p,before)

@pytest.mark.parametrize('operation',['fit','step','backward'])
def test_training_disabled(operation):
    config=json.loads(Path('configs/research_d02b9/training_mock.json').read_text())
    trainer=PreparedTrainer(config)
    with pytest.raises(PermissionError):getattr(trainer,operation)()
    config['execution_enabled']=True
    with pytest.raises(ValueError):validate_config(config)
