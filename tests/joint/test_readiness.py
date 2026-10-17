"""READY-N01--N07: actual recorded runs, never skip/xfail failed G2 gates."""
import gc
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from mfvlr.integration_config import load_integration_config
from mfvlr.config import ConfigError
from mfvlr.prompts import PromptBuilder,PromptMetadata
from mfvlr.tokenizer import TokenizerAdapter
from tools.phase2d2_fixtures import fixture,CASES
from tools.phase2d2_diagnostics import GradientStream,write

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('MFVLR_PHASE2D2_EVIDENCE_ROOT', str(ROOT/'artifacts/phase2d2')))


def read(path):return json.loads(path.read_text())


def test_ready_n01_preservation_and_scoped_configuration():
    baseline=read(OUT/'baseline_manifest.json')
    mismatches=[p for p,h in baseline.items() if not (ROOT/p).is_file() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not mismatches
    for name in ('reference','debug'):
        c=load_integration_config(ROOT/f'configs/integration_{name}.yaml')
        assert c['approval']['optimizer_updates'] is False and c['approval']['P2D15']=='UNAPPROVED'
    import tempfile,yaml,copy
    original=load_integration_config(ROOT/'configs/integration_reference.yaml')
    with tempfile.TemporaryDirectory(dir=OUT) as directory:
        assert Path(directory).resolve().is_relative_to(OUT.resolve())
        for mutation in ('count','temperature_exemption','unknown','optimizer','bool_type','missing'):
            c=copy.deepcopy(original)
            if mutation=='count':c['components']['full_unique_parameters']-=1
            if mutation=='temperature_exemption':c['verification']['N2_exempts_new_objectives']=True
            if mutation=='unknown':c['unknown']=1
            if mutation=='optimizer':c['runtime']['optimizer_updates']=1
            if mutation=='bool_type':c['schema_version']=True
            if mutation=='missing':del c['fixtures']
            path=Path(directory)/'bad.yaml';path.write_text(yaml.safe_dump(c))
            with pytest.raises(ConfigError):load_integration_config(path)


@pytest.mark.parametrize('case',list(CASES))
def test_ready_n02_full_g2_no_update(case):
    directory=OUT/'g2'/('final_'+case);summary=read(directory/'summary.json')
    f=fixture(case)
    assert summary['fixture']==f.provenance
    assert summary['parameter_bytes_unchanged'] and summary['parameter_identities_unchanged']
    assert summary['all_valid_values_and_gradients_finite'] and summary['zero_optimizer_updates']
    for name in ('fd','lr','cmc','fl','ar','kl','total'):
        record=read(directory/f'{name}.json')
        assert len(record['activations'])==19 and all(r['finite'] for r in record['activations'].values())
        assert all(r.get('finite',True) for r in record['gradients']['parameters'].values())
    # Failed elementwise comparison remains a failing mandatory case.
    assert summary['status']=='PASS',f"{case}: unchanged FP32 gradient-additivity gate failed; see recorded comparison"


def scientific(value):
    if isinstance(value,dict):return {k:scientific(v) for k,v in value.items() if k not in ('elapsed_seconds','seconds')}
    if isinstance(value,list):return [scientific(v) for v in value]
    return value


@pytest.mark.parametrize('index',[0,1,2])
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_ready_n03_original_n2_exact_scientific_preservation(index,dtype):
    name=f'fixture{index}_{dtype}_threads4.json'
    current=read(OUT/'n2'/name);previous=read(ROOT/'artifacts/phase2c2_n2_acceptance/diagnostics'/name)
    assert current['status']=='COMPLETED'
    assert scientific(current)==scientific(previous),'materially changed original N2 evidence'


@pytest.mark.parametrize('case',list(CASES))
def test_ready_n04_fixture_and_numerical_records(case):
    f=fixture(case)
    metadata=[PromptMetadata(**{**sample['prompt_metadata'],'sample_id':sample['sample_id']}) for sample in f.provenance['samples']]
    ids=TokenizerAdapter().encode_levels(PromptBuilder().build(metadata)).ids
    assert torch.equal(ids,f.targets.token_ids)
    output=read(OUT/'g2'/('final_'+case)/'total.json')['outputs']
    categories=output['LR_categories']
    assert sum(x['count'] for x in categories.values())==308*f.image.shape[0]
    assert categories['BOS']['count']==categories['EOS']['count']==4*f.image.shape[0]
    assert output['CMC']['tau']>0 and len(output['CMC']['scores'])==f.image.shape[0]
    rec=output['reconstruction']
    assert rec['residual_positive_signs']+rec['residual_negative_signs']+rec['residual_exact_zeros']==rec['elements']
    if case=='local_duplicate':assert torch.equal(ids[0],ids[1]) and not torch.equal(f.image[0],f.image[1])


def test_ready_n05_stream_and_lifecycle():
    model=torch.nn.Linear(1024,512);stream=GradientStream(model,OUT)
    try:
        for name,i in (('fd',1),('lr',2),('cmc',3),('fl',4),('ar',5),('kl',6)):
            for p in model.parameters():p.grad=torch.full_like(p,float(i))
            stream.add(name)
        for p in model.parameters():p.grad=torch.full_like(p,21.)
        result=stream.compare_joint();assert result['allclose']
        assert all(r['max_abs_difference']==0 for r in result['parameters'].values())
    finally:stream.close()
    for case in CASES:assert read(OUT/'g2'/('final_'+case)/'summary.json')['graph_and_hook_lifecycle']=='PASS'


@pytest.mark.parametrize('batch',[1,2,8])
def test_ready_n06_cpu_capacity_reporting(batch):
    if batch in (1,2):
        case='real' if batch==1 else 'real_efs';data=read(OUT/'g2'/('final_'+case)/'resources.json')
        assert data['sampled_peak_rss_bytes']>data['rss_baseline_bytes'] and data['elapsed_seconds']>0
        # This validates an actual resource measurement, not a G2 acceptance pass.
        assert data['status'] in ('PASS','FAIL_GRADIENT_ADDITIVITY')
    else:
        record=read(OUT/'resource_availability.json')['CPU_B8']
        assert record['status']=='NOT RUN' and record['reason'] and record['available_RAM_bytes']>0


def test_ready_n07_gpu_capacity_reporting():
    data=read(OUT/'resource_availability.json')
    assert not torch.cuda.is_available()
    for batch in (1,2,8):
        record=data[f'GPU_B{batch}'];assert record['status']=='NOT RUN' and record['reason']
