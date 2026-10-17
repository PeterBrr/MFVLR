"""PROPOSED N3 controls. Existing I08/I10/READY-N02 gates remain mandatory."""
from fractions import Fraction
import json
import os
from pathlib import Path

import pytest
import torch

from mfvlr.language_encoder import LanguageAttention, initialize_attention
from tools.review_phase2d2_n3_local import explicit_attention, explicit_gradients
from tools.n3_gradient_tools import selftest

ROOT=Path(os.environ.get('MFVLR_N3_REVIEW_ROOT',Path.cwd()))
OUT=Path(os.environ.get('MFVLR_N3_EVIDENCE_ROOT',ROOT/'artifacts/phase2d2_n3'))


@pytest.fixture
def attention_fixture():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    model=LanguageAttention()
    initialize_attention(model,torch.Generator().manual_seed(0))
    model.double()  # Promote the same FP32-initialized values.
    x=(torch.randn(1,308,512,generator=torch.Generator().manual_seed(247))*.2).double().requires_grad_()
    c=torch.randn(1,308,512,generator=torch.Generator().manual_seed(319)).double()
    return model,x,c


def test_n3_c01_exact_rational_key_bias():
    p=[Fraction(1,2),Fraction(1,3),Fraction(1,6)]
    upstream=[2,-3,5]
    for scale in (Fraction(1,10000),Fraction(1),Fraction(10000),Fraction(100000000)):
        ds=[sum(scale*upstream[i]*p[i]*(int(i==j)-p[j]) for i in range(3)) for j in range(3)]
        assert sum(ds)==0
        assert all(sum(q*d for d in ds)==0 for q in (Fraction(1,2),Fraction(2,3)))


def test_n3_c02_fp64_manual_qk_derivatives(attention_fixture):
    model,x,c=attention_fixture
    names=('q.weight','k.weight','q.bias','k.bias')
    pp=dict(model.named_parameters())
    actual=torch.autograd.grad(model(x),tuple(pp[n] for n in names)+(x,),grad_outputs=c)
    with torch.no_grad():expected=explicit_gradients(x.detach(),{n:p.detach() for n,p in pp.items()},c)
    for n,g in zip((*names,'input'),actual):
        torch.testing.assert_close(g,expected[n],atol=1e-10,rtol=1e-8)


@pytest.mark.parametrize('name,seed',[('q.weight',1003),('k.weight',1004)])
def test_n3_c03_fp64_directional_finite_difference(attention_fixture,name,seed):
    model,x,c=attention_fixture;pp=dict(model.named_parameters())
    g=torch.autograd.grad(model(x),pp[name],grad_outputs=c)[0]
    weights={n:p.detach() for n,p in pp.items()}
    direction=torch.randn(weights[name].shape,generator=torch.Generator().manual_seed(seed),dtype=torch.float64)
    direction/=direction.norm();epsilon=1e-6
    with torch.no_grad():
        plus=explicit_attention(x.detach(),{**weights,name:weights[name]+epsilon*direction})[0]
        minus=explicit_attention(x.detach(),{**weights,name:weights[name]-epsilon*direction})[0]
        numerical=((plus*c).sum()-(minus*c).sum())/(2*epsilon)
    torch.testing.assert_close(numerical,(g*direction).sum(),atol=1e-5,rtol=1e-3)


def test_n3_c04_stream_arithmetic_and_injected_error(tmp_path):
    result=selftest(tmp_path)
    assert result['injected_error_detection']=='PASS'
    assert result['exact_sum']==result['chunk_boundary']==result['None_and_zero']=='PASS'


def test_n3_c05_recorded_diagnostics_complete():
    for run in ('real_mkldnn','real_efs_mkldnn','local_duplicate_mkldnn','real_efs_native'):
        directory=OUT/'same_forward'/run
        r=json.loads((directory/'summary.json').read_text())
        s=json.loads((directory/'same_graph.json').read_text())
        c=json.loads((directory/'separate_graphs.json').read_text())
        assert r['status']=='DIAGNOSTIC_COMPLETE_PHASE_BLOCKED'
        assert r['ownership']['total_unique']==154156232
        assert r['parameters_and_identities_unchanged'] and r['zero_optimizer_updates']
        assert r['no_numerical_policy_amendment'] and s['graph_released_before_C']
        assert len(s['graph']['parameters'])==24 and len(s['A_vs_B'])==41
        for v in s['A_vs_B'].values():
            assert v['presence_equal']
            if 'elements' in v:
                assert v['finite'] and v['legacy_atol']==1e-6 and v['legacy_rtol']==1e-5
                assert 0<=v['outside_tolerance_count']<=v['elements']
                assert len(v['directional_absolute_errors'])==4
        for component in c.values():
            assert component['all_full_parameter_gradients_finite']
            assert component['same_forward_values_as_G0'] and component['same_parameter_identities_as_G0']
            assert len(component['all_38_VIM_QK'])==38
            for qk in component['all_38_VIM_QK'].values():
                if qk['allocated']:assert qk['exact_zero']


def test_n3_c06_recorded_cmc_fp64_control():
    report=json.loads((OUT/'chain/float64.json').read_text())
    assert report['status']=='PASS' and report['deep_copied_FP32_weights']
    assert report['no_optimizer_updates'] and not report['acceptance_gate_exempted']
