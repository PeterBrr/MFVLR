"""VIM-T01--T10: approved standalone foundation, no FLT blocks."""
import math

import pytest
import torch

from mfvlr.vim import VisionInjectionDelta, parameter_report

FP64 = {"atol": 1e-10, "rtol": 1e-8}
FP32 = {"atol": 1e-6, "rtol": 1e-5}


def fixture(batch=1, length=3, dtype=torch.float64):
    module = VisionInjectionDelta().to(dtype=dtype)
    with torch.no_grad():
        for layer, scale in zip((module.q, module.k, module.v, module.out), (0.03, 0.04, 0.05, 0.06)):
            layer.weight.copy_(torch.eye(512, dtype=dtype) * scale)
    language = torch.linspace(0.1, 0.7, batch*length*512, dtype=dtype).reshape(batch, length, 512).requires_grad_()
    visual = torch.linspace(0.2, 0.9, batch*512, dtype=dtype).reshape(batch, 1, 512).requires_grad_()
    return module, language, visual


@pytest.mark.parametrize("batch,length", [(1, 1), (1, 308), (2, 308)])
def test_vim_t01_shapes(batch, length):
    module, language, visual = fixture(batch, length, torch.float32)
    trace = module.forward_with_trace(language, visual)
    expected = {"query": (batch,8,length,64), "key": (batch,8,1,64), "value": (batch,8,1,64),
                "logits": (batch,8,length,1), "probabilities": (batch,8,length,1),
                "attended": (batch,length,512), "delta": (batch,length,512)}
    for name, shape in expected.items():
        assert tuple(getattr(trace, name).shape) == shape
        assert torch.isfinite(getattr(trace, name)).all()
    trace.delta.square().mean().backward()
    assert visual.grad.isfinite().all() and visual.grad.norm() > 0
    assert torch.equal(module(language, visual), trace.delta)


def test_vim_t02_published_scale():
    module, language, visual = fixture()
    trace = module.forward_with_trace(language, visual)
    raw = trace.query @ trace.key.transpose(-2, -1)
    assert module.denominator == math.sqrt(8/512) == 0.125
    assert torch.equal(trace.logits, raw * 8)
    assert not torch.allclose(trace.logits, raw / 8, **FP64)
    # Analytical two-key fixture only: production VIM continues to reject two keys.
    scores = torch.tensor([0., 1.], dtype=torch.float64)
    published = (scores / module.denominator).softmax(-1)
    conventional = (scores / math.sqrt(64)).softmax(-1)
    expected = torch.tensor([1/(1+math.exp(8)), 1/(1+math.exp(-8))], dtype=torch.float64)
    torch.testing.assert_close(published, expected, **FP64)
    assert not torch.allclose(published, conventional, **FP64)


def test_vim_t03_singleton_reference():
    module, language, visual = fixture(2, 9)
    trace = module.forward_with_trace(language, visual)
    assert torch.equal(trace.probabilities, torch.ones_like(trace.probabilities))
    # Paper row-vector notation uses transposes of PyTorch's stored linear weights.
    reference = (visual @ module.v.weight.T @ module.out.weight.T).expand_as(language)
    torch.testing.assert_close(trace.delta, reference, **FP64)
    assert torch.equal(trace.delta[:, 0], trace.delta[:, -1])


def test_vim_t04_query_invariance():
    module, language, visual = fixture()
    before = module(language, visual)
    with torch.no_grad():
        module.q.weight.add_(0.007)
        module.k.weight.sub_(0.011)
    altered = language * 2 + 0.3
    after = module(altered, visual)
    assert torch.equal(before, after)
    torch.testing.assert_close((altered+after)-(language+before), altered-language, **FP64)


def test_vim_t05_analytical_gradients():
    module, language, visual = fixture(2, 4)
    delta = module(language, visual)
    upstream = torch.linspace(0.2, 0.8, delta.numel(), dtype=torch.float64).reshape_as(delta)
    (delta * upstream).sum().backward()
    for tensor in (module.q.weight, module.k.weight, language):
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad) == 0
    summed = upstream.sum(dim=1, keepdim=True)
    expected_visual = summed @ module.out.weight @ module.v.weight
    value = visual.detach() @ module.v.weight.T
    expected_out = summed.reshape(-1,512).T @ value.reshape(-1,512)
    expected_val = (summed @ module.out.weight).reshape(-1,512).T @ visual.detach().reshape(-1,512)
    for actual, expected in ((visual.grad, expected_visual), (module.v.weight.grad, expected_val),
                             (module.out.weight.grad, expected_out)):
        torch.testing.assert_close(actual, expected, **FP64)
        assert torch.isfinite(actual).all() and actual.norm() > 0


def test_vim_t06_residual_ownership():
    module, language, visual = fixture()
    with torch.no_grad():
        module.out.weight.zero_()
    delta = module(language, visual)
    assert torch.count_nonzero(delta) == 0
    # External residual fixture represents either caller position; no LE/LD is built.
    output = language + delta
    assert torch.equal(output, language)
    output.sum().backward()
    assert torch.equal(language.grad, torch.ones_like(language))
    assert not torch.equal(output, 2*language)


def test_vim_t07_partition_parameters_and_initialization():
    torch.manual_seed(79)
    rng = torch.random.get_rng_state().clone()
    module = VisionInjectionDelta()
    assert torch.equal(rng, torch.random.get_rng_state())
    duplicate = VisionInjectionDelta()
    other = VisionInjectionDelta(seed=1)
    generator = torch.Generator().manual_seed(0)
    for (name, parameter), duplicate_parameter, other_parameter in zip(
            module.named_parameters(), duplicate.parameters(), other.parameters()):
        expected = torch.empty_like(parameter)
        torch.nn.init.xavier_uniform_(expected, gain=1, generator=generator)
        assert torch.equal(parameter, expected), name
        assert torch.equal(parameter, duplicate_parameter)
        assert parameter is not duplicate_parameter and parameter.data_ptr() != duplicate_parameter.data_ptr()
        assert not torch.equal(parameter, other_parameter)
    assert all(layer.bias is None for layer in (module.q, module.k, module.v, module.out))
    assert len(list(module.parameters())) == 4
    assert not list(module.buffers())
    report = parameter_report(module)
    assert report == {"query":262144,"key":262144,"value":262144,"output":262144,
                      "total_unique":1048576,"singleton_zero_task_gradient_qk":524288}
    # Conditional 19-instance budget only; no unapproved LE/LD depth allocation.
    assert 19*report["total_unique"] == 19922944
    module, language, visual = fixture(1, 2)
    with torch.no_grad():
        visual.copy_(torch.arange(512, dtype=torch.float64).reshape(1,1,512)/512)
    trace = module.forward_with_trace(language, visual)
    for head in range(8):
        torch.testing.assert_close(trace.value[0,head,0], visual[0,0,head*64:(head+1)*64]*0.05, **FP64)
    torch.testing.assert_close(trace.attended, (visual*0.05).expand_as(language), **FP64)


def test_vim_t08_invalid_and_overflow():
    module = VisionInjectionDelta()
    x, v = torch.ones(1,3,512), torch.ones(1,1,512)
    invalid = [(x,v.expand(1,2,512)), (x,v.squeeze(1)), (x.expand(2,3,512),v),
               (x[:0],v[:0]), (x[:,:0],v), (x[:,:,:511],v), (x.double(),v),
               (x.to(torch.float16),v), (x.to(torch.int64),v), (x*float('nan'),v),
               (x,v*float('inf')), ([],v), (x,torch.empty(1,1,512,device='meta'))]
    for language, visual in invalid:
        with pytest.raises((TypeError, ValueError)):
            module(language, visual)
    for seed in (True, -1, 1.5, 2**63):
        with pytest.raises(ValueError):
            VisionInjectionDelta(seed=seed)
    with torch.no_grad():
        for p in module.parameters():
            p.fill_(1)
    with pytest.raises(FloatingPointError, match="overflow"):
        module(x*1e38, v*1e38)


def test_vim_t09_directional_derivatives():
    module, language, visual = fixture()
    # Mean keeps the finite-difference scalar well-scaled at the unchanged tolerance.
    def objective():
        return (language + module(language, visual)).mean()
    scalar = objective()
    tensors = (visual, module.v.weight, module.out.weight, language, module.q.weight, module.k.weight)
    grads = torch.autograd.grad(scalar, tensors)
    epsilon = 1e-5
    for target, grad in zip(tensors, grads):
        index = (0,)*target.ndim
        original = target[index].item()
        with torch.no_grad():
            target[index] = original + epsilon
            positive = objective().item()
            target[index] = original - epsilon
            negative = objective().item()
            target[index] = original
        actual = torch.tensor((positive-negative)/(2*epsilon), dtype=torch.float64)
        torch.testing.assert_close(actual, grad[index], **FP64)


def test_vim_t10_fixed_batch_independence():
    module = VisionInjectionDelta()
    generator = torch.Generator().manual_seed(113)
    x = torch.randn(2,308,512,generator=generator,requires_grad=True)
    v = torch.randn(2,1,512,generator=generator,requires_grad=True)
    before = module(x,v)
    changed_x, changed_v = x.detach().clone(), v.detach().clone()
    changed_x[1] *= -7
    changed_v[1] += 3
    after = module(changed_x,changed_v)
    assert torch.equal(before[0],after[0])
    assert not torch.equal(before[1],after[1])
    before[0].square().mean().backward()
    assert torch.count_nonzero(x.grad[1]) == torch.count_nonzero(v.grad[1]) == 0
    assert v.grad[0].norm() > 0
