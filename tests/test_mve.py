"""MVE-T01--MVE-T14 from the approved contract, including negative cases."""

import copy
from dataclasses import replace
import gc
import math

import pytest
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from mfvlr.config import ConfigError, load_config, merge_overlay
from mfvlr.mve import MVE, VisualTransformerBlock, parameter_report


def close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.shapes
@pytest.mark.parametrize("batch", [1, 2])
def test_mve_t01_layer_shapes_and_invalid_inputs(model, batch):
    """Every CNN layer/pool and skip shape, with fixed-resolution input checks."""
    observed = []
    handles = []
    for name, layer in model.shared_encoder.cnn.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            handles.append(layer.register_forward_hook(
                lambda module, args, output, name=name: observed.append((name, tuple(output.shape)))
            ))
    image = torch.zeros(batch, 3, 224, 224)
    try:
        with torch.no_grad():
            local = model.shared_encoder.encode_local(image, return_skips=True)
            global_encoding = model.shared_encoder.encode_global(local.local)
    finally:
        for handle in handles:
            handle.remove()
    expected_shapes = []
    for index, (channels, size) in enumerate(((64, 224), (128, 112), (256, 56), (512, 28), (1024, 14))):
        expected_shapes.extend([(batch, channels, size, size)] * 2)
        if index < 4:
            expected_shapes.append((batch, channels, size // 2, size // 2))
    assert [shape for _, shape in observed] == expected_shapes
    assert [tuple(skip.shape) for skip in local.skips] == [
        (batch, 64, 224, 224), (batch, 128, 112, 112),
        (batch, 256, 56, 56), (batch, 512, 28, 28),
    ]
    assert local.local.shape == (batch, 1024, 14, 14)
    assert global_encoding.tokens.shape == (batch, 197, 512)
    assert global_encoding.global_token.shape == (batch, 1, 512)
    for invalid in (torch.zeros(3, 224, 224), torch.zeros(1, 1, 224, 224),
                    torch.zeros(1, 3, 112, 112), torch.empty(0, 3, 224, 224)):
        with pytest.raises(ValueError):
            model.encode_appearance(invalid)
    for layer in model.shared_encoder.cnn.modules():
        if isinstance(layer, nn.Conv2d):
            assert layer.kernel_size == (3, 3) and layer.stride == (1, 1)
            assert layer.padding == (1, 1) and layer.groups == 1 and layer.bias is not None
        if isinstance(layer, nn.ReLU):
            assert not layer.inplace
    print(f"MVE-T01 batch={batch}: local={tuple(local.local.shape)}, tokens={tuple(global_encoding.tokens.shape)}")


@pytest.mark.shapes
def test_mve_t02_row_major_projection_cls_position(model):
    encoder = model.shared_encoder
    local = torch.zeros(1, 1024, 14, 14)
    local[:, 0] = torch.arange(196).reshape(14, 14)
    local[:, 1] = 2.0
    expected_flat = torch.zeros(1, 196, 1024)
    expected_flat[0, :, 0] = torch.arange(196)
    expected_flat[0, :, 1] = 2.0
    captured = []
    handle = encoder.projection.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
    saved_bias = encoder.projection.bias.detach().clone()
    try:
        with torch.no_grad():
            encoder.projection.bias.fill_(0.3)
            tokens = encoder.tokenize(local)
            spatial = F.linear(expected_flat, encoder.projection.weight, encoder.projection.bias)
            expected = torch.cat((spatial, encoder.class_token), dim=1) + encoder.position
        torch.testing.assert_close(captured[0], expected_flat, atol=0, rtol=0)
        close(tokens, expected)
        close(tokens[:, 196:197], encoder.class_token + encoder.position[:, 196:197])
        assert tokens.shape == (1, 197, 512)
    finally:
        handle.remove()
        with torch.no_grad():
            encoder.projection.bias.copy_(saved_bias)


@pytest.mark.backward
def test_mve_t03_independent_transformer_outputs_and_gradients():
    """Independent head-by-head attention, manual normalization and erf GELU."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        block = VisualTransformerBlock().double()
        state = torch.randn(1, 3, 512, dtype=torch.float64, requires_grad=True)

    def norm(x, layer):
        centered = x - x.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        return centered / (variance + 1e-5).sqrt() * layer.weight + layer.bias

    def affine(x, layer):
        return x @ layer.weight.T + layer.bias

    normalized = norm(state, block.ln1)
    q = affine(normalized, block.attention.q)
    k = affine(normalized, block.attention.k)
    v = affine(normalized, block.attention.v)
    heads = []
    for head in range(8):
        head_slice = slice(64 * head, 64 * (head + 1))
        score = (q[..., head_slice] @ k[..., head_slice].transpose(-2, -1)) / 8.0
        probability = torch.exp(score - score.logsumexp(dim=-1, keepdim=True))
        heads.append(probability @ v[..., head_slice])
    y = state + affine(torch.cat(heads, dim=-1), block.attention.out)
    hidden = affine(norm(y, block.ln2), block.fc1)
    gelu = 0.5 * hidden * (1.0 + torch.erf(hidden / math.sqrt(2)))
    expected = y + affine(gelu, block.fc2)
    actual = block(state)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    variables = (state, *block.parameters())
    actual_gradients = torch.autograd.grad(actual.square().mean(), variables)
    expected_gradients = torch.autograd.grad(expected.square().mean(), variables)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-10, rtol=1e-8)
    assert sum(isinstance(module, nn.LayerNorm) for module in block.modules()) == 2
    assert block.fc1.out_features == 2048 and block.activation.approximate == "none"


@pytest.mark.shapes
def test_mve_t04_class_token_is_image_dependent(model, image):
    with torch.no_grad():
        first = model.encode_appearance(image).global_token
        second = model.encode_appearance(1.0 - image).global_token
    assert first.shape == (1, 1, 512)
    assert (first - second).abs().max().item() > 1e-4
    appearance = model.encode_appearance(image.requires_grad_())
    assert appearance.global_token.requires_grad and appearance.global_token.grad_fn is not None
    assert appearance.local.grad_fn is not None


@pytest.mark.shapes
def test_mve_t05_single_owner_and_parameter_count(model, image):
    report = parameter_report(model)
    assert report == {
        "cnn": 18843200, "projection": 524800, "class_token": 512, "position": 100864,
        "transformer_block_1": 3152384, "transformer_block_2": 3152384,
        "transformer_block_3": 3152384, "transformer_block_4": 3152384, "total_unique": 32078912,
    }
    named = list(model.named_parameters(remove_duplicate=False))
    assert len(named) == len({id(parameter) for _, parameter in named})
    assert {key.split(".")[0] for key in model.state_dict()} == {"shared_encoder"}
    identities = []
    handle = model.shared_encoder.cnn.register_forward_pre_hook(
        lambda module, args: identities.append(tuple(id(parameter) for parameter in module.parameters()))
    )
    try:
        with torch.no_grad():
            model(image, 0.75 * image + 0.1)
    finally:
        handle.remove()
    assert len(identities) == 2 and identities[0] == identities[1]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    parameters = optimizer.param_groups[0]["params"]
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert not any(isinstance(module, (nn.ConvTranspose2d, nn.BatchNorm2d, nn.Dropout)) for module in model.modules())
    print(f"MVE-T05 parameter report: {report}")


@pytest.mark.shapes
def test_mve_t06_absolute_residual_and_additive_fusion(model):
    image = torch.zeros(1, 3, 224, 224)
    reconstruction = torch.linspace(-2.0, 2.0, image.numel()).reshape_as(image)
    with torch.no_grad():
        result = model(image, reconstruction)
        torch.testing.assert_close(result.residual_image, reconstruction.abs(), atol=0, rtol=0)
        torch.testing.assert_close(result.fused_global, result.appearance.global_token + result.residual_global, atol=0, rtol=0)
        zero = model(image, image)
    assert result.residual_image.max().item() == 2.0
    assert torch.count_nonzero(zero.residual_image).item() == 0
    assert zero.residual_global.norm().item() > 0


@pytest.mark.shapes
def test_mve_t07_reused_appearance_calls_and_parity(model, image):
    counts = {"local": 0, "global": 0}

    def counter(name):
        def hook(module, inputs, output):
            counts[name] += 1
        return hook

    handles = [model.shared_encoder.cnn.register_forward_hook(counter("local")),
               model.shared_encoder.blocks[0].register_forward_hook(counter("global"))]
    try:
        with torch.no_grad():
            appearance = model.encode_appearance(image)
            reused = model(image, 0.75 * image + 0.1, appearance)
            assert counts == {"local": 2, "global": 2}
            assert reused.appearance is appearance
            recomputed = model(image, 0.75 * image + 0.1)
            assert counts == {"local": 4, "global": 4}
            close(reused.fused_global, recomputed.fused_global)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.backward
def test_mve_t08_external_reconstruction_gradient_path(model, image):
    """Only a test fixture: spatial channel mean + bilinear resize, no decoder."""
    appearance = model.encode_appearance(image)
    fixture = 0.2 + 0.01 * F.interpolate(
        appearance.local.mean(dim=1, keepdim=True), size=(224, 224),
        mode="bilinear", align_corners=False,
    ).expand(-1, 3, -1, -1)
    # Isolate reconstruction dependency; the normal appearance-global path is cut
    # only in this test record, not in the implementation.
    isolated = replace(appearance, global_token=appearance.global_token.detach(), tokens=appearance.tokens.detach())
    result = model(image, fixture, isolated)
    gradients = torch.autograd.grad(
        result.fused_global.square().mean(),
        (fixture, appearance.local, model.shared_encoder.cnn.stages[0][0].weight),
    )
    assert all(torch.isfinite(gradient).all() and gradient.norm() > 0 for gradient in gradients)
    detached = model(image, fixture.detach(), isolated)
    removed = torch.autograd.grad(
        detached.fused_global.square().mean(), (fixture, appearance.local), allow_unused=True,
    )
    assert removed == (None, None)
    print(f"MVE-T08 reconstruction gradient={tuple(gradients[0].shape)}, local gradient={tuple(gradients[1].shape)}, "
          f"norm={gradients[1].norm().item():.8g}; detached local path=None")


@pytest.mark.backward
def test_mve_t09_absolute_derivative_and_no_clipping(model):
    base = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        appearance = model.encode_appearance(base)
    image = base.clone().requires_grad_()
    reconstruction = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0]).repeat(
        math.ceil(image.numel() / 5)
    )[:image.numel()].reshape_as(image).requires_grad_()
    result = model(image, reconstruction, appearance)
    reconstruction_gradient, image_gradient = torch.autograd.grad(result.residual_image.sum(), (reconstruction, image))
    torch.testing.assert_close(reconstruction_gradient, reconstruction.sign(), atol=0, rtol=0)
    torch.testing.assert_close(image_gradient, -reconstruction.sign(), atol=0, rtol=0)
    assert result.residual_image.max().item() == 2 and result.residual_image.min().item() == 0


@pytest.mark.backward
def test_mve_t10_shared_gradient_sum_and_single_update(model, image):
    parameters = (model.shared_encoder.cnn.stages[0][0].weight,
                  model.shared_encoder.projection.weight, model.shared_encoder.class_token)
    other = 1.0 - image

    def objective(value):
        return model.encode_appearance(value).global_token.square().mean()

    combined = torch.autograd.grad(objective(image) + objective(other), parameters)
    first = torch.autograd.grad(objective(image), parameters)
    second = torch.autograd.grad(objective(other), parameters)
    for combined_gradient, first_gradient, second_gradient in zip(combined, first, second):
        close(combined_gradient, first_gradient + second_gradient)
    before = [parameter.detach().clone() for parameter in parameters]
    try:
        optimizer = torch.optim.SGD(parameters, lr=0.001)
        assert len(optimizer.param_groups[0]["params"]) == len({id(parameter) for parameter in parameters})
        for parameter, gradient in zip(parameters, combined):
            parameter.grad = gradient.clone()
        optimizer.step()  # One controlled unit-test update; no training pipeline.
        for parameter, previous, gradient in zip(parameters, before, combined):
            torch.testing.assert_close(parameter, previous.add(gradient, alpha=-0.001), atol=0, rtol=0)
    finally:
        with torch.no_grad():
            for parameter, previous in zip(parameters, before):
                parameter.copy_(previous)
        model.zero_grad(set_to_none=True)
    print("MVE-T10: independent-role gradients sum; one SGD unit-test update verified and restored")


@pytest.mark.serialization
def test_mve_t11_initialization_order_statistics_and_reproducibility(model):
    encoder = model.shared_encoder
    generator = torch.Generator().manual_seed(0)
    # Independent replay in declared contract order, using mathematical bounds.
    for stage in encoder.cnn.stages:
        for layer in stage:
            if isinstance(layer, nn.Conv2d):
                fan_in = layer.in_channels * 9
                expected = torch.empty_like(layer.weight).normal_(0.0, math.sqrt(2 / fan_in), generator=generator)
                torch.testing.assert_close(layer.weight, expected, atol=0, rtol=0)
                assert torch.count_nonzero(layer.bias) == 0
                ratio = layer.weight.std().item() / math.sqrt(2 / fan_in)
                assert 0.85 < ratio < 1.15

    def check_linear(layer):
        bound = math.sqrt(6 / (layer.in_features + layer.out_features))
        expected = torch.empty_like(layer.weight).uniform_(-bound, bound, generator=generator)
        torch.testing.assert_close(layer.weight, expected, atol=0, rtol=0)
        assert layer.weight.abs().max() <= bound + 1e-7
        assert torch.count_nonzero(layer.bias) == 0

    check_linear(encoder.projection)
    for parameter in (encoder.class_token, encoder.position):
        expected = torch.empty_like(parameter).normal_(0.0, 0.02, generator=generator)
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
        assert 0.016 < parameter.std().item() < 0.024
    assert encoder.position.abs().max() > 0.04  # Not a 2-sigma truncated initializer.
    for block in encoder.blocks:
        for layer in (block.ln1, block.ln2):
            assert torch.equal(layer.weight, torch.ones_like(layer.weight))
            assert torch.count_nonzero(layer.bias) == 0
        for layer in (block.attention.q, block.attention.k, block.attention.v, block.attention.out, block.fc1, block.fc2):
            check_linear(layer)
    original_rng = torch.get_rng_state().clone()
    replica = MVE(seed=0)
    assert torch.equal(torch.get_rng_state(), original_rng)
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, replica.state_dict()[key]), key
    del replica
    gc.collect()


@pytest.mark.shapes
def test_mve_t12_modes_and_batch_independence(model, image, root):
    assert not any(isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)) for module in model.modules())
    with torch.no_grad():
        model.train()
        train = model(image, 0.75 * image + 0.1).fused_global
        model.eval()
        evaluation = model(image, 0.75 * image + 0.1).fused_global
        close(train, evaluation)
        batch = torch.cat((image, 1.0 - image), dim=0)
        together = model(batch, 0.75 * batch + 0.1).fused_global
        changed_batch = torch.cat((image, torch.zeros_like(image)), dim=0)
        changed = model(changed_batch, 0.75 * changed_batch + 0.1).fused_global
        torch.testing.assert_close(together[:1], changed[:1], atol=0, rtol=0)
        print(f"MVE-T12 batch independence max abs error={(evaluation - together[:1]).abs().max().item():.8g}")
        # N1 is approved ONLY for this CPU FP32 cross-batch-size comparison.
        # Train/eval keeps the general threshold; fixed-batch independence above
        # remains exact. No other test consumes the N1 threshold.
        amendment = load_config(root / "configs/debug.yaml").proposal["verification"]["mve_t12_cross_batch"]
        assert evaluation.device.type == amendment["device"] == "cpu"
        assert evaluation.dtype == torch.float32 and amendment["dtype"] == "float32"
        torch.testing.assert_close(evaluation, together[:1],
                                   atol=amendment["atol"], rtol=amendment["rtol"])


@pytest.mark.serialization
def test_mve_t13_serialization_and_record_validation(model, image, tmp_path):
    model.eval()
    reconstruction = 0.75 * image + 0.1
    with torch.no_grad():
        expected = model(image, reconstruction).fused_global.clone()
    state = model.state_dict()
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    assert all(key.startswith("shared_encoder.") for key in state)
    checkpoint = tmp_path / "mve.pt"
    torch.save(state, checkpoint)
    restored = MVE(seed=1)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    restored.eval()
    with torch.no_grad():
        actual = restored(image, reconstruction).fused_global
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert list(restored._modules) == ["shared_encoder"]
    assert parameter_report(restored)["total_unique"] == 32078912
    del restored, state
    gc.collect()
    for invalid in (reconstruction.double(), reconstruction.to(torch.int64)):
        with pytest.raises(TypeError):
            model(image, invalid)
    with pytest.raises(ValueError, match="device"):
        model(image, torch.empty(image.shape, device="meta"))
    with pytest.raises(ValueError):
        model(image, torch.zeros(2, 3, 224, 224))
    for value in (float("nan"), float("inf")):
        invalid = reconstruction.clone()
        invalid[0, 0, 0, 0] = value
        with pytest.raises(ValueError, match="finite"):
            model(image, invalid)
    with torch.no_grad():
        record = model.encode_appearance(image)
    with pytest.raises(ValueError, match="skips"):
        model(image, reconstruction, replace(record, skips=()))
    assert not hasattr(model, "appearance_encoding") and not list(model.buffers())


@pytest.mark.serialization
def test_mve_t14_config_overlay_schema_and_scope(root, tmp_path, monkeypatch):
    reference = load_config(root / "configs/paper_reference.yaml")
    debug = load_config(root / "configs/debug.yaml")
    reference.require_mve_approval()
    debug.require_mve_approval()
    for key in ("mve", "initialization", "verification"):
        assert debug.proposal[key] == reference.proposal[key]
    assert debug.proposal["data"]["training_source_generator"] is None
    assert debug.proposal["training"]["max_steps"] == 3
    assert debug.proposal["execution"]["components"] == ["mve"]
    assert not debug.proposal["training"]["enabled"]
    assert reference.data["review"]["decisions"]["A02"] == "PROPOSED"
    assert not reference.data["review"]["mve_approval"]["production_skip_decoder_approved"]
    assert debug.review_chain[-1]["training_authorized"] is False
    for review in debug.review_chain:
        assert review["numerical_amendments"]["N1"]["status"] == "APPROVED"
        assert review["numerical_amendments"]["N1"]["conditions_verified"] is True
    verification = reference.proposal["verification"]
    assert (verification["fp32_atol"], verification["fp32_rtol"]) == (1e-6, 1e-5)
    assert (verification["fp64_atol"], verification["fp64_rtol"]) == (1e-10, 1e-8)
    assert verification["mve_t12_cross_batch"] == {
        "amendment": "N1", "device": "cpu", "dtype": "float32", "atol": 5e-6, "rtol": 1e-5,
    }

    base_value = {"sequence": [1, 2, 3], "scalar": 3, "nested": {"clear": 9, "keep": True}}
    patch = {"sequence": [4], "scalar": None, "nested": {"clear": None}}
    assert merge_overlay(base_value, patch) == {"sequence": [4], "scalar": None, "nested": {"clear": None, "keep": True}}
    assert base_value["sequence"] == [1, 2, 3] and base_value["scalar"] == 3
    with pytest.raises(ConfigError, match="Unknown"):
        merge_overlay(base_value, {"new": 1})
    with pytest.raises(ConfigError):
        merge_overlay(base_value, {"scalar": {"new": 1}})

    ref_raw = yaml.safe_load((root / "configs/paper_reference.yaml").read_text(encoding="utf-8"))
    debug_raw = yaml.safe_load((root / "configs/debug.yaml").read_text(encoding="utf-8"))
    base_path = tmp_path / "base.yaml"
    debug_path = tmp_path / "overlay.yaml"

    def write(value, path):
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    write(ref_raw, base_path)
    debug_raw["base"] = "base.yaml"
    write(debug_raw, debug_path)
    monkeypatch.chdir(tmp_path.parent)
    assert load_config(debug_path).proposal["mve"] == reference.proposal["mve"]
    # Explicit null wins over the existing debug max_steps value, never inherits.
    variant = copy.deepcopy(debug_raw)
    variant["overrides"]["training"]["max_steps"] = None
    write(variant, debug_path)
    assert load_config(debug_path).proposal["training"]["max_steps"] is None
    # Known deferred lists replace, rather than append to, the reference lists.
    variant["overrides"]["data"]["first_evaluation_generators"] = ["declared_test_fixture"]
    write(variant, debug_path)
    assert load_config(debug_path).proposal["data"]["first_evaluation_generators"] == ["declared_test_fixture"]

    invalid_references = []
    variant = copy.deepcopy(ref_raw); variant["proposal"]["verification"]["fp32_atol"] = 5e-6; invalid_references.append(variant)
    variant = copy.deepcopy(ref_raw); variant["proposal"]["verification"]["mve_t12_cross_batch"]["atol"] = 1e-4; invalid_references.append(variant)
    variant = copy.deepcopy(ref_raw); variant["review"]["numerical_amendments"]["N1"]["scope"] = "all_tests"; invalid_references.append(variant)
    variant = copy.deepcopy(ref_raw); variant["unexpected"] = 1; invalid_references.append(variant)
    variant = copy.deepcopy(ref_raw); variant["proposal"]["mve"]["transformer"]["heads"] = 4; invalid_references.append(variant)
    variant = copy.deepcopy(ref_raw); variant["proposal"]["mve"]["input"]["dtype"] = "float64"; invalid_references.append(variant)
    variant = copy.deepcopy(ref_raw); variant["proposal"]["initialization"]["seed"] = False; invalid_references.append(variant)
    variant = copy.deepcopy(ref_raw); variant["review"]["training_authorized"] = True; invalid_references.append(variant)
    variant = copy.deepcopy(ref_raw); variant["review"]["authorized_implementation_scopes"] = ["mve", "vision_decoder"]; invalid_references.append(variant)
    variant = copy.deepcopy(ref_raw); variant["review"]["approved_decisions"] = ["A02"]; invalid_references.append(variant)
    variant = copy.deepcopy(ref_raw); variant["schema_version"] = 2; invalid_references.append(variant)
    variant = copy.deepcopy(ref_raw); variant["paper_explicit"]["vim_equation_10"] = "corrected_attention"; invalid_references.append(variant)
    variant = copy.deepcopy(ref_raw); del variant["proposal"]["mve"]["fusion"]; invalid_references.append(variant)
    for index, variant in enumerate(invalid_references):
        write(variant, base_path)
        with pytest.raises(ConfigError):
            load_config(base_path)
    write(ref_raw, base_path)

    invalid_overlays = []
    variant = copy.deepcopy(debug_raw); del variant["review"]["numerical_amendments"]; invalid_overlays.append(variant)
    variant = copy.deepcopy(debug_raw); variant["review"]["numerical_amendments"]["N1"]["conditions_verified"] = False; invalid_overlays.append(variant)
    variant = copy.deepcopy(debug_raw); variant["overrides"]["verification"] = {"fp32_atol": 5e-6}; invalid_overlays.append(variant)
    variant = copy.deepcopy(debug_raw); variant["overrides"]["mve"] = {"transformer": {"depth": 2}}; invalid_overlays.append(variant)
    variant = copy.deepcopy(debug_raw); variant["overrides"]["mve"] = {"bogus": 1}; invalid_overlays.append(variant)
    variant = copy.deepcopy(debug_raw); variant["overrides"]["runtime"]["amp"] = "false"; invalid_overlays.append(variant)
    variant = copy.deepcopy(debug_raw); variant["overrides"]["language"]["enabled"] = True; invalid_overlays.append(variant)
    variant = copy.deepcopy(debug_raw); variant["merge_policy"]["propagate_approval"] = True; invalid_overlays.append(variant)
    variant = copy.deepcopy(debug_raw); variant["base"] = "overlay.yaml"; invalid_overlays.append(variant)
    for variant in invalid_overlays:
        write(variant, debug_path)
        with pytest.raises(ConfigError):
            load_config(debug_path)
    variant = copy.deepcopy(debug_raw)
    variant["review"]["implementation_authorized"] = False
    variant["review"]["mve_approval"]["status"] = "PROPOSED"
    write(variant, debug_path)
    unapproved = load_config(debug_path)
    with pytest.raises(ConfigError, match="approval"):
        MVE.from_config(unapproved)
    debug_path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Duplicate"):
        load_config(debug_path)
    print("MVE-T14: strict map/list/null overlay, negative schema/scope/cycle checks, no inherited approval")
