import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mfvlr.vim import VisionInjectionDelta
from mfvlr_research.experiment import (RegionPrompts, ResidualMaskFusion, build_model, build_optimizer, compare_reports,
                                       enable_gradient_checkpointing, evaluate_split, final_iou, load_samples,
                                       make_schedule, neutral_tokens, reconstruct_from, region_prompt, run_experiment,
                                       soft_dice_loss, source_hashes, validate_config)
from mfvlr_research.pilot_skip import SkipMFVLR
from mfvlr_research.pretrained_encoder import ENCODERS, ROOT
from mfvlr_research.text_patch import CLASS_PROMPTS, CLIP_EMBEDDINGS, class_embeddings, patch_classes, text_patch_loss


MODEL = {'variant': 'skip_resize_conv_v2', 'resize_mode': 'nearest', 'language_visual_gradients': 'stop_all',
         'residual_gradient': 'detach_reconstruction', 'skip_connections': 'concat',
         'reconstruction_activation': 'identity', 'mask_background_weight': 1., 'mask_foreground_weight': 4.,
         'mask_loss_weight': 1., 'mask_residual': 'none'}
IMAGE_WEIGHTS = {'fd': .1, 'ar': 1., 'fl': 1., 'lr': 0., 'cmc': 0., 'kl': 0.}


def make_config(**changes):
    config = {'schema': 'mfvlr-experiment-v1', 'name': 'test', 'inputs': 'inputs', 'train_limit': None, 'val_limit': None,
              'model': dict(MODEL), 'loss_weights': dict(IMAGE_WEIGHTS),
              'optimizer': {'image_lr': 1e-4, 'language_lr': 1e-5, 'weight_decay': 0.}, 'scheduler': None,
              'training': {'epochs': 1, 'effective_batch': 4, 'microbatch': 2, 'checkpoint_every': 1, 'threads': 2,
                           'gradient_checkpointing': False, 'gpu_memory_fraction': None}}
    config.update(changes)
    return config


def write_npz(path, count, offset=0):
    generator = np.random.default_rng(offset)
    images = generator.integers(0, 256, (count, 3, 224, 224), dtype=np.uint8)
    masks = np.zeros((count, 224, 224), dtype=np.uint8)
    masks[1::2, 50:150, 50:150] = 1
    labels = np.array([0, 1] * (count // 2))
    ids = np.array([f's{offset}_{i}' for i in range(count)])
    np.savez(path, images=images, masks=masks, labels=labels, sample_ids=ids)


@pytest.fixture
def project(tmp_path):
    (tmp_path / 'inputs').mkdir()
    write_npz(tmp_path / 'inputs/train.npz', 8)
    write_npz(tmp_path / 'inputs/val.npz', 4, offset=100)
    return tmp_path


def image_batch():
    return torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize('change', [
    lambda c: c.update(unknown=1),
    lambda c: c['training'].update(learning_rate=1),
    lambda c: c['model'].update(mask_residual='maybe'),
    lambda c: c['model'].pop('mask_residual'),
    lambda c: c['training'].update(microbatch=1),
    lambda c: c['training'].update(microbatch=3),
    lambda c: c['optimizer'].update(image_lr=0),
    lambda c: c.update(scheduler={'step_size': 0, 'gamma': .1}),
    lambda c: c['training'].update(gpu_memory_fraction=1.5),
    lambda c: c['training'].update(gpu_memory_fraction=0),
    lambda c: c['training'].pop('gpu_memory_fraction'),
])
def test_invalid_configs_are_rejected(change):
    config = make_config()
    change(config)
    with pytest.raises(ValueError):
        validate_config(config)


def test_seed_zero_keeps_accepted_initialization_and_other_seeds_differ():
    settings = {k: v for k, v in MODEL.items() if k != 'mask_residual'}
    reference = SkipMFVLR(settings, IMAGE_WEIGHTS).state_dict()
    seeded = build_model(make_config(), 0).state_dict()
    assert set(seeded) == set(reference)
    assert all(torch.equal(seeded[name], reference[name]) for name in reference)
    other = build_model(make_config(), 1).state_dict()
    for name in ('mve.shared_encoder.projection.weight', 'decoder.trunk.stages.0.conv1.weight', 'heads.detector.weight'):
        assert not torch.equal(other[name], reference[name])


def test_seed_changes_language_initialization():
    config = make_config(loss_weights={'fd': .1, 'ar': 1., 'fl': 1., 'lr': 1e-4, 'cmc': .01, 'kl': .01})
    config['model'].update(vim_residual='double', language_normalization='pre_norm')
    first, second = build_model(config, 1), build_model(config, 2)
    assert not torch.equal(first.flt.vocabulary.weight, second.flt.vocabulary.weight)


def language_config(vim_tokens):
    config = make_config(loss_weights={'fd': .1, 'ar': 1., 'fl': 1., 'lr': 1e-4, 'cmc': .01, 'kl': .01})
    config['model'].update(language_visual_gradients='coupled', vim_residual='single', language_normalization='pre_norm',
                           vim_tokens=vim_tokens)
    return config


@pytest.mark.parametrize('change', [
    lambda c: c['model'].update(vim_tokens='some'),
    lambda c: c['model'].update(vim_residual='double'),
    lambda c: c['model'].update(language_normalization='baseline'),
    lambda c: c.update(loss_weights=dict(IMAGE_WEIGHTS)),
])
def test_invalid_token_vim_configs_are_rejected(change):
    config = language_config('all')
    change(config)
    with pytest.raises(ValueError):
        validate_config(config)


def test_token_vim_is_word_dependent_and_keeps_paired_initialization():
    single = build_model(language_config('cls'), 1).state_dict()
    tokens = build_model(language_config('all'), 1)
    assert set(tokens.state_dict()) == set(single)
    assert all(torch.equal(value, single[name]) for name, value in tokens.state_dict().items())
    vim = tokens.flt.encoder.blocks[0].vim
    language = torch.randn(2, 308, 512, generator=torch.Generator().manual_seed(1))
    visual = torch.randn(2, 197, 512, generator=torch.Generator().manual_seed(2))
    delta = vim(language, visual)
    assert not torch.equal(delta[0, 0], delta[0, 1])
    # Samples must not leak through attention, and Q/K must now learn.
    changed = visual.clone()
    changed[1] += 1
    assert torch.equal(vim(language, changed)[0], delta[0])
    delta.square().sum().backward()
    assert vim.q.weight.grad.abs().sum() > 0 and vim.k.weight.grad.abs().sum() > 0


def test_class_token_config_keeps_its_vim_and_language_route_runs():
    config = language_config('all')
    model = build_model(config, 1).train()
    image = image_batch()
    ids = torch.randint(0, 49408, (2, 308), generator=torch.Generator().manual_seed(4))
    ids[:, 40] = 49407
    output = model.with_language(image, ids, sample_ids=('a', 'b'))
    assert output.language.global_language.shape == (2, 1, 512) and output.language.logits.shape == (2, 308, 49408)
    assert output.predicted_language.shape == (2, 1, 512)
    targets = SimpleNamespace(sample_ids=('a', 'b'), token_ids=ids, labels=torch.tensor([0, 1]),
                              masks=torch.zeros(2, 224, 224, dtype=torch.int64), sample_kinds=('real', 'local'))
    model.compute_objectives(output, targets).total.backward()
    assert model.mve.shared_encoder.blocks[0].fc1.weight.grad.abs().sum() > 0
    assert model.flt.decoder.blocks[6].vim.q.weight.grad.abs().sum() > 0
    assert type(build_model(language_config('cls'), 1).flt.encoder.blocks[0].vim) is VisionInjectionDelta


def test_fusion_starts_at_baseline_and_cannot_change_reconstruction():
    config = make_config()
    baseline = build_model(config, 0).eval()
    config['model']['mask_residual'] = 'concat'
    fused = build_model(config, 0).eval()
    assert all(torch.equal(p, fused.state_dict()[n]) for n, p in baseline.state_dict().items())
    image = image_batch()
    with torch.no_grad():
        expected, actual = baseline.visual_only(image), fused.visual_only(image)
    assert torch.equal(expected.decoded.mask_logits, actual.decoded.mask_logits)
    # The zero output layer must still be trainable, and FL must not reach the AD head.
    output = fused.visual_only(image)
    output.decoded.mask_logits.sum().backward()
    assert fused.mask_fusion.out.weight.grad.abs().sum() > 0
    assert fused.decoder.appearance_head.weight.grad is None


def test_fusion_detaches_residual_and_is_scale_invariant():
    fusion = ResidualMaskFusion(0)
    fusion.out.weight.data.fill_(.1)
    features = torch.zeros(1, 64, 224, 224)
    residual = torch.rand(1, 3, 224, 224, generator=torch.Generator().manual_seed(3)).requires_grad_()
    output = fusion(features, residual)
    output.sum().backward()
    assert residual.grad is None
    with torch.no_grad():
        torch.testing.assert_close(fusion(features, residual * 3), output, atol=1e-4, rtol=1e-4)


def test_gradient_checkpointing_preserves_training_gradients():
    config = make_config()
    plain, checkpointed = build_model(config, 0), build_model(config, 0)
    enable_gradient_checkpointing(checkpointed)
    image = image_batch()
    for model in (plain, checkpointed):
        model.train()
        model.visual_only(image).detection_logits.sum().backward()
    assert torch.equal(plain.mve.shared_encoder.cnn.stages[0][0].weight.grad,
                       checkpointed.mve.shared_encoder.cnn.stages[0][0].weight.grad)


def test_schedule_is_seeded_complete_and_rejects_remainders():
    schedule = make_schedule(8, 4, 2, 5)
    assert schedule == make_schedule(8, 4, 2, 5) and schedule != make_schedule(8, 4, 2, 6)
    assert sorted(i for batch in schedule[:2] for i in batch) == list(range(8))
    with pytest.raises(ValueError):
        make_schedule(10, 4, 1, 0)


def test_load_samples_limit_keeps_classes_balanced(project):
    samples = load_samples(project / 'inputs/train.npz', 4)
    assert [s['label'] for s in samples] == [0, 1, 0, 1]
    assert samples[0]['image'].dtype == torch.float32 and samples[0]['image'].max() <= 1


def test_evaluation_matches_hand_computed_mask_metrics():
    class Perfect(torch.nn.Module):
        def visual_only(self, image):
            from types import SimpleNamespace
            masks = torch.zeros(len(image), 224, 224, dtype=torch.int64)
            masks[1, :112] = 1
            logits = torch.stack((1 - masks, masks), dim=1).float() * 20
            decoded = SimpleNamespace(reconstruction=image, mask_logits=logits)
            return SimpleNamespace(detection_logits=torch.tensor([[9., 0.], [0., 9.]]), decoded=decoded)

    target = torch.zeros(224, 224, dtype=torch.int64)
    target[:56] = 1
    samples = [dict(image=torch.zeros(3, 224, 224), mask=torch.zeros(224, 224, dtype=torch.int64), label=0, sample_id='a', kind='real'),
               dict(image=torch.zeros(3, 224, 224), mask=target, label=1, sample_id='b', kind='local')]
    result = evaluate_split(Perfect(), samples, 'cpu', torch.tensor([1., 4.]), 2)
    assert result['fake_foreground_iou_macro'] == pytest.approx(.5)
    assert result['pixel_precision'] == pytest.approx(.5) and result['pixel_recall'] == 1.
    assert result['real_false_positive_area'] == 0. and result['auc'] == 1.
    # Foreground IoU .5 over the set; background 2*50176-56*224... counted from the same pixels.
    tn = 224 * 224 + 112 * 224
    assert result['pixel_miou_global'] == pytest.approx((.5 + tn / (tn + 56 * 224)) / 2)


def test_resume_reproduces_an_uninterrupted_run(project):
    config = make_config()
    whole = run_experiment(config, 1, project / 'whole', root=project, device='cpu')
    run_experiment(config, 1, project / 'split', root=project, device='cpu', max_updates=1)
    assert json.loads((project / 'split/status.json').read_text())['status'] == 'COMPLETED'
    # A paused run keeps its state and continues exactly where it stopped.
    paused = run_experiment(config, 1, project / 'paused', root=project, device='cpu', budget_seconds=0)
    assert paused['status'] == 'PAUSED_TIME_BUDGET' and (project / 'paused/last.pt').exists()
    resumed = run_experiment(config, 1, project / 'paused', root=project, device='cpu')
    assert resumed['status'] == 'COMPLETED' and not (project / 'paused/last.pt').exists()
    first = torch.load(project / 'whole/final.pt', weights_only=True)['model']
    second = torch.load(project / 'paused/final.pt', weights_only=True)['model']
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert resumed['evaluations'][-1]['val'] == whole['evaluations'][-1]['val']
    steps = [json.loads(line)['step'] for line in (project / 'paused/history.jsonl').read_text().splitlines()]
    assert steps == [1, 2]


def augmentation_sample():
    # The red channel draws the mask, so any geometric mismatch between image and mask shows up.
    mask = torch.zeros(224, 224, dtype=torch.int64)
    mask[40:90, 30:120] = 1
    image = torch.full((3, 224, 224), 64.).div_(255)
    image[0] = mask.float()
    return dict(image=image, mask=mask, label=1, sample_id='s', kind='local')


@pytest.mark.parametrize('mode', ['flip', 'deepfakebench'])
def test_augmentation_is_repeatable_and_leaves_the_source_unchanged(mode):
    from mfvlr_research.augmentation import augment_sample
    sample = augmentation_sample()
    before = sample['image'].clone(), sample['mask'].clone()
    first, second = (augment_sample(sample, mode, 1, 5, 3) for _ in range(2))
    assert torch.equal(first['image'], second['image']) and torch.equal(first['mask'], second['mask'])
    assert torch.equal(sample['image'], before[0]) and torch.equal(sample['mask'], before[1])
    assert first['image'].dtype == torch.float32 and first['image'].shape == (3, 224, 224)
    assert first['mask'].dtype == torch.int64 and set(first['mask'].unique().tolist()) <= {0, 1}
    assert 0 <= first['image'].min() and first['image'].max() <= 1
    assert first['label'] == 1 and first['sample_id'] == 's'
    draws = [augment_sample(sample, mode, 1, update, 3)['mask'] for update in range(40)]
    assert any(not torch.equal(draw, draws[0]) for draw in draws)


def test_flip_moves_image_and_mask_together():
    from mfvlr_research.augmentation import augment_sample
    sample = augmentation_sample()
    outcomes = [augment_sample(sample, 'flip', 0, update, 0) for update in range(20)]
    flipped = [o for o in outcomes if not torch.equal(o['mask'], sample['mask'])]
    assert 0 < len(flipped) < len(outcomes)
    for outcome in flipped:
        assert torch.equal(outcome['mask'], sample['mask'].flip(1)) and torch.equal(outcome['image'], sample['image'].flip(2))


def test_benchmark_rotation_keeps_image_and_mask_aligned():
    from mfvlr_research.augmentation import augment_sample
    sample = augmentation_sample()
    for update in range(40):
        outcome = augment_sample(sample, 'deepfakebench', 0, update, 0)
        hard = outcome['image'][0] > .6
        # Blur, color and JPEG soften edges, so compare the two regions rather than each pixel.
        assert (hard & outcome['mask'].bool()).sum() / (hard | outcome['mask'].bool()).sum() > .8


def test_invalid_augmentation_is_rejected():
    config = make_config()
    config['training']['augmentation'] = 'mixup'
    with pytest.raises(ValueError):
        validate_config(config)


def test_explicit_none_matches_the_default_and_augmentation_changes_training(project):
    default = make_config()
    none, flip = copy.deepcopy(default), copy.deepcopy(default)
    none['training']['augmentation'] = 'none'
    flip['training']['augmentation'] = 'flip'
    states = []
    for name, config in (('default', default), ('none', none), ('flip', flip)):
        run_experiment(config, 1, project / name, root=project, device='cpu')
        states.append(torch.load(project / name / 'final.pt', weights_only=True)['model'])
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
    assert any(not torch.equal(states[0][key], states[2][key]) for key in states[0])


def test_resume_reproduces_augmented_training(project):
    config = make_config()
    config['training']['augmentation'] = 'deepfakebench'
    run_experiment(config, 1, project / 'whole', root=project, device='cpu')
    paused = run_experiment(config, 1, project / 'paused', root=project, device='cpu', budget_seconds=0)
    assert paused['status'] == 'PAUSED_TIME_BUDGET'
    run_experiment(config, 1, project / 'paused', root=project, device='cpu')
    first = torch.load(project / 'whole/final.pt', weights_only=True)['model']
    second = torch.load(project / 'paused/final.pt', weights_only=True)['model']
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_soft_dice_matches_hand_values_and_ignores_real_images():
    masks = torch.zeros((2, 224, 224), dtype=torch.long)
    masks[:, :112] = 1
    perfect = torch.zeros((2, 2, 224, 224))
    perfect[:, 1] = torch.where(masks.bool(), 40., -40.)
    perfect[:, 0] = -perfect[:, 1]
    fake = torch.tensor([1, 1])
    assert soft_dice_loss(perfect, masks, fake).item() == pytest.approx(0, abs=1e-5)
    assert soft_dice_loss(-perfect, masks, fake).item() == pytest.approx(1, abs=1e-3)
    real = torch.tensor([0, 0])
    assert soft_dice_loss(-perfect, masks, real).item() == 0
    mixed = -perfect.clone()
    mixed[0] = perfect[0]
    assert soft_dice_loss(mixed, masks, torch.tensor([1, 0])).item() == pytest.approx(0, abs=1e-5)


def test_invalid_dice_weight_is_rejected():
    for bad in (-1., True, float('nan')):
        with pytest.raises(ValueError):
            validate_config(make_config(model={**MODEL, 'mask_dice_weight': bad}))


def test_zero_dice_matches_the_default_and_dice_changes_training(project):
    default = run_experiment(make_config(), 0, project / 'default', root=project, device='cpu')
    zero = run_experiment(make_config(model={**MODEL, 'mask_dice_weight': 0.}), 0, project / 'zero', root=project, device='cpu')
    dice = run_experiment(make_config(model={**MODEL, 'mask_dice_weight': 1.}), 0, project / 'dice', root=project, device='cpu')
    weights = [torch.load(project / name / 'final.pt', weights_only=True)['model'] for name in ('default', 'zero', 'dice')]
    assert all(torch.equal(weights[0][key], weights[1][key]) for key in weights[0])
    assert any(not torch.equal(weights[0][key], weights[2][key]) for key in weights[0])
    assert default['evaluations'] == zero['evaluations'] and dice['status'] == 'COMPLETED'


needs_resnet = pytest.mark.skipif(not (ROOT / ENCODERS['resnet50_imagenet']).exists(), reason='ImageNet weights not downloaded')


def resnet_config(**optimizer):
    return make_config(model={**MODEL, 'encoder': 'resnet50_imagenet'},
                       optimizer={'image_lr': 1e-4, 'language_lr': 1e-5, 'weight_decay': 0., **optimizer})


def test_invalid_encoder_settings_are_rejected():
    with pytest.raises(ValueError):
        validate_config(make_config(model={**MODEL, 'encoder': 'vgg'}))
    with pytest.raises(ValueError):
        validate_config(make_config(optimizer={'image_lr': 1e-4, 'language_lr': 1e-5, 'weight_decay': 0., 'backbone_lr': 1e-5}))
    with pytest.raises(ValueError):
        validate_config(resnet_config(backbone_lr=0))


def test_explicit_scratch_encoder_matches_the_default():
    default = build_model(make_config(), 1).state_dict()
    explicit = build_model(make_config(model={**MODEL, 'encoder': 'scratch'}), 1).state_dict()
    assert set(default) == set(explicit) and all(torch.equal(default[k], explicit[k]) for k in default)


@needs_resnet
def test_resnet_encoder_keeps_interfaces_imagenet_weights_and_paired_initialization():
    validate_config(resnet_config(backbone_lr=1e-5))
    model, scratch = build_model(resnet_config(), 1), build_model(make_config(), 1)
    imagenet = torch.load(ROOT / ENCODERS['resnet50_imagenet'], weights_only=True)
    cnn = model.mve.shared_encoder.cnn
    assert torch.equal(cnn.stages[1][0].weight, imagenet['conv1.weight'])
    assert torch.equal(cnn.stages[4][0].conv1.weight, imagenet['layer3.0.conv1.weight'])
    # Batch norm is frozen: statistics are buffers, nothing in it is trainable.
    assert not any('bn' in name for name, _ in cnn.named_parameters())
    rest = {k: v for k, v in scratch.state_dict().items() if '.cnn.' not in k}
    assert all(torch.equal(model.state_dict()[k], v) for k, v in rest.items())
    local = model.mve.shared_encoder.encode_local(image_batch())
    assert local.local.shape == (2, 1024, 14, 14)
    assert [tuple(s.shape[1:]) for s in local.skips] == [(64, 224, 224), (128, 112, 112), (256, 56, 56), (512, 28, 28)]
    model.train()
    assert torch.isfinite(model.visual_only(image_batch()).decoded.mask_logits).all()


@needs_resnet
def test_backbone_rate_applies_only_to_pretrained_weights():
    model = build_model(resnet_config(backbone_lr=1e-5), 0)
    optimizer, _ = build_optimizer(model, resnet_config(backbone_lr=1e-5))
    owners = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(owners) == len(set(owners)) == len(list(model.parameters()))
    backbone = {id(p) for p in model.mve.shared_encoder.cnn.pretrained_parameters()}
    rates = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
    assert all(rates[i] == 1e-5 for i in backbone)
    assert rates[id(model.mve.shared_encoder.cnn.stages[0][0].weight)] == 1e-4


@needs_resnet
def test_gradient_checkpointing_preserves_resnet_gradients():
    plain, checkpointed = build_model(resnet_config(), 0), build_model(resnet_config(), 0)
    enable_gradient_checkpointing(checkpointed)
    for model in (plain, checkpointed):
        model.train()
        model.visual_only(image_batch()).decoded.mask_logits.sum().backward()
    for get in (lambda m: m.mve.shared_encoder.cnn.stages[1][0].weight.grad,
                lambda m: m.mve.shared_encoder.cnn.stages[0][0].weight.grad):
        assert torch.allclose(get(plain), get(checkpointed), rtol=1e-4, atol=1e-6)


def test_interrupted_steps_are_dropped_from_history(project):
    config = make_config()
    config['training']['epochs'] = 2
    run_experiment(config, 0, project / 'run', root=project, device='cpu', budget_seconds=None, max_updates=None)
    folder = project / 'partial'
    run_experiment(config, 0, folder, root=project, device='cpu', budget_seconds=0)
    with (folder / 'history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(step=99, losses={})) + '\n')
    run_experiment(config, 0, folder, root=project, device='cpu')
    assert [json.loads(line)['step'] for line in (folder / 'history.jsonl').read_text().splitlines()] == [1, 2, 3, 4]


def test_resume_rejects_a_changed_recipe(project):
    config = make_config()
    run_experiment(config, 0, project / 'run', root=project, device='cpu', budget_seconds=0)
    changed = copy.deepcopy(config)
    changed['optimizer']['image_lr'] = 2e-4
    with pytest.raises(ValueError, match='Resume'):
        run_experiment(changed, 0, project / 'run', root=project, device='cpu')
    status = json.loads((project / 'run/status.json').read_text())
    assert status['status'] == 'FAILED' and 'Resume' in status['error'] and status['time_utc']


@pytest.mark.parametrize('name', ['val.npz', '../train.npz', 'train.npy', 5])
def test_invalid_train_file_is_rejected(name):
    with pytest.raises(ValueError):
        validate_config(make_config(train_file=name))


def test_train_file_selects_the_training_split(project):
    import hashlib
    import shutil
    shutil.copy(project / 'inputs/train.npz', project / 'inputs/train_copy.npz')
    default = run_experiment(make_config(), 0, project / 'default', root=project, device='cpu')
    explicit = run_experiment(make_config(train_file='train.npz'), 0, project / 'explicit', root=project, device='cpu')
    copied = run_experiment(make_config(train_file='train_copy.npz'), 0, project / 'copied', root=project, device='cpu')
    assert default['evaluations'] == explicit['evaluations'] == copied['evaluations']
    write_npz(project / 'inputs/train_other.npz', 8, offset=50)
    other = run_experiment(make_config(train_file='train_other.npz'), 0, project / 'other', root=project, device='cpu')
    assert other['data']['train'] == hashlib.sha256((project / 'inputs/train_other.npz').read_bytes()).hexdigest()
    assert other['evaluations'][-1] != default['evaluations'][-1]
    val = default['evaluations'][-1]['val']
    per_sample = val['per_sample']
    assert len(per_sample['score']) == len(per_sample['value']) == val['count'] == 4
    real = [v for v, label in zip(per_sample['value'], [0, 1, 0, 1]) if label == 0]
    assert abs(np.mean(real) - val['real_false_positive_area']) < 1e-5


def region_config(**model):
    config = language_config('cls')
    config['model'].update(language_prompt='region', language_encoder_input='neutral', **model)
    return config


@pytest.mark.parametrize('change', [
    lambda c: c['model'].update(language_prompt='caption'),
    lambda c: c['model'].update(language_encoder_input='masked'),
    lambda c: c.update(loss_weights=dict(IMAGE_WEIGHTS)),
])
def test_invalid_region_language_configs_are_rejected(change):
    config = region_config()
    change(config)
    with pytest.raises(ValueError):
        validate_config(config)


def test_region_prompt_describes_attribute_place_and_size():
    mask = torch.zeros(224, 224, dtype=torch.int64)
    assert region_prompt('CelebA_HQ/1.jpg', 0, mask)[1] == 'No attribute of this face is edited'
    mask[:30, :] = 1  # 13% of the image, all in the upper third
    assert region_prompt('Diffae/Diffae_Bangs_00012.png', 1, mask)[1:] == (
        'The bangs above the forehead of this face is edited', 'Most edited pixels are in the upper part of the image',
        'The edited area is medium')
    mask[:] = 0
    mask[200:, :20] = 1
    assert region_prompt('Diffae/Diffae_smile_00012.png', 1, mask)[2:] == (
        'Most edited pixels are in the lower part of the image', 'The edited area is small')
    with pytest.raises(ValueError):
        region_prompt('Diffae/Diffae_Age_00012.png', 1, mask)


def test_region_prompt_table_covers_every_text_and_tokenizes():
    prompts = RegionPrompts()
    assert prompts.ids.shape == (1 + 2 * 3 * 3, 308) and len(prompts.rows) == prompts.ids.shape[0]
    assert len({tuple(row.tolist()) for row in prompts.ids}) == prompts.ids.shape[0]
    masks = torch.zeros(2, 224, 224, dtype=torch.int64)
    masks[1, 100:120, :] = 1
    batch = dict(sample_id=['CelebA_HQ/1.jpg', 'Diffae/Diffae_smile_00001.png'], label=torch.tensor([0, 1]), mask=masks)
    ids = prompts.lookup(batch)
    assert torch.equal(ids[0], prompts.ids[0]) and not torch.equal(ids[1], prompts.ids[0])


def test_neutral_encoder_hides_the_target_text_from_the_encoder():
    model = build_model(region_config(), 1).eval()
    image = image_batch()
    prompts = RegionPrompts().ids
    neutral = neutral_tokens()
    with torch.no_grad():
        # Same ids on both sides is exactly the ordinary FLT forward.
        visual = model.visual_only(image).visual.fused_global
        ordinary = model.flt(prompts[:2], visual, return_logits=True)
        same = reconstruct_from(model.flt, prompts[:2], prompts[:2], visual)
        assert torch.equal(ordinary.logits, same.logits)
        model.encoder_ids = neutral
        first = model.with_language(image, prompts[[1, 2]], sample_ids=('a', 'b'))
        second = model.with_language(image, prompts[[3, 4]], sample_ids=('a', 'b'))
    # The encoder output depends only on the image, while the decoder is still teacher-forced on the target.
    assert torch.equal(first.language.global_language, second.language.global_language)
    assert not torch.equal(first.language.logits, second.language.logits)


def test_region_language_run_trains_and_the_image_route_ignores_text(tmp_path):
    (tmp_path / 'inputs').mkdir()
    for name, count, offset in (('train', 8, 0), ('val', 4, 100)):
        write_npz(tmp_path / f'inputs/{name}.npz', count, offset)
        with np.load(tmp_path / f'inputs/{name}.npz') as data:
            arrays = dict(data)
        arrays['sample_ids'] = np.array([f'CelebA_HQ/{offset + i}.jpg' if label == 0 else
                                         f'Diffae/Diffae_{"Bangs" if i % 4 == 1 else "smile"}_{offset + i:05d}.png'
                                         for i, label in enumerate(arrays['labels'])])
        np.savez(tmp_path / f'inputs/{name}.npz', **arrays)
    report = run_experiment(region_config(), 0, tmp_path / 'run', root=tmp_path, device='cpu')
    assert report['status'] == 'COMPLETED' and 'language_probe' in report
    losses = [json.loads(line)['losses'] for line in (tmp_path / 'run/history.jsonl').read_text().splitlines()]
    assert all(np.isfinite(l['lr']) and l['lr'] > 0 for l in losses)


def test_samples_keep_values_of_a_whole_array_conversion(project):
    samples = load_samples(project / 'inputs/train.npz')
    with np.load(project / 'inputs/train.npz') as data:
        images = torch.from_numpy(data['images']).float().div_(255)
        masks = torch.from_numpy(data['masks']).long()
    assert len(samples) == 8 and [s['label'] for s in samples[2:4]] == [0, 1]
    assert all(torch.equal(s['image'], images[i]) and torch.equal(s['mask'], masks[i]) for i, s in enumerate(samples))
    assert samples[-1]['sample_id'] == 's0_7' and samples[-1]['kind'] == 'local'


def test_completed_output_is_never_overwritten(project):
    config = make_config()
    run_experiment(config, 0, project / 'run', root=project, device='cpu')
    with pytest.raises(ValueError, match='completed'):
        run_experiment(config, 0, project / 'run', root=project, device='cpu')


def test_source_hash_keys_are_portable():
    import mfvlr_research.experiment as module
    from pathlib import Path
    root = Path(module.__file__).resolve().parents[2]
    hashes = source_hashes(root)
    assert 'src/mfvlr_research/experiment.py' in hashes and 'src/mfvlr/mve.py' in hashes
    assert not any('\\' in name for name in hashes)


def test_primary_endpoint_and_paired_comparison():
    def report(*values):
        return {'evaluations': [{'epoch': i, 'val': {'fake_foreground_iou_macro': v}} for i, v in enumerate(values)]}
    assert final_iou(report(.9, .1, .2, .3, .4)) == pytest.approx(.3)
    with pytest.raises(ValueError):
        final_iou(report(.9))
    result = compare_reports({0: report(0, .2), 1: report(0, .3)}, {0: report(0, .3), 1: report(0, .25)})
    assert result['mean_difference'] == pytest.approx(.025) and result['candidate_better_in'] == 1
    assert result['paired_seeds'] == 2 and result['sample_sd'] > 0


def diffae_project(tmp_path):
    """Train/val NPZ whose ids carry GenFace names, so attribute-dependent targets can be built."""
    (tmp_path / 'inputs').mkdir()
    for name, count, offset in (('train', 8, 0), ('val', 4, 100)):
        write_npz(tmp_path / f'inputs/{name}.npz', count, offset)
        with np.load(tmp_path / f'inputs/{name}.npz') as data:
            arrays = dict(data)
        arrays['sample_ids'] = np.array([f'CelebA_HQ/{offset + i}.jpg' if label == 0 else
                                         f'Diffae/Diffae_{"Bangs" if i % 4 == 1 else "smile"}_{offset + i:05d}.png'
                                         for i, label in enumerate(arrays['labels'])])
        np.savez(tmp_path / f'inputs/{name}.npz', **arrays)
    return tmp_path


def text_patch_config(mode='clip', weight=1.):
    return make_config(model={**MODEL, 'text_patch': mode, 'text_patch_weight': weight})


@pytest.mark.parametrize('changes', [
    {'text_patch': 'clip'}, {'text_patch': 'random', 'text_patch_weight': 0.}, {'text_patch_weight': 1.},
    {'text_patch': 'bert', 'text_patch_weight': 1.}, {'text_patch': 'clip', 'text_patch_weight': -1.},
])
def test_invalid_text_patch_configs_are_rejected(changes):
    with pytest.raises(ValueError):
        validate_config(make_config(model={**MODEL, **changes}))


def test_class_embeddings_match_the_prompts_and_arms_share_their_geometry_only_in_size():
    stored = json.loads(CLIP_EMBEDDINGS.read_text(encoding='utf-8'))
    assert tuple(stored['prompts']) == CLASS_PROMPTS
    clip, random = class_embeddings('clip'), class_embeddings('random')
    assert clip.shape == random.shape == (3, 768)
    assert torch.allclose(clip.norm(dim=1), torch.ones(3)) and torch.allclose(random.norm(dim=1), torch.ones(3))
    assert torch.equal(random, class_embeddings('random'))
    # CLIP sentences are close to each other; independent random vectors are nearly orthogonal.
    off_diagonal = ~torch.eye(3, dtype=torch.bool)
    assert (clip @ clip.T)[off_diagonal].min() > .5 and (random @ random.T)[off_diagonal].abs().max() < .2


def test_patch_classes_follow_the_diffae_attribute():
    names = ('CelebA_HQ/1.jpg', 'Diffae/Diffae_Bangs_00001.png', 'Diffae/Diffae_smile_00002.png', 'FFHQ/3.jpg')
    assert patch_classes(names, torch.tensor([0, 1, 1, 0])).tolist() == [0, 1, 2, 0]
    with pytest.raises(ValueError):
        patch_classes(('Diffae/Diffae_eyes_1.png',), torch.tensor([1]))


def test_text_patch_loss_prefers_the_right_class_and_place():
    masks = torch.zeros((2, 224, 224), dtype=torch.long)
    masks[1, :112] = 1
    labels, names = torch.tensor([0, 1]), ('CelebA_HQ/1.jpg', 'Diffae/Diffae_smile_00001.png')
    right = torch.full((2, 3, 14, 14), -20.)
    right[0, 0] = right[1, 0, 7:] = 20.
    right[1, 2, :7] = 20.
    wrong_class = right.clone()
    wrong_class[1, 1:] = wrong_class[1, 1:].flip(0)
    assert text_patch_loss(right, masks, labels, names).item() == pytest.approx(0, abs=1e-4)
    assert text_patch_loss(wrong_class, masks, labels, names).item() > 5
    # The edited map alone is right for the wrong class, so its Dice part is still zero.
    assert text_patch_loss(wrong_class, masks, labels, names).item() < text_patch_loss(-right, masks, labels, names).item()


@pytest.mark.parametrize('mode', ['clip', 'random'])
def test_text_patch_starts_at_baseline_and_learns(mode):
    baseline = build_model(make_config(), 0).eval()
    patched = build_model(text_patch_config(mode), 0).eval()
    assert all(torch.equal(p, patched.state_dict()[n]) for n, p in baseline.state_dict().items())
    image = image_batch()
    with torch.no_grad():
        assert torch.equal(baseline.visual_only(image).decoded.mask_logits, patched.visual_only(image).decoded.mask_logits)
    optimizer, _ = build_optimizer(patched, text_patch_config(mode))
    image_group = {id(p) for p in optimizer.param_groups[0]['params']}
    assert all(id(p) in image_group for p in patched.text_patch.parameters())
    patched.train()
    output = patched.visual_only(image)
    output.decoded.mask_logits[:, 1].sum().backward()
    assert patched.text_patch.gain.grad.abs() > 0


def test_text_patch_arms_differ_only_in_the_class_vectors():
    clip, random = (build_model(text_patch_config(mode), 3) for mode in ('clip', 'random'))
    differing = [n for n, p in clip.state_dict().items() if not torch.equal(p, random.state_dict()[n])]
    assert differing == ['text_patch.text']


def test_text_patch_run_trains_with_a_finite_extra_loss(tmp_path):
    project = diffae_project(tmp_path)
    base = run_experiment(make_config(), 0, project / 'base', root=project, device='cpu')
    patched = run_experiment(text_patch_config('clip'), 0, project / 'clip', root=project, device='cpu')
    assert base['status'] == patched['status'] == 'COMPLETED'
    lines = lambda name: [json.loads(line)['losses'] for line in (project / name / 'history.jsonl').read_text().splitlines()]
    assert all(np.isfinite(p['fl']) and p['fl'] > b['fl'] for b, p in zip(lines('base'), lines('clip')))


def write_extracted(root, count=10, truncated=()):
    """A tiny extracted genface50k folder: real JPEGs with empty masks, Diffae PNGs with square masks."""
    import csv, io, subprocess
    from PIL import Image
    rng = np.random.default_rng(7)
    (root / 'img').mkdir(parents=True)
    rows = []
    for i in range(count):
        label = i % 2
        image = Image.fromarray(rng.integers(0, 256, (48 + i, 40, 3), dtype=np.uint8))
        mask = np.zeros((48 + i, 40), np.uint8)
        if label:
            mask[8:30, 6:20] = 255
        name = f'img/Diffae_smile_{i:05d}.png' if label else f'img/{i:05d}.jpg'
        if label:
            image.save(root / name)
        else:
            buffer = io.BytesIO()
            image.save(buffer, 'JPEG')
            data = buffer.getvalue()
            (root / name).write_bytes(data[:-200] if i in truncated else data)
        Image.fromarray(mask).save(root / f'{name}.mask.png')
        rows.append(dict(image=name, mask=f'{name}.mask.png', label=label, source='Diffae' if label else 'FFHQ', annotation=''))
    for split, part in (('train', rows[:count - 4]), ('val', rows[count - 4:])):
        with (root / f'{split}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(part)
    return root


def test_csv_samples_equal_the_packed_npz_and_skip_broken_images(tmp_path):
    import subprocess, sys
    from mfvlr_research.csv_data import CsvSamples
    root = write_extracted(tmp_path / 'ex', count=14, truncated=(2,))
    subprocess.run([sys.executable, str(ROOT / 'tools/pack_genface50k_npz.py'), '--extracted', str(root),
                    '--output', str(tmp_path / 'pkg')], check=True, capture_output=True)
    for split, multiple in (('train', 4), ('val', 1)):
        packed = load_samples(tmp_path / f'pkg/{split}.npz')
        direct = CsvSamples(root, split, multiple=multiple, workers=1)
        assert direct.ids == packed.ids and direct.labels == packed.labels
        for a, b in zip(direct, packed):
            assert torch.equal(a['image'], b['image']) and torch.equal(a['mask'], b['mask'])
    report = CsvSamples(root, 'train', multiple=4, workers=1).report
    # 10 train rows, one truncated JPEG skipped, then cut from 9 to a multiple of 4.
    assert report['count'] == 8 and [s['image'] for s in report['skipped']] == ['img/00002.jpg']
    assert len(report['trimmed']) == 1 and report['count'] % 4 == 0


def test_csv_check_is_cached_and_slices_follow_list_semantics(tmp_path, monkeypatch):
    from mfvlr_research import csv_data
    root = write_extracted(tmp_path / 'ex', count=12)
    first = csv_data.CsvSamples(root, 'train', multiple=4, workers=1)
    monkeypatch.setattr(csv_data, '_check', lambda job: pytest.fail('cached rows were rechecked'))
    again = csv_data.CsvSamples(root, 'train', multiple=4, workers=1)
    assert again.ids == first.ids and again.digest == first.digest
    picked = again[-3:]
    assert [s['sample_id'] for s in picked] == again.ids[-3:]
    assert again[-1]['sample_id'] == again.ids[-1]
    again.prefetch([0, 1])
    assert torch.equal(again[0]['image'], first[0]['image'])


def test_csv_and_npz_routes_train_identically(tmp_path):
    import shutil, subprocess, sys
    root = write_extracted(tmp_path / 'inputs', count=14)
    subprocess.run([sys.executable, str(ROOT / 'tools/pack_genface50k_npz.py'), '--extracted', str(root),
                    '--output', str(tmp_path / 'packed')], check=True, capture_output=True)
    npz = run_experiment(make_config(inputs='packed'), 0, tmp_path / 'npz', root=tmp_path, device='cpu')
    csv_run = run_experiment(make_config(inputs='inputs', train_file='train.csv'), 0, tmp_path / 'csv',
                             root=tmp_path, device='cpu')
    assert npz['status'] == csv_run['status'] == 'COMPLETED' and csv_run['data_checks']['train']['count'] == 8
    weights = [torch.load(tmp_path / name / 'final.pt', weights_only=True)['model'] for name in ('npz', 'csv')]
    assert all(torch.equal(weights[0][key], weights[1][key]) for key in weights[0])
    assert [e['val'] for e in npz['evaluations']] == [e['val'] for e in csv_run['evaluations']]


@pytest.mark.parametrize('name', ['train.csv', 'train_small.csv'])
def test_csv_train_files_are_accepted(name):
    validate_config(make_config(train_file=name))
