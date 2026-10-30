"""Isolated tests; small synthetic model exercises control flow, not A0 convergence."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from mfvlr_research.pilot_data import PilotFaces, freeze_candidates, sha256
from mfvlr_research.pilot_runtime import (auc, metrics, atomic_save, load_checkpoint,
                                        check_finite_tensors, gradient_norm)

ROOT = Path(__file__).resolve().parents[2]


def make_rows(folder, split, n=4):
    rows = []
    for i in range(n):
        label = i % 2
        name = f'Diffae_Bangs_{split}{i}.png' if label else f'{split}{i}.png'
        image, mask = folder / name, folder / ('mask_' + name)
        rgb = np.full((8, 8, 3), i * 20 + {'train': 1, 'val': 2, 'test': 3}[split], dtype=np.uint8)
        target = np.zeros((8, 8), dtype=np.uint8)
        if label:
            target[1:4, 1:4] = 255
        Image.fromarray(rgb).save(image)
        Image.fromarray(target).save(mask)
        rows.append(dict(sample_id=f'{split}/{name}', split=split, image=str(image), mask=str(mask),
            label=label, kind='local' if label else 'real', generator='Diffae' if label else None,
            image_sha256=sha256(image), mask_sha256=sha256(mask),
            decoded_rgb_sha256=hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()))
    return rows


def manifests(folder):
    rows = sum([make_rows(folder, s) for s in ('train', 'val', 'test')], [])
    source = folder / 'checked.jsonl'
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    destination = folder / 'frozen'
    freeze_candidates(source, destination)
    return source, destination


def test_freeze_dedup_cross_split_and_immutable(tmp_path):
    source, destination = manifests(tmp_path)
    rows = [json.loads(x) for x in source.read_text().splitlines()]
    duplicate = dict(rows[0], sample_id='other-name')
    rows.append(duplicate)
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert freeze_candidates(source, destination)['removed'] == ['other-name']
    rows[-1]['split'] = 'test'
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='crosses splits'):
        freeze_candidates(source, destination)
    rows.pop()
    rows[0]['image_sha256'] = '1' * 64
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='replace frozen'):
        freeze_candidates(source, destination)


def test_transforms_tokens_and_mutation(tmp_path):
    _, folder = manifests(tmp_path)
    data = PilotFaces(folder / 'train.jsonl', tokens=True)
    sample = data[1]
    assert sample['image'].shape == (3, 224, 224)
    assert sample['image'].dtype == torch.float32
    assert sample['mask'].shape == (224, 224)
    assert sample['mask'].dtype == torch.int64
    assert set(sample['mask'].unique().tolist()) == {0, 1}
    assert sample['token_ids'].shape == (308,)
    assert sample['token_ids'].dtype == torch.int64
    assert not data[0]['mask'].any()
    assert not torch.equal(data[0]['token_ids'], sample['token_ids'])
    Path(data.rows[1]['image']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed image'):
        data[1]


def test_metrics_ties_and_atomic_checkpoint(tmp_path):
    assert auc([0, 1], [.2, .8]) == 1.
    assert auc([0, 1], [.8, .2]) == 0.
    assert auc([0, 1], [.5, .5]) == .5
    assert auc([0, 0], [.1, .9]) is None
    assert metrics([0, 1, 0, 1], [.1, .9, .7, .2], .5)['confusion'] == dict(tp=1, tn=1, fp=1, fn=1)
    path = tmp_path / 'checkpoint.pt'
    atomic_save({'schema': 'mfvlr-pilot-v1', 'value': torch.tensor([4])}, path)
    assert load_checkpoint(path)['value'].item() == 4
    assert not path.with_suffix('.pt.tmp').exists()


def test_batched_checks_and_gradient_norm():
    a = torch.nn.Parameter(torch.zeros(2))
    b = torch.nn.Parameter(torch.zeros(3))
    a.grad, b.grad = torch.tensor([3., 4.]), torch.tensor([0., 0., 12.])
    check_finite_tensors([a.grad, b.grad, None], 'bad')
    assert gradient_norm([a, b]) == pytest.approx(13., abs=1e-12)
    for invalid in (float('nan'), float('inf'), -float('inf')):
        with pytest.raises(FloatingPointError, match='bad'):
            check_finite_tensors([a.grad, torch.tensor([invalid]), None], 'bad')


class TinyModel(torch.nn.Module):
    """Test double for optimizer/resume orchestration; no production losses replaced."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def visual_only(self, image):
        logits = self.linear(image.mean((2, 3)))
        return SimpleNamespace(detection_logits=logits,
            decoded=SimpleNamespace(mask_logits=logits[:, :, None, None].expand(-1, -1, 224, 224)))

    def with_language(self, image, token_ids, sample_ids):
        return self.visual_only(image)

    def compute_objectives(self, output, targets):
        from mfvlr.objectives import LossValues
        fd = torch.nn.functional.cross_entropy(output.detection_logits, targets.labels)
        zero = fd * 0
        return LossValues(fd, zero, zero, zero, zero, zero, fd)


@pytest.mark.parametrize('scheduled', [False, True])
def test_resume_next_step_matches_uninterrupted(tmp_path, monkeypatch, scheduled):
    """Includes a real optimizer step, RNG restoration and checkpoint cursor replay."""
    _, folder = manifests(tmp_path)
    spec = importlib.util.spec_from_file_location('pilot_train', ROOT / 'train.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    import mfvlr.integration
    monkeypatch.setattr(mfvlr.integration, 'MFVLRIntegration', TinyModel)
    config = json.loads((ROOT / 'configs/diffae_pilot.json').read_text())
    config.update(device='cpu', threads=1, checkpoint_every=1,
                  manifests={s: str(folder / f'{s}.jsonl') for s in ('train', 'val', 'test')})
    config['tiny'] = dict(limit=4, max_steps=2, epochs=2, lr=.001)
    if scheduled:
        # A short interval exercises the same boundary logic as the paper's 15 epochs.
        config['scheduler'] = dict(name='StepLR', step_size=1, gamma=.1)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    # Interrupt just after the epoch-1 checkpoint has advanced the schedule;
    # also interrupt inside epoch 1 to check that partial epochs do not advance it.
    config['tiny'].update(max_steps=3)
    config_path.write_text(json.dumps(config))
    for output, budget, resume in [('full', 3, None), ('resumed', 1, None),
                                   ('resumed', 2, tmp_path / 'resumed/last.pt'),
                                   ('resumed', 3, tmp_path / 'resumed/last.pt')]:
        entry.run(SimpleNamespace(config=config_path, stage='tiny', output=tmp_path / output,
                                  resume=resume, max_steps=budget, check_data=False))
    expected, actual = [load_checkpoint(tmp_path / p / 'last.pt') for p in ('full', 'resumed')]
    assert expected['step'] == actual['step'] == 3
    assert expected['epoch'] == actual['epoch'] == 1
    for name in expected['model']:
        torch.testing.assert_close(expected['model'][name], actual['model'][name], atol=0, rtol=0)
    assert actual['threshold'] == expected['threshold']
    assert actual['scheduler'] == expected['scheduler']
    assert actual['optimizer']['param_groups'][0]['lr'] == pytest.approx(.0001 if scheduled else .001)
    if scheduled:
        assert actual['scheduler']['last_epoch'] == 1
    assert (tmp_path / 'resumed/best.pt').exists()
