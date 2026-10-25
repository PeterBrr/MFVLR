"""Metrics and versioned resumable state for the internal A0 pilot."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import importlib.metadata

import numpy as np
import torch
from torch.utils.data import DataLoader

from .pilot_data import sha256


def check_finite_tensors(tensors, message):
    """Retain every finite check, but synchronize once per device, not per tensor."""
    checks = {}
    for tensor in tensors:
        if torch.is_tensor(tensor):
            checks.setdefault(tensor.device, []).append(torch.isfinite(tensor).all())
    for flags in checks.values():
        if not torch.stack(flags).all().item():
            raise FloatingPointError(message)


def gradient_norm(parameters):
    """FP64 reduction without materializing squared FP64 copies of each gradient."""
    sums = {}
    for parameter in parameters:
        if parameter.grad is not None:
            grad = parameter.grad.detach()
            norm = torch.linalg.vector_norm(grad, dtype=torch.float64)
            sums.setdefault(grad.device, []).append(norm.square())
    return sum(float(torch.stack(values).sum().cpu()) for values in sums.values()) ** .5


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def provenance(root, config):
    files = sorted((root / 'src' / 'mfvlr').glob('*.py'))
    files += sorted((root / 'src' / 'mfvlr_research').glob('pilot_*.py'))
    files += [root / 'train.py', root / 'evaluate.py']
    return {'source': {str(p.relative_to(root)): sha256(p) for p in files},
            'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            'manifests': {s: sha256(root / p) for s, p in config['manifests'].items()},
            'torch': str(torch.__version__), 'numpy': str(np.__version__),
            'python': sys.version,
            'dependencies': {name: importlib.metadata.version(name) for name in ('Pillow', 'ftfy', 'regex', 'wcwidth')},
            'tokenizer_manifest': sha256(root / 'src/mfvlr/tokenizer_assets/manifest.json')}


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if checkpoint.get('schema') != 'mfvlr-pilot-v1':
        raise ValueError('Unsupported pilot checkpoint schema')
    return checkpoint


def auc(labels, scores):
    """Mann-Whitney AUC with average ranks for exact ties."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    positives, negatives = int((labels == 1).sum()), int((labels == 0).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def metrics(labels, scores, threshold):
    y, score = np.asarray(labels), np.asarray(scores)
    if len(y) == 0 or not np.isfinite(score).all():
        raise ValueError('Empty or nonfinite predictions')
    predicted = score >= threshold
    tp = int(((y == 1) & predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    return {'auc': auc(y, score), 'accuracy': (tp + tn) / len(y),
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}, 'threshold': threshold}


def validation_threshold(labels, scores):
    # Fixed grid avoids threshold selection using any test labels.
    candidates = [metrics(labels, scores, float(t)) for t in np.linspace(0, 1, 101)]
    return max(candidates, key=lambda x: (x['f1'], -abs(x['threshold'] - .5)))['threshold']


@torch.inference_mode()
def evaluate_visual(model, dataset, device, batch_size, threshold=None):
    model.eval()
    labels, scores, fake_iou, fake_dice, real_area = [], [], [], [], []
    ce_sum, count = 0., 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        images = batch['image'].to(device)
        output = model.visual_only(images) if hasattr(model, 'visual_only') else model(images)
        logits = output.detection_logits
        mask_logits = output.decoded.mask_logits
        if not torch.isfinite(logits).all() or not torch.isfinite(mask_logits).all():
            raise FloatingPointError('Nonfinite validation predictions')
        targets = batch['label'].to(device)
        ce_sum += float(torch.nn.functional.cross_entropy(logits, targets, reduction='sum').cpu())
        count += len(targets)
        labels.extend(batch['label'].tolist())
        scores.extend(logits.softmax(-1)[:, 1].cpu().tolist())
        predicted = mask_logits.argmax(1).cpu().bool()
        for index, label in enumerate(batch['label'].tolist()):
            p, target = predicted[index], batch['mask'][index].bool()
            if label == 0:
                real_area.append(float(p.float().mean()))
            else:
                intersection, union = int((p & target).sum()), int((p | target).sum())
                size = int(p.sum() + target.sum())
                # Explicit empty/empty convention, separate from real images.
                fake_iou.append(intersection / union if union else 1.)
                fake_dice.append(2 * intersection / size if size else 1.)
    if threshold is None:
        threshold = validation_threshold(labels, scores)
    result = metrics(labels, scores, threshold)
    result.update(count=count, fd_ce=ce_sum / count,
        fake_foreground_iou_macro=float(np.mean(fake_iou)) if fake_iou else None,
        fake_foreground_dice_macro=float(np.mean(fake_dice)) if fake_dice else None,
        real_false_positive_area=float(np.mean(real_area)) if real_area else None)
    return result
