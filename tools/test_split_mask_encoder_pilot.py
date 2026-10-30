"""Small from-scratch GenFace ablation: separate the mask CNN, keep the decoder shared."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.pilot_deps'), str(ROOT / 'src'), str(ROOT / 'tools')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from export_ad_reconstructions import file_hash, rgb_display
from test_ad_overfit import state_hash
from mfvlr.objectives import appearance_reconstruction_loss, forgery_detection_loss
from mfvlr_research.pilot_datasets import dataset_for_config
from mfvlr_research.pilot_runtime import check_finite_tensors, metrics
from mfvlr_research.pilot_variant import visual_model, weighted_mask_loss


class MaskEncoderAblation(nn.Module):
    def __init__(self, config, separate):
        super().__init__()
        self.visual = visual_model(config)
        self.mask_encoder = copy.deepcopy(self.visual.mve.shared_encoder.cnn) if separate else None
        if separate:
            assert state_hash(self.mask_encoder) == state_hash(self.visual.mve.shared_encoder.cnn)
            assert all(a.data_ptr() != b.data_ptr() for a, b in zip(
                self.mask_encoder.parameters(), self.visual.mve.shared_encoder.cnn.parameters()))

    def forward(self, images):
        appearance = self.visual.mve.encode_appearance(images)
        features = self.visual.decoder.trunk(appearance.local)
        reconstruction = self.visual.decoder.appearance_head(features)
        if self.mask_encoder is None:
            mask_features = features
        else:
            local = self.mask_encoder(images, return_skips=False).local
            # The same decoder weights serve both branches, intentionally.
            mask_features = self.visual.decoder.trunk(local)
        mask_logits = self.visual.decoder.mask_head(mask_features)
        encoded = self.visual.mve(images, reconstruction, appearance_encoding=appearance)
        logits = self.visual.detector(encoded.fused_global[:, 0, :])
        return {'reconstruction': reconstruction, 'mask_logits': mask_logits, 'detection_logits': logits}


def objectives(output, batch, device, weights):
    return {
        'ar': appearance_reconstruction_loss(batch['image'].to(device), output['reconstruction']),
        'fd': forgery_detection_loss(output['detection_logits'], batch['label'].to(device)),
        'fl': weighted_mask_loss(output['mask_logits'], batch['mask'].to(device), weights),
    }


def balanced_subset(data, count):
    if count < 4 or count % 4:
        raise ValueError('Subset size must be a positive multiple of four')
    groups = [[i for i, row in enumerate(data.rows) if row['label'] == label] for label in (0, 1)]
    if any(len(group) < count // 2 for group in groups):
        raise ValueError('Not enough balanced samples')
    groups = [[group[i] for i in np.linspace(0, len(group) - 1, count // 2, dtype=int)] for group in groups]
    selected = [i for pair in zip(*groups) for i in pair]
    # Decode/hash verification is performed by the existing dataset reader.
    return [data[i] for i in selected]


def collate(samples):
    return next(iter(DataLoader(samples, batch_size=len(samples), shuffle=False)))


@torch.no_grad()
def evaluate(model, samples, device, weights, batch_size):
    model.eval()
    sums = dict(ar=0., fd=0., fl=0.)
    labels, scores, fake_iou, fake_dice, real_area = [], [], [], [], []
    pixel_counts = dict(tp=0, fp=0, fn=0, tn=0)
    selected = [0, 1, len(samples) - 2, len(samples) - 1]
    preview, offset = {}, 0
    for batch in DataLoader(samples, batch_size=batch_size, shuffle=False):
        output = model(batch['image'].to(device))
        losses = objectives(output, batch, device, weights)
        size = len(batch['image'])
        for name, value in losses.items():
            sums[name] += float(value) * size / len(samples)
        labels.extend(batch['label'].tolist())
        scores.extend(output['detection_logits'].softmax(-1)[:, 1].cpu().tolist())
        predicted = output['mask_logits'].argmax(1).cpu().bool()
        target = batch['mask'].bool()
        pixel_counts['tp'] += int((predicted & target).sum())
        pixel_counts['fp'] += int((predicted & ~target).sum())
        pixel_counts['fn'] += int((~predicted & target).sum())
        pixel_counts['tn'] += int((~predicted & ~target).sum())
        for i, label in enumerate(batch['label'].tolist()):
            p, gt = predicted[i], target[i]
            if label == 0:
                real_area.append(float(p.float().mean()))
            else:
                intersection, union = int((p & gt).sum()), int((p | gt).sum())
                denominator = int(p.sum() + gt.sum())
                fake_iou.append(intersection / union if union else 1.)
                fake_dice.append(2 * intersection / denominator if denominator else 1.)
            if offset + i in selected:
                preview[offset + i] = {
                    'sample_id': batch['sample_id'][i], 'label': label,
                    'image': batch['image'][i].permute(1, 2, 0).numpy(),
                    'reconstruction': output['reconstruction'][i].cpu().permute(1, 2, 0).numpy(),
                    'target': gt.numpy(), 'prediction': p.numpy(),
                    'probability': output['mask_logits'][i].softmax(0)[1].cpu().numpy(),
                }
        offset += size
    tp, fp, fn, tn = (pixel_counts[k] for k in ('tp', 'fp', 'fn', 'tn'))
    fg_union, bg_union = tp + fp + fn, tn + fp + fn
    ious = [x for x in (tp / fg_union if fg_union else None, tn / bg_union if bg_union else None) if x is not None]
    result = {**metrics(labels, scores, .5), 'count': len(samples), 'losses': sums,
              'fake_foreground_iou_macro': float(np.mean(fake_iou)),
              'fake_foreground_dice_macro': float(np.mean(fake_dice)),
              'real_false_positive_area': float(np.mean(real_area)), 'pixel_confusion': pixel_counts,
              'pixel_miou_global': float(np.mean(ious)) if ious else None}
    model.train()
    return result, [preview[i] for i in selected]


def wiring_probe(model, batch, device, weights):
    model.eval()
    output = model(batch['image'].to(device))
    # Verify the existing AD/detection route is exactly preserved.
    with torch.no_grad():
        reference = model.visual(batch['image'].to(device))
        torch.testing.assert_close(output['reconstruction'], reference.decoded.reconstruction, rtol=0, atol=0)
        torch.testing.assert_close(output['detection_logits'], reference.detection_logits, rtol=0, atol=0)
        torch.testing.assert_close(output['mask_logits'], reference.decoded.mask_logits, rtol=0, atol=0)
    named = {
        'image_encoder': model.visual.mve.shared_encoder.cnn.stages[0][0].weight,
        'shared_decoder': model.visual.decoder.trunk.stages[0].projection.weight,
    }
    if model.mask_encoder is not None:
        named['mask_encoder'] = model.mask_encoder.stages[0][0].weight
    values = objectives(output, batch, device, weights)
    result = {}
    for name, loss in values.items():
        grads = torch.autograd.grad(loss, tuple(named.values()), retain_graph=True, allow_unused=True)
        result[name] = {key: None if grad is None else float(grad.norm()) for key, grad in zip(named, grads)}
    if model.mask_encoder is not None:
        assert result['fl']['image_encoder'] is None
        assert result['ar']['mask_encoder'] is None and result['fd']['mask_encoder'] is None
        assert result['fl']['mask_encoder'] > 0
    assert all(result[loss]['shared_decoder'] > 0 for loss in values)
    model.train()
    return result, {key: value.detach().cpu() for key, value in output.items()}


def save_preview(path, baseline, separated):
    sheet = Image.new('RGB', (1344, 280 * len(baseline)), 'white')
    draw = ImageDraw.Draw(sheet)
    raw = {}
    for i, (a, b) in enumerate(zip(baseline, separated)):
        assert a['sample_id'] == b['sample_id']
        np.testing.assert_array_equal(a['image'], b['image'])
        y = 280 * i
        draw.text((5, y + 5), f'{"REAL" if a["label"] == 0 else "DIFFAE"} | {a["sample_id"]}', fill='black')
        panels = [(rgb_display(a['image']), 'Input'),
                  (Image.fromarray(a['target'].astype(np.uint8) * 255), 'GT mask'),
                  (rgb_display(a['reconstruction']), 'Shared CNN: AD'),
                  (rgb_display(b['reconstruction']), 'Separate mask CNN: AD'),
                  (Image.fromarray(a['prediction'].astype(np.uint8) * 255), 'Shared CNN: mask'),
                  (Image.fromarray(b['prediction'].astype(np.uint8) * 255), 'Separate mask CNN: mask')]
        for j, (panel, title) in enumerate(panels):
            draw.text((j * 224 + 5, y + 25), title, fill='black')
            sheet.paste(panel.convert('RGB'), (j * 224, y + 48))
        for prefix, sample in (('baseline', a), ('separate', b)):
            for key in ('image', 'target', 'prediction', 'probability', 'reconstruction'):
                raw[f'{prefix}_{i}_{key}'] = sample[key]
    sheet.save(path.with_suffix('.png'))
    np.savez_compressed(path.with_suffix('.npz'), **raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/genface_split_mask_encoder_pilot.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output is not empty; choose a new directory')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['objectives'] != ['ar', 'fd', 'fl'] or config['seed'] != 0:
        raise ValueError('This controlled experiment requires AR+FD+FL and seed 0')
    micro, effective = config['microbatch_size'], config['effective_batch_size']
    if micro != 2 or effective != 4 or config['train_images'] % effective:
        raise ValueError('Expected microbatch 2, effective batch 4, divisible train set')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    device = torch.device(config['device'])
    train_data = dataset_for_config(ROOT, config, 'train', limit=config['train_pool_limit'])
    val_data = dataset_for_config(ROOT, config, 'val')
    train_samples = balanced_subset(train_data, config['train_images'])
    val_samples = balanced_subset(val_data, config['validation_images'])
    assert not ({s['sample_id'] for s in train_samples} & {s['sample_id'] for s in val_samples})
    rng = np.random.default_rng(config['seed'])
    schedule = []
    while len(schedule) < config['updates']:
        order = rng.permutation(len(train_samples)).tolist()
        schedule.extend(order[i:i + effective] for i in range(0, len(order), effective))
    schedule = schedule[:config['updates']]
    weights = torch.tensor([config['model']['mask_background_weight'], config['model']['mask_foreground_weight']],
                           dtype=torch.float32, device=device)
    args.output.mkdir(parents=True)
    sources = list((ROOT / 'src/mfvlr').glob('*.py')) + list((ROOT / 'src/mfvlr_research').glob('pilot_*.py'))
    sources += [Path(__file__), ROOT / 'tools/test_ad_overfit.py', ROOT / 'tools/export_ad_reconstructions.py']
    report = {'status': 'RUNNING', 'config': config, 'config_sha256': file_hash(args.config),
              'source_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
              'manifest_sha256': {s: file_hash(ROOT / config['manifests'][s]) for s in ('train', 'val')},
              'train_ids': [s['sample_id'] for s in train_samples], 'val_ids': [s['sample_id'] for s in val_samples],
              'schedule': schedule, 'initialization': 'scratch; no trained checkpoint read or loaded',
              'change': 'Independent mask CNN; image/residual encoder and decoder trunk otherwise still shared',
              'metric_policy': 'Detection threshold fixed at 0.5; mask argmax; global pixel mIoU supplements historical fake-only IoU',
              'trials': {}}
    report_path = args.output / 'report.json'
    write = lambda: report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    write()
    initial_common_hash, initial_outputs, initial_metrics = None, None, None
    final_previews = {}
    for name, separate in (('shared_encoder', False), ('separate_mask_encoder', True)):
        torch.manual_seed(0)
        model = MaskEncoderAblation(config, separate).to(device)
        common_hash = state_hash(model.visual)
        if initial_common_hash is None:
            initial_common_hash = common_hash
        else:
            assert common_hash == initial_common_hash
        probe, output = wiring_probe(model, collate(train_samples[:2]), device, weights)
        if initial_outputs is None:
            initial_outputs = output
        else:
            for key in output:
                torch.testing.assert_close(output[key], initial_outputs[key], rtol=0, atol=0)
        trial_dir = args.output / name
        trial_dir.mkdir()
        count = sum(p.numel() for p in model.parameters())
        report['trials'][name] = {'status': 'RUNNING', 'parameters': count, 'initial_common_sha256': common_hash,
                                'initial_wiring_probe': probe, 'history': []}
        trial = report['trials'][name]
        for split, samples in (('train', train_samples), ('val', val_samples)):
            measured, _ = evaluate(model, samples, device, weights, micro)
            trial[f'initial_{split}'] = measured
        if initial_metrics is None:
            initial_metrics = (trial['initial_train'], trial['initial_val'])
        else:
            assert initial_metrics == (trial['initial_train'], trial['initial_val'])
        write()
        options = config['optimizer']
        optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'], weight_decay=options['weight_decay'],
                                     betas=tuple(options['betas']), eps=options['eps'], foreach=False)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        bar = tqdm(enumerate(schedule, 1), total=len(schedule), desc=name, unit='update', dynamic_ncols=True)
        for step, indices in bar:
            optimizer.zero_grad(set_to_none=True)
            observed = dict(ar=0., fd=0., fl=0.)
            for offset in range(0, effective, micro):
                batch = collate([train_samples[i] for i in indices[offset:offset + micro]])
                output = model(batch['image'].to(device))
                losses = objectives(output, batch, device, weights)
                total = sum(losses.values()) * micro / effective
                total.backward()
                for key, value in losses.items():
                    observed[key] += float(value.detach()) * micro / effective
                del output, losses, total
            check_finite_tensors((p.grad for p in model.parameters()), 'Nonfinite pilot gradient')
            optimizer.step()
            check_finite_tensors(model.parameters(), 'Nonfinite pilot parameter')
            bar.set_postfix(ar=f'{observed["ar"]:.4f}', fl=f'{observed["fl"]:.4f}')
            trial['history'].append({'step': step, 'epoch_equivalent': step * effective / len(train_samples),
                                     'batch_losses': observed, 'elapsed_seconds': time.perf_counter() - start})
            if step % config['evaluate_every'] == 0 or step == len(schedule):
                record = {'step': step}
                for split, samples in (('train', train_samples), ('val', val_samples)):
                    measured, preview = evaluate(model, samples, device, weights, micro)
                    record[split] = measured
                    if step == len(schedule):
                        final_previews[(name, split)] = preview
                trial.setdefault('evaluations', []).append(record)
                write()
        trial.update(status='COMPLETED', elapsed_seconds=time.perf_counter() - start,
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device), final_state_sha256=state_hash(model))
        cpu_state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
        torch.save({'schema': 'mfvlr-split-mask-encoder-pilot-v1', 'variant': name, 'model': cpu_state,
                    'config': config, 'step': len(schedule), 'trained_weights_loaded': False}, trial_dir / 'final.pt')
        del cpu_state, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        write()
    for split in ('train', 'val'):
        save_preview(args.output / f'{split}_comparison', final_previews[('shared_encoder', split)],
                     final_previews[('separate_mask_encoder', split)])
    assert all(file_hash(ROOT / relative) == expected for relative, expected in report['source_sha256'].items())
    report.update(status='COMPLETED', common_initial_weights_verified=True, initial_outputs_and_metrics_identical=True,
                  gradient_isolation_verified=True, source_unchanged=True)
    write()
    print(json.dumps({'output': str(args.output.resolve()), 'results': {
        name: {'parameters': trial['parameters'], 'final': trial['evaluations'][-1]}
        for name, trial in report['trials'].items()}}, indent=2))


if __name__ == '__main__':
    main()
