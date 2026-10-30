"""Bounded, separate decoder experiments. Original checkpoint is read-only."""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.pilot_deps'), str(ROOT / 'src'), str(ROOT / 'tools')]

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from diagnose_mask_decoder import map_summary
from mfvlr.integration import MFVLRIntegration, ImageOnlyMFVLR, SuppliedTargets
from mfvlr.objectives import forgery_localization_loss
from mfvlr_research.pilot_data import PilotFaces, sha256
from mfvlr_research.pilot_runtime import load_checkpoint, check_finite_tensors


def load(args, full=False):
    cp = load_checkpoint(args.checkpoint)
    for name, digest in cp['provenance']['source'].items():
        if name.replace('\\', '/').startswith('src/mfvlr/') and sha256(ROOT / name) != digest:
            raise ValueError(f'Production model source changed: {name}')
    limit = max(4, 2 * (args.pair_index + 1)) if full else 4
    data = PilotFaces(ROOT / cp['config']['manifests']['train'], tokens=full, limit=limit)
    if data.digest != cp['provenance']['manifests']['train']:
        raise ValueError('Train manifest changed')
    model = MFVLRIntegration() if full else ImageOnlyMFVLR()
    if full:
        model.load_state_dict(cp['model'], strict=True)
    else:
        weights = {k:v for k,v in cp['model'].items() if k.startswith(('mve.', 'decoder.'))}
        weights.update({k.removeprefix('heads.'):v for k,v in cp['model'].items() if k.startswith('heads.detector.')})
        model.load_state_dict(weights, strict=True)
        del weights
    del cp
    gc.collect()
    model.to(args.device).eval()
    batch = next(iter(DataLoader(data, batch_size=len(data))))
    return model, batch


def pair_metrics(logits, targets):
    probabilities = logits.detach().softmax(1)[:, 1].cpu().numpy()
    pred = logits.detach().argmax(1).cpu()
    fake_iou = []
    for i in (1, 3):
        if i >= len(pred):
            continue
        a, b = pred[i].bool(), targets[i].cpu().bool()
        union = int((a | b).sum())
        fake_iou.append(int((a & b).sum()) / union if union else 1.)
    return {'maps': map_summary(probabilities), 'fake_iou_mean': float(np.mean(fake_iou)),
            'real_false_positive_area': float(pred[::2].float().mean()),
            'probability_spatial_std': float(np.std(probabilities, axis=(1, 2)).mean())}


@torch.no_grad()
def structural(args):
    model, batch = load(args)
    images = batch['image'].to(args.device)
    local = model.mve.shared_encoder.encode_local(images, return_skips=False).local
    target = batch['mask']
    baseline = model.decoder(local).mask_logits
    report = {'samples': batch['sample_id'], 'baseline': pair_metrics(baseline, target),
              'phase_tying': {}, 'encoder_decoder_crossovers': {}}
    originals = [stage.up.weight.detach().clone() for stage in model.decoder.trunk.stages]
    for selected in ([0], [1], [2], [3], [0, 1, 2, 3]):
        for index in selected:
            weight = originals[index]
            tied = weight.mean((-2, -1), keepdim=True).expand_as(weight).clone()
            # Preserve Frobenius norm, so phase averaging is not simply zeroing the layer.
            tied *= weight.norm() / (tied.norm() + 1e-20)
            model.decoder.trunk.stages[index].up.weight.copy_(tied)
        name = '+'.join(f'U{i+1}' for i in selected)
        result = model.decoder(local).mask_logits
        report['phase_tying'][name] = pair_metrics(result, target)
        report['phase_tying'][name]['agreement_with_original'] = float((result.argmax(1) == baseline.argmax(1)).float().mean())
        for index in selected:
            model.decoder.trunk.stages[index].up.weight.copy_(originals[index])
        print('Completed phase control:', name, flush=True)
    if not torch.equal(model.decoder(local).mask_logits, baseline):
        raise AssertionError('Temporary phase controls did not restore original outputs')
    fresh = ImageOnlyMFVLR().to(args.device).eval()
    initial_local = fresh.mve.shared_encoder.encode_local(images, return_skips=False).local
    for name, decoder, features in [('fresh_encoder_trained_decoder', model.decoder, initial_local),
                                    ('trained_encoder_fresh_decoder', fresh.decoder, local),
                                    ('fresh_encoder_fresh_decoder', fresh.decoder, initial_local)]:
        report['encoder_decoder_crossovers'][name] = pair_metrics(decoder(features).mask_logits, target)
    report['restoration_exact'] = True
    return report


def gradients(args):
    model, batch = load(args, full=True)
    # Two examples, one real and one fake, keep the complete FP32 graph on 6GB GPU.
    pair = slice(2 * args.pair_index, 2 * args.pair_index + 2)
    images = batch['image'][pair].to(args.device)
    tokens = batch['token_ids'][pair].to(args.device)
    ids = tuple(batch['sample_id'][pair])
    result = model.with_language(images, tokens, sample_ids=ids)
    losses = model.compute_objectives(result, SuppliedTargets(ids, tokens,
        batch['label'][pair].to(args.device), batch['mask'][pair].to(args.device), tuple(batch['kind'][pair])))
    names = ['U1.up.weight', 'U4.conv2.weight', 'mask_head.weight']
    parameters = [model.decoder.trunk.stages[0].up.weight, model.decoder.trunk.stages[3].conv2.weight,
                  model.decoder.mask_head.weight]
    saved, report = {}, {'samples': ids, 'optimizer_updates': 0, 'losses': {}, 'parameters': {}}
    for name, value in losses.as_dict().items():
        if name == 'total':
            continue
        grads = torch.autograd.grad(value, parameters, allow_unused=True, retain_graph=True)
        check_finite_tensors(grads, 'Nonfinite diagnostic gradient')
        saved[name] = [g.detach().cpu().flatten() if g is not None else None for g in grads]
        report['losses'][name] = float(value.detach())
        print('Measured gradients:', name, flush=True)
        del grads
    for index, name in enumerate(names):
        fg = saved['fl'][index]
        values = {}
        for loss_name, grads in saved.items():
            grad = grads[index]
            values[loss_name] = {'norm': float(grad.double().norm()) if grad is not None else None}
            if grad is not None and fg is not None:
                g, f = grad.double(), fg.double()
                values[loss_name]['cosine_with_FL'] = float(torch.dot(g, f) / (g.norm() * f.norm() + 1e-30))
        total = sum(g[index] for g in saved.values() if g[index] is not None)
        report['parameters'][name] = {'loss_contributions': values,
            'sum_contribution_norm': float(total.double().norm()),
            'sum_cosine_with_FL': float(torch.dot(total.double(), fg.double()) /
                                       (total.double().norm() * fg.double().norm() + 1e-30))}
    return report


def fit(args):
    model, batch = load(args)
    images, targets = batch['image'].to(args.device), batch['mask'].to(args.device)
    with torch.no_grad():
        local = model.mve.shared_encoder.encode_local(images, return_skips=False).local.detach()
    # Decoder copy only. Encoder features, targets and source checkpoint remain fixed.
    decoder = model.decoder
    decoder.train()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, foreach=False)
    foreground = int(targets.sum())
    background = targets.numel() - foreground
    class_weights = torch.tensor([1., background / max(foreground, 1)], device=args.device)
    report = {'samples': batch['sample_id'], 'loss': args.fit_loss, 'lr': args.lr, 'steps': args.steps,
              'foreground_pixel_fraction': foreground / targets.numel(),
              'foreground_pixel_fraction_each': targets.float().mean((1, 2)).cpu().tolist(),
              'class_weights': class_weights.cpu().tolist() if args.fit_loss == 'balanced_FL' else [1., 1.],
              'history': []}
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        decoded = decoder(local)
        loss = (forgery_localization_loss(decoded.mask_logits, targets) if args.fit_loss == 'FL' else
                torch.nn.functional.cross_entropy(decoded.mask_logits, targets, weight=class_weights))
        if step % 10 == 0 or step == args.steps:
            record = {'step': step, 'fl': float(loss.detach()), **pair_metrics(decoded.mask_logits, targets)}
            report['history'].append(record)
            print(args.fit_loss, 'diagnostic step', step, 'loss', record['fl'], 'IoU', record['fake_iou_mean'], flush=True)
        if step == args.steps:
            probabilities = decoded.mask_logits.detach().softmax(1)[:, 1].cpu().numpy()
            for index, p in enumerate(probabilities):
                Image.fromarray(np.round(p * 255).astype(np.uint8)).save(args.output / f'fit_{args.fit_loss}_probability_{index}.png')
            break
        loss.backward()
        check_finite_tensors((p.grad for p in decoder.parameters()), 'Nonfinite fit gradient')
        optimizer.step()
        check_finite_tensors(decoder.parameters(), 'Nonfinite fit parameter')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--mode', choices=['structure', 'gradients', 'fit'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--pair-index', type=int, choices=range(4), default=0)
    parser.add_argument('--fit-loss', choices=['FL', 'balanced_FL'], default='FL')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    args.output.mkdir(parents=True, exist_ok=True)
    before = sha256(args.checkpoint)
    result = {'structure': structural, 'gradients': gradients, 'fit': fit}[args.mode](args)
    if sha256(args.checkpoint) != before:
        raise AssertionError('Source checkpoint changed')
    result.update(checkpoint_sha256=before, source_checkpoint_unchanged=True,
                  scope='bounded diagnostic on selected training samples; not validation or benchmark')
    name = f'gradients_pair{args.pair_index}' if args.mode == 'gradients' and args.pair_index else args.mode
    if args.mode == 'fit' and args.fit_loss != 'FL':
        name += '_' + args.fit_loss
    (args.output / f'{name}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved', args.output / f'{name}.json', flush=True)
