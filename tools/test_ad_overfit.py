"""Bounded AD objective ablations, from scratch or a fixed trained checkpoint."""
import argparse
import gc
import hashlib
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
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from export_ad_reconstructions import file_hash, rgb_display
from mfvlr.objectives import appearance_reconstruction_loss, forgery_detection_loss
from mfvlr_research.pilot_datasets import dataset_for_config
from mfvlr_research.pilot_runtime import load_checkpoint, check_finite_tensors
from mfvlr_research.pilot_variant import visual_model, weighted_mask_loss


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


TRIAL_LOSSES = {'ar_only': ('ar',), 'ar_fd': ('ar', 'fd'),
                'ar_fl': ('ar', 'fl'), 'ar_fd_fl': ('ar', 'fd', 'fl')}


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_model(checkpoint, device, config, initialization):
    if initialization == 'scratch':
        # Construct random weights; never call load_state_dict in this branch.
        return visual_model(config).to(device)
    cp = load_checkpoint(checkpoint)
    model = visual_model(cp['config'])
    state = {}
    for name, tensor in cp['model'].items():
        if name.startswith(('mve.', 'decoder.')):
            state[name] = tensor
        elif name.startswith('heads.detector.'):
            state[name.removeprefix('heads.')] = tensor
    model.load_state_dict(state, strict=True)
    del cp, state
    gc.collect()
    return model.to(device)


def forward_losses(model, batch, device, weights, joint):
    images = batch['image'].to(device)
    if joint:
        result = model(images)
        decoded = result.decoded
        fd = forgery_detection_loss(result.detection_logits, batch['label'].to(device))
        fl = weighted_mask_loss(decoded.mask_logits, batch['mask'].to(device), weights)
    else:
        local = model.mve.shared_encoder.encode_local(images, return_skips=False).local
        decoded = model.decoder(local)
        fd = fl = None
    ar = appearance_reconstruction_loss(images, decoded.reconstruction)
    return {'ar': ar, 'fd': fd, 'fl': fl}, decoded.reconstruction


@torch.no_grad()
def evaluate(model, batches, device, weights):
    model.eval()
    totals = dict(ar=0., fd=0., fl=0.)
    outputs = []
    for batch in batches:
        losses, reconstruction = forward_losses(model, batch, device, weights, True)
        for name, value in losses.items():
            totals[name] += float(value) / len(batches)
        outputs.append(reconstruction.cpu())
    model.train()
    return totals, torch.cat(outputs).permute(0, 2, 3, 1).numpy()


def gradient_probe(model, batch, device, weights):
    model.eval()
    params = {
        'encoder.first_conv.weight': model.mve.shared_encoder.cnn.stages[0][0].weight,
        'decoder.U1.projection.weight': model.decoder.trunk.stages[0].projection.weight,
        'decoder.U4.conv2.weight': model.decoder.trunk.stages[3].conv2.weight,
        'decoder.appearance_head.weight': model.decoder.appearance_head.weight,
    }
    losses, _ = forward_losses(model, batch, device, weights, True)
    saved = {}
    for name, loss in losses.items():
        gradients = torch.autograd.grad(loss, tuple(params.values()), retain_graph=True, allow_unused=True)
        check_finite_tensors(gradients, 'Nonfinite probe gradient')
        saved[name] = [None if g is None else g.detach().cpu().flatten().double() for g in gradients]
    result = {'scope': 'First real/fake training microbatch; raw gradients, not Adam updates',
              'losses': {k: float(v.detach()) for k, v in losses.items()}, 'parameters': {}}
    for i, name in enumerate(params):
        entry = {'norms': {}, 'cosines': {}}
        for loss, gradients in saved.items():
            entry['norms'][loss] = None if gradients[i] is None else float(gradients[i].norm())
        for left, right in (('ar', 'fd'), ('ar', 'fl'), ('fd', 'fl')):
            a, b = saved[left][i], saved[right][i]
            denominator = None if a is None or b is None else float(a.norm() * b.norm())
            entry['cosines'][f'{left}_vs_{right}'] = (
                None if not denominator else float(torch.dot(a, b) / denominator))
        result['parameters'][name] = entry
    del losses, saved
    model.train()
    return result


def export_images(output, inputs, initial, results, sample_ids, labels):
    titles = {'ar_only': 'After: AR only', 'ar_fd': 'After: AR + FD',
              'ar_fl': 'After: AR + FL', 'ar_fd_fl': 'After: AR + FD + FL'}
    sheet = Image.new('RGB', (224 * (2 + len(results)), 280 * len(inputs)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, sample_id in enumerate(sample_ids):
        y = i * 280
        draw.text((5, y + 5), f'{"REAL" if labels[i] == 0 else "DIFFAE"} | {sample_id}', fill='black')
        panels = [(inputs[i], 'Input'), (initial[i], 'Before (same initialization)')]
        panels.extend((array[i], titles[name]) for name, array in results.items())
        for j, (array, title) in enumerate(panels):
            draw.text((j * 224 + 5, y + 25), title, fill='black')
            panel = rgb_display(array)
            sheet.paste(panel, (j * 224, y + 48))
            panel.save(output / f'sample_{i:02d}_column_{j}.png')
    sheet.save(output / 'comparison.png')
    np.savez_compressed(output / 'raw_reconstructions.npz', input=inputs, before=initial, **results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--initialization', choices=['checkpoint', 'scratch'], default='checkpoint')
    parser.add_argument('--trials', nargs='+', choices=list(TRIAL_LOSSES), default=['ar_only', 'ar_fd_fl'])
    args = parser.parse_args()
    if args.steps < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('Steps and learning rate must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output is not empty; choose a new directory')
    if len(set(args.trials)) != len(args.trials) or args.trials[0] != 'ar_only':
        parser.error('Trials must be unique and start with ar_only')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    before_hash = file_hash(args.checkpoint)
    cp = load_checkpoint(args.checkpoint)
    config = cp['config']
    if config.get('model', {}).get('variant') != 'resize_conv_v1':
        raise ValueError('This diagnostic expects the resize-conv variant')
    for relative, expected in cp['provenance']['source'].items():
        if relative.replace('\\', '/').startswith(('src/mfvlr/', 'src/mfvlr_research/')):
            if file_hash(ROOT / relative) != expected:
                raise ValueError(f'Model source changed: {relative}')
    manifest = ROOT / config['manifests']['train']
    if file_hash(manifest) != cp['provenance']['manifests']['train']:
        raise ValueError('Training manifest changed')
    data = dataset_for_config(ROOT, config, 'train', limit=config['pilot']['limit'])
    groups = [[i for i, row in enumerate(data.rows) if row['label'] == label] for label in (0, 1)]
    indices = [groups[0][0], groups[1][0], groups[0][-1], groups[1][-1]]
    if len(set(indices)) != 4:
        raise ValueError('Need four distinct training images')
    batches = list(DataLoader(Subset(data, indices), batch_size=2, shuffle=False, num_workers=0))
    inputs = torch.cat([b['image'] for b in batches]).permute(0, 2, 3, 1).numpy()
    ids = [s for b in batches for s in b['sample_id']]
    labels = [int(x) for b in batches for x in b['label']]
    device = torch.device(args.device)
    settings = config['model']
    if settings['mask_loss_weight'] != 1.:
        raise ValueError('Expected unit mask loss multiplier for this comparison')
    weights = torch.tensor([settings['mask_background_weight'], settings['mask_foreground_weight']],
                           dtype=torch.float32, device=device)
    del cp
    gc.collect()
    args.output.mkdir(parents=True)
    report = {'status': 'RUNNING', 'checkpoint': str(args.checkpoint.resolve()),
              'checkpoint_sha256': before_hash, 'manifest_sha256': file_hash(manifest),
              'initialization': args.initialization, 'trained_weights_loaded': args.initialization == 'checkpoint',
              'reference_checkpoint_role': ('Configuration and provenance only; model weights are newly initialized'
                  if args.initialization == 'scratch' else 'Initial trained model weights and configuration'),
              'sample_ids': ids, 'labels': labels, 'split': 'train',
              'selection': 'First and last real/fake within the actual fixed 1000-image training subset',
              'scope': 'Four-image overfit diagnostic; no validation/test scoring or deployment checkpoint',
              'steps_per_trial': args.steps, 'effective_batch': 4, 'microbatch': 2,
              'optimizer': {'name': 'Adam', 'lr': args.lr, 'weight_decay': 0., 'betas': [.9, .999], 'eps': 1e-8},
              'class_weights': weights.cpu().tolist(),
              'training': 'Fresh optimizer and identical initial weights for each trial; encoder and decoder unfrozen; unit objective weights; no language losses; no sigmoid',
              'display': 'Fixed [0,1] clipping only for PNG; NPZ preserves unclipped float32 output',
              'trials': {}}
    write_json(args.output / 'report.json', report)
    baseline_recon, baseline_hash, final_recon = None, None, {}
    for trial in args.trials:
        torch.manual_seed(0)
        model = load_model(args.checkpoint, device, config, args.initialization).train()
        initial_hash = state_hash(model)
        if baseline_hash is None:
            baseline_hash = initial_hash
        elif initial_hash != baseline_hash:
            raise AssertionError('Trials did not start from identical model parameters')
        initial_metrics, initial_recon = evaluate(model, batches, device, weights)
        if baseline_recon is None:
            baseline_recon = initial_recon
            report['initial_gradient_probe'] = gradient_probe(model, batches[0], device, weights)
        else:
            np.testing.assert_array_equal(initial_recon, baseline_recon)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0., foreach=False)
        history = [{'step': 0, **initial_metrics}]
        start = time.perf_counter()
        bar = tqdm(range(1, args.steps + 1), desc=f'AD diagnostic {trial}', unit='update', dynamic_ncols=True)
        for step in bar:
            optimizer.zero_grad(set_to_none=True)
            mean_ar = 0.
            for batch in batches:
                losses, reconstruction = forward_losses(model, batch, device, weights, trial != 'ar_only')
                total = sum(losses[name] for name in TRIAL_LOSSES[trial]) / len(batches)
                mean_ar += float(losses['ar'].detach()) / len(batches)
                total.backward()
                del losses, reconstruction, total
            check_finite_tensors((p.grad for p in model.parameters()), 'Nonfinite diagnostic gradient')
            optimizer.step()
            check_finite_tensors(model.parameters(), 'Nonfinite diagnostic parameter')
            bar.set_postfix(ar=f'{mean_ar:.5f}')
            if step % 25 == 0 or step == args.steps:
                measured, reconstruction = evaluate(model, batches, device, weights)
                history.append({'step': step, 'elapsed_seconds': time.perf_counter() - start, **measured})
                report['trials'][trial] = {'status': 'RUNNING', 'objectives': TRIAL_LOSSES[trial],
                                          'initial_state_sha256': initial_hash, 'history': history}
                write_json(args.output / 'report.json', report)
                if step == args.steps:
                    final_recon[trial] = reconstruction
        report['trials'][trial] = {'status': 'COMPLETED', 'objectives': TRIAL_LOSSES[trial],
                                  'initial_state_sha256': initial_hash, 'final_state_sha256': state_hash(model),
                                  'history': history,
                                  'elapsed_seconds': time.perf_counter() - start}
        write_json(args.output / 'report.json', report)
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
    export_images(args.output, inputs, baseline_recon, final_recon, ids, labels)
    if file_hash(args.checkpoint) != before_hash:
        raise AssertionError('Original checkpoint changed')
    report.update(status='COMPLETED', source_checkpoint_unchanged=True,
                  identical_initial_outputs_verified=True, identical_initial_weights_verified=True)
    write_json(args.output / 'report.json', report)
    print(json.dumps({'output': str(args.output.resolve()),
                      'initial': report['trials']['ar_only']['history'][0],
                      'final': {k: v['history'][-1] for k, v in report['trials'].items()}}, indent=2))


if __name__ == '__main__':
    main()
