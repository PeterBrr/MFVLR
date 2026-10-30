"""No-update AR/FL gradient audit on common CNN/decoder parameters and fixed train pairs."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.pilot_deps'), str(ROOT / 'src'), str(ROOT / 'tools')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from export_ad_reconstructions import file_hash
from mfvlr.objectives import appearance_reconstruction_loss
from mfvlr_research.pilot_datasets import dataset_for_config
from mfvlr_research.pilot_runtime import load_checkpoint, check_finite_tensors
from mfvlr_research.pilot_variant import visual_model, weighted_mask_loss


def measure_groups(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if name.startswith('mve.shared_encoder.cnn.'):
            stage = name.split('.')[4]
            labels = ('encoder_cnn', f'encoder_stage_{stage}')
        elif name.startswith('decoder.trunk.'):
            stage = name.split('.')[3]
            labels = ('decoder_trunk', f'decoder_stage_{int(stage) + 1}')
        elif name.startswith('decoder.appearance_head.'):
            labels = ('appearance_head',)
        elif name.startswith('decoder.mask_head.'):
            labels = ('mask_head',)
        else:
            continue
        groups[name] = (parameter, labels)
    return groups


def group_statistics(parameters, ar_gradients, fl_gradients, lambdas):
    result = {}
    for (_, (parameter, labels)), ar, fl in zip(parameters.items(), ar_gradients, fl_gradients):
        a = None if ar is None else ar.double()
        f = None if fl is None else fl.double()
        aa = 0. if a is None else float(a.square().sum())
        ff = 0. if f is None else float(f.square().sum())
        af = 0. if a is None or f is None else float((a * f).sum())
        for label in labels:
            row = result.setdefault(label, dict(ar_sq=0., fl_sq=0., dot=0., parameters=0,
                ar_connected_tensors=0, fl_connected_tensors=0, both_connected_tensors=0))
            row['ar_sq'] += aa
            row['fl_sq'] += ff
            row['dot'] += af
            row['parameters'] += parameter.numel()
            row['ar_connected_tensors'] += int(a is not None)
            row['fl_connected_tensors'] += int(f is not None)
            row['both_connected_tensors'] += int(a is not None and f is not None)
    for row in result.values():
        an, fn = math.sqrt(row.pop('ar_sq')), math.sqrt(row.pop('fl_sq'))
        dot = row.pop('dot')
        row.update(ar_norm=an, fl_norm=fn,
                   cosine=None if not an or not fn else dot / (an * fn),
                   raw_fl_over_ar=None if not an else fn / an,
                   weighted_fl_over_ar=None if not an or not lambdas['ar'] else
                       lambdas['fl'] * fn / (lambdas['ar'] * an))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.batches < 1:
        parser.error('--batches must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output is not empty; choose a new directory')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    before_hash = file_hash(args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint['config']
    if config.get('model', {}).get('variant') != 'skip_resize_conv_v2':
        raise ValueError('Expected skip/manual-loss checkpoint')
    for relative, expected in checkpoint['provenance']['source'].items():
        if relative.replace('\\', '/').startswith('src/') and file_hash(ROOT / relative) != expected:
            raise ValueError(f'Checkpoint source mismatch: {relative}')
    manifest = ROOT / config['manifests']['train']
    if file_hash(manifest) != checkpoint['provenance']['manifests']['train']:
        raise ValueError('Training manifest changed')
    data = dataset_for_config(ROOT, config, 'train', limit=checkpoint['recipe']['limit'])
    groups = [[i for i, row in enumerate(data.rows) if row['label'] == label] for label in (0, 1)]
    if min(map(len, groups)) < args.batches:
        raise ValueError('Not enough distinct real/fake pairs')
    selected = [[group[i] for i in np.linspace(0, len(group) - 1, args.batches, dtype=int)] for group in groups]
    order = [i for pair in zip(*selected) for i in pair]
    batches = list(DataLoader(Subset(data, order), batch_size=2, shuffle=False))
    device = torch.device(args.device)
    class_weights = torch.tensor([config['model']['mask_background_weight'],
                                  config['model']['mask_foreground_weight']], device=device)
    report = {'status': 'RUNNING', 'checkpoint': str(args.checkpoint.resolve()),
              'checkpoint_sha256': before_hash, 'step': checkpoint['step'],
              'config': config, 'manifest_sha256': file_hash(manifest),
              'recorded_source': checkpoint['provenance']['source'],
              'scope': 'No optimizer updates, same fixed training pairs at fresh and trained states; raw FP32 gradients reduced in FP64',
              'groups': 'CNN and decoder aggregate groups overlap with their stage groups; do not sum parent and child groups',
              'states': {}}
    args.output.mkdir(parents=True)
    for state in ('fresh', 'checkpoint'):
        torch.manual_seed(0)
        model = visual_model(config).to(device).eval()
        if state == 'checkpoint':
            weights = {}
            for name, tensor in checkpoint['model'].items():
                if name.startswith(('mve.', 'decoder.')):
                    weights[name] = tensor
                elif name.startswith('heads.detector.'):
                    weights[name.removeprefix('heads.')] = tensor
            model.load_state_dict(weights, strict=True)
            del weights
        params = measure_groups(model)
        rows = []
        for batch_number, batch in enumerate(batches):
            images = batch['image'].to(device)
            encoding = model.mve.shared_encoder.encode_local(images, return_skips=True)
            if config['model']['skip_connections'] == 'concat':
                decoded = model.decoder(encoding.local, skips=encoding.skips)
            else:
                decoded = model.decoder(encoding.local)
            if batch_number == 0:
                with torch.no_grad():
                    reference = model(images)
                    torch.testing.assert_close(decoded.reconstruction, reference.decoded.reconstruction, rtol=0, atol=0)
                    torch.testing.assert_close(decoded.mask_logits, reference.decoded.mask_logits, rtol=0, atol=0)
                del reference
            ar = appearance_reconstruction_loss(images, decoded.reconstruction)
            fl = weighted_mask_loss(decoded.mask_logits, batch['mask'].to(device), class_weights)
            targets = tuple(p for p, labels in params.values())
            ar_grads = torch.autograd.grad(ar, targets, retain_graph=True, allow_unused=True)
            check_finite_tensors(ar_grads, 'Nonfinite AR gradient')
            ar_cpu = [None if g is None else g.detach().cpu() for g in ar_grads]
            del ar_grads
            fl_grads = torch.autograd.grad(fl, targets, allow_unused=True)
            check_finite_tensors(fl_grads, 'Nonfinite FL gradient')
            fl_cpu = [None if g is None else g.detach().cpu() for g in fl_grads]
            stats = group_statistics(params, ar_cpu, fl_cpu, config['loss_weights'])
            rows.append({'batch': batch_number, 'sample_ids': batch['sample_id'],
                         'ar': float(ar.detach()), 'fl': float(fl.detach()),
                         'foreground_fraction': batch['mask'].float().mean((1, 2)).tolist(), 'groups': stats})
            print(f'{state}: measured pair {batch_number + 1}/{len(batches)}', flush=True)
            del encoding, decoded, ar, fl, fl_grads, ar_cpu, fl_cpu
        summary = {}
        for group in rows[0]['groups']:
            cosines = [row['groups'][group]['cosine'] for row in rows if row['groups'][group]['cosine'] is not None]
            ratios = [row['groups'][group]['weighted_fl_over_ar'] for row in rows
                      if row['groups'][group]['weighted_fl_over_ar'] is not None]
            summary[group] = {'cosine_mean': float(np.mean(cosines)) if cosines else None,
                              'cosine_min': min(cosines) if cosines else None,
                              'cosine_max': max(cosines) if cosines else None,
                              'negative_batches': sum(x < 0 for x in cosines), 'valid_cosine_batches': len(cosines),
                              'mean_weighted_fl_over_ar': float(np.mean(ratios)) if ratios else None}
        report['states'][state] = {'batches': rows, 'summary': summary, 'direct_route_matches_full_forward': True}
        (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        del params, model
        gc.collect()
        torch.cuda.empty_cache()
    assert file_hash(args.checkpoint) == before_hash
    for relative, expected in report['recorded_source'].items():
        if relative.replace('\\', '/').startswith('src/'):
            assert file_hash(ROOT / relative) == expected
    report.update(status='COMPLETED', source_checkpoint_unchanged=True, optimizer_updates=0)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({s: {g: report['states'][s]['summary'][g] for g in ('encoder_cnn', 'decoder_trunk')}
                      for s in report['states']}, indent=2))


if __name__ == '__main__':
    main()
