"""Six matched GenFace-only training trials for skip, detach, FL weight and language choices."""
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
import torch
from tqdm.auto import tqdm

from export_ad_reconstructions import file_hash
from test_ad_overfit import state_hash
from test_split_mask_encoder_pilot import balanced_subset, collate, evaluate, save_preview
from pilot_checkpointing import enable_checkpointing
from mfvlr.integration import SuppliedTargets
from mfvlr.flt import FineGrainedLanguageTransformer
from mfvlr.objectives import SixObjectives, CrossModalContrastiveLoss, LossValues
from mfvlr_research.pilot_skip import weighted_objective_total
from mfvlr_research.pilot_variant import training_model, weighted_mask_loss
from mfvlr_research.pilot_datasets import dataset_for_config
from mfvlr_research.pilot_optimization import build_optimization
from mfvlr_research.pilot_runtime import check_finite_tensors


TRIALS = (
    ('A_no_skip', 'none', 'coupled', 1., 'off'),
    ('B_skip', 'concat', 'coupled', 1., 'off'),
    ('C_skip_detach', 'concat', 'detach_reconstruction', 1., 'off'),
    ('D_manual_fl', 'concat', 'detach_reconstruction', .25, 'off'),
    ('E_language_legacy', 'concat', 'detach_reconstruction', .25, 'legacy'),
    ('F_language_corrected', 'concat', 'detach_reconstruction', .25, 'corrected'),
)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


class MetricView:
    def __init__(self, model):
        self.model = model

    def eval(self):
        self.model.eval()
        self.rgb_statistics = dict(minimum=float('inf'), maximum=-float('inf'),
                                   outside_count=0, near_saturation_count=0, channel_value_count=0)
        return self

    def train(self):
        self.model.train()
        return self

    def __call__(self, images):
        output = self.model.visual_only(images)
        rgb = output.decoded.reconstruction.detach()
        stats = self.rgb_statistics
        stats['minimum'] = min(stats['minimum'], float(rgb.min()))
        stats['maximum'] = max(stats['maximum'], float(rgb.max()))
        stats['outside_count'] += int(((rgb < 0) | (rgb > 1)).sum())
        stats['near_saturation_count'] += int(((rgb < .01) | (rgb > .99)).sum())
        stats['channel_value_count'] += rgb.numel()
        return {'reconstruction': output.decoded.reconstruction,
                'mask_logits': output.decoded.mask_logits, 'detection_logits': output.detection_logits}


def evaluate_with_rgb(view, samples, weights):
    metric, preview = evaluate(view, samples, 'cuda', weights, 2)
    metric['reconstruction_output'] = dict(view.rgb_statistics)
    return metric, preview


def legacy_losses(model, output, targets, weights):
    # Same historical formulas, including PAD LR/raw-dot CMC; same image FL as F.
    p = output.predictions
    old = SixObjectives.forward(model.objectives,
        detection_logits=p.detection_logits, labels=targets.labels,
        language_logits=output.language.logits, token_ids=targets.token_ids,
        visual=p.visual.fused_global, global_language=output.language.global_language,
        mask_logits=p.decoded.mask_logits, masks=targets.masks,
        image=output.image, reconstruction=p.decoded.reconstruction,
        predicted_language=output.predicted_language, sample_kinds=targets.sample_kinds)
    values = {name: value for name, value in old.as_dict().items() if name != 'total'}
    values['fl'] = weighted_mask_loss(p.decoded.mask_logits, targets.masks, weights)
    return LossValues(**values, total=weighted_objective_total(values, model.objectives.weights))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--train-images', type=int, default=128)
    parser.add_argument('--val-images', type=int, default=128)
    parser.add_argument('--experiment', choices=('six', 'sigmoid'), default='six')
    args = parser.parse_args()
    trials = TRIALS if args.experiment == 'six' else (
        ('D_manual_fl', 'concat', 'detach_reconstruction', .25, 'off'),
        ('G_sigmoid', 'concat', 'detach_reconstruction', .25, 'off'),
    )
    if args.output.exists():
        parser.error('Output exists; select a new run directory')
    if args.updates < 1:
        parser.error('--updates must be positive')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    base = json.loads((ROOT / 'configs/diffae_pilot_skip_detach.json').read_text())
    train_data = dataset_for_config(ROOT, base, 'train', tokens=True, limit=1000)
    val_data = dataset_for_config(ROOT, base, 'val', tokens=True)
    train = balanced_subset(train_data, args.train_images)
    val = balanced_subset(val_data, args.val_images)
    assert not ({s['sample_id'] for s in train} & {s['sample_id'] for s in val})
    random = np.random.default_rng(0)
    schedule = []
    while len(schedule) < args.updates:
        order = random.permutation(len(train)).tolist()
        schedule.extend(order[i:i + 4] for i in range(0, len(order), 4))
    schedule = schedule[:args.updates]
    configs = {}
    for name, skips, residual, fl, language in trials:
        cfg = copy.deepcopy(base)
        cfg['run_name'] = name
        cfg['model'].update(skip_connections=skips, residual_gradient=residual,
                            language_visual_gradients='coupled' if language != 'off' else 'stop_all')
        cfg['loss_weights']['fl'] = fl
        if name == 'G_sigmoid':
            cfg['model']['reconstruction_activation'] = 'sigmoid'
            cfg['protocol']['rgb_output'] = 'Sigmoid at RGB head, before AR and residual; mask logits unchanged'
        if language != 'off':
            cfg['loss_weights'].update(lr=1e-4, cmc=.01, kl=.01)
        cfg['protocol'].update(subset=f'{len(train)} fixed GenFace train, {len(val)} fixed validation; test unused',
            decoder=f'{skips} skips; same shared AD/MD trunk', gradient=residual,
            language=language, losses=str(cfg['loss_weights']),
            scope='From scratch, 100 matched updates by default; single-seed internal ablation, activation checkpointing')
        configs[name] = cfg
    args.output.mkdir(parents=True)
    files = [*ROOT.joinpath('src/mfvlr').glob('*.py'), *ROOT.joinpath('src/mfvlr_research').glob('pilot_*.py')]
    files += [Path(__file__), ROOT / 'tools/pilot_checkpointing.py', ROOT / 'tools/test_split_mask_encoder_pilot.py']
    report = {'status': 'RUNNING', 'experiment': args.experiment,
              'dataset': 'GenFace real + Diffae only', 'test_used': False,
              'initialization': 'fresh seed0; no trained weights loaded', 'microbatch': 2, 'effective_batch': 4,
              'cmc_negatives': 'local microbatch of 2; accumulation does not create a four-sample similarity matrix',
              'updates_per_trial': args.updates, 'activation_checkpointing': True,
              'train_ids': [s['sample_id'] for s in train], 'val_ids': [s['sample_id'] for s in val],
              'schedule': schedule, 'configs': configs, 'trials': {},
              'manifest_sha256': {s: file_hash(ROOT / base['manifests'][s]) for s in ('train', 'val')},
              'source_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in files}}
    write_json(args.output / 'report.json', report)
    previews = {}
    hashes = {}
    for name, skips, residual, fl, language in trials:
        cfg = configs[name]
        folder = args.output / name
        folder.mkdir()
        write_json(folder / 'config.json', cfg)
        trial = report['trials'][name] = {'status': 'STARTING', 'language_mode': language, 'history': []}
        write_json(args.output / 'report.json', report)
        model = optimizer = view = None
        try:
            torch.manual_seed(0)
            model = training_model(cfg)
            if language == 'legacy':
                model.flt = FineGrainedLanguageTransformer(seed=0)
                model.objectives.cmc = CrossModalContrastiveLoss()
            trial['initial_state_sha256'] = state_hash(model)
            if args.experiment == 'sigmoid':
                hashes[name] = trial['initial_state_sha256']
                assert len(set(hashes.values())) == 1, 'Sigmoid/control initial weights differ'
            trial['image_initial_hashes'] = {key: state_hash(module) for key, module in
                (('encoder', model.mve), ('decoder', model.decoder), ('detector', model.heads.detector))}
            if language in ('legacy', 'corrected'):
                hashes[language] = trial['initial_state_sha256']
                if language == 'corrected':
                    assert hashes['legacy'] == hashes['corrected'], 'E/F initial weights differ'
            trial['parameters'] = sum(p.numel() for p in model.parameters())
            model.cuda().train()
            trial['checkpointed_stages'] = enable_checkpointing(model)
            weights = torch.tensor([1., 4.], device='cuda')
            view = MetricView(model)
            trial['initial_val'], _ = evaluate_with_rgb(view, val, weights)
            optimizer, _ = build_optimization(model, cfg, {'lr': 1e-5})
            torch.cuda.reset_peak_memory_stats()
            trial['status'] = 'RUNNING'
            start = time.perf_counter()
            write_json(args.output / 'report.json', report)
            bar = tqdm(enumerate(schedule, 1), total=len(schedule), desc=name, unit='update', dynamic_ncols=True)
            for step, selected in bar:
                optimizer.zero_grad(set_to_none=True)
                means = {}
                for offset in (0, 2):
                    batch = collate([train[i] for i in selected[offset:offset + 2]])
                    images, tokens = batch['image'].cuda(), batch['token_ids'].cuda()
                    ids = tuple(batch['sample_id'])
                    output = model.with_language(images, tokens, sample_ids=ids)
                    targets = SuppliedTargets(ids, tokens, batch['label'].cuda(), batch['mask'].cuda(), tuple(batch['kind']))
                    losses = (legacy_losses(model, output, targets, weights) if language == 'legacy'
                              else model.compute_objectives(output, targets))
                    (losses.total / 2).backward()
                    for key, value in losses.as_dict().items():
                        means[key] = means.get(key, 0.) + float(value.detach()) / 2
                    del output, targets, losses
                check_finite_tensors((p.grad for p in model.parameters()), 'Nonfinite ablation gradient')
                optimizer.step()
                check_finite_tensors(model.parameters(), 'Nonfinite ablation parameter')
                trial['history'].append({'step': step, 'losses': means, 'seconds': time.perf_counter() - start})
                bar.set_postfix(ar=f'{means["ar"]:.4f}', fl=f'{means["fl"]:.4f}', lr=f'{means["lr"]:.1f}')
                if step % 10 == 0:
                    write_json(args.output / 'report.json', report)
                if step in (50, args.updates):
                    optimizer.zero_grad(set_to_none=True)
                    measured, preview = evaluate_with_rgb(view, val, weights)
                    trial.setdefault('evaluations', []).append({'step': step, 'val': measured})
                    if step == args.updates:
                        previews[name] = preview
                    write_json(args.output / 'report.json', report)
            trial['final_train'], _ = evaluate_with_rgb(view, train, weights)
            trial['seconds'] = time.perf_counter() - start
            trial['peak_cuda_allocated_MiB'] = torch.cuda.max_memory_allocated() / 2**20
            cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save({'schema': 'mfvlr-genface-controlled-ablation-v1', 'config': cfg,
                        'language_mode': language, 'model': cpu_state, 'step': args.updates}, folder / 'final.pt')
            del cpu_state
            trial['status'] = 'COMPLETED'
        except Exception as error:
            trial.update(status='FAILED', error=f'{type(error).__name__}: {error}')
            print(f'{name} FAILED: {trial["error"]}', flush=True)
        finally:
            output = targets = losses = images = tokens = None
            del optimizer, view, model
            gc.collect()
            torch.cuda.empty_cache()
            write_json(args.output / 'report.json', report)
    for left, right in zip([t[0] for t in trials[:-1]], [t[0] for t in trials[1:]]):
        if left in previews and right in previews:
            save_preview(args.output / f'val_{left}_vs_{right}', previews[left], previews[right])
    assert all(file_hash(ROOT / relative) == expected for relative, expected in report['source_sha256'].items())
    report.update(status='COMPLETED' if all(t['status'] == 'COMPLETED' for t in report['trials'].values()) else 'COMPLETED_WITH_FAILURES',
                  source_unchanged=True)
    if args.experiment == 'sigmoid':
        report['sigmoid_initial_weights_identical'] = len(set(hashes.values())) == 1
    else:
        report['language_initial_weights_identical'] = hashes.get('legacy') == hashes.get('corrected')
    write_json(args.output / 'report.json', report)
    print(json.dumps({'status': report['status'], 'output': str(args.output.resolve())}, indent=2))


if __name__ == '__main__':
    main()
