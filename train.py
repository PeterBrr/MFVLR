"""Train an internal dataset pilot; dataset/model/gradient/loss policy is explicit."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / '.pilot_deps'), str(ROOT / 'src')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from mfvlr_research.pilot_datasets import dataset_for_config, audit_source_groups
from mfvlr_research.pilot_runtime import (atomic_save, provenance, rng_state, restore_rng,
                                        load_checkpoint, evaluate_visual, check_finite_tensors, gradient_norm)
from mfvlr_research.pilot_optimization import (validate_optimization, build_optimization,
                                              restore_scheduler)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', type=Path, default=ROOT / 'configs/diffae_pilot.json')
    result.add_argument('--stage', choices=['tiny', 'pilot'], default='tiny')
    result.add_argument('--output', type=Path)
    result.add_argument('--resume', type=Path)
    result.add_argument('--check-data', action='store_true', help='Verify all selected files and tokens without constructing a model')
    result.add_argument('--max-steps', type=int, help='Total step budget including resumed steps')
    return result


def run(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['schema'] != 'mfvlr-internal-pilot-v1' or config['seed'] != 0:
        raise ValueError('This A0 pilot fixes schema and initialization seed=0')
    recipe = dict(config[args.stage])
    from mfvlr_research.pilot_variant import variant_settings, training_model, language_enabled
    settings = variant_settings(config)
    needs_language = language_enabled(config)
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError('--max-steps must be positive')
        recipe['max_steps'] = args.max_steps
    batch_size = config['batch_size']
    validate_optimization(config, recipe)
    if type(batch_size) is not int or batch_size < 2 or config['checkpoint_every'] < 1:
        raise ValueError('Training requires batch >=2 for CMC and a positive checkpoint interval')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    dataset = dataset_for_config(ROOT, config, 'train', tokens=needs_language, limit=recipe['limit'])
    validation = dataset_for_config(ROOT, config, 'val')
    # Audit all partitions' content/ID memberships before model allocation.
    test = dataset_for_config(ROOT, config, 'test')
    partitions = [('train', dataset_for_config(ROOT, config, 'train')), ('val', validation), ('test', test)]
    source_audit = audit_source_groups(partitions)
    seen_ids, seen_content = set(), set()
    for part, data in partitions:
        if any(r['split'] != part for r in data.rows):
            raise ValueError('Manifest split mismatch')
        for row in data.rows:
            if row['sample_id'] in seen_ids or row['decoded_rgb_sha256'] in seen_content:
                raise ValueError('Cross-split overlap')
            seen_ids.add(row['sample_id'])
            seen_content.add(row['decoded_rgb_sha256'])
    if args.check_data:
        for data in (dataset, validation, test):
            for index in range(len(data)):
                data[index]
        print(json.dumps({'status': 'DATA_CHECK_PASS', 'train': len(dataset), 'val': len(validation), 'test': len(test),
                          'subject_disjoint_verified': False, **source_audit, 'model_executed': False}), flush=True)
        return
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    run_name = args.stage if settings is None else f'{args.stage}_{settings["variant"]}'
    if 'run_name' in config:
        run_name = config['run_name'] if args.stage == 'pilot' else f'tiny_{config["run_name"]}'
    output = args.output or ROOT / config.get('output_root', 'artifacts/diffae_pilot/runs') / run_name
    output = output.resolve()
    if args.output is None and not args.resume and output.exists() and any(output.iterdir()):
        base, number = output, 1
        while output.exists():
            output = base.with_name(f'{base.name}_{number:03d}')
            number += 1
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output.iterdir()):
        raise ValueError('Selected --output directory is not empty; choose a new --output for a fresh run')
    print(f'Output: {output}', flush=True)
    run_contract = {'config': config, 'stage': args.stage, 'recipe': recipe}
    pins = provenance(ROOT, run_contract['config'])
    (output / 'run.json').write_text(json.dumps({**run_contract, 'provenance': pins, **source_audit}, indent=2), encoding='utf-8')
    status = {'status': 'STARTING', 'step': 0, 'phase': 'construct_model'}
    start_time = time.time()
    try:
        from mfvlr.integration import SuppliedTargets
        model = training_model(config).to(device)
        model.train()
        optimizer, scheduler = build_optimization(model, config, recipe)
        epoch, offset, step, best, threshold = 0, 0, 0, None, .5
        if args.resume:
            checkpoint = load_checkpoint(args.resume)
            if checkpoint['provenance'] != pins or checkpoint['stage'] != args.stage or checkpoint['device'] != str(device):
                raise ValueError('Resume source/data/runtime/stage/device provenance differs')
            if checkpoint['recipe'] != {k:v for k,v in recipe.items() if k != 'max_steps'}:
                raise ValueError('Resume recipe differs')
            model.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch, offset, step = checkpoint['epoch'], checkpoint['offset'], checkpoint['step']
            restore_scheduler(scheduler, checkpoint.get('scheduler'), epoch)
            best, threshold = checkpoint['best'], checkpoint['threshold']
            restore_rng(checkpoint['rng'])
            del checkpoint
        status['step'] = step
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        def save(path):
            atomic_save({'schema': 'mfvlr-pilot-v1', 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'rng': rng_state(), 'epoch': epoch, 'offset': offset, 'step': step, 'best': best,
                'threshold': threshold, 'config': config, 'stage': args.stage, 'device': str(device),
                'recipe': {k:v for k,v in recipe.items() if k != 'max_steps'}, 'provenance': pins}, path)

        total_steps = (len(dataset) // batch_size) * recipe['epochs']
        if recipe['max_steps'] is not None:
            total_steps = min(total_steps, recipe['max_steps'])
        with tqdm(total=max(step, total_steps), initial=step, unit='step', dynamic_ncols=True,
                  ascii=True, desc=f'Train {args.stage}') as progress, \
                (output / 'history.jsonl').open('a', encoding='utf-8') as history:
            if config.get('validation_before_training', False) and args.stage == 'pilot' and not args.resume:
                progress.set_description('Initial validation', refresh=True)
                initial = evaluate_visual(model, validation, device, batch_size, threshold=.5)
                history.write(json.dumps({'event': 'initial_validation', 'step': step,
                                          'validation': initial}) + '\n')
                history.flush()
                model.train()
            while epoch < recipe['epochs'] and (recipe['max_steps'] is None or step < recipe['max_steps']):
                progress.set_description(f'{args.stage} | Epoch {epoch + 1}/{recipe["epochs"]}', refresh=False)
                order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(config['seed'] + epoch)).tolist()
                usable = len(order) - len(order) % batch_size
                if usable == 0:
                    raise ValueError('Training subset smaller than batch size')
                loader = DataLoader(Subset(dataset, order[offset:usable]), batch_size=batch_size, shuffle=False,
                                    num_workers=0, drop_last=True, generator=torch.Generator().manual_seed(epoch))
                previous_step_end = time.perf_counter()
                for batch in loader:
                    batch_ready = time.perf_counter()
                    status['phase'] = 'forward'
                    optimizer.zero_grad(set_to_none=True)
                    images = batch['image'].to(device)
                    tokens = (batch['token_ids'].to(device) if needs_language else
                              torch.zeros((len(images), 308), dtype=torch.long, device=device))
                    ids = tuple(batch['sample_id'])
                    result = model.with_language(images, tokens, sample_ids=ids)
                    losses = model.compute_objectives(result, SuppliedTargets(ids, tokens,
                        batch['label'].to(device), batch['mask'].to(device), tuple(batch['kind'])))
                    values = losses.as_dict()
                    scalars = dict(zip(values, torch.stack([v.detach() for v in values.values()]).cpu().tolist()))
                    if not all(np.isfinite(v) for v in scalars.values()):
                        raise FloatingPointError('Nonfinite loss; backward not attempted')
                    forward_end = time.perf_counter()
                    status['phase'] = 'backward'
                    losses.total.backward()
                    check_finite_tensors((p.grad for p in model.parameters()),
                                         'Nonfinite gradient; optimizer step not attempted')
                    grad_norm = gradient_norm(model.parameters())
                    backward_end = time.perf_counter()
                    status['phase'] = 'optimizer_step'
                    optimizer.step()
                    check_finite_tensors(model.parameters(), 'Nonfinite updated parameter; retain previous checkpoint')
                    check_finite_tensors((v for state in optimizer.state.values() for v in state.values()),
                                         'Nonfinite optimizer state; retain previous checkpoint')
                    optimizer_end = time.perf_counter()
                    step += 1
                    offset += len(ids)
                    status['step'] = step
                    record = {'step': step, 'epoch': epoch, 'offset': offset, 'losses': scalars,
                        'grad_norm': grad_norm, 'lr': optimizer.param_groups[0]['lr'],
                        'timings': {'data_seconds': batch_ready - previous_step_end,
                            'forward_loss_seconds': forward_end - batch_ready,
                            'backward_checks_seconds': backward_end - forward_end,
                            'optimizer_checks_seconds': optimizer_end - backward_end},
                        'elapsed_seconds': time.time() - start_time,
                        'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0}
                    history.write(json.dumps(record) + '\n')
                    history.flush()
                    progress.set_postfix(loss=f'{scalars["total"]:.4g}', fd=f'{scalars["fd"]:.4g}',
                                         fl=f'{scalars["fl"]:.4g}', lr=f'{optimizer.param_groups[0]["lr"]:.2g}',
                                         refresh=False)
                    progress.update(1)
                    del result, losses, values, images, tokens, batch
                    if step % config['checkpoint_every'] == 0:
                        status['phase'] = 'save_checkpoint'
                        save_started = time.perf_counter()
                        save(output / 'last.pt')
                        history.write(json.dumps({'event': 'checkpoint', 'step': step,
                                                  'seconds': time.perf_counter() - save_started}) + '\n')
                        history.flush()
                    previous_step_end = time.perf_counter()
                    if recipe['max_steps'] is not None and step >= recipe['max_steps']:
                        break
                if offset >= usable:
                    status['phase'] = 'validation'
                    progress.set_description(f'{args.stage} | Validation epoch {epoch + 1}', refresh=True)
                    optimizer.zero_grad(set_to_none=True)
                    validation_result = evaluate_visual(model, validation, device, batch_size)
                    threshold = validation_result['threshold']
                    rank = (validation_result['auc'], -validation_result['fd_ce'])
                    improved = best is None or rank > tuple(best)
                    if improved:
                        best = rank
                    epoch += 1
                    offset = 0
                    # Advance once per completed epoch, never per batch or partial epoch.
                    # With step_size=15, epochs 1--15 use the initial LR; epoch 16 uses LR/10.
                    if scheduler is not None:
                        scheduler.step()
                    history.write(json.dumps({'epoch_completed': epoch, 'validation': validation_result,
                                              'lr_next_epoch': optimizer.param_groups[0]['lr']}) + '\n')
                    history.flush()
                    if improved:
                        save(output / 'best.pt')
                    model.train()
                save(output / 'last.pt')
        status.update(status='COMPLETED', step=step, epoch=epoch, offset=offset, phase='finished')
    except BaseException as error:
        status.update(status='STOPPED', error=type(error).__name__, message=str(error))
        raise
    finally:
        status['elapsed_seconds'] = time.time() - start_time
        (output / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')


if __name__ == '__main__':
    run(parser().parse_args())
