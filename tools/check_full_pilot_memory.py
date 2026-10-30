"""Bounded memory/numerical admission for two real full-language pilot updates."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.pilot_deps'), str(ROOT / 'src')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch.utils.data import DataLoader
from mfvlr.integration import SuppliedTargets
from mfvlr_research.pilot_variant import training_model
from mfvlr_research.pilot_datasets import dataset_for_config
from mfvlr_research.pilot_runtime import check_finite_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-activations', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    config = json.loads((ROOT / 'configs/diffae_pilot_skip_detach.json').read_text())
    config['model']['language_visual_gradients'] = 'coupled'
    config['loss_weights'].update(lr=1e-4, cmc=.01, kl=.01)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    data = dataset_for_config(ROOT, config, 'train', tokens=True, limit=4)
    batches = list(DataLoader(data, batch_size=2, shuffle=False))
    report = {'status': 'RUNNING', 'config': config, 'scope': 'Two disposable admission updates, not a pilot result', 'updates': []}
    try:
        model = training_model(config).cuda().train()
        if args.checkpoint_activations:
            from pilot_checkpointing import enable_checkpointing
            report['checkpointed_stages'] = enable_checkpointing(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=.001, foreach=False)
        torch.cuda.reset_peak_memory_stats()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            means = {}
            for batch in batches:
                images, tokens = batch['image'].cuda(), batch['token_ids'].cuda()
                ids = tuple(batch['sample_id'])
                out = model.with_language(images, tokens, sample_ids=ids)
                losses = model.compute_objectives(out, SuppliedTargets(ids, tokens,
                    batch['label'].cuda(), batch['mask'].cuda(), tuple(batch['kind'])))
                (losses.total / 2).backward()
                for name, value in losses.as_dict().items():
                    means[name] = means.get(name, 0.) + float(value.detach()) / 2
                del out, losses
            check_finite_tensors((p.grad for p in model.parameters()), 'Nonfinite full pilot gradients')
            optimizer.step()
            check_finite_tensors(model.parameters(), 'Nonfinite full pilot parameters')
            report['updates'].append(means)
        report['status'] = 'PASS'
    except torch.OutOfMemoryError as error:
        report.update(status='CUDA_OOM', error=str(error).split('\n')[0])
    except Exception as error:
        report.update(status='FAILED', error=f'{type(error).__name__}: {error}')
    report['peak_cuda_allocated_MiB'] = torch.cuda.max_memory_allocated() / 2**20
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
