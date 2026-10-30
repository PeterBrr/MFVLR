"""Matched FD-weight overfit trials, explicit flushed logs and FD/AR/FL probes."""
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
import torch
import run_loss_overfit as base
from measure_ar_fl_gradients import group_statistics, measure_groups

base.TRIALS = {
    'FD_0': dict(ar=1., fd=0., fl=1.),
    'FD_0p1': dict(ar=1., fd=.1, fl=1.),
    'FD_0p25': dict(ar=1., fd=.25, fl=1.),
    'FD_1': dict(ar=1., fd=1., fl=1.),
}
OUTPUT = None
START = time.monotonic()


def log(event, **fields):
    row = dict(event=event, elapsed_seconds=round(time.monotonic() - START, 2), **fields)
    line = json.dumps(row, allow_nan=False)
    print(line, flush=True)
    if OUTPUT is not None:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with (OUTPUT / 'live_log.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
            stream.flush()


class LiveProgress:
    def __init__(self, iterable, total, desc, **kwargs):
        self.iterable, self.total, self.name, self.step = iterable, total, desc, 0
        log('trial_train_start', trial=desc, updates=total)

    def __iter__(self):
        for step, selected in self.iterable:
            self.step = step
            yield step, selected

    def set_postfix(self, values):
        log('train', trial=self.name, step=self.step, total=self.total,
            losses={k: float(v) for k, v in values.items()})


def pair_statistics(parameters, left, right, left_weight, right_weight):
    values = group_statistics(parameters, left, right, dict(ar=left_weight, fl=right_weight))
    result = {}
    for group, row in values.items():
        result[group] = dict(left_norm=row['ar_norm'], right_norm=row['fl_norm'],
            cosine=row['cosine'], weighted_left_norm=left_weight * row['ar_norm'],
            weighted_right_norm=right_weight * row['fl_norm'],
            left_connected_tensors=row['ar_connected_tensors'],
            right_connected_tensors=row['fl_connected_tensors'],
            both_connected_tensors=row['both_connected_tensors'])
    return result


def fd_gradient_probe(model, samples, device, weights, lambdas, pairs):
    model.eval()
    parameters = measure_groups(model)
    rows = []
    log('gradient_probe_start', weights=lambdas, pairs=pairs)
    for index in np.linspace(0, 7, pairs, dtype=int):
        batch = base.collate(samples[2 * index:2 * index + 2])
        image, target = batch['image'].to(device), batch['mask'].to(device)
        output = model(image)
        values = dict(ar=base.appearance_reconstruction_loss(image, output.decoded.reconstruction),
                      fl=base.weighted_mask_loss(output.decoded.mask_logits, target, weights),
                      fd=base.forgery_detection_loss(output.detection_logits, batch['label'].to(device)))
        gradients = {}
        for i, key in enumerate(('ar', 'fl', 'fd')):
            grads = torch.autograd.grad(values[key], tuple(p for p, _ in parameters.values()),
                                        retain_graph=i < 2, allow_unused=True)
            base.check_finite_tensors(grads, 'Nonfinite ' + key + ' diagnostic gradient')
            gradients[key] = [None if g is None else g.detach().cpu() for g in grads]
            del grads
        statistics = {}
        for left, right in (('ar', 'fl'), ('fd', 'fl'), ('fd', 'ar')):
            statistics[left + '_' + right] = pair_statistics(parameters, gradients[left], gradients[right],
                                                           lambdas[left], lambdas[right])
        # Detach is intentionally retained; FD must not reach decoder parameters.
        assert all(g is None for (_, (_, labels)), g in zip(parameters.items(), gradients['fd'])
                   if any(label.startswith(('decoder_', 'appearance_head', 'mask_head')) for label in labels))
        row = dict(sample_ids=batch['sample_id'], losses={k: float(v.detach()) for k,v in values.items()},
                   pairs=statistics)
        rows.append(row)
        log('gradient_pair', pair=len(rows), sample_ids=row['sample_ids'],
            fd_fl_cnn_cosine=statistics['fd_fl']['encoder_cnn']['cosine'],
            fd_ar_cnn_cosine=statistics['fd_ar']['encoder_cnn']['cosine'],
            ar_fl_trunk_cosine=statistics['ar_fl']['decoder_trunk']['cosine'])
        del output, values, gradients
    model.train()
    return rows


_write_json = base.write_json
_last_state = {}


def write_report(path, report):
    global OUTPUT
    OUTPUT = path.parent
    relative = str(Path(__file__).relative_to(ROOT))
    report['source_sha256'][relative] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['experiment'] = 'FD weights0/.1/.25/1; AR=FL=1; no architecture changes'
    report['gradient_probe'] = 'Raw AR/FL/FD gradients at initial/final state; CNN common to all; inactive FD still probed; no optimizer update'
    report['logging'] = 'One flushed JSON line per update, metrics every50, probe/phase messages; live_log.jsonl'
    _write_json(path, report)
    for name, trial in report['trials'].items():
        evaluations = trial.get('evaluations', [])
        marker = (trial['status'], len(evaluations))
        if _last_state.get(name) != marker:
            _last_state[name] = marker
            log('trial_state', trial=name, status=trial['status'],
                saved_step=len(trial['history']), error=trial.get('error'))
            if evaluations:
                log('metrics', trial=name, **evaluations[-1])
    _write_json(OUTPUT / 'progress.json', dict(status=report['status'],
        trials={name: dict(status=t['status'], step=len(t['history']),
                           last_evaluation=t.get('evaluations', [None])[-1]) for name,t in report['trials'].items()}))


if __name__ == '__main__':
    base.tqdm = LiveProgress
    base.gradient_probe = fd_gradient_probe
    base.write_json = write_report
    log('bootstrap', python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        experiment=base.TRIALS)
    base.main()
