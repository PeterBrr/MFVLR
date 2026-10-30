"""Fixed 16-image GenFace loss ablation, portable to a private Kaggle notebook."""
import argparse
import copy
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
import torch
from tqdm.auto import tqdm

from mfvlr.objectives import appearance_reconstruction_loss, forgery_detection_loss
from mfvlr_research.pilot_variant import visual_model, weighted_mask_loss
from mfvlr_research.pilot_runtime import check_finite_tensors
from test_split_mask_encoder_pilot import collate, evaluate
from test_ad_overfit import state_hash
from measure_ar_fl_gradients import measure_groups, group_statistics
from pilot_checkpointing import enable_checkpointing

TRIALS = {'FL_only': {'ar': 0., 'fd': 0., 'fl': 1.},
          'AR_only': {'ar': 1., 'fd': 0., 'fl': 0.},
          'AR_FL': {'ar': 1., 'fd': 0., 'fl': 1.},
          'AR_FD_FL': {'ar': 1., 'fd': 1., 'fl': 1.}}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def load_samples(path):
    with np.load(path, allow_pickle=False) as data:
        images = torch.from_numpy(data['images'].copy()).float().div_(255)
        masks = torch.from_numpy(data['masks'].copy()).long()
        labels, ids = data['labels'].tolist(), data['sample_ids'].tolist()
    assert images.shape == (16, 3, 224, 224) and masks.shape == (16, 224, 224)
    assert labels == [0, 1] * 8 and len(set(ids)) == 16
    assert set(masks.unique().tolist()) <= {0, 1}
    assert not masks[::2].any() and all(masks[i].any() for i in range(1, 16, 2))
    return [dict(image=images[i], mask=masks[i], label=labels[i], sample_id=ids[i],
                 kind='real' if labels[i] == 0 else 'local') for i in range(16)]


class View:
    def __init__(self, model):
        self.model = model

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def __call__(self, image):
        result = self.model(image)
        return dict(reconstruction=result.decoded.reconstruction,
                    mask_logits=result.decoded.mask_logits, detection_logits=result.detection_logits)


def direct_decode(model, image):
    local = model.mve.shared_encoder.encode_local(image, return_skips=True)
    return model.decoder(local.local, skips=local.skips)


def gradient_probe(model, samples, device, weights, lambdas, pairs):
    """Diagnostic AR/FL derivatives, even when one objective is disabled in training."""
    model.eval()
    parameters = measure_groups(model)
    rows = []
    for index in np.linspace(0, 7, pairs, dtype=int):
        batch = collate(samples[2 * index:2 * index + 2])
        image, target = batch['image'].to(device), batch['mask'].to(device)
        decoded = direct_decode(model, image)
        ar = appearance_reconstruction_loss(image, decoded.reconstruction)
        fl = weighted_mask_loss(decoded.mask_logits, target, weights)
        tensors = tuple(p for p, _ in parameters.values())
        a = torch.autograd.grad(ar, tensors, retain_graph=True, allow_unused=True)
        check_finite_tensors(a, 'Nonfinite AR probe')
        a_cpu = [None if x is None else x.detach().cpu() for x in a]
        del a
        f = torch.autograd.grad(fl, tensors, allow_unused=True)
        check_finite_tensors(f, 'Nonfinite FL probe')
        f_cpu = [None if x is None else x.detach().cpu() for x in f]
        rows.append(dict(sample_ids=batch['sample_id'], ar=float(ar.detach()), fl=float(fl.detach()),
                         groups=group_statistics(parameters, a_cpu, f_cpu, lambdas)))
        del decoded, ar, fl, f, a_cpu, f_cpu
    model.train()
    return rows


def export_preview(folder, preview):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (224 * 4, 280 * len(preview)), 'white')
    draw, raw = ImageDraw.Draw(canvas), {}
    for i, sample in enumerate(preview):
        y = 280 * i
        draw.text((5, y + 5), sample['sample_id'], fill='black')
        for col, (key, title) in enumerate((('image', 'Input'), ('reconstruction', 'AD'),
                                           ('target', 'GT'), ('prediction', 'Predicted mask'))):
            array = sample[key]
            values = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
            canvas.paste(Image.fromarray(values).convert('RGB'), (224 * col, y + 48))
            draw.text((224 * col + 5, y + 25), title, fill='black')
        for key in ('image', 'reconstruction', 'target', 'prediction', 'probability'):
            raw[f'{i}_{key}'] = sample[key]
    canvas.save(folder / 'preview.png')
    np.savez_compressed(folder / 'preview.npz', **raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=400)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gradient-pairs', type=int, default=4)
    parser.add_argument('--trials', nargs='+', choices=tuple(TRIALS), default=list(TRIALS))
    args = parser.parse_args()
    if args.output.exists() or args.updates < 1 or not 1 <= args.gradient_pairs <= 8:
        parser.error('Choose a new output, positive updates and 1..8 gradient pairs')
    if len(set(args.trials)) != len(args.trials):
        parser.error('Duplicate trial')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    samples = load_samples(args.data)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    assert config['model']['skip_connections'] == 'concat'
    assert config['model']['residual_gradient'] == 'detach_reconstruction'
    assert config['model'].get('reconstruction_activation', 'identity') == 'identity'
    rng, schedule = np.random.default_rng(0), []
    while len(schedule) < args.updates:
        order = rng.permutation(16).tolist()
        schedule.extend(order[i:i + 4] for i in range(0, 16, 4))
    schedule = schedule[:args.updates]
    args.output.mkdir(parents=True)
    source_files = [*ROOT.joinpath('src/mfvlr').glob('*.py'),
                    *ROOT.joinpath('src/mfvlr').glob('*.json'),
                    *ROOT.joinpath('src/mfvlr_research').glob('pilot_*.py'),
                    Path(__file__), ROOT / 'tools/measure_ar_fl_gradients.py',
                    ROOT / 'tools/test_split_mask_encoder_pilot.py',
                    ROOT / 'tools/test_ad_overfit.py', ROOT / 'tools/export_ad_reconstructions.py',
                    ROOT / 'tools/pilot_checkpointing.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    report = dict(status='RUNNING', dataset='GenFace real + Diffae only',
                  scope='Training-set overfit diagnostic only; no val/test or generalization claim',
                  sample_ids=[s['sample_id'] for s in samples], seed=0,
                  data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(), source_sha256=hashes,
                  environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                                   device=str(args.device), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None),
                  optimizer=dict(name='Adam', lr=1e-4, weight_decay=0., foreach=False),
                  updates=args.updates, schedule=schedule, microbatch=2, effective_batch=4,
                  pixel_weights=[1., 4.], gradient_probe='Raw AR/FL measured without optimizer updates at step0 and final; inactive objectives still probed',
                  checkpoint_schema='mfvlr-loss-overfit-v1 (image-only model, custom schema)', trials={})
    write_json(args.output / 'report.json', report)
    initial_hash = None
    for name in args.trials:
        folder = args.output / name
        folder.mkdir()
        lambdas = TRIALS[name]
        trial = report['trials'][name] = dict(status='STARTING', loss_weights=lambdas, history=[])
        model = optimizer = None
        try:
            torch.manual_seed(0)
            model = visual_model(config).to(args.device).train()
            trial['initial_state_sha256'] = state_hash(model)
            if initial_hash is None:
                initial_hash = trial['initial_state_sha256']
            assert trial['initial_state_sha256'] == initial_hash
            trial['checkpointed_stages'] = enable_checkpointing(model)
            weights = torch.tensor([1., 4.], device=args.device)
            trial['initial_metrics'], _ = evaluate(View(model), samples, args.device, weights, 2)
            trial['gradients_initial'] = gradient_probe(model, samples, args.device, weights, lambdas, args.gradient_pairs)
            # Verify the faster AR/FL-only path agrees with the full visual graph.
            with torch.no_grad():
                image = samples[0]['image'][None].to(args.device)
                direct, full = direct_decode(model, image), model(image).decoded
                torch.testing.assert_close(direct.reconstruction, full.reconstruction, rtol=0, atol=0)
                torch.testing.assert_close(direct.mask_logits, full.mask_logits, rtol=0, atol=0)
            del direct, full, image
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0., foreach=False)
            trial['status'] = 'RUNNING'
            start = time.perf_counter()
            write_json(args.output / 'report.json', report)
            bar = tqdm(enumerate(schedule, 1), total=args.updates, desc=name, unit='update')
            for step, selected in bar:
                optimizer.zero_grad(set_to_none=True)
                means = {key: 0. for key in lambdas}
                for offset in (0, 2):
                    batch = collate([samples[i] for i in selected[offset:offset + 2]])
                    image, masks = batch['image'].to(args.device), batch['mask'].to(args.device)
                    if lambdas['fd']:
                        output = model(image)
                        decoded = output.decoded
                    else:
                        output, decoded = None, direct_decode(model, image)
                    values = {}
                    if lambdas['ar']:
                        values['ar'] = appearance_reconstruction_loss(image, decoded.reconstruction)
                    if lambdas['fl']:
                        values['fl'] = weighted_mask_loss(decoded.mask_logits, masks, weights)
                    if lambdas['fd']:
                        values['fd'] = forgery_detection_loss(output.detection_logits, batch['label'].to(args.device))
                    total = sum(lambdas[key] * value for key, value in values.items())
                    (total / 2).backward()
                    for key, value in values.items():
                        means[key] += float(value.detach()) / 2
                    del output, decoded, values, total
                check_finite_tensors((p.grad for p in model.parameters()), 'Nonfinite training gradient')
                optimizer.step()
                check_finite_tensors(model.parameters(), 'Nonfinite training parameter')
                trial['history'].append(dict(step=step, losses=means))
                bar.set_postfix({key: f'{value:.4f}' for key, value in means.items()})
                if step % 10 == 0:
                    write_json(args.output / 'report.json', report)
                if step % 50 == 0 or step == args.updates:
                    optimizer.zero_grad(set_to_none=True)
                    metrics, preview = evaluate(View(model), samples, args.device, weights, 2)
                    pixel = metrics['pixel_confusion']
                    metrics['pixel_recall'] = pixel['tp'] / max(1, pixel['tp'] + pixel['fn'])
                    metrics['pixel_precision'] = pixel['tp'] / max(1, pixel['tp'] + pixel['fp'])
                    trial.setdefault('evaluations', []).append(dict(step=step, training_set=metrics))
                    export_preview(folder, preview)
                    write_json(args.output / 'report.json', report)
            trial['seconds_train_and_evaluation'] = time.perf_counter() - start
            trial['gradients_final'] = gradient_probe(model, samples, args.device, weights, lambdas, args.gradient_pairs)
            torch.save(dict(schema='mfvlr-loss-overfit-v1', config=config, loss_weights=lambdas,
                            step=args.updates, model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            provenance=dict(data=report['data_sha256'], source=hashes)), folder / 'final.pt')
            trial['status'] = 'COMPLETED'
        except Exception as error:
            trial.update(status='FAILED', error=f'{type(error).__name__}: {error}')
            import traceback
            traceback.print_exc()
        finally:
            output = decoded = values = total = None
            del model, optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            write_json(args.output / 'report.json', report)
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == value for path, value in hashes.items())
    report.update(status='COMPLETED' if all(t['status'] == 'COMPLETED' for t in report['trials'].values()) else 'COMPLETED_WITH_FAILURES',
                  source_unchanged=True)
    write_json(args.output / 'report.json', report)
    print(json.dumps(dict(status=report['status'], output=str(args.output)), indent=2), flush=True)
    if report['status'] != 'COMPLETED':
        raise RuntimeError('One or more trials failed; inspect report.json')


if __name__ == '__main__':
    main()
