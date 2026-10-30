"""Larger GenFace FD comparison with held-out validation and paired random seeds."""
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
import torch
from mfvlr.mve import MVE
from mfvlr.heads import DetectionHeads
from mfvlr_research.pilot_skip import SkipResizeDecoder
from mfvlr_research.pilot_variant import visual_model
import run_loss_overfit as base
import run_fd_weight_ablation as diagnostic
from pilot_checkpointing import enable_checkpointing


def seeded_model(config, seed):
    """Preserve seed0 behavior, but actually vary component initializers for other seeds."""
    torch.manual_seed(seed)
    model = visual_model(config)
    if seed:
        model.mve = MVE(seed=seed)
        model.decoder = SkipResizeDecoder(config['model']['resize_mode'], seed=seed,
            reconstruction_activation='identity')
        model.detector = DetectionHeads(seed=seed).detector
    return model


def samples_from_npz(path, limit=None):
    with np.load(path, allow_pickle=False) as data:
        labels, ids = data['labels'].tolist(), data['sample_ids'].tolist()
        selected = np.arange(len(labels)) if limit is None else np.sort(np.concatenate([
            np.array([i for i, label in enumerate(labels) if label == cls])[
                np.linspace(0, labels.count(cls)-1, limit//2, dtype=int)] for cls in (0, 1)]))
        images = torch.from_numpy(data['images'][selected].copy()).float().div_(255)
        masks = torch.from_numpy(data['masks'][selected].copy()).long()
    ids, labels = [ids[i] for i in selected], [labels[i] for i in selected]
    assert len(ids) % 4 == 0 and len(set(ids)) == len(ids)
    assert labels == [0, 1] * (len(labels)//2)
    assert set(masks.unique().tolist()) <= {0, 1} and not masks[::2].any()
    return [dict(image=images[i], mask=masks[i], label=labels[i], sample_id=ids[i],
                 kind='real' if labels[i] == 0 else 'local') for i in range(len(ids))]


def evaluate(model, samples, device, weights):
    metric, preview = base.evaluate(base.View(model), samples, device, weights, 2)
    p = metric['pixel_confusion']
    metric.update(pixel_recall=p['tp']/max(1, p['tp']+p['fn']),
                  pixel_precision=p['tp']/max(1, p['tp']+p['fp']))
    return metric, preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--fd-weights', type=float, nargs='+', default=[.1, 1.])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--val-limit', type=int)
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--gradient-pairs', type=int, default=4)
    parser.add_argument('--save-epoch-checkpoints', action='store_true')
    args = parser.parse_args()
    if not args.fd_weights or len(set(args.fd_weights)) != len(args.fd_weights) or any(not np.isfinite(w) or w < 0 for w in args.fd_weights):
        parser.error('FD weights must be finite, nonnegative and unique')
    if args.output.exists() or args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Select a fresh output, positive epochs and unique seeds')
    if any(seed < 0 for seed in args.seeds) or not 1 <= args.gradient_pairs <= 8:
        parser.error('Invalid seed/gradient pair count')
    if args.max_updates is not None and args.max_updates < 1:
        parser.error('max-updates must be positive')
    for limit in (args.train_limit, args.val_limit):
        if limit is not None and (limit < 16 or limit % 4):
            parser.error('Smoke limits must be multiples of4 and at least16')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True)
    diagnostic.OUTPUT = args.output
    log = diagnostic.log
    log('load_data_start', inputs=str(args.inputs), seeds=args.seeds)
    train = samples_from_npz(args.inputs / 'train.npz', args.train_limit)
    val = samples_from_npz(args.inputs / 'val.npz', args.val_limit)
    assert not ({s['sample_id'] for s in train} & {s['sample_id'] for s in val})
    config = json.loads((args.inputs / 'config.json').read_text())
    assert config['model']['skip_connections'] == 'concat'
    assert config['model']['residual_gradient'] == 'detach_reconstruction'
    assert config['model']['reconstruction_activation'] == 'identity'
    pair_indices = np.linspace(0, len(train)//2-1, 8, dtype=int)
    probes = [train[2*i+j] for i in pair_indices for j in (0, 1)]
    source_files = [*ROOT.joinpath('src/mfvlr').glob('*.py'), *ROOT.joinpath('src/mfvlr').glob('*.json'),
                    *ROOT.joinpath('src/mfvlr_research').glob('pilot_*.py'),
                    *(ROOT/'tools'/name for name in ('run_fd_validation_pilot.py', 'run_loss_overfit.py',
                        'run_fd_weight_ablation.py', 'measure_ar_fl_gradients.py', 'pilot_checkpointing.py',
                        'test_ad_overfit.py', 'export_ad_reconstructions.py', 'test_split_mask_encoder_pilot.py'))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    report = dict(status='RUNNING', dataset='GenFace real + Diffae only', test_used=False,
        protocol='FD sweep, lambdaAR=lambdaFL=1; final epoch primary; fixed threshold0.5/mask argmax; no augmentation',
        fd_weights=args.fd_weights,
        train_ids=[s['sample_id'] for s in train], val_ids=[s['sample_id'] for s in val],
        source_sha256=hashes, config=config, epochs=args.epochs, seeds=args.seeds,
        effective_batch=4, microbatch=2, optimizer=dict(name='Adam', lr=1e-4, weight_decay=0., foreach=False),
        smoke=any(x is not None for x in (args.train_limit, args.val_limit, args.max_updates)),
        data_sha256={s:hashlib.sha256((args.inputs/(s+'.npz')).read_bytes()).hexdigest() for s in ('train','val')},
        environment=dict(torch=torch.__version__, python=sys.version, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None), schedules={}, trials={})
    write = lambda: base.write_json(args.output/'report.json', report)
    write()
    log('data_ready', train_count=len(train), val_count=len(val), test_used=False)
    seed_hashes = {}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        schedule = []
        for _ in range(args.epochs):
            order = rng.permutation(len(train)).tolist()
            schedule.extend(order[i:i+4] for i in range(0, len(train), 4))
        if args.max_updates:
            schedule = schedule[:args.max_updates]
        report['schedules'][str(seed)] = schedule
        for fd in args.fd_weights:
            name = f'seed_{seed}_fd_{str(fd).replace(".","p")}'
            folder = args.output/name
            folder.mkdir()
            lambdas = dict(ar=1., fl=1., fd=fd)
            t = report['trials'][name] = dict(status='STARTING', seed=seed, loss_weights=lambdas,
                                             history=[], evaluations=[])
            write()
            log('trial_start', trial=name, seed=seed, lambda_fd=fd)
            model = optimizer = None
            try:
                model = seeded_model(config, seed).to(args.device).train()
                t['initial_state_sha256'] = base.state_hash(model)
                if seed in seed_hashes:
                    assert seed_hashes[seed] == t['initial_state_sha256']
                else:
                    seed_hashes[seed] = t['initial_state_sha256']
                t['checkpointed_stages'] = enable_checkpointing(model)
                weights = torch.tensor([1.,4.], device=args.device)
                log('initial_validation_start', trial=name)
                t['initial_val'], _ = evaluate(model, val, args.device, weights)
                t['gradients_initial'] = diagnostic.fd_gradient_probe(model, probes, args.device, weights,
                                                                     lambdas, args.gradient_pairs)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0., foreach=False)
                t['status']='RUNNING'
                write()
                for step, selected in enumerate(schedule,1):
                    optimizer.zero_grad(set_to_none=True)
                    means = {key:0. for key in lambdas}
                    for offset in (0,2):
                        batch=base.collate([train[i] for i in selected[offset:offset+2]])
                        image,masks=batch['image'].to(args.device),batch['mask'].to(args.device)
                        out=model(image)
                        values=dict(ar=base.appearance_reconstruction_loss(image,out.decoded.reconstruction),
                            fl=base.weighted_mask_loss(out.decoded.mask_logits,masks,weights),
                            fd=base.forgery_detection_loss(out.detection_logits,batch['label'].to(args.device)))
                        total=sum(lambdas[key]*v for key,v in values.items())
                        (total/2).backward()
                        for key,v in values.items(): means[key]+=float(v.detach())/2
                        del out,values,total
                    base.check_finite_tensors((p.grad for p in model.parameters()), 'Nonfinite pilot gradient')
                    optimizer.step()
                    base.check_finite_tensors(model.parameters(), 'Nonfinite pilot parameter')
                    epoch=(step-1)//(len(train)//4)+1
                    t['history'].append(dict(step=step,epoch=epoch,losses=means))
                    log('train',trial=name,step=step,total=len(schedule),epoch=epoch,losses=means)
                    if step%10==0: write()
                    if step%(len(train)//4)==0 or step==len(schedule):
                        optimizer.zero_grad(set_to_none=True)
                        log('evaluate_start',trial=name,step=step,epoch=epoch)
                        train_metric,train_preview=evaluate(model,train,args.device,weights)
                        val_metric,val_preview=evaluate(model,val,args.device,weights)
                        t['evaluations'].append(dict(step=step,epoch=epoch,train=train_metric,val=val_metric))
                        val_folder=folder/'val'; val_folder.mkdir(exist_ok=True)
                        base.export_preview(val_folder,val_preview)
                        if step==len(schedule):
                            train_folder=folder/'train'; train_folder.mkdir(exist_ok=True)
                            base.export_preview(train_folder,train_preview)
                        write()
                        log('metrics',trial=name,step=step,epoch=epoch,train=train_metric,val=val_metric)
                        if args.save_epoch_checkpoints:
                            temporary=folder/'last.tmp.pt'
                            torch.save(dict(schema='mfvlr-fd-epoch-state-v1',seed=seed,step=step,epoch=epoch,
                                config=config,loss_weights=lambdas,model=model.state_dict(),optimizer=optimizer.state_dict(),
                                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                                schedule=schedule,provenance=dict(source=hashes,data=report['data_sha256']),
                                history=t['history'],evaluations=t['evaluations']),temporary)
                            temporary.replace(folder/'last.pt')
                            log('checkpoint_saved',trial=name,step=step,epoch=epoch,path=str(folder/'last.pt'))
                t['gradients_final']=diagnostic.fd_gradient_probe(model,probes,args.device,weights,lambdas,args.gradient_pairs)
                torch.save(dict(schema='mfvlr-fd-validation-pilot-v1',seed=seed,step=len(schedule),config=config,
                    loss_weights=lambdas,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    provenance=dict(source=hashes,data=report['data_sha256'])),folder/'final.pt')
                t['status']='COMPLETED'
                log('trial_complete',trial=name,step=len(schedule))
            except Exception as error:
                import traceback
                traceback.print_exc()
                t.update(status='FAILED',error=f'{type(error).__name__}: {error}')
                log('trial_failed',trial=name,error=t['error'])
            finally:
                out=values=total=v=None
                del model,optimizer
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                write()
    assert len(set(seed_hashes.values()))==len(seed_hashes), 'Different seeds unexpectedly share initialization'
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report.update(status='COMPLETED' if all(t['status']=='COMPLETED' for t in report['trials'].values()) else 'COMPLETED_WITH_FAILURES',
                  source_unchanged=True,paired_initialization_verified=len(args.fd_weights)>1,different_seed_initializations_verified=True)
    write()
    log('experiment_complete',status=report['status'])
    if report['status']!='COMPLETED': raise RuntimeError('Pilot contains failed trial(s)')


if __name__=='__main__': main()
