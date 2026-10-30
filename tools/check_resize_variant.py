"""Full-model forward/backward and inference parity check, with zero optimizer updates."""
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
from mfvlr_research.pilot_data import PilotFaces
from mfvlr_research.pilot_runtime import check_finite_tensors
from mfvlr_research.pilot_variant import training_model, visual_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/diffae_pilot_resize_conv.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['model']['language_visual_gradients'] != 'stop_all':
        parser.error('This check expects the stop_all policy')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(config['device'])
    dataset = PilotFaces(ROOT / config['manifests']['train'], tokens=True, limit=2)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    images, tokens = batch['image'].to(device), batch['token_ids'].to(device)
    ids = tuple(batch['sample_id'])
    model = training_model(config).to(device).train()
    output = model.with_language(images, tokens, sample_ids=ids)
    losses = model.compute_objectives(output, SuppliedTargets(ids, tokens, batch['label'].to(device),
        batch['mask'].to(device), tuple(batch['kind'])))
    parameters = [model.mve.shared_encoder.cnn.stages[0][0].weight,
                  model.decoder.trunk.stages[0].projection.weight, model.decoder.mask_head.weight,
                  model.flt.encoder.blocks[0].ff.fc1.weight, model.heads.adapter.weight,
                  model.objectives.cmc.log_temperature]
    names = ['visual_encoder', 'decoder_projection', 'mask_head', 'language_encoder', 'adapter', 'temperature']
    report = {'variant': config['model'], 'losses': {k:float(v.detach()) for k,v in losses.as_dict().items()},
              'optimizer_updates': 0, 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'gradient_paths': {}}
    for name in ('lr', 'cmc', 'kl', 'fl', 'fd', 'ar'):
        grads = torch.autograd.grad(getattr(losses, name), parameters, allow_unused=True, retain_graph=True)
        check_finite_tensors(grads, 'Nonfinite diagnostic gradient')
        report['gradient_paths'][name] = {n: (float(g.norm()) if g is not None else None) for n,g in zip(names, grads)}
        if name in ('lr', 'cmc', 'kl'):
            assert all(g is None for g in grads[:3]), f'{name} still reaches visual parameters'
        if name == 'fl':
            assert all(g is not None and bool(g.count_nonzero()) for g in grads[:3]), 'FL path disconnected'
        if name == 'lr':
            assert grads[3] is not None and bool(grads[3].count_nonzero()), 'Language stopped training'
        if name == 'kl':
            assert grads[4] is not None and bool(grads[4].count_nonzero()), 'Adapter stopped training'
        print('Verified gradient paths:', name, flush=True)
        del grads
    losses.total.backward()
    check_finite_tensors((p.grad for p in model.parameters()), 'Nonfinite total gradient')
    model.zero_grad(set_to_none=True)
    del output, losses
    deployed = visual_model(config).to(device).eval()
    state = {k:v for k,v in model.state_dict().items() if k.startswith(('mve.', 'decoder.'))}
    state.update({k.removeprefix('heads.'):v for k,v in model.state_dict().items() if k.startswith('heads.detector.')})
    deployed.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        full, visual = model.visual_only(images), deployed(images)
        report['inference_parity'] = {'detection_max_abs_error': float((full.detection_logits - visual.detection_logits).abs().max()),
                                     'mask_max_abs_error': float((full.decoded.mask_logits - visual.decoded.mask_logits).abs().max())}
        assert torch.equal(full.detection_logits, visual.detection_logits)
        assert torch.equal(full.decoded.mask_logits, visual.decoded.mask_logits)
    report.update(status='PASS', trainable_parameters=sum(p.numel() for p in model.parameters()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
