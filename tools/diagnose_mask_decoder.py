"""No-update decoder audit: input sensitivity, periodicity, wiring and FL/AR gradients."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.pilot_deps'), str(ROOT / 'src')]

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from mfvlr.integration import ImageOnlyMFVLR
from mfvlr.objectives import forgery_localization_loss, appearance_reconstruction_loss
from mfvlr_research.pilot_data import PilotFaces, sha256
from mfvlr_research.pilot_runtime import load_checkpoint


def stats(tensor):
    x = tensor.detach().float()
    rms = x.square().mean().sqrt()
    delta = (x - x.mean(0, keepdim=True)).square().mean().sqrt()
    return dict(mean=float(x.mean()), std=float(x.std()), rms=float(rms),
                between_image_delta_rms=float(delta), relative_between_image_delta=float(delta / (rms + 1e-12)),
                zero_fraction=float((x == 0).float().mean()))


def map_summary(maps):
    template = maps.mean(0)
    binary = maps >= .5
    majority = binary.mean(0) >= .5
    pixel_frequency = binary.mean(0)
    periodicity = {}
    center = template[16:-16, 16:-16]
    for period in (2, 4, 8, 16):
        phases = np.array([[center[i::period, j::period].mean() for j in range(period)] for i in range(period)])
        reconstructed = np.tile(phases, (center.shape[0] // period, center.shape[1] // period))
        periodicity[str(period)] = dict(explained_spatial_variance=float(1 - np.var(center - reconstructed) / (np.var(center) + 1e-20)),
                                      phase_mean_min=float(phases.min()), phase_mean_max=float(phases.max()))
    return dict(samples=len(maps), probability_min=float(maps.min()), probability_max=float(maps.max()),
                probability_mean=float(maps.mean()), mean_between_image_probability_std=float(maps.std(0).mean()),
                unique_masks=len({m.tobytes() for m in binary}),
                agreement_with_majority_template=float((binary == majority).mean()),
                invariant_pixels_fraction=float(((pixel_frequency == 0) | (pixel_frequency == 1)).mean()),
                periodicity=periodicity)


@torch.no_grad()
def trace(model, images):
    local = model.mve.shared_encoder.encode_local(images, return_skips=False).local
    activations = {'encoder.local': stats(local)}
    x = local
    for index, stage in enumerate(model.decoder.trunk.stages):
        x = stage(x)
        activations[f'decoder.U{index + 1}'] = stats(x)
    logits = model.decoder.mask_head(x)
    activations['mask.logit_margin'] = stats(logits[:, 1:2] - logits[:, 0:1])
    return local, logits, activations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--exported-masks', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint['config']
    torch.set_num_threads(4)
    device = torch.device(args.device)
    dataset = PilotFaces(ROOT / config['manifests']['test'], limit=16)
    if dataset.digest != checkpoint['provenance']['manifests']['test']:
        raise ValueError('Test manifest digest mismatch')
    model = ImageOnlyMFVLR()
    visual = {k:v for k,v in checkpoint['model'].items() if k.startswith(('mve.', 'decoder.'))}
    visual.update({k.removeprefix('heads.'):v for k,v in checkpoint['model'].items() if k.startswith('heads.detector.')})
    source_step = checkpoint['step']
    model.load_state_dict(visual, strict=True)
    del checkpoint, visual
    model.to(device).eval()
    batch = next(iter(DataLoader(dataset, batch_size=16, shuffle=False)))
    images = batch['image'].to(device)
    # Limit full-resolution activations to two samples at a time.
    all_local, all_probability = [], []
    for i in range(0, len(images), 2):
        local, logits, layers = trace(model, images[i:i + 2])
        all_local.append(local)
        all_probability.append(logits.softmax(1)[:, 1].cpu().numpy())
        if i == 0:
            first_layers = layers
    locals = torch.cat(all_local)
    probabilities = np.concatenate(all_probability)
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                  step=source_step, samples=batch['sample_id'], optimizer_updates=0,
                  trained=map_summary(probabilities), first_real_fake_pair_activations=first_layers)
    with torch.no_grad():
        reference = model.decoder(locals[:2]).mask_logits
        repeated = model.decoder(locals[:2]).mask_logits
        swapped = model.decoder(locals[:2].flip(0)).mask_logits.flip(0)
        report['wiring'] = dict(repeat_max_abs_difference=float((reference - repeated).abs().max()),
                                batch_swap_max_abs_difference=float((reference - swapped).abs().max()))
        controls = {'zero_local': torch.zeros_like(locals[:2]),
                    'spatial_and_batch_mean_local': locals.mean((0, 2, 3), keepdim=True).expand_as(locals[:2]),
                    'zero_images_encoded': model.mve.shared_encoder.encode_local(torch.zeros_like(images[:2]), return_skips=False).local}
        report['controls'] = {}
        template = probabilities.mean(0) >= .5
        for name, local in controls.items():
            p = model.decoder(local).mask_logits.softmax(1)[:, 1].cpu().numpy()
            report['controls'][name] = {'summary': map_summary(p),
                'binary_agreement_with_trained_template': float(((p >= .5) == template).mean()),
                'mean_absolute_probability_difference_to_template': float(np.abs(p - probabilities.mean(0)).mean())}
        # Same decoder input, zero biases as a diagnostic ablation; restored immediately.
        saved_biases = [(m.bias, m.bias.detach().clone()) for m in model.decoder.modules()
                        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)) and m.bias is not None]
        try:
            for bias, _ in saved_biases:
                bias.zero_()
            p = model.decoder(locals[:2]).mask_logits.softmax(1)[:, 1].cpu().numpy()
            report['controls']['zero_decoder_biases'] = {'summary': map_summary(p),
                'binary_agreement_with_original_pair': float(((p >= .5) == (probabilities[:2] >= .5)).mean())}
        finally:
            for bias, value in saved_biases:
                bias.copy_(value)
        fresh = ImageOnlyMFVLR().to(device).eval()
        _, initial_logits, initial_layers = trace(fresh, images[:2])
        report['initialized_control'] = {'maps': map_summary(initial_logits.softmax(1)[:, 1].cpu().numpy()),
            'activations': initial_layers,
            'mask_head_weight_change_l2': float((model.decoder.mask_head.weight - fresh.decoder.mask_head.weight).norm()),
            'mask_head_bias_change_l2': float((model.decoder.mask_head.bias - fresh.decoder.mask_head.bias).norm())}
        del fresh
    # Isolated decoder objective gradients: no model/optimizer updates.
    local = locals[:2].detach().requires_grad_(True)
    decoded = model.decoder(local)
    fl = forgery_localization_loss(decoded.mask_logits, batch['mask'][:2].to(device))
    ar = appearance_reconstruction_loss(images[:2], decoded.reconstruction)
    names = ['local_input', 'U1.up.weight', 'U4.conv2.weight', 'mask_head.weight', 'appearance_head.weight']
    targets = [local, model.decoder.trunk.stages[0].up.weight, model.decoder.trunk.stages[3].conv2.weight,
               model.decoder.mask_head.weight, model.decoder.appearance_head.weight]
    gf = torch.autograd.grad(fl, targets, allow_unused=True, retain_graph=True)
    ga = torch.autograd.grad(ar, targets, allow_unused=True)
    report['isolated_decoder_gradients'] = {'FL': float(fl.detach()), 'AR': float(ar.detach()), 'parameters': {}}
    for name, f, a in zip(names, gf, ga):
        info = {'FL_norm': float(f.norm()) if f is not None else None,
                'AR_norm': float(a.norm()) if a is not None else None}
        if f is not None and a is not None:
            info['FL_AR_cosine'] = float((f.flatten() @ a.flatten()) / (f.norm() * a.norm() + 1e-20))
        report['isolated_decoder_gradients']['parameters'][name] = info
    if args.exported_masks:
        paths = sorted(args.exported_masks.glob('*.png'))
        exported = np.stack([np.asarray(Image.open(p)) > 0 for p in paths])
        mean = exported.mean(0)
        report['all_exported_masks'] = dict(count=len(paths), unique=len({m.tobytes() for m in exported}),
            agreement_with_majority_template=float((exported == (mean >= .5)).mean()),
            invariant_pixels_fraction=float(((mean == 0) | (mean == 1)).mean()),
            foreground_fraction_mean=float(exported.mean()),
            foreground_fraction_std_between_images=float(exported.mean((1, 2)).std()))
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(probabilities.mean(0) * 255).astype(np.uint8)).save(args.output / 'mean_foreground_probability.png')
    Image.fromarray(((probabilities >= .5).mean(0) >= .5).astype(np.uint8) * 255).save(args.output / 'majority_mask.png')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
