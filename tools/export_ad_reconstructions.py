"""Visualize AD reconstructions without changing model outputs or checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.pilot_deps'), str(ROOT / 'src')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Subset

from mfvlr_research.pilot_datasets import dataset_for_config
from mfvlr_research.pilot_runtime import load_checkpoint
from mfvlr_research.pilot_variant import visual_model


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rgb_display(values):
    # Fixed [0,1] display range; no per-image normalization or sigmoid.
    return Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--samples-per-class', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.samples_per_class < 1:
        parser.error('--samples-per-class must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output is not empty; choose a new directory')

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint['config']
    manifest = ROOT / config['manifests'][args.split]
    if file_hash(manifest) != checkpoint['provenance']['manifests'][args.split]:
        raise ValueError('Manifest differs from checkpoint')
    for relative, expected in checkpoint['provenance']['source'].items():
        normalized = relative.replace('\\', '/')
        if normalized.startswith(('src/mfvlr/', 'src/mfvlr_research/')):
            if file_hash(ROOT / relative) != expected:
                raise ValueError(f'Model source changed: {relative}')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset = dataset_for_config(ROOT, config, args.split)
    selected = []
    for label in (0, 1):
        candidates = [i for i, row in enumerate(dataset.rows) if row['label'] == label]
        if len(candidates) < args.samples_per_class:
            raise ValueError(f'Not enough class {label} examples')
        positions = np.linspace(0, len(candidates) - 1, args.samples_per_class, dtype=int)
        selected.extend(candidates[p] for p in positions)

    model = visual_model(config)
    weights = {}
    for name, tensor in checkpoint['model'].items():
        if name.startswith(('mve.', 'decoder.')):
            weights[name] = tensor
        elif name.startswith('heads.detector.'):
            weights[name.removeprefix('heads.')] = tensor
    model.load_state_dict(weights, strict=True)
    step = checkpoint['step']
    del checkpoint, weights
    device = torch.device(args.device)
    model.to(device).eval()

    for folder in ('input', 'reconstruction', 'residual_x4', 'raw', 'comparison'):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    sheet = Image.new('RGB', (896, 280 * len(selected)), 'white')
    draw = ImageDraw.Draw(sheet)
    samples = []
    loader = DataLoader(Subset(dataset, selected), batch_size=config['batch_size'],
                        shuffle=False, num_workers=0)
    for batch in loader:
        images = batch['image'].to(device)
        # IE -> shared decoder trunk -> appearance head, identical to visual_route.
        # No residual encoder, detector, tokenizer or language forward is needed.
        appearance = model.mve.encode_appearance(images)
        decoded = model.decoder.decode_appearance(appearance)
        reconstruction = decoded.reconstruction.cpu().permute(0, 2, 3, 1).numpy()
        inputs = batch['image'].permute(0, 2, 3, 1).numpy()
        if not np.isfinite(reconstruction).all():
            raise FloatingPointError('Nonfinite AD output')
        for i, sample_id in enumerate(batch['sample_id']):
            number = len(samples)
            label = int(batch['label'][i])
            safe_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', sample_id)
            filename = f'{number:02d}_{Path(safe_id).stem}.png'
            rgb, recon = inputs[i], reconstruction[i]
            residual = np.abs(recon - rgb)
            input_panel = rgb_display(rgb)
            recon_panel = rgb_display(recon)
            residual_panel = rgb_display(residual * 4)
            target_panel = Image.fromarray(batch['mask'][i].numpy().astype(np.uint8) * 255)
            input_panel.save(args.output / 'input' / filename)
            recon_panel.save(args.output / 'reconstruction' / filename)
            residual_panel.save(args.output / 'residual_x4' / filename)
            np.savez_compressed(args.output / 'raw' / Path(filename).with_suffix('.npz'),
                                input=rgb, reconstruction=recon, residual=residual)
            strip = Image.new('RGB', (896, 248), 'white')
            title = ImageDraw.Draw(strip)
            for j, (panel, name) in enumerate(zip(
                    (input_panel, recon_panel, residual_panel, target_panel),
                    ('Input A', 'AD reconstruction B', '|B-A| x4 (display)', 'GT mask'))):
                title.text((j * 224 + 5, 5), name, fill='black')
                strip.paste(panel.convert('RGB'), (j * 224, 24))
            strip.save(args.output / 'comparison' / filename)
            draw.text((5, number * 280 + 6),
                      f'{"REAL" if label == 0 else "DIFFAE"} | {sample_id}', fill='black')
            sheet.paste(strip, (0, number * 280 + 28))
            samples.append({'sample_id': sample_id, 'label': label, 'filename': filename,
                            'mae_raw': float(residual.mean()),
                            'mse_raw': float(np.square(recon - rgb).mean()),
                            'reconstruction_min': float(recon.min()),
                            'reconstruction_max': float(recon.max()),
                            'fraction_values_outside_0_1': float(((recon < 0) | (recon > 1)).mean())})
    sheet.save(args.output / 'preview.png')
    metadata = {'checkpoint': str(args.checkpoint.resolve()),
                'checkpoint_sha256': file_hash(args.checkpoint), 'step': step,
                'split': args.split, 'manifest_sha256': file_hash(manifest),
                'selection': 'Evenly spaced manifest positions within each class; no metric-based selection',
                'input_protocol': 'Existing dataset RGB bilinear 224x224, float32 [0,1]',
                'route': 'mve.encode_appearance -> decoder.decode_appearance -> reconstruction',
                'display': 'AD clipped to [0,1] for PNG only; abs(B-A)*4 clipped to [0,1]; no sigmoid or per-image normalization',
                'raw': 'NPZ float32 HWC input, reconstruction, abs residual, without display clipping',
                'not_a_benchmark': True, 'samples': samples}
    (args.output / 'export.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output.resolve()), 'count': len(samples),
                      'samples': samples}, indent=2))


if __name__ == '__main__':
    main()
