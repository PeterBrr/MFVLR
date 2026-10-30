"""Export 224x224 foreground predictions and visual comparisons from a pilot checkpoint."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / '.pilot_deps'), str(ROOT / 'src')]

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mfvlr_research.pilot_variant import visual_model
from mfvlr_research.pilot_data import sha256
from mfvlr_research.pilot_datasets import dataset_for_config
from mfvlr_research.pilot_runtime import load_checkpoint


def comparison(rgb, target, prediction):
    overlay = rgb.copy()
    foreground = prediction > 0
    overlay[foreground] = np.round(.55 * rgb[foreground] + .45 * np.array([255, 0, 0])).astype(np.uint8)
    panels = [Image.fromarray(rgb), Image.fromarray(target), Image.fromarray(prediction), Image.fromarray(overlay)]
    strip = Image.new('RGB', (896, 246), 'white')
    draw = ImageDraw.Draw(strip)
    for index, (panel, title) in enumerate(zip(panels, ['Input', 'Ground truth', 'Prediction', 'Prediction overlay'])):
        draw.text((index * 224 + 5, 5), title, fill='black')
        strip.paste(panel.convert('RGB'), (index * 224, 22))
    return strip, Image.fromarray(overlay)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output is not empty; choose a new directory')
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint['config']
    manifest = ROOT / config['manifests'][args.split]
    if sha256(manifest) != checkpoint['provenance']['manifests'][args.split]:
        raise ValueError('Manifest differs from checkpoint')
    for relative, expected in checkpoint['provenance']['source'].items():
        normalized = relative.replace('\\', '/')
        if normalized.startswith(('src/mfvlr/', 'src/mfvlr_research/')) and sha256(ROOT / relative) != expected:
            raise ValueError(f'Model source changed: {relative}')
    torch.set_num_threads(config['threads'])
    dataset = dataset_for_config(ROOT, config, args.split)
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
    for name in ('predicted', 'ground_truth', 'overlay', 'comparison'):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    preview, rows = {0: [], 1: []}, []
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    for batch in tqdm(loader, desc='Export masks', unit='batch', dynamic_ncols=True):
        output = model(batch['image'].to(device))
        if not torch.isfinite(output.decoded.mask_logits).all():
            raise FloatingPointError('Nonfinite mask logits')
        predictions = output.decoded.mask_logits.argmax(1).cpu().numpy().astype(np.uint8) * 255
        for i, sample_id in enumerate(batch['sample_id']):
            label = int(batch['label'][i])
            basename = re.sub(r'[^A-Za-z0-9_.-]+', '_', sample_id)
            filename = f'{len(rows):04d}_{Path(basename).stem}.png'
            rgb = batch['image'][i].permute(1, 2, 0).mul(255).round().byte().numpy()
            target = batch['mask'][i].numpy().astype(np.uint8) * 255
            prediction = predictions[i]
            strip, overlay = comparison(rgb, target, prediction)
            Image.fromarray(prediction).save(args.output / 'predicted' / filename)
            Image.fromarray(target).save(args.output / 'ground_truth' / filename)
            overlay.save(args.output / 'overlay' / filename)
            strip.save(args.output / 'comparison' / filename)
            if len(preview[label]) < 4:
                preview[label].append(strip.copy())
            rows.append({'filename': filename, 'sample_id': sample_id, 'label': label,
                         'predicted_foreground_fraction': float((prediction > 0).mean()),
                         'target_foreground_fraction': float((target > 0).mean())})
    with (args.output / 'index.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    previews = preview[0] + preview[1]
    fake_name = next((r['generator'] for r in dataset.rows if r['label'] == 1), 'FAKE')
    sheet = Image.new('RGB', (896, len(previews) * 268), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, panel in enumerate(previews):
        draw.text((5, i * 268 + 5), 'REAL' if i < len(preview[0]) else fake_name, fill='black')
        sheet.paste(panel, (0, i * 268 + 22))
    sheet.save(args.output / 'preview.jpg', quality=90)
    metadata = {'count': len(rows), 'split': args.split, 'checkpoint': str(args.checkpoint),
                'checkpoint_sha256': sha256(args.checkpoint), 'step': step,
                'mask_size': [224, 224], 'mask_encoding': 'PNG grayscale: 0=background, 255=foreground',
                'decision': 'argmax over two mask logits, same as evaluate.py; no detection gating',
                'comparison_columns': ['input', 'ground truth', 'prediction', 'prediction overlay'],
                'resizing': 'training protocol at 224x224; exports are not original-resolution masks'}
    (args.output / 'export.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'exported': len(rows), 'output': str(args.output.resolve())}, indent=2))


if __name__ == '__main__':
    main()
