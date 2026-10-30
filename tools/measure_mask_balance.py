"""Count binary train-mask pixels using the pilot subset and resize convention."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.pilot_deps'), str(ROOT / 'src')]

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from mfvlr_research.pilot_data import PilotFaces, sha256


def measure(config_path, stage):
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    dataset = PilotFaces(ROOT / config['manifests']['train'], limit=config[stage]['limit'])
    counts = {label: dict(images=0, foreground_pixels=0, background_pixels=0, empty_masks=0)
              for label in (0, 1)}
    fractions = []
    for row in tqdm(dataset.rows, desc='Count train-mask pixels', unit='mask'):
        if row['split'] != 'train':
            raise ValueError('Only training samples may determine weights')
        blob = Path(row['mask']).read_bytes()
        if hashlib.sha256(blob).hexdigest() != row['mask_sha256']:
            raise ValueError(f'Changed mask: {row["sample_id"]}')
        with Image.open(io.BytesIO(blob)) as image:
            if list(image.size) != row['image_size']:
                raise ValueError('Mask/source dimensions disagree with checked manifest')
            mask = np.asarray(image)
            if mask.ndim == 3:
                if mask.shape[2] != 3 or not np.all(mask == mask[:, :, :1]):
                    raise ValueError('Non-grayscale mask channels')
                mask = mask[:, :, 0]
            if mask.ndim != 2 or not set(np.unique(mask).tolist()) <= {0, 1, 255}:
                raise ValueError('Unexpected mask values')
            if row['label'] == 0 and mask.any():
                raise ValueError('Real mask contains foreground')
            binary = Image.fromarray((mask > 0).astype(np.uint8))
            resized = np.asarray(binary.resize((224, 224), Image.Resampling.NEAREST))
        fg = int(np.count_nonzero(resized))
        entry = counts[row['label']]
        entry['images'] += 1
        entry['foreground_pixels'] += fg
        entry['background_pixels'] += resized.size - fg
        entry['empty_masks'] += int(fg == 0)
        if row['label'] == 1:
            fractions.append(fg / resized.size)
    fg = sum(v['foreground_pixels'] for v in counts.values())
    bg = sum(v['background_pixels'] for v in counts.values())
    if not fg or not bg:
        raise ValueError('Both pixel classes must exist to balance weights')
    assert fg + bg == len(dataset) * 224 * 224
    selected_ids = [r['sample_id'] for r in dataset.rows]
    return dict(config=str(config_path), config_sha256=sha256(config_path), stage=stage,
                manifest_sha256=dataset.digest, samples=len(dataset), mask_size=[224, 224],
                selected_ids_sha256=hashlib.sha256(json.dumps(selected_ids).encode()).hexdigest(),
                by_image_class=counts, foreground_pixels=fg, background_pixels=bg,
                foreground_fraction=fg / (fg + bg), background_fraction=bg / (fg + bg),
                proposed_class_weights={'background': 1., 'foreground': bg / fg},
                weighted_foreground_share_at_proposed_weights=(bg / fg * fg) / (bg + bg / fg * fg),
                fake_mask_foreground_fraction_quantiles=dict(zip(['min', 'p25', 'median', 'p75', 'max'],
                    np.quantile(fractions, [0, .25, .5, .75, 1]).tolist())),
                scope='train masks only; no model, GPU, validation or test; config not modified',
                seconds=time.perf_counter() - started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/diffae_pilot_resize_conv.json')
    parser.add_argument('--stage', choices=['tiny', 'pilot'], default='pilot')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/diffae_pilot/mask_balance_pilot.json')
    args = parser.parse_args()
    result = measure(args.config, args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
