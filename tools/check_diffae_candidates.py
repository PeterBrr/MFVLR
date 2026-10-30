"""Validate candidate files without resizing, modifying, or loading the model."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def check(folder):
    counts = collections.Counter()
    failures, collisions = [], []
    seen = {}
    with (folder / 'checked_candidates.jsonl').open('w', encoding='utf-8') as out:
        for split in ('train', 'val', 'test'):
            for line in (folder / f'{split}.candidates.jsonl').read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                try:
                    with Image.open(row['image']) as image:
                        image.load()
                        rgb = np.asarray(image.convert('RGB'))
                        size = image.size
                        digest = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
                    with Image.open(row['mask']) as image:
                        image.load()
                        if image.size != size:
                            raise ValueError('Image/mask dimensions differ')
                        mask = np.asarray(image)
                        if mask.ndim == 3:
                            if mask.shape[2] != 3 or not np.all(mask == mask[:, :, :1]):
                                raise ValueError('Non-grayscale mask channels')
                            mask = mask[:, :, 0]
                        if mask.ndim != 2 or not set(np.unique(mask).tolist()) <= {0, 1, 255}:
                            raise ValueError('Unexpected mask values')
                        if row['label'] == 0 and mask.any():
                            raise ValueError('Real mask has foreground')
                        if row['label'] == 1 and (not mask.any() or (mask > 0).all()):
                            raise ValueError('Local mask has no partial foreground')
                        row['foreground_fraction'] = float((mask > 0).mean())
                    if digest in seen:
                        collisions.append(dict(first=seen[digest], second=row['sample_id'], split=split))
                    seen[digest] = dict(sample_id=row['sample_id'], split=split)
                    row['decoded_rgb_sha256'] = digest
                    row['image_size'] = size
                    for field in ('image', 'mask'):
                        row[field + '_sha256'] = hashlib.sha256(Path(row[field]).read_bytes()).hexdigest()
                    out.write(json.dumps(row) + '\n')
                    counts[split + '_decoded_valid'] += 1
                except Exception as error:
                    failures.append(dict(sample_id=row['sample_id'], split=split, error=str(error)))
                counts['checked'] += 1
                if counts['checked'] % 500 == 0:
                    print(f"Checked {counts['checked']}", flush=True)
    report = dict(status='NOT_TRAIN_READY', counts=dict(counts), failures=failures,
        identical_decoded_images=collisions,
        pending=['mask provenance', 'source/identity split audit', 'label mapping provenance',
                 'approved deterministic preprocessing', 'tokenizer environment', 'numerical/resource gate'])
    (folder / 'decode_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('failures','identical_decoded_images')}, indent=2))
    print('Failures:', len(failures), 'identical images:', len(collisions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    check(parser.parse_args().folder)
