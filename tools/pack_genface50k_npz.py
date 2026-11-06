"""Turn the extracted genface50k zip into the train.npz/val.npz the experiment runner reads.

Same tensors as PilotFaces: RGB bilinear-resized to 224, mask > 0 nearest-resized to 224,
uint8 storage. Runs on the HPC side after `unzip genface50k.zip`; written through memmaps so
peak RAM stays near one image. Test is never packed.

An image or mask that does not fully decode (a truncated JPEG passes PIL's verify) is skipped and
listed in package_report.json. Train is then cut to a multiple of the batch, from the end of the
interleaved order, where only surplus reals sit; the runner needs an exact multiple.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
# One preprocessing for both routes: the runner's CSV reader uses the same function and order.
from mfvlr_research.csv_data import load_row as load, ordered_rows, sample_id  # noqa: E402


def pack(root, split, output, limit=None, multiple=1):
    with (root / f'{split}.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if limit:
        # Balanced subset: the first limit/2 rows of each class in csv order.
        rows = [r for label in ('0', '1') for r in [x for x in rows if x['label'] == label][:limit // 2]]
    order = ordered_rows(rows)
    count = len(order)
    scratch = output / f'.{split}_tmp'
    scratch.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(scratch / 'images.npy', 'w+', np.uint8, (count, 3, 224, 224))
    masks = np.lib.format.open_memmap(scratch / 'masks.npy', 'w+', np.uint8, (count, 224, 224))
    labels, ids, skipped = [], [], []
    for i, row in enumerate(order):
        try:
            image, mask, label = load(root, row)
        except (OSError, SyntaxError) as error:
            skipped.append(dict(image=row['image'], error=f'{type(error).__name__}: {error}'))
            print(f'{split}: skipped {row["image"]} ({error})', flush=True)
            continue
        images[len(labels)], masks[len(labels)] = image, mask
        labels.append(label)
        ids.append(sample_id(row))
        if (i + 1) % 2000 == 0:
            print(f'{split}: {i + 1}/{count}', flush=True)
    kept = len(labels) - len(labels) % multiple
    trimmed = ids[kept:]
    labels, ids = labels[:kept], ids[:kept]
    if len(set(ids)) != kept:
        raise ValueError(f'{split}: repeated sample IDs')
    images.flush()
    masks.flush()
    target = output / f'{split}.npz'
    np.savez(target, images=images[:kept], masks=masks[:kept], labels=np.array(labels), sample_ids=np.array(ids))
    del images, masks
    for name in ('images.npy', 'masks.npy'):
        (scratch / name).unlink()
    scratch.rmdir()
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            digest.update(block)
    return dict(count=kept, real=labels.count(0), fake=labels.count(1), sha256=digest.hexdigest(),
                skipped=skipped, trimmed=trimmed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extracted', type=Path, required=True, help='Folder holding train.csv, val.csv and the image folders')
    parser.add_argument('--output', type=Path, required=True, help='Package folder; point a config\'s "inputs" here')
    parser.add_argument('--train-limit', type=int, default=None, help='Optional balanced subset of train')
    parser.add_argument('--batch-multiple', type=int, default=4, help='Train count is cut to a multiple of this')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {split: pack(args.extracted, split, args.output, args.train_limit if split == 'train' else None,
                          args.batch_multiple if split == 'train' else 1)
              for split in ('train', 'val')}
    (args.output / 'package_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
