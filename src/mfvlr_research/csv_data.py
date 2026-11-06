"""Read a GenFace split straight from the extracted folder (train.csv/val.csv), one image at a time.

Same tensors as the packed NPZ: `load_row` is the function tools/pack_genface50k_npz.py packs with,
and rows come in the same interleaved real/fake order. Decoding costs a few CPU milliseconds per
image while the GPU needs about eight images a second, so nothing is packed: no extra disk, and
RAM holds only the batches in flight.

Every image and mask is fully decoded once before training (a truncated JPEG passes PIL's verify).
Rows that fail are skipped, train is cut to a multiple of the batch from the end of the order, and
the result is cached next to the CSV, keyed by its SHA256, so a resubmitted job does not recheck.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

PREFIX = dict(Diffae='Diffae', CelebA_HQ='CelebA_HQ', FFHQ='FFHQ')


def load_row(root, row):
    """RGB bilinear-resized to 224 and mask > 0 nearest-resized to 224, both uint8, as PilotFaces."""
    with Image.open(root / row['image']) as source:
        size = source.size
        rgb = np.array(source.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)
    with Image.open(root / row['mask']) as source:
        if source.size != size:
            raise ValueError(f'{row["image"]}: image/mask dimensions differ')
        mask = np.asarray(source.convert('L'))
    # PNG masks are 0/1 or 0/255; JPEG masks carry compression noise, so those use the midpoint.
    binary = mask > (0 if set(np.unique(mask).tolist()) <= {0, 1, 255} else 127)
    label = int(row['label'])
    if (label == 0) == bool(binary.any()):
        raise ValueError(f'{row["image"]}: mask does not match label {label}')
    resized = np.array(Image.fromarray(binary.astype(np.uint8)).resize((224, 224), Image.Resampling.NEAREST), dtype=np.uint8)
    return rgb.transpose(2, 0, 1), resized, label


def sample_id(row):
    return f'{PREFIX[row["source"]]}/{Path(row["image"]).name}'


def ordered_rows(rows):
    """Alternate real/fake so any prefix stays balanced; surplus rows of one class come last."""
    real, fake = [r for r in rows if r['label'] == '0'], [r for r in rows if r['label'] == '1']
    return [r for pair in zip(real, fake) for r in pair] + real[len(fake):] + fake[len(real):]


def _check(job):
    root, row = job
    try:
        load_row(root, row)
    except (OSError, SyntaxError) as error:
        return f'{type(error).__name__}: {error}'
    return None


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def usable_rows(root, split, multiple=1, workers=None):
    """Rows that decode, in pack order and cut to a multiple; cached per CSV content."""
    root = Path(root)
    text = (root / f'{split}.csv').read_text(encoding='utf-8')
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    cache = root / f'.{split}_usable_{digest[:16]}_x{multiple}.json'
    if cache.exists():
        return json.loads(cache.read_text(encoding='utf-8'))
    rows = ordered_rows(list(csv.DictReader(text.splitlines())))
    workers = workers or available_cpus()
    jobs = [(root, row) for row in rows]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            errors = list(pool.map(_check, jobs, chunksize=64))
    else:
        errors = [_check(job) for job in jobs]
    kept = [row for row, error in zip(rows, errors) if error is None]
    cut = len(kept) - len(kept) % multiple
    result = dict(split=split, csv_sha256=digest, rows=kept[:cut], trimmed=[sample_id(r) for r in kept[cut:]],
                  skipped=[dict(image=row['image'], error=error) for row, error in zip(rows, errors) if error])
    try:
        cache.write_text(json.dumps(result), encoding='utf-8')
    except OSError:
        pass  # a full or read-only data folder only costs a recheck next time
    return result


class CsvSamples:
    """The Samples interface (len, int/slice indexing, ids, labels) over decoded-on-demand rows."""
    def __init__(self, root, split, *, multiple=1, limit=None, workers=None, threads=4):
        self.root = Path(root)
        info = usable_rows(self.root, split, multiple, workers)
        rows = info['rows']
        if limit is not None:
            # Evenly spaced rows of each class, as load_samples picks from an NPZ.
            labels = [int(r['label']) for r in rows]
            keep = sorted(i for cls in (0, 1) for i in np.array([i for i, label in enumerate(labels) if label == cls])[
                np.linspace(0, labels.count(cls) - 1, limit // 2, dtype=int)].tolist())
            rows = [rows[i] for i in keep]
        self.rows = rows
        self.labels = [int(r['label']) for r in rows]
        self.ids = [sample_id(r) for r in rows]
        self.report = dict(count=len(rows), real=self.labels.count(0), fake=self.labels.count(1),
                           csv_sha256=info['csv_sha256'], skipped=info['skipped'], trimmed=info['trimmed'])
        self.digest = hashlib.sha256(json.dumps([info['csv_sha256'], limit, self.ids]).encode('utf-8')).hexdigest()
        self._threads = ThreadPoolExecutor(threads)
        self._pending = {}

    def __len__(self):
        return len(self.rows)

    def _load(self, index):
        image, mask, label = load_row(self.root, self.rows[index])
        return dict(image=torch.from_numpy(image).float().div_(255), mask=torch.from_numpy(mask).long(), label=label,
                    sample_id=self.ids[index], kind='real' if label == 0 else 'local')

    def prefetch(self, indices):
        """Start decoding rows that will be asked for next, while the GPU works on the current ones."""
        for index in indices:
            if index not in self._pending:
                self._pending[index] = self._threads.submit(self._load, index)

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            self.prefetch(indices)
            return [self[i] for i in indices]
        if index < 0:
            index += len(self)
        pending = self._pending.pop(index, None)
        return pending.result() if pending is not None else self._load(index)

    def __iter__(self):
        return (self[i] for i in range(len(self)))
