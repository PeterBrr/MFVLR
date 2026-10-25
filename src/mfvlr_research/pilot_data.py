"""File-backed Diffae pilot inputs; independent of historical preparation gates."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'Empty manifest: {path}')
    ids, digests = set(), set()
    for row in rows:
        if row['sample_id'] in ids or row['decoded_rgb_sha256'] in digests:
            raise ValueError('Duplicate sample ID or decoded image in manifest')
        ids.add(row['sample_id'])
        digests.add(row['decoded_rgb_sha256'])
        if row['label'] == 0:
            if row['kind'] != 'real' or row['generator'] is not None:
                raise ValueError('Invalid real metadata')
        elif row['label'] == 1:
            if row['kind'] != 'local' or row['generator'] != 'Diffae' or not Path(row['image']).name.startswith('Diffae_'):
                raise ValueError('This pilot supports Real/Diffae only')
        else:
            raise ValueError('Labels must be 0=real / 1=fake')
        for name in ('image_sha256', 'mask_sha256', 'decoded_rgb_sha256'):
            if len(row[name]) != 64 or any(c not in '0123456789abcdef' for c in row[name]):
                raise ValueError(f'Invalid digest: {name}')
    return rows


def freeze_candidates(source, destination):
    """Deduplicate within split; reject any exact cross-split overlap, never move rows."""
    rows = [json.loads(x) for x in Path(source).read_text(encoding='utf-8').splitlines() if x.strip()]
    by_split = {s: [] for s in ('train', 'val', 'test')}
    seen, ids, removed = {}, set(), []
    for row in rows:
        split, digest = row['split'], row['decoded_rgb_sha256']
        if split not in by_split or row['sample_id'] in ids:
            raise ValueError('Invalid split or repeated sample ID')
        ids.add(row['sample_id'])
        if digest in seen:
            if seen[digest] != split:
                raise ValueError('Exact image content crosses splits; resolve before training')
            removed.append(row['sample_id'])
            continue
        seen[digest] = split
        by_split[split].append(row)
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    contents = {}
    for split, values in by_split.items():
        if not values or {r['label'] for r in values} != {0, 1}:
            raise ValueError(f'{split} must contain both classes')
        contents[split] = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in values).encode()
    # Check all destinations before writing any manifest.
    for split, data in contents.items():
        path = output / f'{split}.jsonl'
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f'Refusing to replace frozen manifest: {path}')
    result = {'source_sha256': sha256(source), 'removed': removed, 'splits': {},
              'protocol': 'internal Diffae pilot; no verified identity mapping or mask provenance'}
    for split, data in contents.items():
        path = output / f'{split}.jsonl'
        if not path.exists():
            path.write_bytes(data)
        read_manifest(path)
        result['splits'][split] = {'count': len(by_split[split]), 'sha256': sha256(path)}
    (output / 'manifest_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


class PilotFaces(Dataset):
    def __init__(self, manifest, *, tokens=False, limit=None):
        self.manifest = Path(manifest)
        self.digest = sha256(manifest)
        self.rows = read_manifest(manifest)
        if limit is not None:
            if limit < 2 or limit % 2:
                raise ValueError('Subset size must be positive and even')
            groups = [[r for r in self.rows if r['label'] == label][:limit // 2] for label in (0, 1)]
            if any(len(g) != limit // 2 for g in groups):
                raise ValueError('Not enough samples for balanced subset')
            self.rows = [r for pair in zip(*groups) for r in pair]
        self.token_ids = None
        if tokens:
            from mfvlr.prompts import PromptBuilder, PromptMetadata
            from mfvlr.tokenizer import TokenizerAdapter
            builder, tokenizer = PromptBuilder(), TokenizerAdapter()
            metadata = [PromptMetadata('real', 'real'), PromptMetadata('diffae', 'fake', 'AM', 'diffusion', 'Diffae')]
            self.token_ids = tokenizer.encode_levels(builder.build(metadata)).ids

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        blobs = {}
        for key in ('image', 'mask'):
            blobs[key] = Path(row[key]).read_bytes()
            if hashlib.sha256(blobs[key]).hexdigest() != row[key + '_sha256']:
                raise ValueError(f'Changed {key}: {row["sample_id"]}')
        with Image.open(io.BytesIO(blobs['image'])) as source:
            original_size = source.size
            rgb = np.array(source.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), copy=True)
        with Image.open(io.BytesIO(blobs['mask'])) as source:
            if source.size != original_size:
                raise ValueError('Image/mask dimensions differ')
            mask = np.asarray(source)
            if mask.ndim == 3:
                if mask.shape[2] != 3 or not np.all(mask == mask[:, :, :1]):
                    raise ValueError('Mask channels disagree')
                mask = mask[:, :, 0]
            if mask.ndim != 2 or not set(np.unique(mask).tolist()) <= {0, 1, 255}:
                raise ValueError('Mask must contain binary 0/1 or 0/255 values')
            if row['label'] == 0 and mask.any():
                raise ValueError('Real mask contains foreground')
            binary = Image.fromarray((mask > 0).astype(np.uint8))
            resized = np.array(binary.resize((224, 224), Image.Resampling.NEAREST), dtype=np.int64, copy=True)
        sample = {'image': torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255),
                  'mask': torch.from_numpy(resized), 'label': row['label'],
                  'sample_id': row['sample_id'], 'kind': row['kind']}
        if self.token_ids is not None:
            sample['token_ids'] = self.token_ids[row['label']].clone()
        return sample
