"""Supplied tensor contracts only: no decoding, cropping or mask generation."""
import hashlib
import json
import torch
from torch.utils.data import Dataset, DataLoader
from mfvlr_research.d02b8.contracts import dataset as validate_metadata

def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def manifest_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def validate_manifest(rows):
    validate_metadata(rows)
    sources = {}
    for row in rows:
        digest = row['source_sha256']
        if digest in sources and sources[digest] != row['split']:
            raise ValueError('Duplicate source bytes across splits')
        sources[digest] = row['split']
    return manifest_hash(rows)

class InMemoryFaces(Dataset):
    """Entire manifest must be supplied before selecting a partition.

    Source hash refers to supplied float32 tensor bytes, not an image file.
    Subject/group correctness and preprocessing approval need external review.
    """
    def __init__(self, rows, tensors, split, *, expected_manifest_sha256):
        if validate_manifest(rows) != expected_manifest_sha256:
            raise ValueError('Manifest provenance mismatch')
        self.manifest_sha256 = expected_manifest_sha256
        if split not in ('train', 'validation', 'test') or set(tensors) != {r['sample_id'] for r in rows}:
            raise ValueError('Partition or sample coverage')
        self.samples = []
        for row in rows:
            image, mask = tensors[row['sample_id']]
            if image.device.type != 'cpu' or image.dtype != torch.float32 or tuple(image.shape) != (3,224,224):
                raise ValueError('Already prepared CPU float32 RGB required')
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError('Supplied RGB range [0,1] required')
            if mask.device.type != 'cpu' or mask.dtype != torch.int64 or tuple(mask.shape) != (224,224) or not ((mask == 0) | (mask == 1)).all():
                raise ValueError('Supplied integer binary mask required')
            if row['kind'] == 'real' and mask.any() or row['kind'] == 'efs' and not mask.all():
                raise ValueError('Supplied mask conflicts with declared semantics')
            if tensor_hash(image) != row['source_sha256'] or tensor_hash(mask) != row['mask_sha256']:
                raise ValueError('Tensor provenance mismatch')
            if row['split'] == split:
                self.samples.append(dict(image=image.detach().clone(), mask=mask.detach().clone(),
                    label=row['label'], sample_id=row['sample_id'], subject_group=row['subject_group']))
        if not self.samples:
            raise ValueError('Empty selected partition')
    def __len__(self): return len(self.samples)
    def __getitem__(self, index):
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k,v in self.samples[index].items()}

def loader(dataset, *, batch_size=1, seed=0):
    if not isinstance(dataset, InMemoryFaces) or type(batch_size) is not int or batch_size not in (1,2) or type(seed) is not int:
        raise ValueError('Bounded in-memory interface only')
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))
