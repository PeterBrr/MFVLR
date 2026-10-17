"""Tensor-only checkpoint directory. No pickle or executable model objects.

    Trust comes from an independently supplied manifest SHA256. Size limits
    must be set by the caller before loading; hashes do not confer trust.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from mfvlr_research.d02b8.contracts import sha

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def metadata(value):
    keys = {'schema','variant','scope','update_count','numerical_status','source_sha256',
            'config_sha256','dataset_sha256','tokenizer_sha256','loss_contract_sha256'}
    if type(value) is not dict or set(value) != keys:
        raise ValueError('Complete checkpoint metadata required')
    if value['schema'] != 'D02B9-tensor-v1' or value['variant'] != 'A0' or value['scope'] not in ('mock','research'):
        raise ValueError('Checkpoint schema/scope')
    if type(value['update_count']) is not int or value['update_count'] != 0 or value['numerical_status'] not in ('FAIL','NOT RUN'):
        raise ValueError('No training or acceptance claim authorized')
    for name in keys:
        if name.endswith('sha256'): sha(value[name])

def save(model, directory, info):
    metadata(info)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    rows = {}
    for index, (name, tensor) in enumerate(model.state_dict().items()):
        if tensor.device.type != 'cpu' or tensor.dtype not in (torch.float32, torch.float64, torch.int64, torch.bool):
            raise ValueError('Explicit CPU tensor dtype required')
        if not torch.isfinite(tensor).all(): raise ValueError('Nonfinite checkpoint')
        path = root / f'{index:05d}.npy'
        with path.open('xb') as stream: np.save(stream, tensor.detach().numpy(), allow_pickle=False)
        rows[name] = dict(file=path.name, sha256=digest(path), bytes=path.stat().st_size,
                          shape=list(tensor.shape), dtype=str(tensor.numpy().dtype))
    document = dict(metadata=info, tensors=rows)
    path = root / 'manifest.json'
    with path.open('x', encoding='utf-8') as stream: json.dump(document, stream, indent=2)
    return digest(path)

def load(model, directory, *, trusted_manifest_sha256, max_bytes):
    sha(trusted_manifest_sha256)
    if type(max_bytes) is not int or max_bytes <= 0: raise ValueError('Explicit positive resource limit')
    root = Path(directory).resolve()
    manifest = root / 'manifest.json'
    if manifest.is_symlink() or manifest.stat().st_size > 2**20 or digest(manifest) != trusted_manifest_sha256:
        raise ValueError('Untrusted manifest')
    doc = json.loads(manifest.read_text(encoding='utf-8'))
    if set(doc) != {'metadata','tensors'}: raise ValueError('Unknown checkpoint fields')
    metadata(doc['metadata'])
    current = model.state_dict()
    if set(doc['tensors']) != set(current): raise ValueError('State keys mismatch')
    total = 0
    loaded = {}
    for index, (name, target) in enumerate(current.items()):
        row = doc['tensors'][name]
        if set(row) != {'file','sha256','bytes','shape','dtype'} or row['file'] != f'{index:05d}.npy':
            raise ValueError('Invalid tensor inventory/path')
        path = root / row['file']
        if path.is_symlink() or path.resolve().parent != root: raise ValueError('External tensor path')
        if type(row['bytes']) is not int or row['bytes'] <= 0: raise ValueError('Invalid size')
        total += row['bytes']
        if total > max_bytes or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Tensor size/hash/resource mismatch')
        if row['shape'] != list(target.shape) or row['dtype'] != str(target.cpu().numpy().dtype):
            raise ValueError('Target tensor layout mismatch')
        array = np.load(path, allow_pickle=False, mmap_mode='r')
        if list(array.shape) != row['shape'] or str(array.dtype) != row['dtype'] or not np.isfinite(array).all():
            raise ValueError('Invalid tensor contents')
        loaded[name] = torch.from_numpy(np.array(array, copy=True))
    # Reject contradictory aliases before loading; keep the caller's Parameter identities.
    aliases = {}
    for name, tensor in current.items():
        key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
        if key in aliases and not torch.equal(loaded[name], loaded[aliases[key]]):
            raise ValueError('Contradictory tied tensor values')
        aliases[key] = name
    model.load_state_dict(loaded, strict=True, assign=False)
    return doc['metadata']
