"""Closed metadata schema; registration is never execution authorization."""
import hashlib
import json
import math
import re
from pathlib import Path

ORDER = ('fd', 'lr', 'cmc', 'fl', 'ar', 'kl')
PASSES = ORDER + ('total',)
VARIANTS = ('A0', 'P3', 'Q3')
PARTITIONS = ('G2-primary', 'G2-controls', 'NG2')
COUNTS = {'A0': (508, 154156232), 'P3': (550, 154177736), 'Q3': (550, 154177736)}
BACKENDS = {'C4M': (4, 4, True), 'C4N': (4, 4, False), 'C1M': (1, 4, True)}

class InvalidRecord(ValueError):
    pass

def exact_keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise InvalidRecord(f'Expected exactly {sorted(keys)}')

def sha(value):
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise InvalidRecord('Missing/invalid SHA-256')
    return value

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)  # Validate before creating the exclusive record.
    with path.open('xb') as f:
        f.write(data + b'\n')

def schedule():
    rows = []
    cases = ('real', 'efs', 'local', 'real_efs', 'real_local', 'local_duplicate')
    for prefix, partition, backend in (('G', PARTITIONS[0], 'C4M'), ('C', PARTITIONS[1], 'C4N')):
        rows += [(f'{prefix}{i+1:02}', partition, fixture, 1 if i < 3 else 2, backend)
                 for i, fixture in enumerate(cases)]
    rows.append(('C07', PARTITIONS[1], 'real_efs', 2, 'C1M'))
    rows += [(f'N{i+1:02}', 'NG2', f'NG2-{i%3}', 2, 'C4M' if i < 3 else 'C4N') for i in range(6)]
    return [dict(experiment_id=f'D02B-{row}-{v}', row=row, partition=p, fixture=f,
                 batch=b, backend=be, variant=v) for row, p, f, b, be in rows for v in VARIANTS]

def participation(name):
    """Static graph connectivity, not an assertion of nonzero magnitude."""
    if name.startswith('mve.shared_encoder.cnn.') or name.startswith('decoder.trunk.'):
        connected = set(ORDER)
    elif name.startswith('mve.') or name.startswith('decoder.appearance_head.'):
        connected = {'fd', 'lr', 'cmc', 'ar', 'kl'} if name.startswith('decoder.') else {'fd', 'lr', 'cmc', 'kl'}
    elif name.startswith('decoder.mask_head.'):
        connected = {'fl'}
    elif name.startswith('heads.detector.'):
        connected = {'fd'}
    elif name.startswith('heads.adapter.'):
        connected = {'kl'}
    elif name.startswith('objectives.cmc.'):
        connected = {'cmc'}
    elif name.startswith('flt.decoder'):
        connected = {'lr'}
    elif name.startswith('flt.encoder') or name == 'flt.vocabulary.weight':
        connected = {'lr', 'cmc', 'kl'}
    else:
        raise InvalidRecord(f'Unknown parameter owner: {name}')
    connected.add('total')
    structural = bool(re.search(r'\.vim\.[qk]\.weight$', name))
    return {p: ('zero' if structural else 'allocated') if p in connected else 'none' for p in PASSES}

def category(name):
    if re.search(r'\.vim\.[qk]\.weight$', name):
        return 'singleton_vim_qk'
    if '.attention.' in name or '_attention.' in name:
        if name.endswith('.k.bias'):
            return 'ordinary_key_bias_cancellation'
        if name.endswith(('.q.weight', '.k.weight')):
            return 'ordinary_qk_weight'
    return 'other'

def validate_inventory(entries, variant, expected):
    """Compare full names/shapes with a separately pinned historical roster."""
    if variant not in VARIANTS or type(entries) is not list:
        raise InvalidRecord('Inventory/variant')
    if len(entries) != COUNTS[variant][0] or len(expected) != len(entries):
        raise InvalidRecord('Parameter tensor count')
    names, objects, storages = set(), set(), set()
    total = 0
    for entry, reference in zip(entries, expected):
        exact_keys(entry, ('name', 'shape', 'dtype', 'sha256', 'object_id', 'storage_id'))
        name, shape = entry['name'], entry['shape']
        if name != reference['name'] or shape != reference['shape'] or entry['dtype'] != 'float32':
            raise InvalidRecord('Parameter name/shape/dtype')
        if type(shape) is not list or any(type(n) is not int or n <= 0 for n in shape):
            raise InvalidRecord('Shape')
        for value, seen in ((name, names), (entry['object_id'], objects), (entry['storage_id'], storages)):
            if type(value) is not str or not value or value in seen:
                raise InvalidRecord('Duplicate/missing parameter or storage owner')
            seen.add(value)
        sha(entry['sha256'])
        participation(name)
        total += math.prod(shape)
    if total != COUNTS[variant][1] or 'flt.vocabulary.weight' not in names:
        raise InvalidRecord('Unique elements/vocabulary owner')
    if sum(category(n) == 'singleton_vim_qk' for n in names) != 38:
        raise InvalidRecord('VIM Q/K coverage')

def validate_cell(record, *, approved_partition, pinned, expected_inventory):
    """pinned is a separate trusted pre-forward seal, never taken from record itself.

    D02B1 supplies no executable seals. Validation may be exercised on mock seals.
    Matching this schema cannot authorize dispatch.
    """
    fields = ('identity', 'hashes', 'backend', 'seeds', 'ownership', 'vocabulary_tied', 'loss_order', 'hash_algorithms')
    exact_keys(record, fields)
    identity = record['identity']
    if identity not in schedule() or type(identity.get('batch')) is not int:
        raise InvalidRecord('Unregistered exact cell identity')
    if approved_partition not in PARTITIONS or identity['partition'] != approved_partition:
        raise InvalidRecord('Partition not approved for this metadata selection')
    hashes = record['hashes']
    if record['hash_algorithms'] != {'files': 'sha256-file-bytes', 'inputs': 'sha256-contiguous-little-endian-bytes',
                                     'state': 'sha256-ordered-named-parameter-name-plus-bytes',
                                     'inventory': 'sha256-canonical-json'}:
        raise InvalidRecord('Unspecified or changed digest algorithms')
    exact_keys(hashes, ('source', 'config', 'protocol', 'driver', 'inputs', 'state', 'inventory'))
    for key in ('source', 'config', 'protocol', 'driver', 'state', 'inventory'):
        sha(hashes[key])
    exact_keys(hashes['inputs'], ('image', 'tokens', 'mask', 'labels', 'sample_order'))
    for value in hashes['inputs'].values():
        sha(value)
    intra, inter, mkldnn = BACKENDS[identity['backend']]
    backend = dict(device='cpu', dtype='float32', intra=intra, inter=inter, mkldnn=mkldnn,
                   deterministic=True, matmul='highest', attention='explicit',
                   python='3.14.3', torch='2.12.1+cpu', platform='Windows')
    if canonical(record['backend']) != canonical(backend):
        raise InvalidRecord('Backend cohort mismatch')
    flt_seed = 6101 + int(identity['fixture'][-1]) if identity['partition'] == 'NG2' else 0
    if record['seeds'] != dict(mve=0, vd=0, heads=0, flt=flt_seed) or any(type(v) is not int for v in record['seeds'].values()):
        raise InvalidRecord('Initialization seeds')
    if record['vocabulary_tied'] is not True or record['loss_order'] != list(ORDER):
        raise InvalidRecord('Tying/loss order')
    validate_inventory(record['ownership'], identity['variant'], expected_inventory)
    if fingerprint(record['ownership']) != hashes['inventory']:
        raise InvalidRecord('Inventory hash mismatch')
    # Exact approved seal covers values, source, state, input and identity together.
    if fingerprint(record) != sha(pinned):
        raise InvalidRecord('Provenance seal mismatch')
    return fingerprint(record)

def verify_files(root, inventory):
    root = Path(root).resolve()
    if not inventory:
        raise InvalidRecord('Empty provenance inventory')
    for rel, digest in inventory.items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or file_hash(path) != sha(digest):
            raise InvalidRecord(f'File provenance mismatch: {rel}')

