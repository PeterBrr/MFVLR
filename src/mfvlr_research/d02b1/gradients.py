"""Bounded-memory FP32 gradient records and FP64 accumulation; array input only.

No automatic differentiation, model construction, objective implementation, or
numerical workers are present. None is recorded distinctly from allocated zero.
"""
import hashlib
import copy
import itertools
import math
from pathlib import Path
import numpy as np
from .schema import InvalidRecord, ORDER, PASSES, category, file_hash, write_new

CHUNK = 262144
DIRECTION_SEEDS = (5101, 5102, 5103, 5104)
ATOL, RTOL = 1e-6, 1e-5

class GradientStore:
    def __init__(self, directory, specifications):
        """specifications maps every parameter to shape and seven expected presences.

        Production-sized metadata must first pass validate_inventory. Synthetic
        array tests use this storage algorithm directly without scientific seals.
        """
        if not specifications:
            raise InvalidRecord('Empty parameter inventory')
        self.specs = copy.deepcopy(specifications)
        for name, spec in specifications.items():
            if not isinstance(name, str) or not name or set(spec) != {'shape', 'presence'}:
                raise InvalidRecord('Parameter specification')
            if any(type(x) is not int or x <= 0 for x in spec['shape']):
                raise InvalidRecord('Parameter shape')
            if set(spec['presence']) != set(PASSES) or any(x not in ('allocated', 'zero', 'none') for x in spec['presence'].values()):
                raise InvalidRecord('Complete gradient connectivity map required')
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        self.records = {}
        self.cursor = 0
        self.sequence = [(p, n) for p in PASSES for n in self.specs]
        write_new(self.root / 'specifications.json', specifications)

    def put(self, loss, name, array):
        if self.cursor >= len(self.sequence) or (loss, name) != self.sequence[self.cursor]:
            raise InvalidRecord('Duplicate, omitted, or out-of-order gradient pass/parameter')
        spec = self.specs[name]
        expectation = spec['presence'][loss]
        if (array is None) != (expectation == 'none'):
            raise InvalidRecord('Unexpected gradient disconnection/connection')
        record = {'allocated': array is not None, 'shape': list(spec['shape']), 'expectation': expectation}
        stem = f'{self.cursor:05d}'
        if array is not None:
            if not isinstance(array, np.ndarray) or array.dtype != np.dtype('float32') or list(array.shape) != list(spec['shape']) or not array.flags.c_contiguous:
                raise InvalidRecord('Expected contiguous FP32 gradient with exact shape')
            flat = array.reshape(-1)
            path = self.root / (stem + '.f32')
            squared, maximum, nonzero = 0., 0., 0
            with path.open('xb') as f:
                for start in range(0, flat.size, CHUNK):
                    chunk = flat[start:start+CHUNK]
                    if not np.isfinite(chunk).all():
                        raise InvalidRecord('Nonfinite gradient; partial attempt preserved')
                    if (expectation == 'zero' or category(name) == 'singleton_vim_qk') and np.count_nonzero(chunk):
                        raise InvalidRecord('Nonzero singleton VIM structural gradient')
                    x = chunk.astype(np.float64)
                    squared += float(x @ x)
                    maximum = max(maximum, float(np.abs(x).max()))
                    nonzero += int(np.count_nonzero(x))
                    f.write(chunk.astype('<f4', copy=False).tobytes())
            record.update(path=path.name, sha256=file_hash(path), elements=flat.size,
                          bytes=path.stat().st_size, finite=True, nonzero=nonzero,
                          l2=math.sqrt(squared), rms=math.sqrt(squared/flat.size), max_abs=maximum)
        self.records[(loss, name)] = record
        write_new(self.root / (stem + '.json'), dict(loss=loss, parameter=name, **record))
        self.cursor += 1

    def complete(self):
        if self.cursor != len(self.sequence):
            raise InvalidRecord('Incomplete seven-pass gradient evidence')

    def open_gradient(self, loss, name):
        r = self.records[(loss, name)]
        if not r['allocated']:
            return None
        path = self.root / r['path']
        if path.stat().st_size != r['bytes'] or file_hash(path) != r['sha256']:
            raise InvalidRecord('Gradient shard incomplete/corrupted')
        return path.open('rb')

    def analyze(self):
        self.complete()
        result = {}
        for name, spec in self.specs.items():
            result[name] = self._parameter(name, spec)
            write_new(self.root / (hashlib.sha256(name.encode()).hexdigest() + '.analysis.json'), result[name])
        return {'parameters': result, 'parameter_tensors': len(result),
                'status': 'FAIL' if any(r['outside_tolerance_count'] for r in result.values()) else 'PASS',
                'reference': 'FP64 ordered accumulation of isolated FP32 gradients; not full FP64 autodiff',
                'elementwise': {'actual': 'FP32', 'reference': 'FP32(FP64 ordered sum)', 'atol': ATOL, 'rtol': RTOL},
                'metric_reference': 'unrounded FP64 ordered sum', 'chunk': CHUNK, 'direction_seeds': DIRECTION_SEEDS}

    def _parameter(self, name, spec):
        count = math.prod(spec['shape'])
        handles = []
        sum_path = self.root / (hashlib.sha256(name.encode()).hexdigest() + '.sum.f64')
        pairs = list(itertools.combinations(ORDER, 2))
        pair_dot = {p: 0. for p in pairs}
        salt = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
        generators = [np.random.default_rng(s + salt) for s in DIRECTION_SEEDS]
        r = dict(elements=count, shape=list(spec['shape']), mathematical_class=category(name),
                 outside_tolerance_count=0, max_abs_error=0., max_relative_nonzero_reference=0.,
                 actual_squared=0., reference_squared=0., error_squared=0., dot=0.,
                 reference_zero_coordinates=0, reference_zero_actual_nonzero=0, sign_disagreements=0,
                 reference_max_abs=0., direction_actual_raw=[0.]*4, direction_reference_raw=[0.]*4)
        try:
            for p in PASSES:
                handles.append(self.open_gradient(p, name))
            with sum_path.open('xb') as out:
                for start in range(0, count, CHUNK):
                    n = min(CHUNK, count-start)
                    xs = []
                    for handle in handles:
                        if handle is None:
                            xs.append(np.zeros(n, dtype=np.float64))
                        else:
                            raw = handle.read(n*4)
                            if len(raw) != n*4:
                                raise InvalidRecord('Truncated gradient')
                            xs.append(np.frombuffer(raw, dtype='<f4').astype(np.float64))
                    ref = np.zeros(n, dtype=np.float64)
                    for x in xs[:6]:
                        ref += x  # Literal fd,lr,cmc,fl,ar,kl order.
                    out.write(ref.astype('<f8', copy=False).tobytes())
                    actual, delta = xs[6], xs[6]-ref
                    rounded = ref.astype(np.float32)
                    if not np.isfinite(rounded).all():
                        raise InvalidRecord('Nonfinite rounded additivity reference')
                    a32 = actual.astype(np.float32)
                    close = np.abs(a32-rounded) <= np.float32(ATOL) + np.float32(RTOL)*np.abs(rounded)
                    r['outside_tolerance_count'] += int((~close).sum())
                    r['max_abs_error'] = max(r['max_abs_error'], float(np.abs(delta).max()))
                    r['reference_max_abs'] = max(r['reference_max_abs'], float(np.abs(ref).max()))
                    nz = ref != 0
                    if nz.any():
                        r['max_relative_nonzero_reference'] = max(r['max_relative_nonzero_reference'], float((np.abs(delta[nz])/np.abs(ref[nz])).max()))
                    for key, value in (('actual_squared', actual@actual), ('reference_squared', ref@ref), ('error_squared', delta@delta), ('dot', actual@ref)):
                        r[key] += float(value)
                    r['reference_zero_coordinates'] += int((~nz).sum())
                    r['reference_zero_actual_nonzero'] += int(((~nz)&(actual != 0)).sum())
                    r['sign_disagreements'] += int((actual*ref < 0).sum())
                    for i, generator in enumerate(generators):
                        direction = (generator.integers(0, 2, size=n, dtype=np.int8)*2-1).astype(np.float64)
                        r['direction_actual_raw'][i] += float(actual@direction)
                        r['direction_reference_raw'][i] += float(ref@direction)
                    for pair in pairs:
                        pair_dot[pair] += float(xs[ORDER.index(pair[0])]@xs[ORDER.index(pair[1])])
        finally:
            for handle in handles:
                if handle is not None:
                    handle.close()
        records = {p: self.records[(p, name)] for p in PASSES}
        refnorm = math.sqrt(r['reference_squared'])
        anorm = math.sqrt(r['actual_squared'])
        r.update(passes=records, gradient_l2=anorm, reference_l2=refnorm,
                 reference_rms=refnorm/math.sqrt(count), error_l2=math.sqrt(r['error_squared']),
                 outside_proportion=r['outside_tolerance_count']/count,
                 relative_l2=math.sqrt(r['error_squared'])/refnorm if refnorm else None,
                 relative_l2_reason=None if refnorm else 'zero reference norm',
                 cosine=r['dot']/(anorm*refnorm) if anorm*refnorm else None,
                 cosine_reason=None if anorm*refnorm else 'zero actual or reference norm',
                 sum_path=sum_path.name, sum_sha256=file_hash(sum_path))
        r['cancellation_numerator'] = sum(records[p].get('l2', 0.) for p in ORDER)
        r['cancellation_denominator'] = refnorm
        r['cancellation_ratio'] = r['cancellation_numerator']/refnorm if refnorm else None
        r['cancellation_reason'] = None if refnorm else 'zero ordered sum norm'
        for k in ('actual', 'reference'):
            r[f'direction_{k}'] = [v/math.sqrt(count) for v in r[f'direction_{k}_raw']]
        r['direction_absolute_error'] = [abs(a-b) for a,b in zip(r['direction_actual'], r['direction_reference'])]
        r['loss_pairs'] = {}
        for a, b in pairs:
            denominator = records[a].get('l2', 0.)*records[b].get('l2', 0.)
            reason = 'unused gradient' if not records[a]['allocated'] or not records[b]['allocated'] else 'allocated zero norm'
            r['loss_pairs'][f'{a}:{b}'] = {'dot': pair_dot[(a,b)], 'cosine': pair_dot[(a,b)]/denominator if denominator else None,
                                         'undefined_reason': None if denominator else reason}
        if r['mathematical_class'] == 'ordinary_key_bias_cancellation':
            r['analytical_zero_residuals'] = {p: {'allocated': records[p]['allocated'], 'l2': records[p].get('l2'), 'max_abs': records[p].get('max_abs')} for p in PASSES}
        return r

def state_snapshot(arrays):
    """Synthetic state/ownership control: raw bytes + shape/dtype + exact identities."""
    entries = list(arrays.items())
    for i, (_, value) in enumerate(entries):
        if not isinstance(value, np.ndarray) or not np.isfinite(value).all():
            raise InvalidRecord('Finite arrays required')
        if any(np.shares_memory(value, other) for _, other in entries[:i]):
            raise InvalidRecord('Illegal storage alias')
    return {n: {'object_id': id(a), 'pointer': a.__array_interface__['data'][0],
                'shape': list(a.shape), 'dtype': str(a.dtype), 'sha256': hashlib.sha256(a.tobytes()).hexdigest()} for n,a in entries}

def require_state_unchanged(before, arrays):
    if before != state_snapshot(arrays):
        raise InvalidRecord('Parameter/state bytes or ownership changed')
