"""Variant-invariant comparisons on supplied arrays; no model execution."""
import numpy as np
from .schema import InvalidRecord, VARIANTS, exact_keys

INVARIANT_LOSSES = ('fd', 'cmc', 'fl', 'ar', 'kl')

def common_state_parity(a, b):
    """Metadata-only parity/ownership audit. Inputs must be validated inventories.

    Does not infer live ownership from hashes; distinct identifiers are required.
    The future model adapter must observe them directly in the same process.
    """
    aa, bb = {x['name']:x for x in a}, {x['name']:x for x in b}
    if len(aa)!=len(a) or len(bb)!=len(b):
        raise InvalidRecord('Repeated parameter names')
    common = aa.keys() & bb.keys()
    if len(common) not in (508,550):
        raise InvalidRecord('Incomplete common parameter inventory')
    for kind in ('object_id','storage_id'):
        if {x[kind] for x in a} & {x[kind] for x in b}:
            raise InvalidRecord('Variants must own distinct parameters and storage')
    for n in common:
        if any(aa[n][k]!=bb[n][k] for k in ('shape','dtype','sha256')):
            raise InvalidRecord(f'Common state mismatch: {n}')
    return {'common_parameters':len(common),'equal_hashes_and_shapes':True,'disjoint_recorded_owners':True,
            'basis':'supplied metadata; live graph ownership not executed in D02B1'}

def pair_identity(a, b):
    fields = ('variant', 'fixture', 'backend', 'batch', 'inputs_hash', 'common_state_hash', 'protocol_hash')
    exact_keys(a, fields)
    exact_keys(b, fields)
    if a['variant'] not in VARIANTS or b['variant'] not in VARIANTS or a['variant'] == b['variant']:
        raise InvalidRecord('Distinct registered variants required')
    from .schema import sha
    for record in (a, b):
        for key in ('inputs_hash', 'common_state_hash', 'protocol_hash'):
            sha(record[key])
    if any(a[k] != b[k] for k in fields if k != 'variant'):
        raise InvalidRecord('Unmatched scientific fixture/backend/common state')

def exact_arrays(a, b):
    if a is None or b is None:
        if a is not b:
            raise InvalidRecord('Gradient presence differs')
        return True
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray) or a.dtype != b.dtype or a.shape != b.shape:
        raise InvalidRecord('Pair shape/dtype')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise InvalidRecord('Nonfinite pair')
    return bool(np.array_equal(a, b))

def invariant_pair(a_id, b_id, outputs_a, outputs_b, gradients_a, gradients_b, common_names):
    pair_identity(a_id, b_id)
    for mapping in (outputs_a, outputs_b, gradients_a, gradients_b):
        exact_keys(mapping, INVARIANT_LOSSES)
    result = {}
    for loss in INVARIANT_LOSSES:
        exact_keys(gradients_a[loss], common_names)
        exact_keys(gradients_b[loss], common_names)
        result[loss] = {'output_exact': exact_arrays(outputs_a[loss], outputs_b[loss]),
                        'parameters': {name: exact_arrays(gradients_a[loss][name], gradients_b[loss][name]) for name in common_names}}
    return {'status': 'PASS' if all(r['output_exact'] and all(r['parameters'].values()) for r in result.values()) else 'FAIL',
            'comparisons': result, 'variant_dependent_excluded': ['lr', 'total']}

def same_sample_pair(a, b, *, kind):
    """B1/B2 sample arrays already sliced, with matching cotangents/reductions.

    Fixed-batch isolation is exact. Cross-batch FP32 comparisons retain the
    original threshold and their FAIL result; no new N2 waiver is applied.
    """
    if kind == 'cmc_objective':
        raise InvalidRecord('CMC is intentionally batch coupled; no equality criterion')
    if kind not in ('fixed_batch_output', 'fixed_batch_derivative', 'cross_batch_output', 'cross_batch_derivative'):
        raise InvalidRecord('Pairing kind')
    equality = exact_arrays(a, b)
    if a is None:
        raise InvalidRecord('A disconnected surrogate cannot prove sample isolation')
    delta = a.astype(np.float64)-b.astype(np.float64)
    ref = b.astype(np.float64)
    outside = np.abs(a-b) > np.float32(1e-6)+np.float32(1e-5)*np.abs(b)
    return {'kind': kind, 'exact': equality, 'max_abs': float(np.abs(delta).max()),
            'outside_count': int(outside.sum()), 'outside_proportion': float(outside.mean()),
            'relative_l2': float(np.linalg.norm(delta)/np.linalg.norm(ref)) if np.any(ref) else None,
            'status': ('PASS' if equality else 'FAIL') if kind.startswith('fixed') else ('FAIL' if outside.any() else 'PASS'),
            'scope': 'sample output/derivative only, not a loss/additivity acceptance amendment'}

