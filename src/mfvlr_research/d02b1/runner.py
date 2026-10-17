"""Metadata-only runner boundary. No CLI, YAML execution flag, model import or worker."""
from pathlib import Path
from .schema import InvalidRecord, schedule, validate_cell, write_new

class ExecutionNotAuthorized(PermissionError):
    pass

def dispatch(*args, **kwargs):
    # Intentionally unconditional. A future implementation requires new code review.
    raise ExecutionNotAuthorized('B0-D03 not authorized; D02B1 has no worker dispatch')

def continue_after_numerical_failure(*args, **kwargs):
    raise ExecutionNotAuthorized('B0-D08 is not authorized')

def prepare(record, *, approved_partition, pinned, expected_inventory, coordinator):
    identity = validate_cell(record, approved_partition=approved_partition, pinned=pinned,
                             expected_inventory=expected_inventory)
    path = coordinator.reserve(record['identity']['experiment_id'], identity)
    write_new(path / 'registration.json', record)
    return {'path': str(path), 'execution': 'NOT RUN', 'dispatch': 'DISABLED'}

def status_record(*, provenance, structure, finite, complete, numerical, resource):
    if any(type(x) is not bool for x in (provenance, structure, finite, complete)):
        raise InvalidRecord('Gate booleans required')
    if numerical not in ('PASS', 'FAIL', 'NOT RUN') or resource not in ('PASS', 'FAIL/OOM', 'INSUFFICIENT', 'NOT VERIFIED'):
        raise InvalidRecord('Unknown status')
    hard = not all((provenance, structure, finite, complete))
    return {'execution': 'HARD STOP' if hard else 'NOT RUN',
            'structural': 'PASS' if structure else 'FAIL', 'provenance': 'PASS' if provenance else 'FAIL',
            'numerical': numerical, 'resource': resource,
            'continuation': 'DISABLED', 'dependent_work_blocked': hard or numerical != 'PASS' or resource != 'PASS'}

def coverage():
    return [dict(c, execution='NOT RUN', structural='NOT RUN', numerical='NOT RUN', resource='NOT VERIFIED') for c in schedule()]

def assert_mock_selection(nodeids):
    if not nodeids or any(not n.replace('\\', '/').startswith('tests/research_d02b1/') for n in nodeids):
        raise InvalidRecord('D02B1 verification selects only isolated new mock tests; legacy tests forbidden')

def require_complete(record, required_parameters, required_stages):
    """No output subset may masquerade as a full scientific cell."""
    from .schema import exact_keys, PASSES
    if not isinstance(required_parameters, dict) or not required_parameters:
        raise InvalidRecord('Sealed parameter shape/connectivity specifications required')
    exact_keys(record, ('gradients', 'stages', 'losses', 'state_before', 'state_after', 'ownership_before', 'ownership_after'))
    if record['state_before'] != record['state_after'] or record['ownership_before'] != record['ownership_after']:
        raise InvalidRecord('State/ownership mutation')
    for value in record['state_before'].values():
        from .schema import sha
        sha(value)
    if not record['state_before'] or set(record['state_before']) != set(required_parameters):
        raise InvalidRecord('State inventory incomplete')
    if set(record['stages']) != set(required_stages) or set(record['gradients']) != set(required_parameters):
        raise InvalidRecord('Incomplete stage/parameter outputs')
    if set(record['ownership_before']) != set(required_parameters):
        raise InvalidRecord('Incomplete ownership output')
    for name, stages in record['stages'].items():
        exact_keys(stages, ('shape', 'finite', 'rms', 'max_abs', 'l2'))
        if stages['finite'] is not True:
            raise InvalidRecord(f'Nonfinite stage: {name}')
        if isinstance(required_stages, dict) and stages['shape'] != required_stages[name]:
            raise InvalidRecord(f'Wrong stage shape: {name}')
        import math
        if any(type(stages[k]) not in (float,int) or not math.isfinite(stages[k]) or stages[k]<0 for k in ('rms','max_abs','l2')):
            raise InvalidRecord('Invalid activation statistics')
    for parameter, gradient in record['gradients'].items():
        if not isinstance(gradient, dict) or not {'passes','loss_pairs','direction_actual','direction_reference'}.issubset(gradient):
            raise InvalidRecord(f'Incomplete parameter analysis: {parameter}')
        exact_keys(gradient['passes'], PASSES)
        spec = required_parameters[parameter]
        for loss, entry in gradient['passes'].items():
            expected = spec['presence'][loss]
            base = {'allocated', 'shape', 'expectation'}
            extra = {'path','sha256','elements','bytes','finite','nonzero','l2','rms','max_abs'}
            exact_keys(entry, base if expected == 'none' else base | extra)
            if entry['allocated'] is not (expected != 'none') or entry['expectation'] != expected or entry['shape'] != spec['shape']:
                raise InvalidRecord('Gradient presence/shape/expectation mismatch')
            if expected != 'none':
                from .schema import sha
                sha(entry['sha256'])
                elements = math.prod(spec['shape'])
                if entry['finite'] is not True or entry['elements'] != elements or entry['bytes'] != elements*4:
                    raise InvalidRecord('Incomplete gradient data')
                if type(entry['nonzero']) is not int or not 0 <= entry['nonzero'] <= elements:
                    raise InvalidRecord('Invalid nonzero count')
                if any(type(entry[k]) not in (int,float) or not math.isfinite(entry[k]) or entry[k]<0 for k in ('l2','rms','max_abs')):
                    raise InvalidRecord('Nonfinite gradient statistics')
                if expected == 'zero' and any(entry[k] != 0 for k in ('nonzero','l2','rms','max_abs')):
                    raise InvalidRecord('Structural zero failure')
        from itertools import combinations
        from .schema import ORDER
        exact_keys(gradient['loss_pairs'], (a+':'+b for a,b in combinations(ORDER,2)))
        if len(gradient['direction_actual']) != 4 or len(gradient['direction_reference']) != 4:
            raise InvalidRecord('Missing preregistered direction')
        for values in (gradient['direction_actual'],gradient['direction_reference']):
            if any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
                raise InvalidRecord('Nonfinite direction')
    exact_keys(record['losses'], PASSES)
    import math
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in record['losses'].values()):
        raise InvalidRecord('Nonfinite scalar objective')
