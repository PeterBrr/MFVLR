"""Frozen coverage specification and finite statistics validation."""
import math
from .schema import InvalidRecord, exact_keys

def stage_shapes(batch):
    if type(batch) is not int or batch not in (1, 2):
        raise InvalidRecord('Batch must be 1 or 2')
    b = batch
    stages = {
        'appearance.local': [b,1024,14,14], 'residual.local': [b,1024,14,14],
        'appearance.global': [b,1,512], 'residual.global': [b,1,512], 'Iv': [b,1,512],
        'Ipre': [b,3,224,224], 'residual.image': [b,3,224,224],
        'logits_fd': [b,2], 'logits_fl': [b,2,224,224], 'adapter': [b,1,512],
        'low': [b,308,512], 'LE.input': [b,308,512], 'Tl': [b,1,512],
        'LD.shifted_lookup': [b,308,512], 'LD.input': [b,308,512],
        'logits_lr': [b,308,49408],
    }
    stages.update({f'LE.{i}.output': [b,308,512] for i in range(12)})
    stages.update({f'LD.{i}.{s}': [b,308,512] for i in range(7) for s in ('self', 'memory', 'ff', 'vim', 'output')})
    return stages

def validate_stages(records, batch):
    expected = stage_shapes(batch)
    exact_keys(records, expected)
    for name, shape in expected.items():
        r = records[name]
        exact_keys(r, ('shape', 'finite', 'rms', 'max_abs', 'l2'))
        if r['shape'] != shape or r['finite'] is not True:
            raise InvalidRecord(f'Stage shape/finiteness: {name}')
        if any(type(r[k]) not in (int, float) or not math.isfinite(r[k]) or r[k] < 0 for k in ('rms','max_abs','l2')):
            raise InvalidRecord(f'Nonfinite/invalid stage statistic: {name}')
