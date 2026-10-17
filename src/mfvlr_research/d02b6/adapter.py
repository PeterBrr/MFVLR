"""Research-only G2 evidence adapter; model dispatch is unconditionally disabled.

No torch import, model construction, backward, subprocess, CLI or YAML switch.
Loss bindings describe the accepted callable boundaries without reimplementing
objectives. Only small independent NumPy mock records can be consumed here.
"""
import copy
import math
from pathlib import Path
import numpy as np
from ..d02b1 import schema as s
from ..d02b1.gradients import GradientStore, state_snapshot, require_state_unchanged
from ..d02b1.runner import ExecutionNotAuthorized
from ..d02b1.safety import Reservations, Coordinator, archive_attempt, verify_replay
from ..d02b1.pairing import common_state_parity, invariant_pair
from ..d02b1.stages import stage_shapes, validate_stages

LOSS_BINDINGS = {
    'fd': ('mfvlr.objectives.forgery_detection_loss', ('detection_logits','labels')),
    'lr': ('mfvlr.objectives.language_reconstruction_loss', ('language_logits','token_ids')),
    'cmc': ('root.objectives.cmc', ('visual','global_language')),
    'fl': ('mfvlr.objectives.forgery_localization_loss', ('mask_logits','masks')),
    'ar': ('mfvlr.objectives.appearance_reconstruction_loss', ('image','reconstruction')),
    'kl': ('mfvlr.objectives.knowledge_distillation_loss', ('global_language','predicted_language')),
}
FIELDS = tuple(dict.fromkeys(k for _, args in LOSS_BINDINGS.values() for k in args)) + ('sample_kinds',)

def dispatch(*args, **kwargs):
    raise ExecutionNotAuthorized('B0-D03 disabled: D02B6 permits static/mock verification only')

def continue_after_failure(*args, **kwargs):
    raise ExecutionNotAuthorized('B0-D08 disabled: retain FAIL and stop dependent work')

def primary_matrix():
    return [dict(c, execution='NOT RUN', numerical='NOT RUN', structural='NOT RUN',
                 resource='NOT VERIFIED') for c in s.schedule() if c['partition']=='G2-primary']

def cell_plan(cell_id):
    matches=[c for c in s.schedule() if c['experiment_id']==cell_id and c['partition']=='G2-primary']
    if len(matches)!=1: raise s.InvalidRecord('Only exact G01-G06 primary identities')
    cell=matches[0]
    return dict(identity=cell, passes=list(s.PASSES), parameter_inventory=s.COUNTS[cell['variant']],
                required_stages=stage_shapes(cell['batch']),
                loss_bindings=copy.deepcopy(LOSS_BINDINGS), execution='NOT RUN',
                dispatch='DISABLED', continuation='DISABLED',
                timing_plan=['W:no-grad','T1:joint','T2:joint','T3:joint'],
                timing_execution='NOT RUN; separate execution approval required')

def bind_arguments(fields):
    """Identity-preserving bindings only: no detach, casts, objectives or tensor operations."""
    s.exact_keys(fields,FIELDS)
    return {loss:tuple(fields[name] for name in args) for loss,(_,args) in LOSS_BINDINGS.items()}

def validate_preflight(record, *, pinned, roster, source_root, source_hashes,
                       reference_inventory, ram, disk):
    cell_plan(record['identity']['experiment_id'])
    s.verify_files(source_root,source_hashes)
    seal=s.validate_cell(record,approved_partition='G2-primary',pinned=pinned,expected_inventory=roster)
    parity=common_state_parity(reference_inventory,record['ownership'])
    reservation=Reservations().check(record['identity']['batch'],ram,disk)
    return dict(seal=seal,parity=parity,resource=reservation,
                basis='Validated supplied metadata only; not a live model observation',execution='NOT RUN')

def seal_pass(record,expected,path):
    """Persist observed records only after exact agreement with independently pinned fields."""
    s.exact_keys(record,('cell','pass','state','ownership','inputs','source','backend','resource'))
    cell_plan(record['cell'])
    if record['pass'] not in s.PASSES: raise s.InvalidRecord('Unregistered scientific pass')
    for key in ('state','ownership','inputs','source'):
        if not isinstance(record[key],dict) or not record[key]:raise s.InvalidRecord('Missing seal map')
        for digest in record[key].values():s.sha(digest)
    if s.canonical(record)!=s.canonical(expected):raise s.InvalidRecord('Pre-forward seal mismatch')
    s.write_new(path,record)

def failure_record(path, *, provenance, structural, finite, complete, numerical, resource):
    if any(type(v) is not bool for v in (provenance,structural,finite,complete)):
        raise s.InvalidRecord('Exact gate booleans required')
    if numerical not in ('PASS','FAIL','NOT RUN') or resource not in ('PASS','INSUFFICIENT','FAIL/OOM','NOT VERIFIED'):
        raise s.InvalidRecord('Invalid status')
    hard=not all((provenance,structural,finite,complete)) or resource!='PASS'
    result=dict(execution='HARD STOP' if hard else 'NOT RUN',provenance=provenance,
                structural=structural,finite=finite,complete=complete,numerical=numerical,
                resource=resource,dependent_work='BLOCKED',B0_D03='DISABLED',B0_D08='DISABLED')
    s.write_new(path,result)  # Store genuine failure before reporting the stop.
    return result

class MockEvidenceAdapter:
    """Exercise frozen stream with tiny arrays only. Cannot ingest full-model gradients.

    Metadata may cover all 550 names, but each mock parameter has <=16 elements
    and the entire state has <=4096. Never a model execution or scientific PASS.
    """
    def __init__(self,path,specs,state):
        if set(specs)!=set(state) or len(specs)>550 or sum(a.size for a in state.values())>4096:
            raise s.InvalidRecord('Small mock inventory only')
        for n,a in state.items():
            if type(a) is not np.ndarray or a.dtype!=np.float32 or a.size>16 or list(a.shape)!=specs[n]['shape']:
                raise s.InvalidRecord('Small independent FP32 mock arrays only')
        self.state=state;self.before=state_snapshot(state);self.specs=copy.deepcopy(specs)
        self.store=GradientStore(Path(path)/'gradients',specs);self.cursor=0;self.scalars={}

    def consume(self,loss,gradients,scalar):
        if self.cursor>=7 or loss!=s.PASSES[self.cursor]:raise s.InvalidRecord('Seven-pass order')
        require_state_unchanged(self.before,self.state)
        s.exact_keys(gradients,self.specs)
        if type(scalar) not in (int,float) or not math.isfinite(scalar):raise s.InvalidRecord('Nonfinite scalar')
        for name in self.specs:self.store.put(loss,name,gradients[name])
        self.scalars[loss]=scalar;self.cursor+=1

    def finish(self):
        require_state_unchanged(self.before,self.state)
        result=self.store.analyze()
        result.update(basis='Independent small-array mock only; no G2 execution',losses=self.scalars,
                      execution='NOT RUN',dependent_work='BLOCKED',B0_D08='DISABLED')
        s.write_new(self.store.root.parent/'mock_result.json',result)
        return result
