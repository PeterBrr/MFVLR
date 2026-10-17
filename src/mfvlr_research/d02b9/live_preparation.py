"""Unexecuted full-model recipe. No enabled live dispatcher in this phase.

    A later reviewed execution wrapper must implement operational ownership,
    RAM/disk admission, sampled resource monitoring and source/backend seals.
    This factory is deliberately blocked before importing any MFVLR module.
"""
from .same_forward import dispatch_real

CELL = 'D02B-G01-A0-SAME-FORWARD'
PARAMETERS = 508
ELEMENTS = 154156232
STATE = '1865527facba9adc8a14d443c112d492698d275ce4e8beaa784aee61a514895a'

def execute(*args, **kwargs):
    return dispatch_real(*args, **kwargs)

def _build_frozen_reference():
    """Prepared lazy factory, NOT called in D02B9 verification."""
    import gc
    import json
    from pathlib import Path
    from mfvlr.integration import MFVLRIntegration, ownership_report
    from mfvlr_research.decoder_a11 import build_variant, parity_report
    from tools.phase2d2_diagnostics import state_hash
    from tools.phase2d2_fixtures import fixture
    reference = MFVLRIntegration()
    if state_hash(reference) != STATE: raise RuntimeError('Seed-0 state mismatch')
    model = build_variant(reference,'A0')
    parity = parity_report(reference,model)
    ownership_report(model)
    f = fixture('real')
    historical = json.loads(Path('artifacts/phase2d2_d02b7_r2/attempt_01/fixture.json').read_text(encoding='utf-8'))
    if f.provenance != historical['provenance']: raise RuntimeError('Original G01 fixture mismatch')
    del reference
    gc.collect()
    return model,f,parity

def _prepared_body(model, fixture, destination, seal):
    """Unexecuted recipe, for a future authorized wrapper only.

    `seal` must verify frozen sources/config/backend, owner identity, resources,
    fixture bytes, state and roster. No caller-accessible approval flag exists.
    Tests exercise the traversal kernel, never this full-model recipe.
    """
    from pathlib import Path
    from tools.phase2d2_diagnostics import state_hash, stats
    from mfvlr_research.d02b7.capture import Capture
    from mfvlr_research.d02b7.arithmetic import audit
    from mfvlr_research.d02b1.schema import participation, write_new
    from mfvlr_research.d02b1.gradients import GradientStore
    from .same_forward import _traverse, ORDER
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    named = tuple(model.named_parameters())
    if len(named) != PARAMETERS or sum(p.numel() for _,p in named) != ELEMENTS or state_hash(model) != STATE:
        raise RuntimeError('Frozen A0 roster/state mismatch')
    if model.flt.vocabulary_output_weight is not model.flt.vocabulary.weight:
        raise RuntimeError('Vocabulary ownership')
    if any(p.grad is not None for _,p in named): raise RuntimeError('Dirty parameter gradients')
    seal('before_single_forward')
    store = GradientStore(root/'gradient_stream', {
        name: dict(shape=list(p.shape), presence=participation(name)) for name,p in named})
    with Capture(model, stats) as capture:
        output = model.with_language(fixture.image, fixture.targets.token_ids,
                                     sample_ids=fixture.targets.sample_ids)
        write_new(root/'activations.json', capture.finish(output))
        values = model.compute_objectives(output, fixture.targets).as_dict()
        arithmetic = audit(output, fixture.targets, model, values)
        write_new(root/'live_arithmetic.json', arithmetic)
        if not arithmetic['all_exact']: raise RuntimeError('Live objective arithmetic')
        seal('before_traversals')
        passes = _traverse(named, {n:values[n] for n in ORDER}, store)
        write_new(root/'passes.json', passes)
    del values, output
    seal('after_traversals')
    if state_hash(model) != STATE or any(p.grad is not None for _,p in named):
        raise RuntimeError('State or parameter-gradient mutation')
    analysis = store.analyze()
    write_new(root/'additivity.json', analysis)
    from .comparison import compare_streams
    paired = compare_streams(root/'gradient_stream',
        Path('artifacts/phase2d2_d02b7_r2/attempt_01/gradient_stream'))
    write_new(root/'separate_forward_pairing.json', paired)
    return analysis

def resource_readiness(available_ram, free_disk):
    """Proposed conservative admission screen, NOT a capacity guarantee."""
    if any(type(x) is not int or x < 0 for x in (available_ram,free_disk)):
        raise ValueError('Measured nonnegative integer byte counts required')
    return dict(resource='SCREEN_MET' if available_ram >= 8*1024**3 and free_disk >= 8*1024**3 else 'WAIT',
                capacity='NOT VERIFIED', execution='DISABLED', dependent_work='BLOCKED')

def validate_plan(plan):
    required = {'cell','execution_enabled','B0_D08','parameter_tensors','elements',
                'forward_count','gradient_passes','retain_graph','state_sha256',
                'atol','rtol','backend','fixture','resource_capacity_verified'}
    if set(plan) != required: raise ValueError('Unknown or missing execution-plan fields')
    expected = dict(cell=CELL, execution_enabled=False, B0_D08=False,
        parameter_tensors=PARAMETERS, elements=ELEMENTS, forward_count=1,
        gradient_passes=['fd','lr','cmc','fl','ar','kl','total'],
        retain_graph=[True]*6+[False], state_sha256=STATE, atol=1e-6, rtol=1e-5,
        backend='Windows CPU C4M', fixture='original G01 real B1', resource_capacity_verified=False)
    if plan != expected: raise ValueError('Frozen scope mismatch')
    return 'PREPARED; REAL EXECUTION DISABLED'
