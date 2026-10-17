"""Scoped N2 acceptance runner; production modules and historical evidence are immutable.

Run prepare once, then audit, targeted, suite, diagnostics, in that order.
All child stdout/stderr and commands are preserved separately. No test filters
are used for the complete suite. Numerical diagnostics use the unchanged
review helper, with explicit destinations under the new acceptance directory.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'.phase2c1_deps'), str(ROOT/'src')]
OUT = ROOT/'artifacts/phase2c2_n2_acceptance'
BASE = '484efd03bb95725e353c7c2257147f8ee2fe8914d1d5d98abcbf9f2a99c8e6d8'
TARGET = ROOT/'tests/language/test_flt.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def prepare():
    assert sha(TARGET) == BASE, 'Approved patch baseline hash mismatch; no changes made'
    assert not (OUT/'approval.json').exists(), 'Never overwrite an acceptance record'
    history = OUT/'history'
    history.mkdir(parents=True, exist_ok=True)
    before = {}
    for directory in ('src','tests','configs','docs','tools','artifacts'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and OUT not in path.parents and '__pycache__' not in path.parts:
                before[path.relative_to(ROOT).as_posix()] = sha(path)
    write(OUT/'baseline_manifest.json', before)
    (history/'test_flt_before_n2.py.txt').write_bytes(TARGET.read_bytes())
    for name in ('proposed_t17.patch','proposal_status.json'):
        (history/name).write_bytes((ROOT/'artifacts/phase2c2_n2'/name).read_bytes())
    original = TARGET.read_text().splitlines(keepends=True)
    patch = (history/'proposed_t17.patch').read_text().splitlines(keepends=True)
    assert patch[:3] == ['--- a/tests/language/test_flt.py\n',
                        '+++ b/tests/language/test_flt.py\n', '@@ -235,13 +235,83 @@\n']
    old = [line[1:] for line in patch[3:] if line[0] in ' -']
    new = [line[1:] for line in patch[3:] if line[0] in ' +']
    assert len(old) == 13 and len(new) == 83 and original[234:247] == old
    result = ''.join(original[:234]+new+original[247:])
    # Preserve the original file's newline convention as well as all other bytes.
    newline = b'\r\n' if b'\r\n' in TARGET.read_bytes() else b'\n'
    TARGET.write_bytes(result.encode('utf-8').replace(b'\n', newline))
    write(OUT/'approval.json', {
        'amendment':'N2','status':'APPROVED','applied':True,'date':'2026-10-11',
        'authority':'Explicit user instruction: Phase 2C.2 N2 Acceptance Implementation',
        'scope':'ONLY FLT-T17 cross-batch FP32 verification policy',
        'patch_sha256':sha(history/'proposed_t17.patch'), 'baseline_test_sha256':BASE,
        'patched_test_sha256':sha(TARGET),'exact_proposed_patch_applied':True,
        'historical_PROPOSED_comment_retained':'Literal approved patch; superseded by this approval record',
        'FP32_diagnostic_tolerance':{'atol':1e-6,'rtol':1e-5},
        'FP64_mandatory_tolerance':{'atol':1e-10,'rtol':1e-8},
        'MVE_N1_applied_to_FLT':False,'architecture_changes_authorized':False,
        'production_changes_authorized':False,'training_authorized':False,
        'final_user_acceptance':'PENDING review of completed verification'})
    print('Baseline verified; exact approved patch applied.', flush=True)


def command_run(name, command):
    logs = OUT/'logs'; logs.mkdir(parents=True, exist_ok=True)
    assert not (logs/f'{name}.execution.json').exists(), 'Use a new run name; preserve existing logs'
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)])
    start = time.perf_counter()
    with (logs/f'{name}.stdout.log').open('w',encoding='utf-8') as stdout, \
         (logs/f'{name}.stderr.log').open('w',encoding='utf-8') as stderr:
        completed = subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    record = {'command':command,'cwd':str(ROOT),'PYTHONPATH':env['PYTHONPATH'],
              'exit_code':completed.returncode,'elapsed_seconds':time.perf_counter()-start}
    write(logs/f'{name}.execution.json',record)
    print(json.dumps(record),flush=True)
    if completed.returncode: raise RuntimeError(f'{name} failed; inspect preserved stdout/stderr')


def audit():
    import gc
    import torch
    from tools import review_phase2c2_n2 as h
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    checks = {}
    def rejects(label, error, function):
        try: function()
        except error: checks[label] = True
        else: raise AssertionError(label)
    a = torch.tensor([0.,2.,1.]); b = torch.tensor([0.,1.,0.])
    metric = h.compare(a,b,atol=1e-6,rtol=1e-5)
    assert metric['outside_tolerance_count']==2 and metric['elements']==3
    assert metric['max_absolute_difference']==1 and metric['relative_l2_difference']==2**.5
    assert metric['zero_reference_nonzero_difference_count']==1 and metric['max_relative_difference_is_infinite']
    checks['compare_known_statistics'] = True
    rejects('compare_shape_rejection',AssertionError,lambda:h.compare(a,b[:1],1e-6,1e-5))
    rejects('compare_dtype_rejection',AssertionError,lambda:h.compare(a,b.double(),1e-6,1e-5))
    for value in (float('nan'),float('inf')):
        rejects(f'compare_nonfinite_{value}',FloatingPointError,
                lambda:h.compare(torch.tensor([value]),torch.zeros(1),1e-6,1e-5))
    small = OUT/'audit_roundtrip.json'
    h.write(small,metric);assert json.loads(small.read_text())==metric
    rejects('write_nonfinite_rejection',ValueError,lambda:h.write(OUT/'must_not_exist.json',{'x':float('nan')}))
    assert not (OUT/'must_not_exist.json').exists()
    checks['write_json_roundtrip'] = True
    ids,v,description = h.fixture(0)
    model = h.FineGrainedLanguageTransformer(seed=0)
    before = h.state_hash(model); identities=[id(p) for p in model.parameters()]
    capture = h.forward_capture(model,ids,v)
    expected = {'vocabulary','encoder.input','decoder.input'} | {'output.'+n for n in h.FIELDS}
    expected |= {f'encoder.blocks.{i}' for i in range(12)} | {f'decoder.blocks.{i}' for i in range(7)}
    assert set(capture)==expected and len(capture)==27
    for name,value in capture.items():
        shape = (2,308,49408) if name=='output.logits' else ((2,1,512) if name=='output.global_language' else (2,308,512))
        assert tuple(value.shape)==shape and value.isfinite().all()
        assert not value.requires_grad and value.grad_fn is None
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    checks['all_27_stages_shapes_finite_no_graph_hooks_removed'] = True
    del capture;gc.collect()
    lookup_records=[]
    def lookup_hook(module,args,output):
        lookup_records.append({'requires_grad':output.requires_grad,'grad_fn':type(output.grad_fn).__name__,
                               'shape':list(output.shape)})
    hook=model.vocabulary.register_forward_hook(lookup_hook)
    try: gradients=h.cross_sample_gradients(model,ids,v)
    finally: hook.remove()
    assert lookup_records==[{'requires_grad':True,'grad_fn':'EmbeddingBackward0','shape':[2,308,512]}]
    assert len(gradients)==2
    for record in gradients:
        assert record['other_lookup_gradient_exact_zero'] and record['other_visual_gradient_exact_zero']
        assert record['own_lookup_gradient']['l2_norm']>0 and record['own_visual_gradient']['l2_norm']>0
    assert all(p.grad is None for p in model.parameters())
    checks['genuine_lookup_and_both_direction_gradients'] = True
    reference=copy.deepcopy(model).double()
    assert reference.vocabulary_output_weight is reference.vocabulary.weight
    assert all(a is not b and torch.equal(a.float(),b) for a,b in zip(reference.parameters(),model.parameters()))
    assert h.state_hash(model)==before and identities==[id(p) for p in model.parameters()]
    checks['fp64_deepcopy_exact_fp32_initialization_and_original_identity'] = True
    metadata=h.backend()
    assert metadata['torch']==torch.__version__ and metadata['num_threads']==4
    assert metadata['device']=='cpu' and metadata['build_config'] and metadata['deterministic_algorithms']
    checks['backend_matches_actual_runtime'] = True
    write(OUT/'helper_audit.json',{'status':'PASS','checks':checks,'capture_keys':sorted(expected),
        'lookup_records':lookup_records,'cross_sample_gradients':gradients,'backend':metadata,
        'fixture':description,'fp32_initialized_state_sha256':before,'parameter_report':h.parameter_report(model),
        'helper_source_sha256':sha(ROOT/'tools/review_phase2c2_n2.py')})
    print(json.dumps({'helper_audit':'PASS','checks':checks}),flush=True)


def pytest_run(complete):
    name='complete' if complete else 'new_t17'
    junit=OUT/'junit'/f'{name}.xml';junit.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pytest']
    command += ['tests'] if complete else ['tests/language/test_flt.py','-k','t17']
    command += ['-vv','-s',f'--junitxml={junit}']
    # Unchanged integration tests emit to historical destinations. Save original
    # bytes before execution, retain fresh measurements separately, restore originals.
    historical={p:p.read_bytes() for p in (ROOT/'artifacts/phase2c2/gradients').glob('*.json')}
    for p,data in historical.items():
        destination=OUT/'history/gradients'/p.name
        if not destination.exists(): destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
    try: command_run(name,command)
    finally:
        if complete:
            for p,data in historical.items():
                fresh=OUT/'integration_gradients'/p.name;fresh.parent.mkdir(parents=True,exist_ok=True)
                fresh.write_bytes(p.read_bytes())
                p.write_bytes(data)


def diagnostics():
    destination=OUT/'diagnostics';destination.mkdir(parents=True,exist_ok=True)
    for fixture in range(3):
        for dtype in ('float32','float64'):
            name=f'fixture{fixture}_{dtype}_threads4'
            path=destination/f'{name}.json'
            command_run(name,[sys.executable,str(ROOT/'tools/review_phase2c2_n2.py'),
                '--worker',str(fixture),'--dtype',dtype,'--threads','4','--output',str(path)])
    write(destination/'provenance.json',{
        'N2_status':'APPROVED','patch_applied':True,'fresh_after_complete_suite':True,
        'note':'Immutable review worker retains its historical review_only/PROPOSED/N2_applied=false labels. '
               'These describe the diagnostic routine, not active test policy. Approval is recorded separately.',
        'worker_source_sha256':sha(ROOT/'tools/review_phase2c2_n2.py'),
        'FP64_method':'Worker promotes the same seed-0 FP32-initialized values. Mandatory pytest separately '
                      'deep-copies the live FP32 model, verifies every promoted value, and compares all stages.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','audit','targeted','suite','diagnostics'))
    stage=parser.parse_args().stage
    if stage=='prepare': prepare()
    elif stage=='audit': audit()
    elif stage=='targeted': pytest_run(False)
    elif stage=='suite': pytest_run(True)
    else: diagnostics()
