"""Prepared actual seal adapter. Never instantiated by D02B10 tests/dispatch."""
import hashlib
import json
from pathlib import Path
from .operations import execute_real, write_new

class FrozenAdapter:
    scope='live-unapproved'
    def __init__(self,root,pin_path,expected_pin_sha256):
        self.root=Path(root).resolve()
        if Path.cwd().resolve()!=self.root:raise RuntimeError('Frozen recipe requires repository CWD')
        data=Path(pin_path).read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected_pin_sha256:raise RuntimeError('Trusted pin-file hash required')
        self.pins=json.loads(data)
        if set(self.pins)!={'files','historical_result','cell'} or self.pins['cell']!='D02B-G01-A0-SAME-FORWARD':raise RuntimeError('Pin schema')
        if self.pins['historical_result']!={'failed_tensors':293,'outside_coordinates':8772234}:raise RuntimeError('Historical result changed')
        self.parity=None;self.forward_calls=0
    def verify_static(self):
        from mfvlr_research.d02b1.schema import verify_files
        verify_files(self.root,self.pins['files'])
    def build(self):
        import torch
        torch.set_num_threads(4);torch.set_num_interop_threads(4)
        torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.mkldnn.enabled=True
        from tools.review_phase2c2_n2 import backend
        from mfvlr_research.d02b2.contract import bundle
        if backend()!=bundle()['fixtures']['F0']['backend']:raise RuntimeError('Pinned Windows CPU C4M backend mismatch')
        from mfvlr_research.d02b9.live_preparation import _build_frozen_reference
        model,fixture,self.parity=_build_frozen_reference()
        return model,fixture
    def snapshot(self,model,fixture):
        import torch
        from tools.review_phase2c2_n2 import backend
        from tools.phase2d2_fixtures import tensor_hash
        from tools.phase2d2_diagnostics import state_hash
        from mfvlr.integration import ownership_report
        from mfvlr_research.d02b1.schema import validate_inventory
        from mfvlr_research.d02b9.live_preparation import STATE
        from mfvlr_research.d02b2.contract import bundle
        expected=json.loads((self.root/'artifacts/phase2d2_d02b7_r2/attempt_01/fixture.json').read_text())
        inputs=dict(image=tensor_hash(fixture.image),tokens=tensor_hash(fixture.targets.token_ids),
            mask=tensor_hash(fixture.targets.masks),labels=tensor_hash(fixture.targets.labels),
            sample_order=list(fixture.targets.sample_ids),kinds=list(fixture.targets.sample_kinds),
            sample_order_sha256=hashlib.sha256(json.dumps(list(fixture.targets.sample_ids),separators=(',',':')).encode()).hexdigest())
        if inputs!=expected['inputs'] or fixture.provenance!=expected['provenance']:raise RuntimeError('Actual fixture/input mismatch')
        rows=[dict(name=n,shape=list(p.shape),dtype='float32',sha256=tensor_hash(p),object_id=str(id(p)),
                   storage_id=str(p.untyped_storage().data_ptr())) for n,p in model.named_parameters(remove_duplicate=False)]
        if any(p.dtype!=torch.float32 or p.device.type!='cpu' or not p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError('FP32 CPU clean trainable ownership')
        roster=json.loads((self.root/'tests/research_d02b1/fixtures/historical_rosters.json').read_text())['rosters']['A0']
        validate_inventory(rows,'A0',roster);ownership_report(model)
        if model.flt.vocabulary_output_weight is not model.flt.vocabulary.weight:raise RuntimeError('Vocabulary tie')
        if state_hash(model)!=STATE or backend()!=bundle()['fixtures']['F0']['backend']:raise RuntimeError('State/backend seal')
        return dict(state=STATE,ownership=rows,inputs=inputs,fixture=fixture.provenance,backend=backend(),parity=self.parity)
    def run(self,model,fixture,path,seal,check_monitor):
        from mfvlr_research.d02b9.live_preparation import _prepared_body
        hooks=[]
        def monitor_forward(module,args):check_monitor()
        def monitor_gradient(gradient):
            check_monitor()
            return gradient
        def count_forward(module,args):
            self.forward_calls+=1
            if self.forward_calls!=1:raise RuntimeError('Extra full-language forward')
        try:
            hooks.extend(m.register_forward_pre_hook(monitor_forward) for m in model.modules())
            hooks.extend(p.register_hook(monitor_gradient) for p in model.parameters())
            hooks.append(model.flt.register_forward_pre_hook(count_forward))
            result=_prepared_body(model,fixture,path,seal)
            if self.forward_calls!=1:raise RuntimeError('Missing full-language forward')
            write_new(Path(path)/'actual_forward_count.json',dict(full_language_forwards=self.forward_calls))
            return result
        finally:
            for hook in hooks:hook.remove()
    def verify_complete(self,path,analysis):
        from mfvlr_research.d02b1.schema import participation,PASSES
        if analysis['parameter_tensors']!=508 or analysis['elementwise']!={'actual':'FP32','reference':'FP32(FP64 ordered sum)','atol':1e-6,'rtol':1e-5}:raise RuntimeError('Comparison contract changed')
        roster=json.loads((self.root/'tests/research_d02b1/fixtures/historical_rosters.json').read_text())['rosters']['A0']
        names=[r['name'] for r in roster]
        if set(analysis['parameters'])!=set(names):raise RuntimeError('Incomplete full roster')
        records=list((path/'gradient_stream').glob('[0-9][0-9][0-9][0-9][0-9].json'))
        if len(records)!=508*7:raise RuntimeError('Incomplete raw records')
        for index,(loss,name) in enumerate((l,n) for l in PASSES for n in names):
            row=json.loads((path/'gradient_stream'/f'{index:05d}.json').read_text())
            if row['loss']!=loss or row['parameter']!=name or row['expectation']!=participation(name)[loss]:raise RuntimeError('Raw participation/order mismatch')
        for filename in ('passes.json','activations.json','live_arithmetic.json','additivity.json','separate_forward_pairing.json','actual_forward_count.json'):
            if not (path/filename).is_file():raise RuntimeError('Missing required output '+filename)
        passes=json.loads((path/'passes.json').read_text())
        if [p['loss'] for p in passes]!=list(PASSES) or [p['retain_graph'] for p in passes]!=[True]*6+[False]:raise RuntimeError('Traversal roster')
        paired=json.loads((path/'separate_forward_pairing.json').read_text())
        if len(paired)!=508*7 or any(p['structural']!='PASS' for p in paired):raise RuntimeError('Historical pairing/structure')
        from .evidence import validate_stream
        write_new(path/'raw_integrity.json',validate_stream(path/'gradient_stream',roster,analysis))

def execute(*args,**kwargs):
    return execute_real(*args,**kwargs)

def _run_live(expected_pin_sha256):
    """Complete internal wiring for future reviewed authorization; NOT dispatched.

    Fixed global cell root prevents choosing another root to evade its lease.
    This function must not be invoked during D02B10.
    """
    from .operations import _orchestrate
    root=Path(__file__).resolve().parents[3]
    parent=root/'artifacts/phase2d2_d02b10/live_attempts'
    parent.mkdir(exist_ok=True)
    adapter=FrozenAdapter(root,root/'configs/research_d02b10/live_pins.json',expected_pin_sha256)
    return _orchestrate(parent,'attempt_01',adapter)
