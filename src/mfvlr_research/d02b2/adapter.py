"""Pinned A1.1 adapter plan and deliberately blocked real construction boundary."""
from .contract import (VARIANTS, InvalidEvidence, ExecutionDisabled, bundle, norm_inventory,
                       validate_request, verify_sources)

class FrozenN2Adapter:
    def __init__(self,variant):
        if type(variant) is not str or variant not in VARIANTS: raise InvalidEvidence('Only frozen A0/P3/Q3')
        self.variant=variant

    def plan(self):
        return {'variant':self.variant,'reference':'mfvlr.flt.FineGrainedLanguageTransformer(seed=0)',
                'builder':'mfvlr_research.decoder_a11.build_variant',
                'builder_sha256':bundle()['sources']['src/mfvlr_research/decoder_a11.py'],
                'build_order':'CPU FP32 reference -> frozen build_variant -> deep-copy candidate -> FP64 promotion if requested',
                'norms':norm_inventory(self.variant),'execution':'NOT RUN','dispatch':'DISABLED'}

    def prepare(self,record,pinned_sha256,root):
        if record.get('variant')!=self.variant: raise InvalidEvidence('Adapter/record variant mismatch')
        validate_request(record,pinned_sha256)
        audit=verify_sources(root)
        return {'plan':self.plan(),'source_audit':audit,'execution':'NOT RUN'}

    def build(self,reference):
        # Deliberate unconditional barrier BEFORE any import or builder invocation.
        # The frozen binding below documents the future call; no callback/flag bypass.
        raise ExecutionDisabled('Real N2 construction requires separate approval and reviewed gate change')
        from mfvlr_research.decoder_a11 import build_variant
        return build_variant(reference,self.variant)

    def promoted_copy(self,candidate):
        raise ExecutionDisabled('Real FP64 model promotion is not authorized')
        import copy
        return copy.deepcopy(candidate).double()

    def execute(self,*args,**kwargs):
        raise ExecutionDisabled('No numerical N2 execution in D02B2')

def dispatch(*args,**kwargs):
    raise ExecutionDisabled('No CLI/YAML/record may authorize N2 execution')
