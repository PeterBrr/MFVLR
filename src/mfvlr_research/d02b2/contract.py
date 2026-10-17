"""Frozen N2 metadata and input verification; no random fixture regeneration."""
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

BUNDLE_SHA256='3e4bc2ab73652b3c62d246e4c61edd5c12f0a56e84c5968bde2042ed199bf0d2'
VARIANTS=('A0','P3','Q3')
PRECISIONS=('float32','float64')
FP32=(1e-6,1e-5)
FP64=(1e-10,1e-8)
COUNTS={'A0':(387,112758272),'P3':(429,112779776),'Q3':(429,112779776)}

class InvalidEvidence(ValueError): pass
class ExecutionDisabled(PermissionError): pass

def exact(value,keys):
    keys=set(keys)
    if type(value) is not dict or set(value)!=keys: raise InvalidEvidence(f'Expected exact keys {sorted(keys)}')

def sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None: raise InvalidEvidence('Missing/invalid SHA-256')
    return value

def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def seal(value): return hashlib.sha256(canonical(value)).hexdigest()
def file_hash(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def tensor_hash(array): return hashlib.sha256(array.tobytes(order='C')).hexdigest()

def bundle():
    path=Path(__file__).with_name('frozen_inputs.json')
    if file_hash(path)!=BUNDLE_SHA256: raise InvalidEvidence('Frozen input bundle changed')
    return json.loads(path.read_text(encoding='utf-8'))

def verify_sources(root):
    root=Path(root).resolve()
    pins=bundle()['sources']
    for rel,digest in pins.items():
        path=(root/rel).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=digest: raise InvalidEvidence('Frozen source/evidence changed: '+rel)
    return {'files_checked':len(pins),'status':'PASS','basis':'file integrity only; no model execution'}

def load_fixture(name):
    if name not in ('F0','F1','F2'): raise InvalidEvidence('Unknown fixture')
    desc=bundle()['fixtures'][name]['fixture_description']
    ids=np.array(desc['ids'],dtype='<i8')
    visual=np.array(desc['fp32_visual'],dtype='<f4')
    if ids.shape!=(2,308) or visual.shape!=(2,1,512) or not np.isfinite(visual).all(): raise InvalidEvidence('Fixture shape/finite')
    if tensor_hash(ids)!=desc['ids_sha256'] or tensor_hash(visual)!=desc['fp32_visual_sha256']: raise InvalidEvidence('Fixture bytes')
    if np.any(ids<0) or np.any(ids>=49408): raise InvalidEvidence('Vocabulary range')
    ids.flags.writeable=False; visual.flags.writeable=False
    return ids,visual

def fixture_views(name):
    ids,visual=load_fixture(name)
    return {'B2':(ids,visual),'B1_sample0':(ids[:1],visual[:1]),'B1_sample1':(ids[1:2],visual[1:2])}

def fixed_batch_change(ids,visual,sample):
    if type(sample) is not int or sample not in (0,1) or ids.shape!=(2,308) or visual.shape!=(2,1,512): raise InvalidEvidence('Fixed-batch fixture')
    changed_ids=ids.copy(); changed_visual=visual.copy()
    changed_ids[sample,250]=1234; changed_visual[sample]+=np.float32(.5)
    return changed_ids,changed_visual

def stages(fixture,batch):
    if fixture not in ('F0','F1','F2') or type(batch) is not int or batch not in (1,2): raise InvalidEvidence('Stage fixture/batch')
    result={k:[batch,308,512] for k in ('vocabulary','encoder.input','decoder.input')}
    result.update({f'encoder.blocks.{i}':[batch,308,512] for i in range(12)})
    result.update({f'decoder.blocks.{i}':[batch,308,512] for i in range(7)})
    result.update({'output.'+k:[batch,308,512] for k in ('low','encoder_hidden','decoder_hidden')})
    result['output.global_language']=[batch,1,512]; result['output.logits']=[batch,308,49408]
    if fixture=='F0':
        names=('ln','attention.q','attention.k','attention.v','attention.out','vim.q','vim.k','vim.v','vim.out','ff.fc1','ff.gelu','ff.fc2')
        result.update({'encoder.blocks.0.'+n:[batch,308,2048 if n in ('ff.fc1','ff.gelu') else 512] for n in names})
        for n in ('vim.k','vim.v'): result['encoder.blocks.0.'+n]=[batch,1,512]
    return result

def stage_shape(name,fixture,batch):
    # VIM K/V projections receive the singleton visual token, not 308 tokens.
    result=stages(fixture,batch)[name]
    if name in ('encoder.blocks.0.vim.k','encoder.blocks.0.vim.v'): return [batch,1,512]
    return result

def norm_inventory(variant):
    if variant not in VARIANTS: raise InvalidEvidence('Variant')
    result={f'encoder.blocks.{i}.ln':dict(width=512,eps=1e-5,affine=True,placement='pre_self_attention') for i in range(12)}
    if variant!='A0':
        position='pre' if variant=='P3' else 'post_residual'
        for i in range(7):
            for j,role in enumerate(('self_attention','memory_query','ff'),1):
                result[f'decoder.blocks.{i}.ln{j}']=dict(width=512,eps=1e-5,affine=True,placement=position+'_'+role)
    return result

def schedule():
    return [dict(experiment_id=f'D02B2-N2-{f}-{v}-{d}',fixture=f,variant=v,precision=d,execution='NOT RUN')
            for f in ('F0','F1','F2') for v in VARIANTS for d in PRECISIONS]

def validate_request(record,pinned_sha256):
    exact(record,('experiment_id','fixture','variant','precision','backend','source_hashes','input_hashes',
                  'base_fp32_state_sha256','candidate_fp32_state_sha256','state_hash_algorithm','initialization_seed'))
    identity={k:record[k] for k in ('experiment_id','fixture','variant','precision')}
    if dict(identity,execution='NOT RUN') not in schedule(): raise InvalidEvidence('Unregistered N2 identity')
    data=bundle(); f=data['fixtures'][record['fixture']]; desc=f['fixture_description']
    if canonical(record['backend'])!=canonical(f['backend']): raise InvalidEvidence('Backend cohort mismatch')
    if record['source_hashes']!=data['sources']: raise InvalidEvidence('Complete source/protocol/config/evidence hashes required')
    if record['input_hashes']!={'ids':desc['ids_sha256'],'visual_fp32':desc['fp32_visual_sha256']}:
        raise InvalidEvidence('Input hashes/order mismatch')
    if record['base_fp32_state_sha256']!=f['a0_fp32_state_sha256']: raise InvalidEvidence('Wrong common FP32 initialization')
    sha(record['candidate_fp32_state_sha256'])
    if record['variant']=='A0' and record['candidate_fp32_state_sha256']!=record['base_fp32_state_sha256']: raise InvalidEvidence('A0 state changed')
    if record['state_hash_algorithm']!=data['state_hash_algorithm']: raise InvalidEvidence('State hash algorithm changed')
    if type(record['initialization_seed']) is not int or record['initialization_seed']!=0: raise InvalidEvidence('Seed changed')
    if seal(record)!=sha(pinned_sha256): raise InvalidEvidence('Unsealed provenance')
    return {'metadata':'VALID','execution':'NOT RUN','runtime_observations':'NOT VERIFIED'}

def validate_ownership(observation,variant):
    exact(observation,('parameters','norms','encoder_blocks','decoder_blocks','vocabulary_lookup_id','vocabulary_output_id','vim_instances'))
    if variant not in VARIANTS: raise InvalidEvidence('Variant')
    if observation['encoder_blocks']!=[f'encoder.blocks.{i}' for i in range(12)] or observation['decoder_blocks']!=[f'decoder.blocks.{i}' for i in range(7)]: raise InvalidEvidence('Block coverage/order')
    if canonical(observation['norms'])!=canonical(norm_inventory(variant)): raise InvalidEvidence('LN count/placement/epsilon')
    if observation['vim_instances']!=[f'encoder.blocks.{i}.vim' for i in range(12)]+[f'decoder.blocks.{i}.vim' for i in range(7)]: raise InvalidEvidence('19 independent VIM roles')
    parameters=observation['parameters']; expected={x['name']:x['shape'] for x in bundle()['rosters'][variant]}
    if type(parameters) is not list or len(parameters)!=COUNTS[variant][0]: raise InvalidEvidence('Parameter tensor count')
    names=set(); objects=set(); storages=set(); elements=0
    for p in parameters:
        exact(p,('name','shape','dtype','sha256','object_id','storage_id'))
        if p['name'] not in expected or p['shape']!=expected[p['name']] or p['dtype']!='float32': raise InvalidEvidence('Parameter name/shape/dtype')
        for field,seen in (('name',names),('object_id',objects),('storage_id',storages)):
            if type(p[field]) is not str or not p[field] or p[field] in seen: raise InvalidEvidence('Parameter/storage alias')
            seen.add(p[field])
        sha(p['sha256']); elements+=math.prod(p['shape'])
    if elements!=COUNTS[variant][1] or names!=set(expected): raise InvalidEvidence('Unique element count')
    vocab=next(p for p in parameters if p['name']=='vocabulary.weight')
    if observation['vocabulary_lookup_id']!=vocab['object_id'] or observation['vocabulary_output_id']!=vocab['object_id']: raise InvalidEvidence('Vocabulary tie')
    return {'parameter_tensors':len(parameters),'unique_elements':elements,'basis':'validated supplied metadata, not observed model execution'}

def gradient_role(name):
    if '.vim.q.' in name or '.vim.k.' in name: return 'allocated_exact_zero'
    if name.endswith('.k.bias') and ('attention.' in name): return 'analytical_key_bias_residual'
    if name.endswith(('.q.weight','.k.weight')) and ('attention.' in name): return 'ordinary_qk_weight'
    return 'connected_finite'
