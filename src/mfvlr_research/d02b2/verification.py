"""N2 record/array checks. All model observations must come from a future approved run."""
import math
import numpy as np
from .contract import (InvalidEvidence, exact, FP32, FP64, bundle, stages, gradient_role,
                       canonical, validate_ownership, sha, validate_request, seal)

def compare_arrays(actual,reference,*,category):
    policies={'fp32_cross_batch':(np.dtype('float32'),FP32,'diagnostic_only'),
              'fp64_cross_batch':(np.dtype('float64'),FP64,'mandatory'),
              'fixed_batch':(None,(0.,0.),'mandatory_exact'),
              'permutation':(None,(0.,0.),'mandatory_exact')}
    if category not in policies: raise InvalidEvidence('No policy for this category; no N1/N2 transfer')
    dtype,(atol,rtol),role=policies[category]
    if not isinstance(actual,np.ndarray) or not isinstance(reference,np.ndarray) or actual.shape!=reference.shape or actual.dtype!=reference.dtype or not actual.size:
        raise InvalidEvidence('Comparison shape/dtype/empty')
    if actual.dtype not in (np.dtype('float32'),np.dtype('float64')) or (dtype is not None and actual.dtype!=dtype): raise InvalidEvidence('Wrong precision')
    a=actual.reshape(-1); b=reference.reshape(-1)
    outside=zero_nonzero=0; maximum=relative=sum_d=sum_b=0.
    for start in range(0,a.size,262144):
        aa=a[start:start+262144]; bb=b[start:start+262144]
        if not np.isfinite(aa).all() or not np.isfinite(bb).all(): raise InvalidEvidence('Nonfinite comparison')
        d=aa.astype(np.float64)-bb.astype(np.float64); r=bb.astype(np.float64); nz=r!=0
        if role=='mandatory_exact': bad=aa!=bb
        else:
            with np.errstate(over='ignore'):
                bad=np.abs(aa-bb)>aa.dtype.type(atol)+aa.dtype.type(rtol)*np.abs(bb)
        outside+=int(bad.sum()); zero_nonzero+=int(((~nz)&(d!=0)).sum())
        maximum=max(maximum,float(np.abs(d).max()))
        if nz.any(): relative=max(relative,float((np.abs(d[nz])/np.abs(r[nz])).max()))
        sum_d+=float(d@d); sum_b+=float(r@r)
    return {'category':category,'role':role,'shape':list(actual.shape),'dtype':str(actual.dtype),'elements':a.size,
            'finite':True,'atol':atol,'rtol':rtol,'outside_tolerance_count':outside,'outside_proportion':outside/a.size,
            'max_absolute_difference':maximum,'max_relative_difference_nonzero_reference':relative,
            'zero_reference_nonzero_difference_count':zero_nonzero,'max_relative_difference_is_infinite':zero_nonzero>0,
            'rms_absolute_difference':math.sqrt(sum_d/a.size),'relative_l2_difference':math.sqrt(sum_d/sum_b) if sum_b else None,
            'exact_equal':bool(np.array_equal(a,b)),'numerical':'FAIL' if outside else 'PASS',
            'acceptance':'NOT GRANTED','execution':'ARRAY CHECK ONLY'}

def validate_stages(records,fixture,batch,precision):
    expected=stages(fixture,batch)
    exact(records,expected)
    if precision not in ('float32','float64'): raise InvalidEvidence('Precision')
    for name,record in records.items():
        exact(record,('shape','dtype','finite','l2_norm','rms','max_abs','graph_owned'))
        if record['shape']!=expected[name] or record['dtype']!=precision or record['finite'] is not True or record['graph_owned'] is not False:
            raise InvalidEvidence('Shape, finite value, or retained graph: '+name)
        if any(type(record[k]) not in (int,float) or not math.isfinite(record[k]) or record[k]<0 for k in ('l2_norm','rms','max_abs')):
            raise InvalidEvidence('Invalid stage statistic')

def validate_gradients(records,variant):
    roster={p['name']:p['shape'] for p in bundle()['rosters'][variant]}
    exact(records,roster)
    zeros=0; ordinary={}; key_bias={}
    for name,shape in roster.items():
        r=records[name]
        exact(r,('allocated','shape','finite','nonzero','l2_norm','max_abs'))
        if r['allocated'] is not True or r['shape']!=shape or r['finite'] is not True: raise InvalidEvidence('Disconnected/nonfinite gradient: '+name)
        if type(r['nonzero']) is not int or not 0<=r['nonzero']<=math.prod(shape): raise InvalidEvidence('Nonzero count')
        if any(type(r[k]) not in (int,float) or not math.isfinite(r[k]) or r[k]<0 for k in ('l2_norm','max_abs')): raise InvalidEvidence('Gradient statistic')
        if (r['nonzero']==0)!=(r['l2_norm']==0) or (r['nonzero']==0)!=(r['max_abs']==0): raise InvalidEvidence('Inconsistent zero/nonzero evidence')
        role=gradient_role(name)
        if role=='allocated_exact_zero':
            if r['nonzero']!=0: raise InvalidEvidence('VIM Q/K structural failure')
            zeros+=1
        elif role=='ordinary_qk_weight': ordinary[name]=r
        elif role=='analytical_key_bias_residual': key_bias[name]=r
    if zeros!=38: raise InvalidEvidence('38 VIM Q/K checks required')
    # The original T16 probe requires these positive controls, not every weak Q/K.
    required=('vocabulary.weight','encoder_position','decoder_position','encoder.blocks.0.attention.v.weight',
              'decoder.blocks.0.memory_attention.v.weight','encoder.blocks.0.ff.fc1.weight','decoder.blocks.6.ff.fc2.weight')
    if any(records[n]['l2_norm']<=0 for n in required): raise InvalidEvidence('Connected positive control failed')
    return {'vim_allocated_exact_zero':zeros,'ordinary_qk':ordinary,'ordinary_key_bias_residuals':key_bias,
            'basis':'record validation only, not a new gradient execution'}

def validate_cross_sample(records):
    if type(records) is not list or len(records)!=2: raise InvalidEvidence('Both directions required')
    for sample,r in enumerate(records):
        exact(r,('probe_sample','other_sample','lookup_boundary','visual_connected','lookup_connected',
                 'other_visual_nonzero','other_lookup_nonzero','own_visual_l2','own_lookup_l2','finite'))
        if type(r['probe_sample']) is not int or type(r['other_sample']) is not int or r['probe_sample']!=sample or r['other_sample']!=1-sample:
            raise InvalidEvidence('Sample direction/order')
        if r['lookup_boundary']!='actual_output.low_EmbeddingBackward' or r['visual_connected'] is not True or r['lookup_connected'] is not True:
            raise InvalidEvidence('Detached/surrogate lookup or visual leaf')
        if r['finite'] is not True or type(r['other_visual_nonzero']) is not int or type(r['other_lookup_nonzero']) is not int or r['other_visual_nonzero']!=0 or r['other_lookup_nonzero']!=0:
            raise InvalidEvidence('Cross-sample leakage/nonfinite')
        if any(type(r[k]) not in (int,float) or not math.isfinite(r[k]) or r[k]<=0 for k in ('own_visual_l2','own_lookup_l2')):
            raise InvalidEvidence('Missing own-sample positive control')

def validate_state_preservation(before,after,variant):
    validate_ownership(before,variant); validate_ownership(after,variant)
    if canonical(before)!=canonical(after): raise InvalidEvidence('State/parameter identity changed')

def validate_common_copy(reference,candidate,variant):
    validate_ownership(reference,'A0'); validate_ownership(candidate,variant)
    source={p['name']:p for p in reference['parameters']}; target={p['name']:p for p in candidate['parameters']}
    for key in ('object_id','storage_id'):
        if {p[key] for p in source.values()} & {p[key] for p in target.values()}: raise InvalidEvidence('Candidate aliases reference')
    for name,p in source.items():
        if any(p[k]!=target[name][k] for k in ('shape','dtype','sha256')): raise InvalidEvidence('Common initialization differs')
    for name in set(target)-set(source):
        value=1. if name.endswith('weight') else 0.
        import hashlib
        expected=hashlib.sha256(np.full(512,value,dtype='<f4').tobytes()).hexdigest()
        if target[name]['sha256']!=expected: raise InvalidEvidence('LN initialization differs')

def validate_promotion(fp32_observation,promoted,variant):
    validate_ownership(fp32_observation,variant)
    exact(promoted,('parameters','vocabulary_lookup_id','vocabulary_output_id'))
    source={p['name']:p for p in fp32_observation['parameters']}
    rows=promoted['parameters']
    if type(rows) is not list or len(rows)!=len(source): raise InvalidEvidence('Promotion coverage')
    names=set(); objects=set(); storages=set()
    for r in rows:
        exact(r,('name','shape','dtype','sha256','expected_fp64_sha256','roundtrip_fp32_sha256','object_id','storage_id'))
        if r['name'] not in source or r['name'] in names: raise InvalidEvidence('Promotion names')
        p=source[r['name']]; names.add(r['name'])
        if r['shape']!=p['shape'] or r['dtype']!='float64' or r['roundtrip_fp32_sha256']!=p['sha256']:
            raise InvalidEvidence('Must promote the same FP32 initialization')
        sha(r['sha256'])
        if r['sha256']!=sha(r['expected_fp64_sha256']): raise InvalidEvidence('Not an exact FP32-to-FP64 cast')
        for key,seen in (('object_id',objects),('storage_id',storages)):
            if type(r[key]) is not str or not r[key] or r[key] in seen or r[key] in {q[key] for q in source.values()}: raise InvalidEvidence('Promotion must own disjoint copy')
            seen.add(r[key])
    vocab=next(r for r in rows if r['name']=='vocabulary.weight')
    if promoted['vocabulary_lookup_id']!=vocab['object_id'] or promoted['vocabulary_output_id']!=vocab['object_id']: raise InvalidEvidence('Promoted tie')

def exact_promotion_arrays(original,promoted):
    """Independent bounded byte/value check, exercised only on tiny arrays here."""
    import hashlib
    if not isinstance(original,np.ndarray) or not isinstance(promoted,np.ndarray) or original.shape!=promoted.shape or original.dtype!=np.dtype('float32') or promoted.dtype!=np.dtype('float64'):
        raise InvalidEvidence('Promotion shape/dtype')
    if not original.size or not original.flags.c_contiguous or not promoted.flags.c_contiguous or np.shares_memory(original,promoted):
        raise InvalidEvidence('Independent contiguous promotion required')
    h32=hashlib.sha256(); h64=hashlib.sha256()
    a=original.reshape(-1); b=promoted.reshape(-1)
    for start in range(0,a.size,262144):
        x=a[start:start+262144]; y=b[start:start+262144]
        if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.array_equal(x.astype(np.float64),y):
            raise InvalidEvidence('Promotion differs from exact original FP32 values')
        h32.update(x.astype('<f4',copy=False).tobytes()); h64.update(x.astype('<f8').tobytes())
    return {'original_fp32_sha256':h32.hexdigest(),'expected_fp64_sha256':h64.hexdigest(),
            'basis':'array values verified; not model ownership/initialization by itself'}

def validate_complete_evidence(record,request,pinned_request_sha256):
    """Fail-closed completeness/status validator, tested on explicitly mocked records.

    It does not authorize execution or certify that supplied observations occurred.
    Raw evidence must be sealed/preserved before invoking this validator.
    """
    validate_request(request,pinned_request_sha256)
    exact(record,('request_sha256','fp32_state_before_sha256','fp32_state_after_sha256','fp32_before','fp32_after','promoted_before','promoted_after',
                  'activations','cross_batch','fixed_batch','permutation','full_backward','cross_derivatives'))
    if record['request_sha256']!=seal(request): raise InvalidEvidence('Result/request mismatch')
    if record['fp32_state_before_sha256']!=request['candidate_fp32_state_sha256'] or record['fp32_state_after_sha256']!=request['candidate_fp32_state_sha256']:
        raise InvalidEvidence('Candidate state hash mismatch/mutation')
    variant,fixture,precision=(request[k] for k in ('variant','fixture','precision'))
    validate_state_preservation(record['fp32_before'],record['fp32_after'],variant)
    validate_stages(record['activations'],fixture,2,precision)
    expected=stages(fixture,1)
    if type(record['cross_batch']) is not list or len(record['cross_batch'])!=2: raise InvalidEvidence('Both B1 pairings required')
    mismatches=0
    metric_fields=('category','role','shape','dtype','elements','finite','atol','rtol','outside_tolerance_count',
                   'outside_proportion','max_absolute_difference','max_relative_difference_nonzero_reference',
                   'zero_reference_nonzero_difference_count','max_relative_difference_is_infinite',
                   'rms_absolute_difference','relative_l2_difference','exact_equal','numerical','acceptance','execution')
    for sample,item in enumerate(record['cross_batch']):
        exact(item,('sample','stages'))
        if type(item['sample']) is not int or item['sample']!=sample: raise InvalidEvidence('Sample order')
        exact(item['stages'],expected)
        for name,m in item['stages'].items():
            exact(m,metric_fields)
            if m['shape']!=expected[name] or m['elements']!=math.prod(expected[name]) or m['dtype']!=precision or m['finite'] is not True:
                raise InvalidEvidence('Incomplete comparison dimensions/finiteness')
            atol,rtol=FP32 if precision=='float32' else FP64
            if (m['atol'],m['rtol'])!=(atol,rtol) or m['category']!=('fp32_cross_batch' if precision=='float32' else 'fp64_cross_batch') or m['role']!=('diagnostic_only' if precision=='float32' else 'mandatory'):
                raise InvalidEvidence('Numerical policy changed')
            n=m['outside_tolerance_count']; z=m['zero_reference_nonzero_difference_count']
            if type(n) is not int or not 0<=n<=m['elements'] or type(z) is not int or not 0<=z<=m['elements']:
                raise InvalidEvidence('Mismatch count')
            if m['outside_proportion']!=n/m['elements'] or m['numerical']!=('FAIL' if n else 'PASS'):
                raise InvalidEvidence('Failure count/status concealed')
            if m['acceptance']!='NOT GRANTED' or m['max_relative_difference_is_infinite'] is not (z>0) or type(m['exact_equal']) is not bool:
                raise InvalidEvidence('Comparison status/zero-reference semantics')
            for key in ('max_absolute_difference','max_relative_difference_nonzero_reference','rms_absolute_difference','relative_l2_difference'):
                value=m[key]
                if key=='relative_l2_difference' and value is None: continue
                if type(value) not in (int,float) or not math.isfinite(value) or value<0: raise InvalidEvidence('Invalid comparison metric')
            if m['exact_equal'] and (n or m['max_absolute_difference'] or z): raise InvalidEvidence('False exact-equality claim')
            mismatches+=n
    if type(record['fixed_batch']) is not list or len(record['fixed_batch'])!=2: raise InvalidEvidence('Both changed-sample directions required')
    for sample,item in enumerate(record['fixed_batch']):
        exact(item,('changed_sample','unaffected_sample','exact_equality'))
        if type(item['changed_sample']) is not int or type(item['unaffected_sample']) is not int or (item['changed_sample'],item['unaffected_sample'])!=(sample,1-sample): raise InvalidEvidence('Fixed-batch order')
        exact(item['exact_equality'],expected)
        if any(x is not True for x in item['exact_equality'].values()): raise InvalidEvidence('Fixed-batch leakage')
    exact(record['permutation'],expected)
    if any(x is not True for x in record['permutation'].values()): raise InvalidEvidence('Permutation failure')
    if precision=='float32':
        if record['promoted_before'] is not None or record['promoted_after'] is not None: raise InvalidEvidence('Unexpected promotion')
        exact(record['full_backward'],('probe','gradients'))
        if record['full_backward']['probe']!='mean(logits^2)+mean(LD_hidden^2)+mean(LE_global^2)': raise InvalidEvidence('Probe changed')
        validate_gradients(record['full_backward']['gradients'],variant)
        validate_cross_sample(record['cross_derivatives'])
    else:
        validate_promotion(record['fp32_before'],record['promoted_before'],variant)
        validate_promotion(record['fp32_before'],record['promoted_after'],variant)
        if canonical(record['promoted_before'])!=canonical(record['promoted_after']): raise InvalidEvidence('Promoted state mutation')
        if record['full_backward'] is not None or record['cross_derivatives'] is not None: raise InvalidEvidence('Unregistered FP64 backward sweep')
    return {'record_validation':'COMPLETE','numerical':'FAIL' if mismatches else 'PASS',
            'outside_tolerance_count':mismatches,'policy':'diagnostic_only' if precision=='float32' else 'mandatory',
            'mandatory_numerical_failure':bool(mismatches and precision=='float64'),
            'execution':'NOT VERIFIED BY RECORD VALIDATION','acceptance':'NOT GRANTED','phase2d2':'BLOCKED'}

def n2_state_hash(named_arrays):
    """Original N2 hash convention, streamed over provided contiguous FP32 arrays."""
    import hashlib
    digest=hashlib.sha256(); seen=set()
    for name,array in named_arrays:
        if type(name) is not str or not name or name in seen or not isinstance(array,np.ndarray) or array.dtype!=np.dtype('<f4') or not array.flags.c_contiguous:
            raise InvalidEvidence('State name/dtype/order/contiguity')
        seen.add(name)
        digest.update(name.encode()); digest.update(str(tuple(array.shape)).encode())
        flat=array.reshape(-1)
        for start in range(0,flat.size,262144):
            chunk=flat[start:start+262144]
            if not np.isfinite(chunk).all(): raise InvalidEvidence('Nonfinite state')
            digest.update(chunk.tobytes())
    if not seen: raise InvalidEvidence('Empty state')
    return digest.hexdigest()
