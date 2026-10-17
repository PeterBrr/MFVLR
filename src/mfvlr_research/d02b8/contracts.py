"""Proposed metadata contracts only. No torch, data I/O, optimizer or dispatcher."""
import re
class ContractError(ValueError):pass
def exact(x,keys):
    if type(x) is not dict or set(x)!=set(keys):raise ContractError('Unknown or missing fields')
def sha(x):
    if type(x) is not str or re.fullmatch('[0-9a-f]{64}',x) is None:raise ContractError('SHA256 required')
def config(x):
    exact(x,('schema','mode','variant','execution_enabled','batch','precision','device','loss_order','loss_weights','optimizer','dataset_manifest_sha256'))
    if x['schema']!='D02B8-proposed-v1' or x['mode'] not in ('training','inference') or x['variant']!='A0':raise ContractError('Unapproved schema/mode/variant')
    if x['execution_enabled'] is not False or x['optimizer'] is not None:raise ContractError('Execution and optimizers unauthorized')
    if type(x['batch']) is not int or x['batch'] not in (1,2) or x['precision']!='float32' or x['device']!='cpu':raise ContractError('Unsupported baseline')
    if x['loss_order']!=['fd','lr','cmc','fl','ar','kl'] or x['loss_weights']!=[1,1,1,1,1,1]:raise ContractError('Frozen loss semantics')
    sha(x['dataset_manifest_sha256'])
    return {'validation':'metadata only','execution':'DISABLED','approval':'PROPOSED'}
def sample(x):
    exact(x,('sample_id','split','subject_group','source_sha256','image_shape','image_dtype','label','kind','mask_shape','mask_dtype','mask_sha256','mask_semantics','prompt_metadata_sha256','preprocessing_contract_sha256'))
    if any(type(x[k]) is not str or not x[k].strip() for k in ('sample_id','subject_group')):raise ContractError('Stable identifiers required')
    if x['split'] not in ('train','validation','test') or x['kind'] not in ('real','efs','local'):raise ContractError('Explicit partition/kind required')
    if type(x['label']) is not int or x['label']!=(0 if x['kind']=='real' else 1):raise ContractError('Real/Fake semantics')
    if x['image_shape']!=[3,224,224] or x['image_dtype']!='float32' or x['mask_shape']!=[224,224] or x['mask_dtype']!='int64':raise ContractError('Shape/type')
    expected={'real':'all_background','efs':'all_foreground','local':'supplied_binary_localization'}
    if x['mask_semantics']!=expected[x['kind']]:raise ContractError('No inferred mask preprocessing')
    for k in ('source_sha256','mask_sha256','prompt_metadata_sha256','preprocessing_contract_sha256'):sha(x[k])
    return 'METADATA_VALID; tensor values and preprocessing NOT VERIFIED'
def dataset(rows):
    seen=set();groups={}
    if type(rows) is not list or not rows:raise ContractError('Empty dataset contract')
    for r in rows:
        sample(r)
        if r['sample_id'] in seen:raise ContractError('Duplicate sample')
        seen.add(r['sample_id']);groups.setdefault(r['subject_group'],set()).add(r['split'])
    if any(len(s)>1 for s in groups.values()):raise ContractError('Cross-split subject leakage')
    return 'METADATA_VALID; no loader or preprocessing implemented'
def checkpoint(x):
    exact(x,('schema','variant','parameter_tensors','elements','state_sha256','source_manifest_sha256','config_sha256','tokenizer_manifest_sha256','dataset_manifest_sha256','loss_contract_sha256','backend_sha256','initialization_seed','update_count','optimizer_state','numerical_status'))
    if x['schema']!='D02B8-proposed-v1' or x['variant']!='A0' or x['parameter_tensors']!=508 or x['elements']!=154156232:raise ContractError('Full A0 ownership')
    if type(x['initialization_seed']) is not int or x['initialization_seed']!=0 or type(x['update_count']) is not int or x['update_count']!=0 or x['optimizer_state'] is not None:raise ContractError('No updates authorized')
    if x['numerical_status'] not in ('FAIL','NOT RUN'):raise ContractError('Cannot claim acceptance')
    for k,v in x.items():
        if k.endswith('sha256'):sha(v)
    return 'METADATA_VALID; no checkpoint loading/saving or model construction'
