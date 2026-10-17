"""Mock-only handoff validation; no dataset or training implementation."""
def validate_batch(batch):
    import torch
    required={'image','mask','label','sample_ids','token_sample_ids','token_ids','kinds'}
    if set(batch)!=required:raise ValueError('Missing/unknown handoff fields')
    image,mask,labels,tokens=(batch[k] for k in ('image','mask','label','token_ids'))
    b=image.shape[0]
    if image.dtype!=torch.float32 or tuple(image.shape)!=(b,3,224,224) or b not in (1,2):raise ValueError('Image contract')
    if not torch.isfinite(image).all() or image.min()<0 or image.max()>1:raise ValueError('Supplied RGB range')
    if labels.dtype!=torch.int64 or tuple(labels.shape)!=(b,) or not ((labels==0)|(labels==1)).all():raise ValueError('Class semantics')
    if mask.dtype!=torch.int64 or tuple(mask.shape)!=(b,224,224) or not ((mask==0)|(mask==1)).all():raise ValueError('Mask semantics')
    if tokens.dtype!=torch.int64 or tuple(tokens.shape)!=(b,308) or tokens.min()<0 or tokens.max()>=49408:raise ValueError('Four-segment token contract')
    if any(t.device.type!='cpu' for t in (image,mask,labels,tokens)):raise ValueError('Mock CPU handoff only')
    if any(not isinstance(batch[k],(tuple,list)) or any(type(v) is not str or not v.strip() for v in batch[k]) for k in ('sample_ids','token_sample_ids')):raise ValueError('Explicit sample ID sequences')
    if batch['sample_ids']!=batch['token_sample_ids'] or len(batch['sample_ids'])!=b or len(set(batch['sample_ids']))!=b:raise ValueError('Sample/token pairing')
    if len(batch['kinds'])!=b:raise ValueError('Kinds')
    for i,kind in enumerate(batch['kinds']):
        if kind not in ('real','efs','local') or labels[i].item()!=(0 if kind=='real' else 1):raise ValueError('Kind/label')
        if kind=='real' and mask[i].any() or kind=='efs' and not mask[i].all():raise ValueError('Supplied mask/kind contradiction')
    return 'MOCK_HANDOFF_VALID; token bytes still require approved tokenizer provenance'

def invoke_mock(model,batch):
    if getattr(model,'scope',None)!='synthetic':raise PermissionError('Mock interface only')
    validate_batch(batch)
    output=model.with_language(batch['image'],batch['token_ids'],sample_ids=batch['sample_ids'])
    return model.compute_objectives(output,batch)
