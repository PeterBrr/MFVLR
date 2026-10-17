"""Independent live FP32 arithmetic reconciliation on detached observed tensors.

No production loss helper is called by this observer. Category regrouping
differences are recorded, never used to change LR or its reduction.
"""
def audit(output,targets,root,observed):
    import torch
    with torch.no_grad():
        p=output.predictions;language=output.language
        fdlog=p.detection_logits.detach().log_softmax(1)
        fd=-fdlog[torch.arange(len(targets.labels)),targets.labels].mean()
        ids=targets.token_ids
        logp=language.logits.detach().log_softmax(-1)
        nll=-logp.gather(-1,ids.unsqueeze(-1)).squeeze(-1)
        per_image=nll.sum(1);lr=per_image.mean()
        categories={};masks={'bos':ids==49406,'eos':ids==49407,'pad':ids==0}
        masks['content']=~(masks['bos']|masks['eos']|masks['pad'])
        for name,mask in masks.items():
            values=nll[mask]
            categories[name]=dict(count=int(mask.sum()),fp64_nll_sum=float(values.double().sum()),fp32_nll_sum=float(values.sum()))
        raw=p.visual.fused_global.detach()[:,0]@language.global_language.detach()[:,0].T
        tau=root.objectives.cmc.log_temperature.detach().exp();scores=raw/tau
        row=scores.log_softmax(1);col=scores.T.log_softmax(1)
        cmc=(-row.diagonal().mean()-col.diagonal().mean())/2
        flp=p.decoded.mask_logits.detach().log_softmax(1)
        fl=-flp.gather(1,targets.masks[:,None]).squeeze(1).mean()
        difference=output.image.detach()-p.decoded.reconstruction.detach();ar=difference.square().mean()
        text=(language.global_language.detach()/.5).log_softmax(-1)
        vision=(output.predicted_language.detach()/.5).log_softmax(-1)
        kl=(text.exp()*(text-vision)).sum(-1)[:,0].mean()
        refs=dict(fd=fd,lr=lr,cmc=cmc,fl=fl,ar=ar,kl=kl)
        for name,x in refs.items():
            if not torch.isfinite(x):raise FloatingPointError('Nonfinite independent arithmetic '+name)
        comparisons={n:dict(observed=float(observed[n].detach()),reference=float(x),exact=bool(torch.equal(observed[n].detach(),x))) for n,x in refs.items()}
        total=observed['fd'].detach();steps=[float(total)]
        for n in ('lr','cmc','fl','ar','kl'):
            total=total+observed[n].detach();steps.append(float(total))
        comparisons['total']=dict(observed=float(observed['total'].detach()),reference=float(total),exact=bool(torch.equal(total,observed['total'].detach())))
        scalar_sum=sum(v['fp64_nll_sum'] for v in categories.values())/ids.shape[0]
        return dict(comparisons=comparisons,all_exact=all(r['exact'] for r in comparisons.values()),
            literal_fp32_total_steps=steps,lr=dict(categories=categories,per_image_fp32=per_image.tolist(),
                batch_mean_fp32=float(lr),category_sum_fp64=scalar_sum,category_regrouping_difference=scalar_sum-float(lr),
                token_nll_fp32=nll.tolist(),token_ids=ids.tolist(),all_positions=int(ids.numel())),
            fd=dict(labels=targets.labels.tolist(),class_order=['Real','Fake']),
            cmc=dict(raw_dot=raw.tolist(),scores=scores.tolist(),temperature=float(tau),positive_indices=list(range(len(ids))),
                row_entropy=(-(row.exp()*row).sum(1)).tolist(),column_entropy=(-(col.exp()*col).sum(1)).tolist()),
            fl=dict(mask_classes=targets.masks.unique().tolist(),foreground_pixels=int(targets.masks.sum()),reduction='mean batch and spatial'),
            ar=dict(image_min=float(output.image.min()),image_max=float(output.image.max()),reconstruction_min=float(p.decoded.reconstruction.min()),reconstruction_max=float(p.decoded.reconstruction.max())),
            kl=dict(temperature=.5,direction='text||adapted_vision',text_entropy=(-(text.exp()*text).sum(-1)).tolist(),vision_entropy=(-(vision.exp()*vision).sum(-1)).tolist(),
                live_text_connected=language.global_language.grad_fn is not None,live_adapter_connected=output.predicted_language.grad_fn is not None),
            basis='Independent operations on actual detached outputs; no extra model forward/backward; no new tolerance')
