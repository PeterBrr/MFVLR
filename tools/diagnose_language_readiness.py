"""Compare single/double VIM residual using two fixed train images; no optimizer/test."""
import gc
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.pilot_deps'),str(ROOT/'src'),str(ROOT/'tools')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from mfvlr_research.pilot_variant import visual_model
from mfvlr_research.pilot_language import EosFineGrainedLanguageTransformer, nonpad_language_reconstruction_loss, NormalizedCrossModalContrastiveLoss
from mfvlr.prompts import PromptBuilder, PromptMetadata
from mfvlr.tokenizer import TokenizerAdapter
from mfvlr.heads import DetectionHeads
from mfvlr.objectives import knowledge_distillation_loss
from run_loss_overfit import load_samples, write_json

def rms(x):return float(x.detach().double().square().mean().sqrt())
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output
    assert not out.exists(),'Choose a fresh output'
    out.mkdir(parents=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    path=ROOT/'artifacts/genface5000_epoch30_kaggle_20261027/downloaded/fd_validation_results/seed_0_fd_0p1/final.pt'
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    samples=load_samples(ROOT/'artifacts/kaggle_loss_overfit_20261026/package_001/train16.npz')[:2]
    visual=visual_model(ckpt['config']);visual.load_state_dict(ckpt['model'],strict=True);visual.to(device).eval()
    with torch.no_grad(): features=visual(torch.stack([s['image'] for s in samples]).to(device)).visual.fused_global.detach()
    del visual,ckpt;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    prompts=PromptBuilder().build([PromptMetadata('real','real'),PromptMetadata('fake','fake','AM','diffusion','Diffae')])
    ids=TokenizerAdapter().encode_levels(prompts).ids.to(device)
    report=dict(status='RUNNING',scope='Two fixed training images; frozen epoch30 visual weights, fresh language seed0; no optimizer, no test',
        sample_ids=[s['sample_id'] for s in samples],visual_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        visual_rms=rms(features),effective_cmc_batch=2,language_weights=dict(lr=1e-4,cmc=.01,kl=.01),modes={})
    write=lambda:write_json(out/'report.json',report)
    write()
    for mode in ('single','double','double_pre_norm'):
        print('START '+mode,flush=True)
        flt=EosFineGrainedLanguageTransformer(seed=0,vim_residual='single' if mode=='single' else 'double',
            language_normalization='pre_norm' if mode=='double_pre_norm' else 'baseline').to(device).eval()
        flt.requires_grad_(False)
        adapter=DetectionHeads(seed=0).to(device).eval();adapter.requires_grad_(False)
        cmc=NormalizedCrossModalContrastiveLoss().to(device)
        stats={};handles=[]
        def hook(name):
            def collect(module,args,value):stats[name]=dict(rms=rms(value),maximum=float(value.detach().abs().max()))
            return collect
        for part in ('encoder','decoder'):
            for i,b in enumerate(getattr(flt,part).blocks):handles.append(b.register_forward_hook(hook(f'{part}.{i+1}')))
        feature=features.clone().requires_grad_(True)
        output=flt(ids,feature)
        for h in handles:h.remove()
        losses=dict(lr=nonpad_language_reconstruction_loss(output.logits,ids),
            cmc=cmc(feature,output.global_language),
            kl=knowledge_distillation_loss(output.global_language,adapter.align_language(feature)))
        gradients={}
        for name,loss in losses.items():
            grad=torch.autograd.grad(loss,feature,retain_graph=True)[0]
            assert torch.isfinite(grad).all()
            gradients[name]=dict(norm=float(grad.double().norm()),weighted_norm=float(grad.double().norm())*report['language_weights'][name])
        row=dict(activation=stats,logit_rms=rms(output.logits),losses={k:float(v.detach()) for k,v in losses.items()},
                 gradients_at_visual_token=gradients,perturbations={})
        baseline=output.logits.detach().clone()
        del output,losses,feature,grad;gc.collect()
        with torch.no_grad():
            for name,f in [('zero',torch.zeros_like(features)),('swapped',features.flip(0))]:
                alt=flt(ids,f)
                row['perturbations'][name]=dict(lr=float(nonpad_language_reconstruction_loss(alt.logits,ids)),
                    logit_rmse_vs_matched=rms(alt.logits-baseline))
                del alt
        report['modes'][mode]=row;write();print(json.dumps(dict(mode=mode,**{k:v for k,v in row.items() if k!='activation'})),flush=True)
        del flt,adapter,cmc,baseline;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    report['status']='COMPLETED';write();print(out/'report.json',flush=True)

if __name__=='__main__':main()
