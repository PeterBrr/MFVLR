"""Initialization probe: per-block effect of single vs double VIM residual under
pre_norm, and how much label information the two GenFace real/Diffae prompts
leave for LR and CMC. No training, no dataset images, no checkpoint.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'.pilot_deps'), str(ROOT/'.phase2c1_deps'), str(ROOT/'src')]
import numpy as np
import torch
from mfvlr.tokenizer import TokenizerAdapter
from mfvlr.prompts import PromptBuilder, PromptMetadata
from mfvlr_research.pilot_language import EosFineGrainedLanguageTransformer, nonpad_language_reconstruction_loss

BOS = 49406


def prompt_ids():
    records = [PromptMetadata('real', 'real'), PromptMetadata('diffae', 'fake', 'AM', 'diffusion', 'Diffae')]
    prompts = PromptBuilder().build(records)
    return prompts, TokenizerAdapter().encode_levels(prompts).ids


def block_probe(ids, visual, mode):
    model = EosFineGrainedLanguageTransformer(seed=0, vim_residual=mode, language_normalization='pre_norm')
    blocks = [(f'LE{j+1}', b) for j, b in enumerate(model.encoder.blocks)]
    blocks += [(f'LD{j+1}', b) for j, b in enumerate(model.decoder.blocks)]
    rms = {}
    hooks = [b.register_forward_hook(lambda m, i, o, k=k: rms.__setitem__(k, float(o.detach().pow(2).mean().sqrt())))
             for k, b in blocks]
    out = model(ids, visual)
    for h in hooks:
        h.remove()
    loss = nonpad_language_reconstruction_loss(out.logits, ids)
    loss.backward()
    base_t, base_l = out.global_language.detach(), out.logits.detach()
    rows = []
    with torch.no_grad():
        for name, block in blocks:
            grad = float(sum(p.grad.norm()**2 for p in block.parameters() if p.grad is not None)**.5)
            factor, original = block.residual_factor, block.forward
            block.forward = lambda *a, f=factor: f*a[0]  # drop this block's updates, keep residual scaling
            ablated = model(ids, visual)
            block.forward = original
            rows.append(dict(block=name, stream_rms=rms[name], lr_grad_norm=grad,
                             global_language_rel_change=float((ablated.global_language-base_t).norm()/base_t.norm()),
                             logits_rel_change=float((ablated.logits-base_l).norm()/base_l.norm())))
    return dict(mode=mode, normalization='pre_norm', lr_nonpad=float(loss), blocks=rows)


def prompt_information(ids):
    target = ids.numpy()
    shifted = np.concatenate([np.full((2, 1), BOS), target[:, :-1]], 1)
    nonpad = target != 0
    required = [t for t in range(target.shape[1]) if nonpad[:, t].any()
                and (shifted[0, :t+1] == shifted[1, :t+1]).all() and target[0, t] != target[1, t]]
    return dict(nonpad_targets=nonpad.sum(1).tolist(), label_required_positions=required)


def schedule_pairs(train_count=5000, epochs=15):
    """Reproduces tools/run_language5000.py schedule; train labels alternate 0,1."""
    rng = np.random.default_rng(0)
    same = total = both = steps = 0
    for _ in range(epochs):
        order = rng.permutation(train_count)
        for i in range(0, train_count, 4):
            flags = [order[i] % 2 == order[i+1] % 2, order[i+2] % 2 == order[i+3] % 2]
            same += sum(flags); total += 2; both += all(flags); steps += 1
    return dict(identical_prompt_microbatch_fraction=same/total, steps_with_both_microbatches_identical=both/steps,
                validation_pairs='val alternates labels [0,1,...]; language_eval pairs are always real/fake')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--visual-rms', type=float, default=9.92, help='per-element RMS of the synthetic fused visual token')
    args = parser.parse_args()
    assert not args.output.exists()
    torch.manual_seed(0)
    prompts, ids = prompt_ids()
    visual = torch.randn(2, 1, 512)*args.visual_rms
    report = dict(scope='initialization only, seed0, CPU FP32, synthetic visual token; not a trained-model measurement',
                  torch=torch.__version__, visual_rms=args.visual_rms, prompts=prompts,
                  prompt_information=prompt_information(ids), schedule=schedule_pairs(),
                  probes=[block_probe(ids, visual, mode) for mode in ('single', 'double')])
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for probe in report['probes']:
        print(probe['mode'], 'LE1/LE12 grad ratio %.0f' % (probe['blocks'][0]['lr_grad_norm']/probe['blocks'][11]['lr_grad_norm']))
    print(json.dumps(report['prompt_information']), json.dumps(report['schedule']))


if __name__ == '__main__':
    main()
