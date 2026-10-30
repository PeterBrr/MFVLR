"""Real scratch FLT forward/backward check for EOS, non-PAD LR and normalized CMC."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.pilot_deps'), str(ROOT / 'src')]

import torch
from mfvlr.prompts import PromptBuilder, PromptMetadata
from mfvlr.tokenizer import TokenizerAdapter, EOS, PAD
from mfvlr_research.pilot_language import (EosFineGrainedLanguageTransformer,
    NormalizedCrossModalContrastiveLoss, nonpad_language_reconstruction_loss)
from mfvlr_research.pilot_runtime import check_finite_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new report')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    prompts = PromptBuilder().build([PromptMetadata('real', 'real'),
        PromptMetadata('diffae', 'fake', 'AM', 'diffusion', 'Diffae')])
    ids = TokenizerAdapter().encode_levels(prompts).ids
    model = EosFineGrainedLanguageTransformer(seed=0).eval()
    cmc = NormalizedCrossModalContrastiveLoss()
    visual = torch.randn(2, 1, 512, requires_grad=True)
    output = model(ids, visual)
    last = torch.arange(308).expand_as(ids).masked_fill(ids != EOS, -1).amax(1)
    expected = output.encoder_hidden[torch.arange(2), last].unsqueeze(1)
    assert torch.equal(expected, output.global_language)
    lr = nonpad_language_reconstruction_loss(output.logits, ids)
    contrastive = cmc(visual, output.global_language)
    token_gradient = torch.autograd.grad(lr, output.logits, retain_graph=True)[0]
    assert token_gradient[ids == PAD].count_nonzero() == 0
    assert token_gradient[ids == EOS].count_nonzero() > 0
    del token_gradient
    before = visual.detach().clone()
    torch.testing.assert_close(cmc.scores(visual * 7, output.global_language * 11),
                               cmc.scores(visual, output.global_language))
    assert torch.equal(visual.detach(), before)
    gradients = torch.autograd.grad(lr + contrastive, [model.encoder.blocks[0].attention.q.weight,
        model.decoder.blocks[-1].ff.fc1.weight, visual, cmc.log_temperature])
    check_finite_tensors(gradients, 'Nonfinite language backward')
    assert all(g.count_nonzero() > 0 for g in gradients)
    report = {'status': 'PASS', 'device': 'cpu', 'scope': 'Real FLT with offline real/Diffae prompts; random fixed visual conditioning; no images or optimizer updates',
              'token_shape': list(ids.shape), 'last_eos_positions': last.tolist(),
              'global_shape': list(output.global_language.shape), 'target_pad_count': int((ids == PAD).sum()),
              'nonpad_lr': float(lr.detach()), 'normalized_cmc': float(contrastive.detach()),
              'last_eos_verified': True, 'pad_target_gradient_zero': True,
              'eos_target_gradient_nonzero': True, 'cmc_scale_invariance_verified': True,
              'inputs_not_normalized_in_place': True, 'backward_finite_and_connected': True,
              'optimizer_updates': 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
