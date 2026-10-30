"""Evaluate a pilot checkpoint using only the image route; threshold comes from validation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / '.pilot_deps'), str(ROOT / 'src')]

import torch
from mfvlr_research.pilot_data import sha256
from mfvlr_research.pilot_datasets import dataset_for_config
from mfvlr_research.pilot_runtime import load_checkpoint, evaluate_visual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error('Output must differ from the checkpoint')
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint['config']
    torch.set_num_threads(config['threads'])
    manifest = ROOT / config['manifests'][args.split]
    if sha256(manifest) != checkpoint['provenance']['manifests'][args.split]:
        raise ValueError('Evaluation manifest changed since training')
    dataset = dataset_for_config(ROOT, config, args.split)
    from mfvlr_research.pilot_variant import visual_model
    for folder in ('mfvlr', 'mfvlr_research'):
        for relative, expected in checkpoint['provenance']['source'].items():
            if relative.replace('\\', '/').startswith('src/' + folder + '/') and sha256(ROOT / relative) != expected:
                raise ValueError('Model/evaluation source differs from checkpoint provenance')
    model = visual_model(config)
    visual = {}
    for name, value in checkpoint['model'].items():
        if name.startswith(('mve.', 'decoder.')):
            visual[name] = value
        elif name.startswith('heads.detector.'):
            visual[name.removeprefix('heads.')] = value
    model.load_state_dict(visual, strict=True)
    device = torch.device(args.device)
    model.to(device)
    threshold = checkpoint['threshold']
    if args.split == 'test' and checkpoint['best'] is None:
        raise ValueError('Checkpoint has no completed training-epoch validation; complete an epoch before testing')
    trained_steps = checkpoint['step']
    del checkpoint, visual
    result = evaluate_visual(model, dataset, device, config['batch_size'],
                             None if args.split == 'val' else threshold)
    result.update(split=args.split, step=trained_steps, checkpoint_sha256=sha256(args.checkpoint),
                  protocol=config['protocol'], subject_disjoint_verified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
