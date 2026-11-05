"""Encode the text-patch prompts once with frozen OpenAI CLIP ViT-L/14; training never loads CLIP.

Writes src/mfvlr_research/assets/text_patch_clip_vitl14.json: the prompts, the checkpoint SHA256
and one unit-norm 768-d embedding per prompt. Row order is the class order of TextPatchHead.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from mfvlr_research.text_patch import CLASS_PROMPTS  # noqa: E402

# The SHA256 that OpenAI publishes in the ViT-L/14 download URL.
OPENAI_VITL14_SHA256 = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path.home() / '.cache/clip/ViT-L-14.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'src/mfvlr_research/assets/text_patch_clip_vitl14.json')
    args = parser.parse_args()
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != OPENAI_VITL14_SHA256:
        raise SystemExit(f'{args.checkpoint} has SHA256 {digest}, not the OpenAI ViT-L/14 release')
    import open_clip
    from open_clip.openai import load_openai_model
    model = load_openai_model(str(args.checkpoint), device='cpu').eval()
    with torch.no_grad():
        text = model.encode_text(open_clip.tokenize(list(CLASS_PROMPTS))).float()
    text = torch.nn.functional.normalize(text, dim=-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        model='OpenAI CLIP ViT-L/14 text encoder (frozen)', checkpoint_sha256=digest,
        prompts=list(CLASS_PROMPTS), embeddings=[[round(v, 7) for v in row] for row in text.tolist()],
        cosine=[[round(v, 4) for v in row] for row in (text @ text.T).tolist()]), indent=1) + '\n', encoding='utf-8')
    print(args.output, 'cosine between prompts:', (text @ text.T).round(decimals=3).tolist())


if __name__ == '__main__':
    main()
