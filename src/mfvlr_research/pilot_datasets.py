"""Explicit dataset adapters; the historical Diffae reader remains unchanged."""
import json
from pathlib import Path
import re

import torch

from .pilot_data import PilotFaces, sha256


FFPP = 'ffpp_deepfakes_v1'
PROMPT_POLICY = 'ffpp_l1_l2_pad308_v1'


def ffpp_prompt_tokens():
    """Two 77-slot prompts, then 154 PAD slots to retain the existing 308-token model.

    Only L1/L2 text is supplied (MFVLR p.10). Fixed length and PAD treatment are
    declared engineering choices; no generator/family prompts are invented.
    """
    from mfvlr.tokenizer import TokenizerAdapter, BOS, EOS
    tokenizer = TokenizerAdapter()
    prompts = [('A photo of a real face', 'A photo of a real face'),
               ('A photo of a fake face', 'A photo of an identity swapped face')]
    ids = torch.zeros((2, 308), dtype=torch.int64)
    for label, levels in enumerate(prompts):
        for level, text in enumerate(levels):
            payload = tokenizer.encode_text(text)
            if len(payload) > 75:
                raise ValueError('FF++ prompt exceeds its declared 77-slot segment')
            framed = torch.tensor([BOS, *payload, EOS], dtype=torch.int64)
            ids[label, level * 77:level * 77 + len(framed)] = framed
    return ids


def read_ffpp_manifest(path):
    rows = [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
    if not rows:
        raise ValueError('Empty FF++ manifest')
    ids, content = set(), set()
    for row in rows:
        if row.get('dataset') != FFPP or row.get('split') not in ('train', 'val', 'test'):
            raise ValueError('Invalid FF++ dataset/split')
        label = row.get('label')
        if type(label) is not int or label not in (0, 1):
            raise ValueError('FF++ labels must be 0=real, 1=fake')
        if (row.get('kind'), row.get('generator')) != (('real', None) if label == 0 else ('local', 'Deepfakes')):
            raise ValueError('FF++ kind/generator does not match its label')
        sources = row.get('source_video_ids')
        if not isinstance(sources, list) or len(sources) != (1 if label == 0 else 2) or not all(
                isinstance(x, str) and re.fullmatch(r'\d{3}', x) for x in sources):
            raise ValueError('Invalid FF++ source video IDs')
        if row.get('video_id') != '_'.join(sources):
            raise ValueError('FF++ video ID does not match source video IDs')
        for name in ('image_sha256', 'mask_sha256', 'decoded_rgb_sha256'):
            if not isinstance(row.get(name), str) or not re.fullmatch(r'[0-9a-f]{64}', row[name]):
                raise ValueError('Invalid FF++ digest')
        if row['sample_id'] in ids or row['decoded_rgb_sha256'] in content:
            raise ValueError('Duplicate FF++ sample/content')
        ids.add(row['sample_id'])
        content.add(row['decoded_rgb_sha256'])
    return rows


class FFPPFaces(PilotFaces):
    """Reuse checked RGB/mask transforms; supply FF++ metadata and text separately."""
    def __init__(self, manifest, *, tokens=False, limit=None):
        self.manifest = Path(manifest)
        self.digest = sha256(manifest)
        self.rows = read_ffpp_manifest(manifest)
        if limit is not None:
            if type(limit) is not int or limit < 2 or limit % 2:
                raise ValueError('FF++ subset size must be positive and even')
            groups = [[r for r in self.rows if r['label'] == label][:limit // 2] for label in (0, 1)]
            if any(len(group) != limit // 2 for group in groups):
                raise ValueError('Not enough FF++ samples for balanced subset')
            self.rows = [r for pair in zip(*groups) for r in pair]
        self.token_ids = ffpp_prompt_tokens() if tokens else None


def dataset_for_config(root, config, split, *, tokens=False, limit=None):
    settings = config.get('dataset')
    if settings is None:
        owner = PilotFaces
    elif settings == {'name': FFPP, 'prompt_policy': PROMPT_POLICY}:
        owner = FFPPFaces
    else:
        raise ValueError('Unsupported dataset or prompt policy; no label fallback')
    return owner(Path(root) / config['manifests'][split], tokens=tokens, limit=limit)


def audit_source_groups(partitions):
    """Validate all supplied FF++ partitions, independently of sampled frame hashes."""
    if not all(isinstance(data, FFPPFaces) for _, data in partitions):
        if any(isinstance(data, FFPPFaces) for _, data in partitions):
            raise ValueError('Cannot mix FF++ and Diffae partitions')
        return {'source_video_disjoint_verified': False}
    membership, counts = {}, {}
    for split, data in partitions:
        group = set()
        for row in data.rows:
            if row['split'] != split:
                raise ValueError('FF++ manifest split mismatch')
            for source in row['source_video_ids']:
                if source in membership and membership[source] != split:
                    raise ValueError('FF++ source video crosses splits')
                membership[source] = split
                group.add(source)
        counts[split] = len(group)
    return {'source_video_disjoint_verified': True, 'source_video_counts': counts}
