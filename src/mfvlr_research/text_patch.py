"""Text-patch localization head: visual patch tokens scored against fixed sentence embeddings.

In the spirit of VLAForge (arXiv 2603.24454, Eq. 9-10) and HiFi-Net++'s LFLE (arXiv 2410.23556):
each 14x14 patch is projected into a text-embedding space and classified by cosine similarity
against one frozen embedding per class, the map is supervised with the GT mask, and its edited
log-odds are added to the mask logits through a zero-initialized gain, so an untrained head
leaves the baseline prediction unchanged.

`clip` uses frozen CLIP ViT-L/14 text embeddings, precomputed by tools/make_text_patch_embeddings.py.
`random` is the control: fixed random unit vectors of the same size. The visual encoder is trained
from scratch, not in CLIP space, and the projection is learned, so the two arms differ only in the
geometry of the class vectors; a gain of clip over random is what would credit the language prior.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# Row order is the class order: 0 unedited, then one class per GenFace Diffae attribute.
CLASS_PROMPTS = ('an unedited region of a real face photo',
                 'a region of a face photo where the bangs above the forehead were edited',
                 'a region of a face photo where the smile around the mouth was edited')
ATTRIBUTE_CLASSES = {'bangs': 1, 'smile': 2}
TEXT_PATCH_MODES = ('none', 'clip', 'random')
CLIP_EMBEDDINGS = Path(__file__).with_name('assets') / 'text_patch_clip_vitl14.json'
GRID, PATCH = 14, 16


def class_embeddings(mode):
    """[3,768] unit rows; both arms are the same for every training seed."""
    if mode == 'clip':
        stored = json.loads(CLIP_EMBEDDINGS.read_text(encoding='utf-8'))
        if tuple(stored['prompts']) != CLASS_PROMPTS:
            raise ValueError(f'{CLIP_EMBEDDINGS} was made for other prompts; rerun tools/make_text_patch_embeddings.py')
        text = torch.tensor(stored['embeddings'], dtype=torch.float32)
    elif mode == 'random':
        text = torch.randn(len(CLASS_PROMPTS), 768, generator=torch.Generator().manual_seed(0))
    else:
        raise ValueError(f'text_patch mode must be one of {TEXT_PATCH_MODES[1:]}')
    return F.normalize(text, dim=-1)


def patch_classes(sample_ids, labels):
    """The edited class of each image: 0 for real, the Diffae attribute's row otherwise."""
    classes = []
    for name, label in zip(sample_ids, labels.tolist()):
        if label == 0:
            classes.append(0)
            continue
        attribute = Path(name).stem.split('_')[1].casefold()
        if attribute not in ATTRIBUTE_CLASSES:
            raise ValueError(f'{name}: no text-patch class for attribute {attribute!r}')
        classes.append(ATTRIBUTE_CLASSES[attribute])
    return torch.tensor(classes, device=labels.device)


class TextPatchHead(nn.Module):
    def __init__(self, mode, seed=0):
        super().__init__()
        self.register_buffer('text', class_embeddings(mode))
        with torch.random.fork_rng(devices=[]):
            self.norm = nn.LayerNorm(512)
            self.project = nn.Linear(512, self.text.shape[1])
            nn.init.xavier_uniform_(self.project.weight, generator=torch.Generator().manual_seed(seed))
            nn.init.zeros_(self.project.bias)
        # CLIP's initial temperature 0.07; clamped at CLIP's maximum scale of 100.
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, tokens):
        """[B,197,512] tokens, spatial first and class token last, -> [B,3,14,14] class logits."""
        patches = F.normalize(self.project(self.norm(tokens[:, :GRID * GRID])), dim=-1)
        logits = self.logit_scale.clamp(max=math.log(100)).exp() * patches @ self.text.T
        return logits.transpose(1, 2).reshape(tokens.shape[0], -1, GRID, GRID)

    def mask_bias(self, logits, size):
        """Edited-vs-unedited log-odds, upsampled, as a correction to the foreground mask logit."""
        odds = torch.logsumexp(logits[:, 1:], dim=1, keepdim=True) - logits[:, :1]
        odds = F.interpolate(odds, size=size, mode='bilinear', align_corners=False)
        return torch.cat((torch.zeros_like(odds), self.gain * odds), dim=1)


def text_patch_loss(logits, masks, labels, sample_ids):
    """Soft cross-entropy on per-patch edited fractions plus soft Dice of the edited map on fake images."""
    fraction = F.avg_pool2d(masks[:, None].float(), PATCH)[:, 0]
    classes = patch_classes(sample_ids, labels)
    target = torch.zeros_like(logits)
    target[:, 0] = 1 - fraction
    target.scatter_add_(1, classes[:, None, None, None].expand(-1, 1, GRID, GRID), fraction[:, None])
    log_probability = logits.log_softmax(dim=1)
    cross_entropy = -(target * log_probability).sum(dim=1).mean()
    fake = labels == 1
    if not fake.any():
        return cross_entropy
    edited, truth = 1 - log_probability[fake, 0].exp(), fraction[fake]
    overlap = (edited * truth).sum(dim=(1, 2))
    dice = (1 - (2 * overlap + 1) / (edited.sum(dim=(1, 2)) + truth.sum(dim=(1, 2)) + 1)).mean()
    return cross_entropy + dice
