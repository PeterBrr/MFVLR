"""ImageNet ResNet-50 in place of the from-scratch U-Net contracting path.

An improvement outside the MFVLR paper, which trains from scratch. The decoder and
transformer keep their interfaces: skips of 64x224, 128x112, 256x56 and 512x28, and
a 1024x14x14 bottleneck, which is exactly ResNet-50 up to layer3. Batch norm stays
frozen because a microbatch of 2 cannot estimate batch statistics.
"""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.ops import FrozenBatchNorm2d

from mfvlr.mve import DoubleConv, LocalEncoding

ENCODERS = {'scratch': None, 'resnet50_imagenet': 'artifacts/pretrained/resnet50-11ad3fa6.bin'}
WEIGHTS_SHA256 = {'resnet50_imagenet': '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'}
ROOT = Path(__file__).resolve().parents[2]


class ResNetLocalEncoder(nn.Module):
    """Same contract as UNetLocalEncoder; `stages` lets gradient checkpointing wrap each stage."""

    def __init__(self, weights_path, seed):
        super().__init__()
        data = Path(weights_path).read_bytes()
        if hashlib.sha256(data).hexdigest() != WEIGHTS_SHA256['resnet50_imagenet']:
            raise ValueError(f'Unexpected ResNet-50 weights: {weights_path}')
        network = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
        network.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        with torch.random.fork_rng(devices=[]):
            # ResNet has no full-resolution feature, so the 224 skip keeps a scratch stem.
            full = DoubleConv(3, 64)
            self.lift = nn.Conv2d(64, 128, 1)
            generator = torch.Generator().manual_seed(seed)
            for layer in full:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu', generator=generator)
                    nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(self.lift.weight, generator=generator)
            nn.init.zeros_(self.lift.bias)
        self.stages = nn.ModuleList([full, nn.Sequential(network.conv1, network.bn1, network.relu),
                                     nn.Sequential(network.maxpool, network.layer1), network.layer2, network.layer3])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def pretrained_parameters(self):
        return [p for stage in self.stages[1:] for p in stage.parameters()]

    def forward(self, image, return_skips=True):
        full = self.stages[0](image)
        half = self.stages[1]((image - self.mean) / self.std)
        quarter = self.stages[2](half)
        eighth = self.stages[3](quarter)
        local = self.stages[4](eighth)
        skips = (full, self.lift(half), quarter, eighth) if return_skips else ()
        return LocalEncoding(local=local, skips=skips)


def install_encoder(model, encoder, seed):
    """Swap the shared CNN; the transformer and decoder keep the baseline initialization."""
    if encoder not in ENCODERS:
        raise ValueError(f'model.encoder must be one of {sorted(ENCODERS)}')
    if encoder == 'scratch':
        return
    model.mve.shared_encoder.cnn = ResNetLocalEncoder(ROOT / ENCODERS[encoder], seed)
