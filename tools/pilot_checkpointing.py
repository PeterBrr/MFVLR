"""Recompute training activations without changing module parameters/state keys."""
import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_stage(module):
    original = module.forward

    def forward(*args, **kwargs):
        if module.training and torch.is_grad_enabled():
            return checkpoint(original, *args, use_reentrant=False, **kwargs)
        return original(*args, **kwargs)

    module.forward = forward


def enable_checkpointing(model):
    stages = [*model.mve.shared_encoder.cnn.stages, *model.mve.shared_encoder.blocks,
              *model.decoder.trunk.stages]
    if getattr(model, 'flt', None) is not None:
        stages += [*model.flt.encoder.blocks, *model.flt.decoder.blocks]
    for stage in stages:
        checkpoint_stage(stage)
    return len(stages)
