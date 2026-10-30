"""Schedule boundaries and optimizer grouping, without expensive model training."""
import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from mfvlr_research.pilot_optimization import build_optimization, validate_optimization

ROOT = Path(__file__).resolve().parents[2]


def paper_config():
    return json.loads((ROOT / 'configs/diffae_pilot_paper_optimizer.json').read_text(encoding='utf-8'))


def model_with_temperature():
    model = nn.Module()
    model.linear = nn.Linear(1, 1)
    model.objectives = nn.Module()
    model.objectives.cmc = nn.Module()
    model.objectives.cmc.log_temperature = nn.Parameter(torch.tensor(-2.6))
    return model


def test_paper_optimizer_all_groups_and_epoch_boundaries():
    config = paper_config()
    model = model_with_temperature()
    optimizer, scheduler = build_optimization(model, config, config['pilot'])
    assert type(optimizer) is torch.optim.Adam
    assert [g['weight_decay'] for g in optimizer.param_groups] == [.001, .001]
    assert optimizer.param_groups[1]['params'][0] is model.objectives.cmc.log_temperature
    rates = []
    for _ in range(31):
        rates.append(optimizer.param_groups[0]['lr'])
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    assert rates[:15] == pytest.approx([1e-4] * 15)
    assert rates[15:30] == pytest.approx([1e-5] * 15)
    assert rates[30] == pytest.approx(1e-6)


def test_legacy_recipe_keeps_temperature_exemption_and_constant_lr():
    config = json.loads((ROOT / 'configs/diffae_pilot_small.json').read_text())
    optimizer, scheduler = build_optimization(model_with_temperature(), config, config['pilot'])
    assert scheduler is None
    assert [g['weight_decay'] for g in optimizer.param_groups] == [.001, 0.]
    assert [g['lr'] for g in optimizer.param_groups] == [1e-5, 1e-5]


@pytest.mark.parametrize('schedule', [
    {'name': 'StepLR', 'step_size': 0, 'gamma': .1},
    {'name': 'StepLR', 'step_size': True, 'gamma': .1},
    {'name': 'StepLR', 'step_size': 15, 'gamma': 0},
    {'name': 'StepLR', 'step_size': 15, 'gamma': 1},
    {'name': 'StepLR', 'step_size': 15, 'gamma': float('nan')},
    {'name': 'ReduceLROnPlateau', 'step_size': 15, 'gamma': .1},
])
def test_invalid_schedule_rejected_before_training(schedule):
    config = copy.deepcopy(paper_config())
    config['scheduler'] = schedule
    with pytest.raises(ValueError):
        validate_optimization(config, config['pilot'])
