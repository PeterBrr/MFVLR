"""Explicit optimizer/scheduler policy; legacy pilot recipes remain unchanged."""
import math

import torch


def _number(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (value < 0 if zero else value <= 0)):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')


def validate_optimization(config, recipe):
    settings = config['optimizer']
    if settings.get('name', 'Adam') != 'Adam':
        raise ValueError('Only the explicitly supported Adam optimizer is available')
    _number(recipe['lr'], 'learning rate')
    _number(settings['weight_decay'], 'weight decay', zero=True)
    _number(settings.get('temperature_weight_decay', 0.), 'temperature weight decay', zero=True)
    _number(settings['eps'], 'Adam epsilon')
    betas = settings['betas']
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError('Adam betas must contain two values')
    for value in betas:
        _number(value, 'Adam beta', zero=True)
        if value >= 1:
            raise ValueError('Adam betas must be less than one')
    schedule = config.get('scheduler')
    if schedule is None:
        return
    if (not isinstance(schedule, dict)
            or set(schedule) != {'name', 'step_size', 'gamma'}
            or schedule['name'] != 'StepLR'):
        raise ValueError('Scheduler must specify StepLR name, step_size and gamma')
    if type(schedule['step_size']) is not int or schedule['step_size'] < 1:
        raise ValueError('Scheduler step_size must be a positive integer number of completed epochs')
    _number(schedule['gamma'], 'scheduler gamma')
    if schedule['gamma'] >= 1:
        raise ValueError('Scheduler gamma must be less than one')


def build_optimization(model, config, recipe):
    validate_optimization(config, recipe)
    settings = config['optimizer']
    regular, temperature = [], []
    for name, parameter in model.named_parameters():
        (temperature if name == 'objectives.cmc.log_temperature' else regular).append(parameter)
    optimizer = torch.optim.Adam([
        {'params': regular, 'weight_decay': settings['weight_decay']},
        {'params': temperature, 'weight_decay': settings.get('temperature_weight_decay', 0.)},
    ], lr=recipe['lr'], betas=tuple(settings['betas']), eps=settings['eps'], foreach=False)
    schedule = config.get('scheduler')
    scheduler = None if schedule is None else torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=schedule['step_size'], gamma=schedule['gamma'])
    return optimizer, scheduler


def restore_scheduler(scheduler, saved_state, completed_epochs):
    """Never silently restart an enabled schedule when resuming a checkpoint."""
    if scheduler is None:
        if saved_state is not None:
            raise ValueError('Checkpoint scheduler differs from the configured schedule')
        return
    if saved_state is None:
        raise ValueError('Checkpoint is missing the configured scheduler state')
    if saved_state.get('last_epoch') != completed_epochs:
        raise ValueError('Scheduler state and completed epoch count differ')
    scheduler.load_state_dict(saved_state)
