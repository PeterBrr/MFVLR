"""One-forward traversal engine. Only bounded synthetic execution is exposed."""
import time
import torch
from mfvlr_research.d02b1.gradients import GradientStore

ORDER = ('fd', 'lr', 'cmc', 'fl', 'ar', 'kl')

def dispatch_real(*args, **kwargs):
    raise PermissionError('D02B9 real G01-A0 execution requires separate authorization; B0-D08 disabled')

def _traverse(named_parameters, losses, store):
    """Prepared full-roster kernel; callers must seal ownership/state/provenance.

    No forward here. Gradients are streamed before the next traversal. This is
    not an authorized live dispatcher and performs no automatic continuation.
    """
    if tuple(losses) != ORDER:
        raise ValueError('Literal six-loss order required')
    names, parameters = zip(*named_parameters)
    if len(set(names)) != len(names) or len({id(p) for p in parameters}) != len(parameters):
        raise ValueError('Duplicate parameter ownership')
    if len({p.untyped_storage().data_ptr() for p in parameters}) != len(parameters):
        raise ValueError('Aliased parameter storage')
    if any(not p.requires_grad or p.dtype != torch.float32 or p.device.type != 'cpu' for p in parameters):
        raise ValueError('Trainable CPU FP32 parameters required')
    for value in losses.values():
        if value.ndim != 0 or not value.requires_grad or not torch.isfinite(value).item():
            raise ValueError('Connected finite scalar required')
    total = losses[ORDER[0]]
    for name in ORDER[1:]:
        total = total + losses[name]
    if not torch.isfinite(total).item():
        raise FloatingPointError('Nonfinite total')
    values = dict(losses, total=total)
    records = []
    for index, (loss, value) in enumerate(values.items()):
        started = time.perf_counter()
        gradients = torch.autograd.grad(value, parameters, allow_unused=True,
            materialize_grads=False, retain_graph=index < 6, create_graph=False)
        for name, gradient in zip(names, gradients):
            store.put(loss, name, None if gradient is None else gradient.detach().contiguous().numpy())
        records.append(dict(loss=loss, scalar=float(value.detach()), retain_graph=index < 6,
                            seconds=time.perf_counter()-started))
        del gradients, gradient
    store.complete()
    return records

def run_mock(forward, named_parameters, specifications, destination):
    """Explicit small-graph test interface; cannot admit a full MFVLR roster."""
    parameters = tuple(named_parameters)
    if not parameters or len(parameters) > 32 or sum(p.numel() for _, p in parameters) > 4096:
        raise PermissionError('Synthetic mock size limit')
    if any(p.grad is not None for _, p in parameters):
        raise ValueError('Clean .grad buffers required')
    if tuple(specifications) != tuple(n for n, _ in parameters):
        raise ValueError('Complete ordered roster required')
    before = [p.detach().clone() for _, p in parameters]
    store = GradientStore(destination, specifications)
    losses = forward()  # exactly one invocation; never retry on failure
    try:
        records = _traverse(parameters, losses, store)
        if any(p.grad is not None or not torch.equal(p, old) for (_, p), old in zip(parameters, before)):
            raise RuntimeError('Parameter or .grad mutation')
        return dict(execution='COMPLETE', passes=records, analysis=store.analyze())
    finally:
        # No returned tensor keeps the retained graph alive after this function.
        del losses

def historical_decisions(current, historical):
    """Read-only pairing, never reinterpret a historical failure as accepted."""
    if set(current) != set(historical):
        raise ValueError('Complete historical roster required')
    return {name: {'historical': historical[name], 'same_forward': current[name],
                   'changed': current[name] != historical[name]} for name in current}
