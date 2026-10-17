"""Independent arithmetic fixtures. No research variant forward is an oracle."""
import math
import torch


def norm_forward(x, gamma, beta):
    mean = x.sum(-1, keepdim=True) / 512
    centered = x - mean
    variance = centered.square().sum(-1, keepdim=True) / 512
    return gamma * centered / (variance + 1e-5).sqrt() + beta


def norm_vjp(x, gamma, cotangent):
    mean = x.sum(-1, keepdim=True) / 512
    centered = x - mean
    inverse = ((centered.square().sum(-1, keepdim=True) / 512) + 1e-5).rsqrt()
    normalized = centered * inverse
    u = cotangent * gamma
    dx = inverse * (u - u.sum(-1, keepdim=True) / 512
                    - normalized * (u * normalized).sum(-1, keepdim=True) / 512)
    axes = tuple(range(x.ndim - 1))
    return dx, (cotangent * normalized).sum(axes), cotangent.sum(axes)


def attention(query, memory, weights, causal):
    q = query @ weights["q.weight"].T + weights["q.bias"]
    k = memory @ weights["k.weight"].T + weights["k.bias"]
    v = memory @ weights["v.weight"].T + weights["v.bias"]
    q, k, v = [t.reshape(t.shape[0], t.shape[1], 8, 64).transpose(1, 2) for t in (q, k, v)]
    scores = q @ k.transpose(-1, -2) / 8
    if causal:
        rows = torch.arange(query.shape[1])[:, None]
        cols = torch.arange(memory.shape[1])[None, :]
        scores = scores.masked_fill(cols > rows, -torch.inf)
    exponents = (scores - scores.max(-1, keepdim=True).values).exp()
    probability = exponents / exponents.sum(-1, keepdim=True)
    context = (probability @ v).transpose(1, 2).reshape_as(query)
    return context @ weights["out.weight"].T + weights["out.bias"]


def functional_block(x, memory, visual, parameters, variant):
    def w(prefix): return {k[len(prefix)+1:]: v for k, v in parameters.items() if k.startswith(prefix + ".")}
    def norm(t, i): return norm_forward(t, parameters[f"ln{i}.weight"], parameters[f"ln{i}.bias"])
    a = x + attention(norm(x, 1) if variant == "P3" else x,
                      norm(x, 1) if variant == "P3" else x, w("self_attention"), True)
    if variant == "Q3": a = norm(a, 1)
    b = a + attention(norm(a, 2) if variant == "P3" else a, memory, w("memory_attention"), False)
    if variant == "Q3": b = norm(b, 2)
    f = norm(b, 3) if variant == "P3" else b
    pre = f @ parameters["ff.fc1.weight"].T + parameters["ff.fc1.bias"]
    gelu = .5 * pre * (1 + torch.erf(pre / math.sqrt(2)))
    c = b + gelu @ parameters["ff.fc2.weight"].T + parameters["ff.fc2.bias"]
    if variant == "Q3": c = norm(c, 3)
    # Singleton softmax equals one. Independent forward value, not a graph oracle.
    delta = (visual @ parameters["vim.v.weight"].T) @ parameters["vim.out.weight"].T
    return c + delta
