"""Flow matching in the pi0 / openpi convention.

x_tau = tau * eps + (1 - tau) * a,  target velocity u = eps - a,
tau ~ Beta(1.5, 1) * 0.999 + 0.001; sampling integrates tau: 1 -> 0 in NUM_STEPS Euler steps.
"""
import torch

NUM_STEPS = 10


def sample_tau(n, device, gen=None):
    # Beta(1.5, 1) via inverse CDF: F(x) = x^1.5
    u = torch.rand(n, device=device, generator=gen)
    return u.pow(1 / 1.5) * 0.999 + 0.001


def noisy(a, eps, tau):
    t = tau.view(-1, *([1] * (a.dim() - 1)))
    return t * eps + (1 - t) * a, eps - a


def flow_loss(velocity_fn, a, mask, gen=None, tau=None, eps=None):
    """Per-sample masked MSE between predicted and target velocity."""
    if tau is None:
        tau = sample_tau(a.shape[0], a.device, gen)
    if eps is None:
        eps = torch.randn(a.shape, device=a.device, generator=gen)
    x, u = noisy(a, eps, tau)
    err = (velocity_fn(x, tau) - u).pow(2).mean(-1)          # (B, H)
    return (err * mask).sum(-1) / mask.sum(-1).clamp(min=1)  # (B,)


def flow_sample(velocity_fn, shape, device, gen=None):
    x = torch.randn(shape, device=device, generator=gen)
    dt = -1.0 / NUM_STEPS
    tau = torch.ones(shape[0], device=device)
    for _ in range(NUM_STEPS):
        x = x + dt * velocity_fn(x, tau)
        tau = tau + dt
    return x
