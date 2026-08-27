"""Four ways to emit a continuous action, on identical features.

This is the comparison the repo exists for. Every head takes the same 128
dimensional vision language feature and produces a 2D action. They differ only
in how the action distribution is represented.

  DiscreteBins   RT-2 style. Quantise each dimension into K bins and predict
                 them as tokens, autoregressively across dimensions so the
                 second dimension is conditioned on the first. Multimodal by
                 construction, since a categorical can put mass on two bins. The
                 cost is quantisation error, which floors precision at half a
                 bin width no matter how good the model is.

  Regression     A direct MLP trained with mean squared error. One forward pass,
                 the fastest thing here, and unimodal by construction: the
                 minimiser of MSE is the conditional *mean*, so on two-mode data
                 it emits the average of the modes. On this task that points at
                 the obstacle.

  DiffusionHead  A DDPM over the action vector, conditioned on the feature.
                 Multimodal, and the standard answer in robot learning. The cost
                 is that sampling needs many denoising steps, which sets a
                 ceiling on control rate.

  FlowHead       Conditional flow matching, the pi-0 style action expert.
                 Multimodal like diffusion but trained against a straight
                 interpolant, so it integrates accurately in far fewer steps.
                 The claim being tested is that it keeps diffusion's
                 multimodality at closer to regression's latency.

Each exposes `loss(feat, action)` and `sample(feat, steps)`, and reports the
number of network evaluations a sample costs, since that is what decides the
achievable control frequency.

**Action chunking.** Every head predicts `chunk` consecutive actions at once and
the controller executes them open loop before replanning. This is not a
performance optimisation, it is a correctness fix for multimodal policies. The
demonstrator's choice of side is not observable, so a head that samples
independently at every timestep can draw "go left" and then "go right" and
dither in place. Committing to a chunk makes the sampled mode persist for as
long as the chunk lasts. ACT and pi-0 both do this and the reason is exactly
this failure.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class Head(nn.Module):
    name = "base"
    multimodal = False

    def nfe(self, steps: int) -> int:
        raise NotImplementedError


class RegressionHead(Head):
    name = "regression"
    multimodal = False

    def __init__(self, dim, act_dim=2, hidden=128, chunk=1):
        super().__init__()
        self.chunk, self.act_dim = chunk, act_dim
        out = act_dim * chunk
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out), nn.Tanh())

    def loss(self, feat, action):
        return ((self.net(feat) - action.flatten(1)) ** 2).mean()

    @torch.no_grad()
    def sample(self, feat, steps=1):
        return self.net(feat).view(-1, self.chunk, self.act_dim)

    def nfe(self, steps=1):
        return 1


class DiscreteBins(Head):
    name = "discrete bins"
    multimodal = True

    def __init__(self, dim, act_dim=2, bins=21, hidden=128, chunk=1):
        super().__init__()
        self.bins, self.act_dim, self.chunk = bins, act_dim, chunk
        n_out = act_dim * chunk
        self.n_out = n_out
        # autoregressive over the flattened chunk: token i sees all before it
        self.nets = nn.ModuleList([
            nn.Sequential(nn.Linear(dim + i, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, bins)) for i in range(n_out)])
        self.register_buffer("centers", torch.linspace(-1, 1, bins))

    def quantise(self, action):
        return (action.unsqueeze(-1) - self.centers).abs().argmin(-1)

    def loss(self, feat, action):
        flat = action.flatten(1)
        idx = self.quantise(flat)
        total, prefix = 0.0, feat
        for i, net in enumerate(self.nets):
            total = total + nn.functional.cross_entropy(net(prefix), idx[:, i])
            prefix = torch.cat([prefix, flat[:, i:i + 1]], -1)
        return total / self.n_out

    @torch.no_grad()
    def sample(self, feat, steps=1):
        out, prefix = [], feat
        for net in self.nets:
            probs = net(prefix).softmax(-1)
            val = self.centers[torch.multinomial(probs, 1).squeeze(-1)]
            out.append(val)
            prefix = torch.cat([prefix, val.unsqueeze(-1)], -1)
        return torch.stack(out, -1).view(-1, self.chunk, self.act_dim)

    def nfe(self, steps=1):
        return self.n_out

    @property
    def quantisation_error(self):
        """Half a bin width: the precision floor this head cannot beat."""
        return 1.0 / (self.bins - 1)


def _time_embed(t, dim=32):
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, dtype=t.dtype) / half)
    ang = t.view(-1, 1) * freqs.view(1, -1) * 1000.0
    return torch.cat([ang.sin(), ang.cos()], -1)


class _Denoiser(nn.Module):
    def __init__(self, dim, act_dim, hidden, t_dim=32):
        super().__init__()
        self.t_dim = t_dim
        self.net = nn.Sequential(nn.Linear(dim + act_dim + t_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, act_dim))

    def forward(self, feat, x, t):
        return self.net(torch.cat([feat, x, _time_embed(t, self.t_dim)], -1))


class DiffusionHead(Head):
    name = "diffusion"
    multimodal = True

    def __init__(self, dim, act_dim=2, hidden=128, T=50, chunk=1):
        super().__init__()
        self.T, self.chunk = T, chunk
        act_dim = act_dim * chunk
        self.net = _Denoiser(dim, act_dim, hidden)
        betas = torch.linspace(1e-4, 0.05, T)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("abar", torch.cumprod(1 - betas, 0))
        self.act_dim = act_dim

    def loss(self, feat, action):
        action = action.flatten(1)
        b = action.shape[0]
        t = torch.randint(0, self.T, (b,))
        noise = torch.randn_like(action)
        ab = self.abar[t].unsqueeze(-1)
        x_t = ab.sqrt() * action + (1 - ab).sqrt() * noise
        pred = self.net(feat, x_t, t.float() / self.T)
        return ((pred - noise) ** 2).mean()

    @torch.no_grad()
    def sample(self, feat, steps=None):
        steps = steps or self.T
        x = torch.randn(feat.shape[0], self.act_dim)
        idx = torch.linspace(self.T - 1, 0, steps).long()
        for k, t in enumerate(idx):
            eps = self.net(feat, x, torch.full((feat.shape[0],), float(t) / self.T))
            ab = self.abar[t]
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            if k + 1 < len(idx):
                ab_prev = self.abar[idx[k + 1]]
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
            else:
                x = x0
        return x.view(-1, self.chunk, self.act_dim // self.chunk)

    def nfe(self, steps=None):
        return steps or self.T


class FlowHead(Head):
    name = "flow (pi-0 style)"
    multimodal = True

    def __init__(self, dim, act_dim=2, hidden=128, chunk=1):
        super().__init__()
        self.chunk = chunk
        self.act_dim = act_dim * chunk
        self.net = _Denoiser(dim, self.act_dim, hidden)

    def loss(self, feat, action):
        action = action.flatten(1)
        b = action.shape[0]
        x0 = torch.randn_like(action)
        t = torch.rand(b)
        x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * action
        return ((self.net(feat, x_t, t) - (action - x0)) ** 2).mean()

    @torch.no_grad()
    def sample(self, feat, steps=5):
        x = torch.randn(feat.shape[0], self.act_dim)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((feat.shape[0],), i * dt)
            x = x + dt * self.net(feat, x, t)
        return x.clamp(-1, 1).view(-1, self.chunk, self.act_dim // self.chunk)

    def nfe(self, steps=5):
        return steps


HEADS = {
    "regression": RegressionHead,
    "discrete bins": DiscreteBins,
    "diffusion": DiffusionHead,
    "flow (pi-0 style)": FlowHead,
}
