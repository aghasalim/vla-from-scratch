"""Scripted demonstrations that are deliberately multimodal.

The demonstrator picks a side, left or right, uniformly at random, and routes
around the obstacle on that side before going to the target. Both routes are
correct and roughly equally long, so the conditional action distribution given
the observation genuinely has two modes.

This is the condition the whole repo depends on. Averaging the two modes points
straight at the obstacle, which is exactly what a regression head trained with
mean squared error will do, and it is why diffusion and flow heads exist. If the
demonstrations were unimodal, a regression head would win and the comparison
would say nothing.
"""
from __future__ import annotations

import torch

from .envs import OBSTACLE_HALF_H, OBSTACLE_HALF_W, OBSTACLE_Y, ReachEnv


def waypoint_route(agent, target, side):
    """Where to steer next: around the obstacle on `side`, then to the target."""
    detour_x = torch.where(side > 0, OBSTACLE_HALF_W + 0.22, -(OBSTACLE_HALF_W + 0.22))
    below = agent[:, 1] < OBSTACLE_Y + OBSTACLE_HALF_H + 0.05
    # while below the obstacle, head for the detour gap; after that, the target
    wp_x = torch.where(below, detour_x, target[:, 0])
    wp_y = torch.where(below, torch.full_like(agent[:, 1], OBSTACLE_Y + OBSTACLE_HALF_H + 0.18),
                       target[:, 1])
    return torch.stack([wp_x, wp_y], -1)


@torch.no_grad()
def collect(n: int, horizon: int = 24, seed: int = 0, held_out=None, train=True,
            noise: float = 0.04, chunk: int = 1):
    """Return dict of tensors. `action` is (N, chunk, 2): the next `chunk` actions.

    Targets are the actions the demonstrator took from this state onward, padded
    at the end of the episode by repeating the last one.
    """
    env = ReachEnv(n, horizon=horizon, seed=seed, held_out_pairs=held_out, train=train)
    g = torch.Generator().manual_seed(seed + 1)
    side = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)

    imgs, cols, shps, acts, states = [], [], [], [], []
    for _ in range(horizon):
        obs = env.observe()
        wp = waypoint_route(env.agent, env.target_pos(), side)
        d = wp - env.agent
        a = (d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        a = (a + noise * torch.randn(a.shape, generator=g)).clamp(-1, 1)
        imgs.append(obs["image"])
        cols.append(obs["color"])
        shps.append(obs["shape"])
        states.append(obs["state"])
        acts.append(a)
        env.step(a)

    acts_t = torch.stack(acts, 1)                              # (n, T, 2)
    windows = []
    for t in range(horizon):
        idx = [min(t + k, horizon - 1) for k in range(chunk)]
        windows.append(acts_t[:, idx])                          # (n, chunk, 2)
    chunked = torch.stack(windows, 1).flatten(0, 1)             # (n*T, chunk, 2)

    return {
        "image": torch.stack(imgs, 1).flatten(0, 1),
        "color": torch.stack(cols, 1).flatten(0, 1),
        "shape": torch.stack(shps, 1).flatten(0, 1),
        "state": torch.stack(states, 1).flatten(0, 1),
        "action": chunked,
        "side": side.repeat_interleave(horizon),
        "success_rate": env.success().float().mean().item(),
    }
