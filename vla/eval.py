"""Closed loop rollouts, and the latency that decides control rate."""
from __future__ import annotations

import time

import torch

from .envs import ReachEnv


@torch.no_grad()
def rollout(encoder, head, n=256, horizon=24, seed=0, steps=None,
            held_out=None, train=True):
    env = ReachEnv(n, horizon=horizon, seed=seed, held_out_pairs=held_out, train=train)
    obs = env.observe()
    blocked_total = torch.zeros(n)
    chunk = getattr(head, "chunk", 1)
    t = 0
    while t < horizon:
        # Replan, then execute the whole chunk open loop. That commitment is
        # what stops a multimodal head from re-drawing its mode every step.
        feat = encoder(obs["image"], obs["color"], obs["shape"], obs["state"])
        plan = head.sample(feat, steps) if steps else head.sample(feat)
        for k in range(min(chunk, horizon - t)):
            obs, _, blocked = env.step(plan[:, k])
            blocked_total += blocked.float()
            t += 1
    return {
        "success": env.success().float().mean().item(),
        "blocked_steps": blocked_total.mean().item(),
        "final_dist": (env.agent - env.target_pos()).norm(dim=-1).mean().item(),
    }


@torch.no_grad()
def latency(encoder, head, n=64, steps=None, repeats=12):
    """Seconds per action, and the control frequency that implies."""
    env = ReachEnv(n, seed=0)
    obs = env.observe()
    feat = encoder(obs["image"], obs["color"], obs["shape"], obs["state"])
    for _ in range(3):
        head.sample(feat, steps) if steps else head.sample(feat)
    chunk = getattr(head, "chunk", 1)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        head.sample(feat, steps) if steps else head.sample(feat)
        # one forward pass yields `chunk` actions, so amortise over the chunk
        ts.append((time.perf_counter() - t0) / (n * chunk))
    ts.sort()
    per_action = ts[len(ts) // 2]
    return {"latency_s": per_action, "max_hz": 1.0 / per_action,
            "chunk": chunk,
            "nfe": head.nfe(steps) if steps else head.nfe()}
