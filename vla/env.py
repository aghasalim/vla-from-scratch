"""A 2D tabletop reach task, rendered as images.

Written by hand rather than pulled from LIBERO or ManiSkill, for the same reason
the pendulum in the world-model repo is: the question here is about action heads
and language generalisation, and a simulator I control lets me hold out specific
object attributes and specific phrasings, which is exactly the split the
generalisation claim needs.

A scene holds several distractor objects and one target. Each object has a
colour and a shape, both of which appear in the instruction. The policy sees an
image and an instruction, and outputs a 2D velocity for the gripper. Success is
ending the episode near the object the instruction names, which means a policy
that ignores the language and drives to the nearest object fails most of the
time by construction.
"""
from __future__ import annotations

import torch

COLORS = {"red": (1.0, 0.2, 0.2), "green": (0.2, 0.9, 0.3),
          "blue": (0.3, 0.4, 1.0), "yellow": (1.0, 0.9, 0.2)}
SHAPES = ("square", "circle", "triangle")
RES = 32
SUCCESS_RADIUS = 0.16


def _stamp(canvas, cx, cy, shape, colour, size=0.15):
    """Draw one object into a (3, RES, RES) canvas. Coordinates are in [-1, 1]."""
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, RES), torch.linspace(-1, 1, RES),
                            indexing="ij")
    dx, dy = xs - cx, ys - cy
    if shape == "square":
        mask = (dx.abs() < size) & (dy.abs() < size)
    elif shape == "circle":
        mask = (dx ** 2 + dy ** 2) < size ** 2
    else:                                   # triangle, pointing up
        mask = (dy > -size) & (dy < size) & (dx.abs() < (size - dy) * 0.6)
    for c in range(3):
        canvas[c] = torch.where(mask, torch.full_like(canvas[c], colour[c]), canvas[c])


class TabletopReach:
    """Batched. Every environment in the batch has its own scene and target."""

    action_dim = 2

    def __init__(self, n: int, n_objects: int = 3, horizon: int = 12,
                 combos: list[tuple[str, str]] | None = None, seed: int = 0):
        self.n, self.n_objects, self.horizon = n, n_objects, horizon
        self.combos = combos or [(c, s) for c in COLORS for s in SHAPES]
        self.g = torch.Generator().manual_seed(seed)
        self.reset()

    def reset(self):
        n, k = self.n, self.n_objects
        self.pos = (torch.rand(n, k, 2, generator=self.g) * 1.4) - 0.7
        # keep objects apart so "nearest" is not ambiguous
        for _ in range(6):
            d = self.pos.unsqueeze(2) - self.pos.unsqueeze(1)
            close = (d.norm(dim=-1) < 0.5) & (~torch.eye(k, dtype=torch.bool))
            self.pos = self.pos + 0.12 * (d * close.unsqueeze(-1)).sum(2)
        self.pos = self.pos.clamp(-0.8, 0.8)

        idx = torch.randint(0, len(self.combos), (n, k), generator=self.g)
        self.attrs = idx
        self.target = torch.randint(0, k, (n,), generator=self.g)
        self.gripper = (torch.rand(n, 2, generator=self.g) * 1.2) - 0.6
        self.t = 0
        return self.observe()

    def target_attrs(self):
        i = self.attrs[torch.arange(self.n), self.target]
        return [self.combos[j] for j in i.tolist()]

    def target_pos(self):
        return self.pos[torch.arange(self.n), self.target]

    def observe(self) -> torch.Tensor:
        """(n, 3, RES, RES) images. The gripper is drawn as a white dot."""
        img = torch.zeros(self.n, 3, RES, RES)
        for b in range(self.n):
            for k in range(self.n_objects):
                colour, shape = self.combos[self.attrs[b, k]]
                _stamp(img[b], self.pos[b, k, 0].item(), self.pos[b, k, 1].item(),
                       shape, COLORS[colour])
            _stamp(img[b], self.gripper[b, 0].item(), self.gripper[b, 1].item(),
                   "circle", (1.0, 1.0, 1.0), size=0.07)
        return img

    def step(self, action: torch.Tensor):
        self.gripper = (self.gripper + action.clamp(-1, 1) * 0.35).clamp(-1, 1)
        self.t += 1
        done = self.t >= self.horizon
        return self.observe(), self.success(), done

    def success(self) -> torch.Tensor:
        return (self.gripper - self.target_pos()).norm(dim=-1) < SUCCESS_RADIUS

    def expert_action(self, noise: float = 0.05) -> torch.Tensor:
        """Move toward the target. Used to generate demonstrations."""
        d = self.target_pos() - self.gripper
        a = d / 0.35
        return (a + noise * torch.randn(a.shape, generator=self.g)).clamp(-1, 1)

    def nearest_object_is_target(self) -> torch.Tensor:
        """How often a language-blind 'go to the nearest object' policy would win."""
        d = (self.pos - self.gripper.unsqueeze(1)).norm(dim=-1)
        return d.argmin(dim=1) == self.target
