"""A 2D reaching task with an obstacle, written directly.

The agent starts at the bottom, three objects sit at the top, and a language
instruction names one of them. Actions are continuous 2D velocities. An obstacle
sits in the middle of the workspace.

The obstacle is the entire point. A demonstrator can go around it to the left or
to the right, and both are correct. That makes the demonstration distribution
**multimodal**, which is the condition under which action head design actually
matters: a regression head trained on both modes learns their average, which
drives straight into the obstacle. Without multimodality a regression head wins
trivially and the comparison in this repo would be meaningless.

Observations are a small rendered image plus the instruction tokens, so the
policy has to read the scene rather than being handed object coordinates.
"""
from __future__ import annotations

import torch

GRID = 24
COLORS = ["red", "green", "blue"]
SHAPES = ["square", "circle", "triangle"]
OBSTACLE_Y = 0.0
OBSTACLE_HALF_W = 0.30
OBSTACLE_HALF_H = 0.10
SUCCESS_RADIUS = 0.16
MAX_SPEED = 0.16


class ReachEnv:
    """Batched. State is agent xy; objects are fixed per episode."""

    act_dim = 2

    def __init__(self, n: int, horizon: int = 24, seed: int = 0,
                 held_out_pairs: set | None = None, train: bool = True):
        self.n, self.horizon = n, horizon
        self.g = torch.Generator().manual_seed(seed)
        self.held_out = held_out_pairs or set()
        self.train = train
        self.reset()

    def _sample_scene(self):
        """Three objects with distinct (colour, shape) pairs, one is the target."""
        n = self.n
        allowed = [(c, s) for c in range(3) for s in range(3)
                   if ((c, s) in self.held_out) != self.train]
        if len(allowed) < 3:
            # A scene needs three distinct objects. The earlier version took
            # randperm(len(allowed))[:3], which silently returned fewer indices
            # when the split was small and left the remaining slots at their
            # zero initialised value, i.e. a (red, square) object that the split
            # was supposed to exclude. That is a held out pair leaking into the
            # evaluation set without any error, which is the worst kind.
            raise ValueError(
                f"split has only {len(allowed)} allowed (colour, shape) pairs but a "
                f"scene needs 3. train={self.train}, held_out={sorted(self.held_out)}. "
                f"Hold out exactly 3 pairs, or widen the vocabulary.")
        self.obj_color = torch.zeros(n, 3, dtype=torch.long)
        self.obj_shape = torch.zeros(n, 3, dtype=torch.long)
        for i in range(n):
            picks = torch.randperm(len(allowed), generator=self.g)[:3]
            for j, p in enumerate(picks):
                c, s = allowed[int(p)]
                self.obj_color[i, j], self.obj_shape[i, j] = c, s
        xs = torch.tensor([-0.62, 0.0, 0.62])
        self.obj_pos = xs.view(1, 3, 1).repeat(n, 1, 1)
        self.obj_pos = torch.cat([self.obj_pos, torch.full((n, 3, 1), 0.72)], -1)
        self.target = torch.randint(0, 3, (n,), generator=self.g)

    def reset(self):
        self._sample_scene()
        self.agent = torch.stack([
            (torch.rand(self.n, generator=self.g) * 2 - 1) * 0.25,
            torch.full((self.n,), -0.78)], dim=-1)
        self.t = 0
        return self.observe()

    def target_pos(self):
        return self.obj_pos[torch.arange(self.n), self.target]

    def target_tokens(self):
        """(colour_id, shape_id) of the instructed object."""
        i = torch.arange(self.n)
        return self.obj_color[i, self.target], self.obj_shape[i, self.target]

    def render(self):
        """Small RGB image. Objects drawn by colour channel and shape footprint."""
        img = torch.zeros(self.n, 3, GRID, GRID)

        def to_px(xy):
            return ((xy + 1) * 0.5 * (GRID - 1)).round().long().clamp(0, GRID - 1)

        # obstacle in all channels, so it reads as a wall rather than an object
        ox0 = to_px(torch.tensor([-OBSTACLE_HALF_W]))[0]
        ox1 = to_px(torch.tensor([OBSTACLE_HALF_W]))[0]
        oy0 = to_px(torch.tensor([OBSTACLE_Y - OBSTACLE_HALF_H]))[0]
        oy1 = to_px(torch.tensor([OBSTACLE_Y + OBSTACLE_HALF_H]))[0]
        img[:, :, oy0:oy1 + 1, ox0:ox1 + 1] = 0.45

        px = to_px(self.obj_pos)                       # (n, 3, 2)
        for i in range(self.n):
            for j in range(3):
                x, y = int(px[i, j, 0]), int(px[i, j, 1])
                c, s = int(self.obj_color[i, j]), int(self.obj_shape[i, j])
                r = 1 if s == 0 else 2                  # square small, others wider
                y0, y1 = max(0, y - r), min(GRID, y + r + 1)
                x0, x1 = max(0, x - r), min(GRID, x + r + 1)
                img[i, c, y0:y1, x0:x1] = 1.0
                if s == 2:                              # triangle: notch a corner
                    img[i, c, y0, x0] = 0.0
            ax, ay = to_px(self.agent[i])
            img[i, :, int(ay), int(ax)] = 0.85
        return img

    def observe(self):
        """Image, instruction tokens, and proprioception.

        The agent's own position is given directly rather than left to be read
        off the image. A real robot has joint encoders, so this is what a VLA
        actually receives. It also removes a bottleneck that has nothing to do
        with the question being studied: the agent is one pixel at 24x24, and
        three stride-2 convolutions into a global average pool destroy that,
        so without proprioception every head is guessing where it is and the
        comparison measures the encoder rather than the action representation.
        """
        col, shp = self.target_tokens()
        return {"image": self.render(), "color": col, "shape": shp,
                "state": self.agent.clone()}

    def hits_obstacle(self, pos):
        return ((pos[:, 0].abs() < OBSTACLE_HALF_W)
                & ((pos[:, 1] - OBSTACLE_Y).abs() < OBSTACLE_HALF_H))

    def step(self, action):
        a = action.clamp(-1, 1) * MAX_SPEED
        nxt = (self.agent + a).clamp(-1, 1)
        blocked = self.hits_obstacle(nxt)
        self.agent = torch.where(blocked.unsqueeze(1), self.agent, nxt)
        self.t += 1
        dist = (self.agent - self.target_pos()).norm(dim=-1)
        return self.observe(), dist, blocked

    def success(self):
        return (self.agent - self.target_pos()).norm(dim=-1) < SUCCESS_RADIUS
