"""Vision encoder, language encoder, and the fusion that conditions the head.

Kept small and identical across every action head, so the comparison in
experiments/ isolates the head. The fusion is FiLM style: the language produces
a per-channel scale and shift applied to the visual features. That is enough to
let the instruction select which object matters, which is the only thing the
language has to do in this task.

There is no pretrained vision-language backbone here, which is the honest
limitation of this repo: the real VLA claim is that web-scale pretraining brings
semantic generalisation, and a model trained only on this task's demonstrations
cannot test that. What it *can* test is the action head comparison and whether
the language pathway generalises over phrasing, both of which are architecture
questions rather than pretraining questions.
"""
from __future__ import annotations

import torch
from torch import nn

from .heads import HEADS
from .language import MAX_LEN, VOCAB


class VisionEncoder(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Linear(64, dim)
        self.dim = dim

    def forward(self, img):
        h = self.net(img)
        return self.out(self.pool(h).flatten(1)), h


class LanguageEncoder(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB) + 1, dim, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(MAX_LEN, dim) * 0.02)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, tokens):
        mask = (tokens != 0).float().unsqueeze(-1)
        h = (self.emb(tokens) + self.pos) * mask
        return self.net(h.sum(1) / mask.sum(1).clamp(min=1))


class SpatialSoftmax(nn.Module):
    """Expected (x, y) of each feature channel.

    A reaching policy needs to know *where* the selected object is. Global
    average pooling, which is what this fused with at first, throws exactly that
    away: it says how much of a feature is present and nothing about position.
    The symptom was unmistakable in hindsight, a behaviour cloning loss of 0.0116
    with a closed loop success rate of 10.7%. The policy had learned the average
    action and could not aim.

    Spatial softmax is the standard fix in visuomotor policies: normalise each
    channel over the image and take the expectation of the coordinate grid, so
    every channel returns a location rather than an intensity.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(temperature).log())

    def forward(self, maps):
        b, c, h, w = maps.shape
        flat = (maps.reshape(b, c, h * w) / self.log_t.exp().clamp(min=1e-3)).softmax(-1)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=maps.device),
                                torch.linspace(-1, 1, w, device=maps.device), indexing="ij")
        x = (flat * xs.reshape(1, 1, -1)).sum(-1)
        y = (flat * ys.reshape(1, 1, -1)).sum(-1)
        return torch.cat([x, y], dim=-1)                 # (b, 2c)


class VLAPolicy(nn.Module):
    def __init__(self, head: str, dim: int = 64, action_dim: int = 2, **head_kw):
        super().__init__()
        self.vision = VisionEncoder(dim)
        self.language = LanguageEncoder(dim)
        self.film = nn.Linear(dim, 2 * 64)              # scale and shift on vision maps
        self.spatial = SpatialSoftmax()
        self.fuse = nn.Sequential(nn.Linear(dim * 2 + 128, 128), nn.SiLU(),
                                  nn.Linear(128, dim))
        self.head = HEADS[head](dim, action_dim, **head_kw)
        self.head_name = self.head.name

    def condition(self, img, tokens):
        v, maps = self.vision(img)
        lang = self.language(tokens)
        scale, shift = self.film(lang).chunk(2, dim=-1)
        modulated = maps * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        keypoints = self.spatial(modulated)              # WHERE the selected features are
        return self.fuse(torch.cat([v, lang, keypoints], dim=-1))

    def loss(self, img, tokens, action):
        return self.head.loss(self.condition(img, tokens), action)

    @torch.no_grad()
    def act(self, img, tokens):
        return self.head.act(self.condition(img, tokens))


class BlindPolicy(VLAPolicy):
    """Identical, but the instruction is zeroed out.

    The control that says whether the language is doing anything. On this task a
    blind policy can only guess which object is meant, so its ceiling is roughly
    one over the number of objects.
    """

    def condition(self, img, tokens):
        return super().condition(img, torch.zeros_like(tokens))
