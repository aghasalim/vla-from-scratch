"""The vision language encoder, at the smallest scale that still has the shape.

A real VLA puts a pretrained 1B to 3B model here. This is a small CNN over the
rendered scene plus embeddings for the instruction tokens, fused into one
feature vector. It is not pretrained and it is not web scale, so it cannot test
the semantic generalisation claim that motivates real VLAs, and the README says
so plainly.

What it can do is hold the perception identical across all four action heads, so
the comparison between them is about the action representation and nothing else.
Every head in heads.py sits on this same frozen-shape feature.
"""
from __future__ import annotations

import torch
from torch import nn


class VisionLanguageEncoder(nn.Module):
    def __init__(self, n_colors=3, n_shapes=3, dim=128, img_channels=3, state_dim=2):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(img_channels, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.color_emb = nn.Embedding(n_colors, 32)
        self.shape_emb = nn.Embedding(n_shapes, 32)
        self.state_enc = nn.Sequential(nn.Linear(state_dim, 32), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(64 + 64 + 32, dim), nn.ReLU(),
                                  nn.Linear(dim, dim), nn.ReLU())
        self.dim = dim

    def forward(self, image, color, shape, state):
        v = self.vision(image)
        lang = torch.cat([self.color_emb(color), self.shape_emb(shape)], -1)
        return self.fuse(torch.cat([v, lang, self.state_enc(state)], -1))
