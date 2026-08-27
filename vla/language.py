"""Instructions, with held out phrasings.

Two generalisation axes, and they test different things:

  unseen objects    a (colour, shape) combination never demonstrated. Tests
                    whether the model composes attributes rather than memorising
                    twelve object identities.
  novel phrasing    a sentence template never seen in training, describing an
                    object it has seen. Tests whether the language pathway
                    generalises over surface form.

The second is the one the VLA bet is really about. A policy that memorises
"the string 'reach the red square' means go to that blob" will fail on a new
template even for a familiar object.
"""
from __future__ import annotations

TRAIN_TEMPLATES = [
    "reach the {colour} {shape}",
    "go to the {colour} {shape}",
    "move to the {colour} {shape}",
    "touch the {colour} {shape}",
]

HELD_OUT_TEMPLATES = [
    "navigate toward the {colour} {shape}",
    "please put the gripper on the {colour} {shape}",
    "the {colour} {shape} is your target",
]

VOCAB = sorted({w for t in TRAIN_TEMPLATES + HELD_OUT_TEMPLATES
                for w in t.replace("{colour}", "").replace("{shape}", "").split()}
               | {"red", "green", "blue", "yellow", "square", "circle", "triangle"})
STOI = {w: i + 1 for i, w in enumerate(VOCAB)}      # 0 is padding
MAX_LEN = 10


def render(colour: str, shape: str, template: str) -> str:
    return template.format(colour=colour, shape=shape)


def tokenize(sentences: list[str]):
    import torch
    out = torch.zeros(len(sentences), MAX_LEN, dtype=torch.long)
    for i, s in enumerate(sentences):
        for j, w in enumerate(s.split()[:MAX_LEN]):
            out[i, j] = STOI.get(w, 0)
    return out
