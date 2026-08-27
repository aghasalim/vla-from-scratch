"""Train all four action heads under identical conditions and compare them.

Same demonstrations, same encoder architecture, same optimiser, same number of
gradient steps. The only thing that varies is how the action is represented.

Three measurements:
  success        closed loop rollout success rate on held-in scenes
  generalisation the same on (colour, shape) pairs never seen in training
  latency        seconds per action and the control rate it implies

    .venv/bin/python -m experiments.main
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from vla.backbone import VisionLanguageEncoder
from vla.data import collect
from vla.eval import latency, rollout
from vla.heads import HEADS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# (colour, shape) pairs withheld from training entirely
HELD_OUT = {(0, 1), (1, 2), (2, 0)}
SAMPLE_STEPS = {"diffusion": [50, 10, 4], "flow (pi-0 style)": [5, 2, 1]}


def train_one(name, cls, data, args, seed):
    torch.manual_seed(seed)
    enc = VisionLanguageEncoder()
    head = cls(enc.dim, chunk=args.chunk)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps, pct_start=0.1)
    n = data["action"].shape[0]
    g = torch.Generator().manual_seed(seed + 3)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        i = torch.randint(0, n, (args.batch,), generator=g)
        feat = enc(data["image"][i], data["color"][i], data["shape"][i], data["state"][i])
        loss = head.loss(feat, data["action"][i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
    return enc, head, time.perf_counter() - t0, loss.item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--demos", type=int, default=600)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    rows, sweep = [], []
    started = time.perf_counter()

    for seed in args.seeds:
        data = collect(args.demos, seed=seed, held_out=HELD_OUT, train=True,
                       chunk=args.chunk)
        print(f"seed {seed}: {data['action'].shape[0]} demo transitions, "
              f"scripted success {data['success_rate']:.3f}")
        for name, cls in HEADS.items():
            enc, head, wall, final_loss = train_one(name, cls, data, args, seed)
            held_in = rollout(enc, head, args.eval_n, seed=seed + 500,
                              held_out=HELD_OUT, train=True)
            unseen = rollout(enc, head, args.eval_n, seed=seed + 900,
                             held_out=HELD_OUT, train=False)
            lat = latency(enc, head)
            rows.append({"head": name, "seed": seed, "chunk": args.chunk,
                         "train_s": wall,
                         "final_loss": final_loss,
                         "success": held_in["success"],
                         "blocked_steps": held_in["blocked_steps"],
                         "final_dist": held_in["final_dist"],
                         "unseen_success": unseen["success"], **lat})
            print(f"  {name:18} success {held_in['success']:.3f}  "
                  f"unseen {unseen['success']:.3f}  "
                  f"blocked {held_in['blocked_steps']:5.2f}  "
                  f"{lat['nfe']:2d} nfe  {lat['max_hz']:7.0f} Hz  {wall:4.0f}s")

            for s in SAMPLE_STEPS.get(name, []):
                r = rollout(enc, head, args.eval_n, seed=seed + 500, steps=s,
                            held_out=HELD_OUT, train=True)
                lt = latency(enc, head, steps=s)
                sweep.append({"head": name, "seed": seed, "steps": s,
                              "success": r["success"], **lt})

    for fname, data_rows in (("heads.csv", rows), ("step-sweep.csv", sweep)):
        p = RESULTS / fname
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in data_rows for k in r}))
            w.writeheader()
            w.writerows(data_rows)
        print(f"wrote {p.relative_to(ROOT)} ({len(data_rows)} rows)")
    (RESULTS / "run-meta.json").write_text(json.dumps({
        **vars(args), "held_out_pairs": sorted(HELD_OUT),
        "wall_clock_s": time.perf_counter() - started,
        "torch": torch.__version__, "device": "cpu"}, indent=1))
    print(f"total {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
