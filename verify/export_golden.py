"""Write the golden reference files the other-language checks read.

Two things come out of here, both computed by importing the repo's own code so
they cannot drift from it:

golden_routes.csv   noise free demonstrator trajectories for a fixed grid of
                    initial conditions. This pins the environment kernel:
                    waypoint routing, the speed clamp, the workspace clamp and
                    the obstacle block. verify/route.c reimplements that kernel
                    and has to reproduce every column.

demo_stats.json     the lateral action statistic the README's multimodality
                    claim rests on, measured on the documented demonstration
                    sets. verify/demomc reruns the same simulation from its own
                    generator and has to land in the same place.

    python verify/export_golden.py [output directory]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from vla.data import collect, waypoint_route
from vla.envs import MAX_SPEED, OBSTACLE_HALF_H, OBSTACLE_Y, ReachEnv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verify"
HELD_OUT = {(0, 1), (1, 2), (2, 0)}
SEEDS = (0, 1, 2)
DEMOS = 800
HORIZON = 24

# the branch condition inside waypoint_route: while the agent is below this it
# is steering for the detour gap, which is the phase the two modes exist in
DETOUR_Y = OBSTACLE_Y + OBSTACLE_HALF_H + 0.05


def golden_routes() -> list[dict]:
    """Noise free trajectories on a grid of (start x, target x, side)."""
    starts = [-0.25, -0.125, 0.0, 0.125, 0.25]
    targets = [-0.62, 0.0, 0.62]
    sides = [-1.0, 1.0]
    combos = [(a, b, c) for a in starts for b in targets for c in sides]
    n = len(combos)
    env = ReachEnv(n, horizon=HORIZON, seed=0)
    env.agent = torch.tensor([[a, -0.78] for a, _, _ in combos])
    tgt = torch.tensor([[b, 0.72] for _, b, _ in combos])
    side = torch.tensor([c for _, _, c in combos])

    rows = []
    for t in range(HORIZON):
        agent = env.agent.clone()
        wp = waypoint_route(agent, tgt, side)
        d = wp - agent
        a = (d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(-1, 1)
        nxt = (agent + a * MAX_SPEED).clamp(-1, 1)
        blocked = env.hits_obstacle(nxt)
        env.agent = torch.where(blocked.unsqueeze(1), agent, nxt)
        for i in range(n):
            rows.append({
                "case": i, "t": t,
                "start_x": combos[i][0], "target_x": combos[i][1],
                "side": combos[i][2],
                "agent_x": float(agent[i, 0]), "agent_y": float(agent[i, 1]),
                "action_x": float(a[i, 0]), "action_y": float(a[i, 1]),
                "blocked": int(blocked[i]),
                "next_x": float(env.agent[i, 0]), "next_y": float(env.agent[i, 1]),
            })
    return rows


def demo_stats() -> dict:
    """Mean lateral action per mode, on the detour phase, per documented seed."""
    per_seed = {}
    for seed in SEEDS:
        d = collect(DEMOS, horizon=HORIZON, seed=seed, held_out=HELD_OUT, train=True)
        ax = d["action"][:, 0, 0]
        side = d["side"]
        below = d["state"][:, 1] < DETOUR_Y
        # Rounded to six decimals. These are float32 reductions, so the last
        # couple of digits depend on the summation order the platform picks,
        # and a byte comparison of this file across machines would fail on
        # noise a hundred times smaller than anything read off it.
        per_seed[str(seed)] = {
            "transitions": int(ax.shape[0]),
            "detour_transitions": int(below.sum()),
            "left_mean": round(float(ax[below & (side < 0)].mean()), 6),
            "right_mean": round(float(ax[below & (side > 0)].mean()), 6),
            "pooled_mean": round(float(ax[below].mean()), 6),
            "scripted_success": round(float(d["success_rate"]), 6),
        }
    return {"demos": DEMOS, "horizon": HORIZON, "detour_y": DETOUR_Y,
            "seeds": list(SEEDS), "per_seed": per_seed}


def main() -> int:
    global OUT
    if len(sys.argv) > 1:
        OUT = Path(sys.argv[1])
        OUT.mkdir(parents=True, exist_ok=True)
    rows = golden_routes()
    cols = list(rows[0])
    with (OUT / "golden_routes.csv").open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(
                str(r[c]) if isinstance(r[c], int) else f"{r[c]:.12g}" for c in cols) + "\n")
    print(f"wrote {OUT / 'golden_routes.csv'} ({len(rows)} rows)")
    (OUT / "demo_stats.json").write_text(json.dumps(demo_stats(), indent=1) + "\n")
    print(f"wrote {OUT / 'demo_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
