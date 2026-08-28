"""Figures from committed CSVs. Nothing is re-run here.

The one exception is the rollout animation, which needs agent positions and no
CSV holds those. It retrains two heads at the documented seed, records the
paths into results/rollout-traces.npz, and refuses to continue unless the
success and collision numbers it reproduces match the committed heads.csv row
exactly. Once that cache is committed the animation redraws from it in a second
and nothing is retrained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.patheffects import withStroke

from bench.style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ORDER = ["regression", "discrete bins", "diffusion", "flow (pi-0 style)"]
TICKS = ["regression", "discrete\nbins", "diffusion", "flow\n(pi-0)"]

# Red for regression and green for flow are the colours the README argument
# already runs on, so they stay. The other two come from the shared palette.
C = {"regression": PALETTE[1], "discrete bins": PALETTE[3],
     "diffusion": PALETTE[0], "flow (pi-0 style)": PALETTE[2]}
SEED_DOT = {"s": 13, "zorder": 6, "color": "white",
            "edgecolors": "#333333", "linewidths": 0.9}


def _bars(ax, t, col, ylabel, fmt="{:.2f}"):
    """Median bar, min to max whisker, and the three seed values on top of it.

    Three seeds is few enough that hiding them behind a summary would be a
    choice about what not to show, so they are drawn.
    """
    top = t[col].max()
    for i, h in enumerate(ORDER):
        s = t[t["head"] == h][col]
        med = s.median()
        ax.bar(i, med, color=C[h], width=0.62, zorder=2)
        ax.errorbar(i, med, yerr=[[med - s.min()], [s.max() - med]],
                    color="#333333", capsize=5, linewidth=1.2, zorder=5)
        ax.scatter([i] * len(s), s, **SEED_DOT)
        ax.text(i, s.max() + top * 0.07, fmt.format(med), ha="center", fontsize=9.5,
                color="#222222")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(TICKS, fontsize=9.5)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, top * 1.22)
    ax.grid(axis="x", visible=False)


def fig_multimodality(out: Path) -> Path:
    """The headline: what mode averaging costs."""
    t = pd.read_csv(RESULTS / "heads.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 5.0))
    _bars(a, t, "blocked_steps", "steps pressed into the wall, per episode")
    titled(a, "Regression drives into the wall it should route around",
           "median steps per 24 step episode, whiskers span 3 seeds, dots are the seeds")
    _bars(b, t, "success", "success rate", fmt="{:.3f}")
    titled(b, "Only flow separates on success as well",
           "fraction of 256 episodes ending within 0.16 of the named object")
    b.text(0.99, 0.96, "scripted demonstrator: 0.988", transform=b.transAxes,
           ha="right", va="top", fontsize=9, color="#5a5a5a")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_latency(out: Path) -> Path:
    """Control rate against success. Up and to the right is better."""
    t = pd.read_csv(RESULTS / "heads.csv")
    sw = pd.read_csv(RESULTS / "step-sweep.csv")
    fig, ax = plt.subplots(figsize=(10.0, 5.6))

    def spread(x, y, colour, size, z, label=None):
        """Median marker with the min to max bar across the 3 seeds.

        Three seeds is not enough for a smooth curve, so the range is drawn
        rather than smoothed away. Flow at one step is the widest of them.
        """
        ax.errorbar(x.median(), y.median(),
                    yerr=[[y.median() - y.min()], [y.max() - y.median()]],
                    color=colour, elinewidth=1.0, capsize=4, alpha=0.5, zorder=z)
        ax.scatter(x.median(), y.median(), s=size, color=colour, label=label,
                   zorder=z + 1, edgecolors="white", linewidths=1.3)

    for h in ORDER:
        s = t[t["head"] == h]
        default = int(s["nfe"].iloc[0])
        spread(s["max_hz"], s["success"], C[h], 170, 5, label=h)
        line = [(s["max_hz"].median(), s["success"].median(), default)]
        # Only the budgets the default is not, so no config is drawn twice.
        for n, g in sw[sw["head"] == h].groupby("steps"):
            if n == default:
                continue
            spread(g["max_hz"], g["success"], C[h], 42, 3)
            line.append((g["max_hz"].median(), g["success"].median(), n))
        if len(line) == 1:      # regression and discrete bins have no budget to sweep
            continue
        line.sort()
        ax.plot([p[0] for p in line], [p[1] for p in line], color=C[h],
                linestyle="--", linewidth=1.4, alpha=0.8, zorder=3)
        for x, y, n in line:
            ax.annotate(f"{n} step" + ("s" if n > 1 else ""), (x, y),
                        textcoords="offset points", xytext=(12, 0), ha="left",
                        va="center", fontsize=8.5, color=C[h], zorder=6,
                        path_effects=[withStroke(linewidth=3, foreground="white")])
    ax.set_xscale("log")
    ax.set_xlim(1.4e4, 3.6e6)
    ax.set_xlabel("max control rate (Hz, action head only, one CPU core, log scale)")
    ax.set_ylabel("success rate")
    ax.set_ylim(0.175, 0.35)
    titled(ax, "Flow keeps the multimodality at close to regression speed",
           "bars span 3 seeds, dashed lines are a guide across sampling budgets")
    ax.legend(loc="upper left", fontsize=9.5, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_generalisation(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "heads.csv")
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    w = 0.34
    for i, h in enumerate(ORDER):
        s = t[t["head"] == h]
        seen, unseen = s["success"].median(), s["unseen_success"].median()
        ax.bar(i - w / 2, seen, w, color=C[h], zorder=2)
        ax.bar(i + w / 2, unseen, w, color=C[h], alpha=0.42, hatch="///",
               edgecolor=C[h], zorder=2)
        ax.text(i - w / 2, seen + 0.008, f"{seen:.3f}", ha="center", fontsize=9.5)
        ax.text(i + w / 2, unseen + 0.008, f"{unseen:.3f}", ha="center", fontsize=9.5,
                color="#444444")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(TICKS, fontsize=9.5)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 0.335)
    ax.grid(axis="x", visible=False)
    titled(ax, "Every head loses ground on unseen (colour, shape) pairs",
           "medians of 3 seeds, and nothing here was pretrained on anything")
    # Neutral swatches: a red or green key would read as a head, not a split.
    ax.legend(handles=[Patch(facecolor="#777777", label="pairs seen in training"),
                       Patch(facecolor="#cccccc", hatch="///", edgecolor="#777777",
                             label="pairs held out entirely")],
              loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# --- the animation ----------------------------------------------------------
SEED = 0
CACHE = "rollout-traces.npz"


def _traces() -> dict:
    """Agent paths for two heads at seed 0. Cached, because training is slow.

    No CSV holds positions, so this is the one figure that has to call the
    training code. It reproduces the committed seed 0 row before it is allowed
    to draw anything, which is what stops the animation from drifting away from
    the table it sits next to.
    """
    cache = RESULTS / CACHE
    if cache.exists():
        return dict(np.load(cache))

    # imported here, not at module scope, so redrawing from the cache stays light
    from experiments.main import HELD_OUT, train_one
    from vla.data import collect
    from vla.eval import rollout
    from vla.heads import HEADS

    meta = json.loads((RESULTS / "run-meta.json").read_text())
    args = argparse.Namespace(**{k: meta[k] for k in
                                 ("demos", "steps", "batch", "lr", "chunk", "eval_n")})
    print(f"no {CACHE}, retraining 2 heads at seed {SEED} (a few minutes)")
    committed = pd.read_csv(RESULTS / "heads.csv")
    data = collect(args.demos, seed=SEED, held_out=HELD_OUT, train=True,
                   chunk=args.chunk)
    saved = {}
    for key, head in (("reg", "regression"), ("flow", "flow (pi-0 style)")):
        enc, h, _, _ = train_one(head, HEADS[head], data, args, SEED)
        r = rollout(enc, h, args.eval_n, seed=SEED + 500, held_out=HELD_OUT,
                    train=True, trace=True)
        row = committed[(committed["head"] == head) & (committed["seed"] == SEED)]
        for col in ("success", "blocked_steps"):
            got, want = r[col], float(row[col].iloc[0])
            if abs(got - want) > 1e-9:
                raise SystemExit(
                    f"{head} at seed {SEED} gave {col} {got}, but heads.csv says "
                    f"{want}. The animation would not be the run in the table, so "
                    f"it is not being written.")
        saved[f"{key}_path"] = r["path"].numpy().astype("float32")
        saved[f"{key}_pressed"] = r["pressed"].numpy()
        saved[f"{key}_success"] = np.float32(r["success"])
    np.savez_compressed(cache, **saved)
    print(f"-> {cache.relative_to(ROOT)} (matches heads.csv at seed {SEED})")
    return saved


def _stage(ax):
    """The workspace: wall in the middle, three object slots along the top."""
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    ax.add_patch(Rectangle((-0.30, -0.10), 0.60, 0.20, facecolor="#9a9a9a",
                           edgecolor="none", zorder=2))
    for x in (-0.62, 0.0, 0.62):
        ax.add_patch(Circle((x, 0.72), 0.16, facecolor="none", edgecolor="#b0b0b0",
                            linestyle=(0, (3, 3)), linewidth=1.0, zorder=2))
        ax.add_patch(Circle((x, 0.72), 0.045, facecolor="#b0b0b0", edgecolor="none",
                            zorder=2))


def fig_rollout(out: Path) -> Path:
    """256 evaluation episodes per head, played back side by side."""
    tr = _traces()
    heads = [("regression", "reg", "Regression drives into the wall",
              "256 evaluation episodes at seed 0"),
             ("flow (pi-0 style)", "flow", "Flow picks a side and goes around",
              "same scenes, same encoder, only the head changed")]
    n_steps = tr["reg_path"].shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 5.0))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.82, bottom=0.12, wspace=0.06)
    art, clocks = {}, []
    for ax, (head, key, title, sub) in zip(axes, heads):
        _stage(ax)
        p = tr[f"{key}_path"]
        free = ax.scatter(p[0, :, 0], p[0, :, 1], s=11, color=C[head], alpha=0.40,
                          linewidths=0, zorder=4)
        stuck = ax.scatter([], [], s=26, color="#111111", alpha=0.9, linewidths=0,
                           zorder=5)
        note = ax.text(0.5, -0.045, "", transform=ax.transAxes, ha="center", va="top",
                       fontsize=9.3, color="#444444")
        done = ax.text(0.5, -0.115, "", transform=ax.transAxes, ha="center", va="top",
                       fontsize=9.3, color="#444444")
        titled(ax, title, sub)
        clocks.append(ax.text(0.97, 0.96, "", transform=ax.transAxes, ha="right",
                              va="top", fontsize=9.3, color="#5a5a5a"))
        art[key] = (free, stuck, note, done)

    hold = 12
    frames = n_steps * 2 + hold

    def draw(f):
        k = min(f // 2, n_steps - 1)
        for _head, key, _title, _sub in heads:
            free, stuck, note, done = art[key]
            p, pr = tr[f"{key}_path"], tr[f"{key}_pressed"]
            free.set_offsets(p[k])
            blocked = pr[k - 1] if k else np.zeros(p.shape[1], dtype=bool)
            stuck.set_offsets(p[k][blocked] if blocked.any() else np.empty((0, 2)))
            note.set_text(f"pressed into the wall so far: "
                          f"{pr[:k].sum(0).mean():.2f} steps per episode")
            done.set_text("" if k < n_steps - 1 else
                          f"ended on the named object: "
                          f"{float(tr[f'{key}_success']):.3f}")
        for c in clocks:
            c.set_text(f"step {k} of {n_steps - 1}")
        return []

    # dpi is set here rather than left to the style: this is a GIF a reviewer
    # opens on a phone, and 170 dpi would be a few megabytes for no gain.
    FuncAnimation(fig, draw, frames=frames, interval=1000 / 12, blit=False).save(
        out, writer=PillowWriter(fps=12), dpi=100)
    plt.close(fig)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for p in (fig_multimodality(RESULTS / "multimodality.png"),
              fig_latency(RESULTS / "latency.png"),
              fig_generalisation(RESULTS / "generalisation.png"),
              fig_rollout(RESULTS / "rollout.gif")):
        print(f"-> {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
