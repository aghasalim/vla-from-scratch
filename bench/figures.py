"""Figures from committed CSVs. Nothing is re-run here."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ORDER = ["regression", "discrete bins", "diffusion", "flow (pi-0 style)"]
C = {"regression": "#b2182b", "discrete bins": "#ef8a62",
     "diffusion": "#2166ac", "flow (pi-0 style)": "#1a9850"}


def _bars(ax, t, col, ylabel, title):
    for i, h in enumerate(ORDER):
        s = t[t["head"] == h][col]
        med = s.median()
        ax.bar(i, med, color=C[h], width=0.65)
        ax.errorbar(i, med, yerr=[[med - s.min()], [s.max() - med]],
                    color="#222222", capsize=5, linewidth=1.2)
        ax.text(i, s.max() * 1.04, f"{med:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([h.replace(" (pi-0 style)", "\n(pi-0)") for h in ORDER], fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")


def fig_multimodality(out: Path) -> Path:
    """The headline: what mode averaging costs."""
    t = pd.read_csv(RESULTS / "heads.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))
    _bars(a, t, "blocked_steps", "steps per episode pressed into the obstacle",
          "Mode averaging, measured\nregression drives into the obstacle it should route around")
    _bars(b, t, "success", "success rate",
          "Task success\nbars are min to max over 3 seeds")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_latency(out: Path) -> Path:
    """Control rate against success. Up and to the right is better."""
    t = pd.read_csv(RESULTS / "heads.csv")
    sw = pd.read_csv(RESULTS / "step-sweep.csv")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for h in ORDER:
        s = t[t["head"] == h]
        ax.scatter(s["max_hz"].median(), s["success"].median(), s=190, color=C[h],
                   label=h, zorder=4, edgecolors="white", linewidths=1.5)
    for h in ("diffusion", "flow (pi-0 style)"):
        s = sw[sw["head"] == h].groupby("steps")[["max_hz", "success"]].median().reset_index()
        ax.plot(s["max_hz"], s["success"], color=C[h], linestyle="--", linewidth=1.4,
                alpha=0.8, zorder=3)
        for _, r in s.iterrows():
            ax.annotate(f"{int(r['steps'])}", (r["max_hz"], r["success"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8, color=C[h])
    ax.set_xscale("log")
    ax.set_xlabel("max control rate (Hz, single CPU core, log scale)")
    ax.set_ylabel("success rate")
    ax.set_title("Control rate against success\n"
                 "dashed lines sweep sampling steps; labels are the step count")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_generalisation(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "heads.csv")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    w = 0.36
    for i, h in enumerate(ORDER):
        s = t[t["head"] == h]
        ax.bar(i - w / 2, s["success"].median(), w, color=C[h], label="seen pairs" if i == 0 else None)
        ax.bar(i + w / 2, s["unseen_success"].median(), w, color=C[h], alpha=0.5,
               hatch="//", label="held out pairs" if i == 0 else None)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([h.replace(" (pi-0 style)", "\n(pi-0)") for h in ORDER], fontsize=8.5)
    ax.set_ylabel("success rate")
    ax.set_title("Held out (colour, shape) pairs\n"
                 "every head drops, and none of them was pretrained on anything")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for p in (fig_multimodality(RESULTS / "multimodality.png"),
              fig_latency(RESULTS / "latency.png"),
              fig_generalisation(RESULTS / "generalisation.png")):
        print(f"-> {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
