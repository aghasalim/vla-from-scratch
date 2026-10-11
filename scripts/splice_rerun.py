"""Put one head's rerun into results/heads.csv without touching the other rows.

    python scripts/splice_rerun.py out/bins/heads.csv

Every row of the rerun replaces the (head, seed) row it matches, except for the
timing columns. Those depend on the machine, and throughput is only comparable
between heads timed on the same one, so they keep the values from the main run.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "results" / "heads.csv"
TIMING = {"latency_s", "max_hz", "train_s"}


def main(rerun: str) -> int:
    rows = list(csv.DictReader(TARGET.open(newline="")))
    fields = list(rows[0])
    with open(rerun, newline="") as fh:
        new = {(r["head"], r["seed"]): r for r in csv.DictReader(fh)}
    hit = 0
    for r in rows:
        n = new.get((r["head"], r["seed"]))
        if n:
            r.update({k: v for k, v in n.items() if k not in TIMING})
            hit += 1
    assert hit == len(new), f"{len(new) - hit} rerun rows matched nothing"
    with TARGET.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"replaced {hit} rows in {TARGET.name}, timing columns kept")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
