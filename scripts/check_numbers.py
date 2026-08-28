"""Fail if a number in README.md no longer matches results/."""
from __future__ import annotations

import collections
import csv
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    heads = list(csv.DictReader((ROOT / "results" / "heads.csv").open()))
    sweep = list(csv.DictReader((ROOT / "results" / "step-sweep.csv").open()))
    body = (ROOT / "README.md").read_text()
    # Detail moved out of the README lives in notes/METHODS.md. A figure quoted
    # there is still a quoted figure and still has to match its source.
    _methods = ROOT / "notes" / "METHODS.md"
    if _methods.exists():
        body += "\n" + _methods.read_text()
    claims, failures = [], []

    by = collections.defaultdict(list)
    for r in heads:
        by[r["head"]].append(r)

    for head, rs in by.items():
        for col, places in (("success", 3), ("unseen_success", 3), ("blocked_steps", 2)):
            v = [float(r[col]) for r in rs]
            claims.append((f"{head} {col}", f"{statistics.median(v):.{places}f}"))
        # the seed ranges the non-overlap argument rests on
        s = [float(r["success"]) for r in rs]
        claims.append((f"{head} success min", f"{min(s):.3f}"))
        claims.append((f"{head} success max", f"{max(s):.3f}"))
        hz = statistics.median(float(r["max_hz"]) for r in rs)
        claims.append((f"{head} max_hz", f"{hz:,.0f}"))

    # Collision ranges: the README quotes regression's own span and then a single
    # envelope across the three multimodal heads, since the non-overlap argument
    # only needs the widest of them. Checking each head's span individually would
    # fail on numbers the text never states.
    reg = [float(r["blocked_steps"]) for r in by["regression"]]
    claims.append(("regression blocked min", f"{min(reg):.2f}"))
    claims.append(("regression blocked max", f"{max(reg):.2f}"))
    mm = [float(r["blocked_steps"]) for h, rs in by.items() if h != "regression" for r in rs]
    claims.append(("multimodal blocked envelope min", f"{min(mm):.2f}"))
    claims.append(("multimodal blocked envelope max", f"{max(mm):.2f}"))

    bs = collections.defaultdict(list)
    for r in sweep:
        bs[(r["head"], int(r["steps"]))].append(r)
    for (head, steps), rs in bs.items():
        claims.append((f"{head} steps={steps} success",
                       f"{statistics.median(float(r['success']) for r in rs):.3f}"))
        claims.append((f"{head} steps={steps} hz",
                       f"{statistics.median(float(r['max_hz']) for r in rs):,.0f}"))

    for label, text in claims:
        if not re.search(r"(?<![\d.,])" + re.escape(text) + r"(?![\d])", body):
            failures.append(f"{label} should read {text}, not found")

    print(f"checked {len(claims)} quoted figures against results/")
    if failures:
        print("\nDRIFT DETECTED:")
        for f in failures[:12]:
            print(f"  - {f}")
        if len(failures) > 12:
            print(f"  ... and {len(failures) - 12} more")
        return 1
    print("no drift")
    # What this does and does not cover, so the green line is not read as more
    # than it is: each figure is recomputed from results/ and looked for in the
    # prose. It cannot catch a wrong number that happens to appear somewhere,
    # it does not check claims written in words (ratios, multiples, ranges),
    # and it does not read notes/LOGBOOK.md.
    print("this checks quoted figures against results/, not claims written in words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
