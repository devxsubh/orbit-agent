"""How much should a satellite check itself? The audit "floor" sweep.

Every photo keeps at least `floor` x the average chance of being sent in full (src/agent/audit.py).
- floor 0: almost only unsure photos are checked (close to agent v4's triage, but still random)
- floor 1: every photo has the same chance (a plain random sample)
A low floor spends the checks where the model is unsure: best when the model is healthy. A high
floor keeps checking photos the model is sure about: needed when the model is confidently wrong.
This sweep measures both sides on the same days and budget (EuroSAT replay, 3 stations x 7 days
x 3 photo orders, 0.25 Mbps), for the healthy head and every failure in src/model/faults.py.

Run from the orbit-agent folder (about 5 min):
    python -m src.eval.audit_floor
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from src.agent.audit import audit
from src.eval.audit_test import SEEDS, eurosat_population, run, summarise
from src.paths import result

ROOT = Path(__file__).resolve().parents[2]
FLOORS = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit floor sweep: efficiency vs safety")
    parser.add_argument("--rate", type=float, default=0.25)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    pop = eurosat_population()
    methods = {f"floor {f:g}": ((lambda P, b, t, r, f=f: audit(P, b, t, r, floor=f)), "difference")
               for f in FLOORS}
    methods.update({f"floor {f:g}, tuned": ((lambda P, b, t, r, f=f: audit(P, b, t, r, floor=f)), "tuned")
                    for f in FLOORS})
    rows = run(pop, "eurosat", ["delhi", "bengaluru", "svalbard"], [args.rate], args.days, SEEDS,
               list(pop.scenarios), methods)
    summary = summarise(rows)
    out = result(f"audit_floor_{datetime.now():%Y%m%d_%H%M}.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n{args.rate:g} Mbps, 3 stations x {args.days} days x {len(SEEDS)} photo orders. "
          "RMSE in points; 'holds' = 90% error bars (scaled+1) that held, all classes / rare classes")
    scen = list(pop.scenarios)
    print(f"{'floor':<16}" + "".join(f"{s[:22]:>24}" for s in scen))
    for m in methods:
        cells = []
        for s in scen:
            mine = [x for x in summary if x["scenario"] == s and x["method"] == m]
            cells.append(f"{np.mean([x['rmse_pts'] for x in mine]):.2f} "
                         f"{np.mean([x['holds_scaled+1'] for x in mine]):.0%}/"
                         f"{np.mean([x['holds_scaled+1_rare'] for x in mine]):.0%}")
        print(f"{m:<16}" + "".join(f"{c:>24}" for c in cells))
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
