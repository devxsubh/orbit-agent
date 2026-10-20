"""Closed loop: a new camera with ZERO labelled photos at launch.

Audit photos are not only checks. The ground reads their true class, so every audit photo is also
a new training label. After each pass the ground retrains the 30 KB head on all audit photos so far
and sends it up (about 30 KB: seconds of uplink). The head reaches the satellite during the next
pass, so it first labels the photos waiting for the pass after that (one pass of delay).
Each pass's head learned only from EARLIER passes, so the error bars stay honest: the photos it is
judged on played no part in training it.

Compared over a 7-day mission (Delhi, 3 photo orders), all with the same bytes:
- closed loop: audit + retrain after every pass
- launch head kept, with audits (honest, but the head stays bad)
- launch head kept, with triage (agent v4's rule: no honest error bar)
- head trained before launch on 1,000 labels of THIS camera (best case, usually not available)
- raw random (no onboard AI)

Run from the orbit-agent folder:
    python -m src.eval.audit_loop                 # EuroSAT photos; the AID head at launch
    python -m src.eval.audit_loop --data real     # real Sentinel-2A photos; the EuroSAT head at launch
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from src.agent.audit import POLICIES, audit
from src.eval.audit_test import (EXP, LEVEL, SEEDS, TILE, eurosat_days, eurosat_population,
                                 real_days, real_population)
from src.eval.estimate import combine, estimate, interval

ROOT = Path(__file__).resolve().parents[2]
C = 300      # head setting (no tuning photos exist at launch; 300 was picked on EuroSAT tuning photos)
DELAY = 1    # passes between training a head and using it


def train(x: np.ndarray, y: np.ndarray):
    """The 30 KB head on the audit photos so far (None until two classes have been seen)."""
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y)) < 2:
        return None
    return LogisticRegression(C=C, max_iter=3000).fit(x, y)


def head_chances(head, x: np.ndarray, K: int) -> np.ndarray:
    """The head's chances for all K classes (0 for classes it hasn't seen; equal if no head yet)."""
    if head is None:
        return np.full((len(x), K), 1 / K)
    P = np.zeros((len(x), K))
    P[:, head.classes_] = head.predict_proba(x)
    return P


def mission(pop, data: str, days, seed: int, sat, station: str, rate: float) -> list[dict]:
    """One 7-day mission. Returns one row per day per method."""
    K = len(pop.classes)
    launch, best = pop.scenarios[pop.launch], pop.scenarios["healthy head"]
    fixed = {"launch head + audit": (launch, "audit (floor 0.5)", "difference"),
             "launch head + triage": (launch, "triage (v4)", "count"),
             "head trained before launch + audit": (best, "audit (floor 0.5)", "difference"),
             "raw random": (best, "raw random", "raw")}
    rng = np.random.default_rng(seed)
    heads = [None]                              # heads[t] = head trained after pass t-1
    seen_x, seen_y = [], []
    rows, t = [], 0
    day_iter = (eurosat_days(pop, station, rate, days, seed, sat) if data == "eurosat"
                else real_days(pop, station, rate, days, sat))
    for day, passes in day_iter:
        idx_all = np.concatenate([idx for _, idx in passes])
        truth = pop.Y[idx_all].mean(0)
        results = {name: [] for name in ["closed loop", *fixed]}
        right = []
        for budget, idx in passes:
            head = heads[max(0, len(heads) - 1 - DELAY)]  # the newest head that has reached the satellite
            P = head_chances(head, pop.x[idx], K)
            right += list(P.argmax(1) == pop.truth[idx]) if head is not None else [False] * len(idx)
            sent = audit(P, budget, TILE, rng, floor=0.5)
            results["closed loop"].append(estimate(P, sent, pop.Y[idx][sent.full], "difference", rng))
            seen_x += list(pop.x[idx][sent.full])        # the audit photos become training labels
            seen_y += list(pop.truth[idx][sent.full])
            heads.append(train(np.array(seen_x), np.array(seen_y)) if seen_x else None)
            for name, (P_all, policy, how) in fixed.items():
                s = POLICIES[policy](P_all[idx], budget, TILE, rng)
                results[name].append(estimate(P_all[idx], s, pop.Y[idx][s.full], how, rng))
            t += 1
        sizes = [len(idx) for _, idx in passes]
        for name, parts in results.items():
            e = combine(parts, sizes)
            lo, hi = interval(e, "scaled+1", LEVEL)  # the default error bar (src/eval/estimate.py)
            rows.append({"data": data, "station": station, "rate_mbps": rate, "seed": seed, "day": day,
                         "method": name, "halfwidth_pts": round(100 * float(((hi - lo) / 2).mean()), 3),
                         "holds": round(float(((lo - 1e-9 <= truth) & (truth <= hi + 1e-9)).mean()), 4),
                         "rmse_pts": round(100 * float(np.sqrt(((e.value - truth) ** 2).mean())), 3),
                         "worst_error_pts": round(100 * float(np.abs(e.value - truth).max()), 3),
                         "head_right": round(float(np.mean(right)), 4) if name == "closed loop" else "",
                         "labels_so_far": len(seen_y) if name == "closed loop" else ""})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed loop: a new camera learns from its own audits")
    parser.add_argument("--data", choices=["eurosat", "real"], default="eurosat")
    parser.add_argument("--station", default="delhi")
    parser.add_argument("--rate", type=float, help="downlink in Mbps (default 0.25 eurosat, 0.05 real)")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    from src.orbit.contact_windows import load_satellite

    sat = load_satellite(tle_path=str(ROOT / "data" / "raw" / "tle" / "40697.tle"))
    if args.data == "eurosat":
        pop, days, rate = eurosat_population(), args.days, args.rate or 0.25
    else:
        from src.data import s2_real
        all_days = sorted({r["day"] for r in s2_real.load_chips()})
        pop = real_population(all_days[:-args.days], all_days[-args.days:])
        days, rate = all_days[-args.days:], args.rate or 0.05
    print(f"\nLaunch head: {pop.launch} ({pop.notes[pop.launch]}). "
          f"Best case: head trained before launch ({pop.notes['healthy head']}).")
    print(f"{args.station.title()}, {rate:g} Mbps, {len(SEEDS)} photo orders. Head retrained after every pass, "
          f"used {DELAY} pass later.")

    rows = []
    for seed in SEEDS:
        rows += mission(pop, args.data, days, seed, sat, args.station, rate)

    methods = list(dict.fromkeys(r["method"] for r in rows))
    day_names = list(dict.fromkeys(r["day"] for r in rows))
    print(f"\n90% error bar half-width in percentage points (mean of {len(SEEDS)} photo orders); "
          f"closed loop: head right % and labels so far")
    print(f"{'day':<12}" + "".join(f"{m[:24]:>26}" for m in methods) + f"{'head right':>12}{'labels':>8}")
    for d in day_names:
        mine = [r for r in rows if r["day"] == d]
        cells = [np.mean([r["halfwidth_pts"] for r in mine if r["method"] == m]) for m in methods]
        loop = [r for r in mine if r["method"] == "closed loop"]
        print(f"{d:<12}" + "".join(f"{c:>26.2f}" for c in cells)
              + f"{np.mean([r['head_right'] for r in loop]):>12.0%}{np.mean([r['labels_so_far'] for r in loop]):>8.0f}")
    print("\nOver the whole mission: how often the 90% error bar held / worst error in a day (points)")
    for m in methods:
        mine = [r for r in rows if r["method"] == m]
        print(f"  {m:<36} holds {np.mean([r['holds'] for r in mine]):.0%}   "
              f"worst error {max(r['worst_error_pts'] for r in mine):5.1f}   "
              f"mean ± {np.mean([r['halfwidth_pts'] for r in mine]):.2f}")

    out = EXP / f"audit_loop_{args.data}_{rate:g}mbps_{datetime.now():%Y%m%d_%H%M}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
