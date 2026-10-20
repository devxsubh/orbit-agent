"""Markdown tables for the audited-downlink results, straight from the newest CSVs.

Run from the orbit-agent folder (no models, a second):
    python -m src.eval.audit_tables
    python -m src.eval.audit_tables --rate 0.1

Prints, ready to paste into docs/experiment_log.md or the paper:
1. Can the ground trust the numbers? (EuroSAT replay, and the real Sentinel-2A replay if run)
2. How much should a satellite check itself? (floor sweep)
3. When does onboard AI beat compression? (R sweep)
4. A new camera learns from its own audits (closed loop)
Each cell: typical error (RMSE, percentage points) / how often the 90% error bar held ("scaled+1" bars;
labels only and triage only have a "pretend every label is right" bar).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MAIN = ["labels only", "triage (v4)", "raw random", "stratified", "audit, floor 0.5"]


def newest(pattern: str) -> Path | None:
    files = sorted(glob.glob(str(ROOT / "experiments" / pattern)), key=os.path.getmtime)
    return Path(files[-1]) if files else None


def rows_of(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def mean(rows, key) -> float:
    return float(np.mean([float(r[key]) for r in rows]))


def honesty(path: Path, rate: float) -> None:
    rows = [r for r in rows_of(path) if float(r["rate_mbps"]) == rate]
    if not rows:
        rates = sorted({float(r["rate_mbps"]) for r in rows_of(path)})
        print(f"(no rows at {rate:g} Mbps in {path.name}; rates there: {rates})\n")
        return
    stations = sorted({r["station"] for r in rows})
    print(f"**{path.name}**, {rate:g} Mbps, stations: {', '.join(stations)}. "
          "Each cell: typical error (points) / 90% error bar held (all classes, rare classes)\n")
    methods = [m for m in MAIN if any(r["method"] == m for r in rows)]
    print("| Onboard model | " + " | ".join(methods) + " |")
    print("|---" * (len(methods) + 1) + "|")
    for scen in dict.fromkeys(r["scenario"] for r in rows):
        cells = []
        for m in methods:
            mine = [r for r in rows if r["scenario"] == scen and r["method"] == m]
            rare = [float(r["holds_scaled+1_rare"]) for r in mine if r["holds_scaled+1_rare"] != ""]
            rare_txt = f", {100 * np.mean(rare):.0f}%" if rare else ""
            cells.append(f"{mean(mine, 'rmse_pts'):.2f} / {100 * mean(mine, 'holds_scaled+1'):.0f}%{rare_txt}")
        print(f"| {scen} | " + " | ".join(cells) + " |")
    print()


def floors(path: Path) -> None:
    rows = rows_of(path)
    scen = list(dict.fromkeys(r["scenario"] for r in rows))
    print(f"**{path.name}**: typical error (points) / 90% bar held, by floor (difference estimator)\n")
    print("| Floor | " + " | ".join(scen) + " |")
    print("|---" * (len(scen) + 1) + "|")
    for m in dict.fromkeys(r["method"] for r in rows):
        if "tuned" in m:
            continue
        cells = [f"{mean([r for r in rows if r['method'] == m and r['scenario'] == s], 'rmse_pts'):.2f} / "
                 f"{100 * mean([r for r in rows if r['method'] == m and r['scenario'] == s], 'holds_scaled+1'):.0f}%"
                 for s in scen]
        print(f"| {m.split()[1]} | " + " | ".join(cells) + " |")
    print()


def rsweep(path: Path) -> None:
    rows = rows_of(path)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    print(f"**{path.name}**: % of photos whose class reached the ground right (strict) / "
          "honest 90% error bar on class shares (± points)\n")
    print("| R | Mbps | " + " | ".join(methods) + " |")
    print("|---" * (len(methods) + 2) + "|")
    for rate in sorted({float(r["rate_mbps"]) for r in rows}, reverse=True):
        mine = [r for r in rows if float(r["rate_mbps"]) == rate]
        cells = []
        for m in methods:
            rs = [r for r in mine if r["method"] == m]
            width = mean(rs, "halfwidth_pts")
            cells.append(f"{mean(rs, 'strict_pct'):.1f}% / " + ("exact" if width < 0.005 else f"±{width:.1f}"))
        print(f"| {mean(mine, 'R'):.0f} | {rate:g} | " + " | ".join(cells) + " |")
    print()


def loop(path: Path) -> None:
    rows = rows_of(path)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    print(f"**{path.name}**: per mission day, typical error (points) / honest error bar (±); "
          "closed loop's head right and labels so far\n")
    print("| Day | " + " | ".join(methods) + " | head right | labels |")
    print("|---" * (len(methods) + 3) + "|")
    for i, d in enumerate(dict.fromkeys(r["day"] for r in rows), 1):
        mine = [r for r in rows if r["day"] == d]
        cells = [f"{mean([r for r in mine if r['method'] == m], 'rmse_pts'):.2f} / "
                 f"±{mean([r for r in mine if r['method'] == m], 'halfwidth_pts'):.2f}" for m in methods]
        cl = [r for r in mine if r["method"] == "closed loop"]
        print(f"| {i} | " + " | ".join(cells) + f" | {100 * mean(cl, 'head_right'):.0f}% | "
              f"{mean(cl, 'labels_so_far'):.0f} |")
    print("\nWhole mission, 90% bar held / worst daily error (points):")
    for m in methods:
        mine = [r for r in rows if r["method"] == m]
        print(f"- {m}: {100 * mean(mine, 'holds'):.0f}% / {max(float(r['worst_error_pts']) for r in mine):.1f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Markdown tables for the audited downlink")
    parser.add_argument("--rate", type=float, default=0.25, help="EuroSAT rate for table 1")
    parser.add_argument("--real-rate", type=float, default=0.05, help="real-replay rate for table 1b")
    args = parser.parse_args()

    print("### 1. Can the ground trust the numbers?\n")
    for pattern, rate in (("audit_summary_eurosat_*.csv", args.rate), ("audit_summary_real_*.csv", args.real_rate)):
        path = newest(pattern)
        if path:
            honesty(path, rate)
    if (path := newest("audit_floor_*.csv")):
        print("### 2. How much should a satellite check itself?\n")
        floors(path)
    if (path := newest("compress_sweep_*.csv")):
        print("### 3. When does onboard AI beat plain compression?\n")
        rsweep(path)
    for data in ("eurosat", "real"):
        if (path := newest(f"audit_loop_{data}_*.csv")):
            print(f"### 4. Closed loop ({data})\n")
            loop(path)


if __name__ == "__main__":
    main()
