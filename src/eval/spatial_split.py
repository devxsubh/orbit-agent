"""Sentinel-2 (EuroSAT) accuracy with a test split by LOCATION, to close the "neighbouring tile" leak.

The overfitting check found 723 of 1,000 EuroSAT final photos have a near-twin in training
(src/eval/overfit_check.py); without them the head drops from 92.7% to 88.4%. EuroSAT cuts many
tiles from the same scenes, so a random split puts neighbours on both sides.

Fix: put every photo in a 10 km x 10 km block (from its real coordinates, src/data/eurosat_geo.py)
and give WHOLE blocks to training, tuning or test. Same sizes as before (test 100/class, tuning
50/class, training 100/class), same frozen SigLIP-base + small head, 3 random block assignments.
A stricter 50 km version is a check. Both sizes were fixed before looking at any result.

Run from the orbit-agent folder (needs python -m src.data.eurosat_geo first; makes ~4,000 new
embeddings the first time, a few minutes on the Mac):
    python -m src.eval.spatial_split

Saves experiments/spatial_split_eurosat_<time>.csv (one row per block size x seed) and
experiments/head_eurosat_siglip-base_n100_spatial_final.csv (seed 42, 10 km: answers in head format).
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data.eurosat import CLASSES, dataset_dir
from src.data.eurosat_geo import OUT as COORDS
from src.eval.overfit_check import TWIN, bootstrap, nearest
from src.model.head import calibration_error, embed, fit_head, save_answers

ROOT = Path(__file__).resolve().parents[2]
SIZES = {"final": 100, "tuning": 50, "train": 100}  # photos per class, as in src/eval/sentinel_test.py
SHARE = {"final": 0.4, "tuning": 0.2, "train": 0.4}  # share of blocks each set gets
BLOCK_KM = [10, 50]
SEEDS = [42, 43, 44]


def load_coords() -> list[dict]:
    if not COORDS.exists():
        raise SystemExit("No coordinates yet: run python -m src.data.eurosat_geo first")
    return list(csv.DictReader(open(COORDS)))


def block_of(row: dict, km: int) -> tuple:
    """The block a photo is in: its map zone (UTM) and which km x km square of it."""
    size = km * 1000
    return row["epsg"], int(float(row["x"]) // size), int(float(row["y"]) // size)


def split(rows: list[dict], km: int, seed: int):
    """(final, tuning, train) as (label, file name) lists; no block is in two sets."""
    blocks = sorted({block_of(r, km) for r in rows})
    random.Random(seed).shuffle(blocks)
    cut1 = int(len(blocks) * SHARE["final"])
    cut2 = cut1 + int(len(blocks) * SHARE["tuning"])
    side = {b: ("final" if i < cut1 else "tuning" if i < cut2 else "train") for i, b in enumerate(blocks)}
    pools = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["file"]):
        pools[(side[block_of(r, km)], r["label"])].append(r["file"])
    rng = random.Random(f"{seed}-{km}")
    sets = {}
    for name, per_class in SIZES.items():
        sets[name] = []
        for label in CLASSES:
            files = pools[(name, label)]
            if len(files) < per_class:
                raise SystemExit(f"{km} km blocks: only {len(files)} {label} photos for {name}")
            sets[name] += [(label, f) for f in rng.sample(files, per_class)]
    return sets["final"], sets["tuning"], sets["train"], len(blocks)


def main() -> None:
    rows = load_coords()
    root = dataset_dir()
    print(f"{len(rows):,} EuroSAT photos with coordinates")
    results, saved = [], False
    for km in BLOCK_KM:
        for seed in SEEDS:
            final, tuning, train, n_blocks = split(rows, km, seed)
            x = {name: embed(photos, "siglip-base", root=root, dataset="eurosat")[0]
                 for name, photos in (("final", final), ("tuning", tuning), ("train", train))}
            head = fit_head(x["train"], [l for l, _ in train], x["tuning"], [l for l, _ in tuning],
                            verbose=False)
            right = head.predict(x["final"]) == np.array([l for l, _ in final])
            chances = head.predict_proba(x["final"]).max(1)
            low, high = bootstrap(right)
            twins = nearest(x["final"], x["train"]) > TWIN
            results.append({"block_km": km, "seed": seed, "blocks": n_blocks,
                            "accuracy": round(right.mean(), 4), "low_95": round(low, 4),
                            "high_95": round(high, 4),
                            "calibration_error": round(calibration_error(chances, right), 4),
                            "final_with_twin": int(twins.sum()),
                            "accuracy_without_twins": round(right[~twins].mean(), 4)})
            r = results[-1]
            print(f"{km:>3} km blocks, seed {seed}: {r['accuracy']:.1%} ({low:.1%}-{high:.1%}), "
                  f"calibration error {r['calibration_error']:.3f}, near-twins in training "
                  f"{r['final_with_twin']}/1000 (without them {r['accuracy_without_twins']:.1%})")
            if km == BLOCK_KM[0] and not saved:
                out = ROOT / "experiments" / "head_eurosat_siglip-base_n100_spatial_final.csv"
                save_answers(out, final, head, x["final"], float("nan"))
                saved = True

    print("\nMean over seeds (random split, same sizes: 92.7%, 723/1000 with a near-twin, 88.4% without):")
    for km in BLOCK_KM:
        mine = [r for r in results if r["block_km"] == km]
        acc = [r["accuracy"] for r in mine]
        print(f"  {km:>3} km blocks: {np.mean(acc):.1%} ± {np.std(acc):.1%} "
              f"(near-twins {np.mean([r['final_with_twin'] for r in mine]):.0f}/1000, "
              f"calibration error {np.mean([r['calibration_error'] for r in mine]):.3f})")
    out = ROOT / "experiments" / f"spatial_split_eurosat_{datetime.now():%Y%m%d_%H%M}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
