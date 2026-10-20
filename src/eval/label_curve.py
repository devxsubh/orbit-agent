"""How many labelled photos does each model need? Frozen head vs tiny CNN.

For 5, 10, 20, 50 and 100 training photos per class, on AID and on Sentinel-2 (EuroSAT), train:
- the decision head: frozen SigLIP-base + logistic regression (src/model/head.py)
- the tiny CNN: MobileNetV3-Small, every weight trained (src/model/cnn.py)
Each size is repeated with 3 different random picks of training photos ("seeds"), so the chart
can show how much the result depends on which photos were picked.
The tuning and FINAL photos never change; tuning picks each model's settings, as before.

Run from the orbit-agent folder:
    python -m src.cloud.kaggle_run --command "src.eval.label_curve"   # full run on a Kaggle GPU
    python -m src.eval.label_curve --quick                            # tiny check on the Mac
    python -m src.eval.plot_label_curve                               # chart -> docs/label_curve.png

Saves experiments/label_curve_<date>.csv (one row per dataset, model, size and seed).
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from src.model import cnn
from src.model.head import best_device, calibration_error, embed, fit_head

ROOT = Path(__file__).resolve().parents[2]
SIZES = [5, 10, 20, 50, 100]  # training photos per class


def subset(pool, n: int, seed: int) -> list[int]:
    """Indexes of n photos per class from the training pool, picked by `seed`."""
    by_class: dict[str, list[int]] = {}
    for i, (label, _) in enumerate(pool):
        by_class.setdefault(label, []).append(i)
    picked = []
    for label, idx in sorted(by_class.items()):
        idx = list(idx)
        random.Random(f"curve-{seed}-{label}").shuffle(idx)
        picked += idx[:n]
    return picked


def run_dataset(dataset: str, sizes: list[int], seeds: list[int], epochs: int, device: str):
    import torch

    pool, tuning, final, folder = cnn.photo_sets(dataset, max(sizes))
    classes = sorted({label for label, _ in pool})
    index = {c: i for i, c in enumerate(classes)}
    print(f"\n=== {dataset}: pool {len(pool):,} training photos, {len(tuning)} tuning, {len(final):,} final ===")

    # Head: embeddings of every photo once (frozen model, saved to disk)
    x_pool, _ = embed(pool, "siglip-base", root=folder, dataset=dataset)
    x_tuning, _ = embed(tuning, "siglip-base", root=folder, dataset=dataset)
    x_final, _ = embed(final, "siglip-base", root=folder, dataset=dataset)
    # CNN: all photos as small images once
    print("Loading photos for the CNN...")
    i_pool, i_tuning, i_final = (cnn.load_images(s, folder) for s in (pool, tuning, final))
    y_pool = torch.tensor([index[l] for l, _ in pool])
    y_tuning = torch.tensor([index[l] for l, _ in tuning])
    truths = np.array([l for l, _ in final])

    rows = []
    for n in sizes:
        for seed in seeds:
            pick = subset(pool, n, seed)
            labels = [pool[i][0] for i in pick]

            head = fit_head(x_pool[pick], labels, x_tuning, [l for l, _ in tuning], verbose=False)
            chances = head.predict_proba(x_final)
            right = head.classes_[chances.argmax(1)] == truths
            rows.append(result(dataset, "head (frozen SigLIP-base)", n, seed, right, chances))

            model = cnn.train("mobilenet_v3_small", classes, i_pool[pick], y_pool[pick],
                              i_tuning, y_tuning, epochs, device, verbose=False, seed=seed)
            t = cnn.best_temperature(cnn.logits_of(model, i_tuning, device), y_tuning)
            chances = torch.softmax(cnn.logits_of(model, i_final, device) / float(t), 1).numpy()
            right = np.array(classes)[chances.argmax(1)] == truths
            rows.append(result(dataset, "tiny CNN (MobileNetV3-Small)", n, seed, right, chances))
            print(f"  {n:>3}/class, seed {seed}: head {rows[-2]['accuracy']:.1%}, "
                  f"CNN {rows[-1]['accuracy']:.1%}", flush=True)
    return rows


def result(dataset, model, n, seed, right, chances) -> dict:
    return {"dataset": dataset, "model": model, "per_class": n, "seed": seed,
            "accuracy": round(float(right.mean()), 4),
            "calibration_error": round(calibration_error(chances.max(1), right), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy vs number of labelled photos")
    parser.add_argument("--datasets", nargs="+", default=["aid", "eurosat"], choices=["aid", "eurosat"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30, help="CNN epochs (best on tuning is kept)")
    parser.add_argument("--quick", action="store_true", help="tiny check: 2 photos/class, 1 seed, 1 epoch")
    args = parser.parse_args()

    sizes, seeds, epochs = ([2], [0], 1) if args.quick else (SIZES, list(range(args.seeds)), args.epochs)
    device = best_device()
    rows = []
    for dataset in args.datasets:
        rows += run_dataset(dataset, sizes, seeds, epochs, device)

    out = ROOT / "experiments" / f"label_curve{'_quick' if args.quick else ''}_{datetime.now():%Y%m%d_%H%M}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
