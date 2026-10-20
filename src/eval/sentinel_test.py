"""Does agent v4's decision head work on REAL Sentinel-2 photos? (EuroSAT: 64x64 px, 10 m per pixel)

Run from the orbit-agent folder:
    python -m src.eval.sentinel_test                 (small head, siglip-base; about 10 min first time)

Three tests:
1. AID head as it is: the head trained on AID (sharp Google Earth photos, 30 classes) labels
   Sentinel-2 photos. Its 30 classes are matched to EuroSAT's 10 (MATCH, fixed before testing).
2. New head on Sentinel-2: same frozen image model, new small classifier trained on EuroSAT photos.
3. Satellite replay with these Sentinel-2 photos: raw vs Gemma agents vs agent v4.

EuroSAT photo sets (each class shuffled the same way every time):
    Gemma's 100 photos (10/class, the 29% baseline)  -> part of the FINAL test
    next 90/class                                    -> rest of the FINAL test (1,000 in total)
    next 50/class                                    -> tuning (picks the classifier setting)
    next 100/class (--train-per-class)               -> training photos for the new head
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.agent.decide import POLICIES, ModelView
from src.data.eurosat import CLASSES, LABELS, dataset_dir, photos_in, sample
from src.eval.run_agent import score
from src.model.head import aid_head, calibration_error, embed, fit_head, save_answers
from src.model.vlm import labels_mentioned
from src.orbit.contact_windows import load_satellite
from src.orbit.replay import build_replay

ROOT = Path(__file__).resolve().parents[2]
GEMMA = ROOT / "experiments" / "baseline_eurosat_gemma3-4b_v1_448px_20261018_1528.csv"  # 29%

# Test 1: AID class -> EuroSAT class. Fixed BEFORE looking at any result.
# EuroSAT "PermanentCrop" (orchards) and "HerbaceousVegetation" have no AID match.
MATCH = {
    "Forest": "Forest",
    "DenseResidential": "Residential", "MediumResidential": "Residential",
    "SparseResidential": "Residential",
    "Industrial": "Industrial",
    "River": "River",
    "Pond": "SeaLake", "Beach": "SeaLake",
    "Farmland": "AnnualCrop",
    "Meadow": "Pasture",
    "Viaduct": "Highway",
}

# Test 3: every photo counts as the same size as in the main AID results (280 KB, a full tile),
# so the downlink is just as tight. (EuroSAT's own JPEGs are only about 3 KB.)
PHOTO_BYTES = 280_000


def photo_sets(train_per_class: int, seed: int = 42):
    """(final, tuning, train) as lists of (label, file name). Final starts with Gemma's photos."""
    root = dataset_dir()
    gemma = {p.name for p, _ in sample(10, seed=seed)}
    final, tuning, train = [], [], []
    for label in CLASSES:
        names = sorted(p.name for p in photos_in(root / label))
        random.Random(f"{seed}-{label}").shuffle(names)
        mine = [n for n in names if n in gemma]
        rest = [n for n in names if n not in gemma]
        final += [(label, n) for n in mine + rest[:90]]
        tuning += [(label, n) for n in rest[90:140]]
        train += [(label, n) for n in rest[140:140 + train_per_class]]
    return final, tuning, train


def test_aid_head(backbone: str, final, x_final, gemma_files: set[str]) -> None:
    """Test 1: the AID head labels Sentinel-2 photos (no retraining)."""
    print("\n=== Test 1: AID head on Sentinel-2 photos, no retraining ===")
    head = aid_head(backbone, train_per_class=20)
    chances = head.predict_proba(x_final)
    targets = sorted(set(MATCH.values()))
    # Chance for each EuroSAT class = sum of the chances of its matching AID classes
    summed = np.stack([chances[:, [i for i, c in enumerate(head.classes_) if MATCH.get(c) == t]].sum(1)
                       for t in targets], axis=1)
    guesses = [targets[i] for i in summed.argmax(1)]
    truths = [label for label, _ in final]
    right = np.array([g == t for g, t in zip(guesses, truths)])
    matchable = np.array([t in targets for t in truths])
    gemma = np.array([n in gemma_files for _, n in final])
    print(f"All 1,000 photos:                    {right.mean():.1%} right "
          f"(random guessing: {1 / len(targets):.0%})")
    print(f"Only the {len(targets)} classes AID has a match for: {right[matchable].mean():.1%}")
    print(f"Gemma's 100 photos:                  {right[gemma].mean():.1%}  (Gemma 3 itself: 29%)")
    print("What the AID head mostly says, per real class:")
    top = head.classes_[chances.argmax(1)]
    for label in CLASSES:
        said = Counter(a for a, t in zip(top, truths) if t == label).most_common(2)
        ok = right[np.array(truths) == label].mean()
        print(f"  {label:<21} {ok:>4.0%} right | says " + ", ".join(f"{a} {n}x" for a, n in said))

    out = ROOT / "experiments" / f"sentinel_test1_aid_head_{backbone}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "truth", "aid_guess", "guess", "correct"])
        for (truth, name), a, g, ok in zip(final, top, guesses, right):
            writer.writerow([name, truth, a, g, bool(ok)])
    print(f"Saved {out.relative_to(ROOT)}")


def test_new_head(backbone, final, x_final, tuning, x_tuning, train, x_train, gemma_files):
    """Test 2: same frozen image model, new small classifier trained on Sentinel-2 photos."""
    print("\n=== Test 2: new head trained on Sentinel-2 photos ===")
    head = fit_head(x_train, [l for l, _ in train], x_tuning, [l for l, _ in tuning])
    chances = head.predict_proba(x_final).max(1)
    right = head.predict(x_final) == np.array([l for l, _ in final])
    gemma = np.array([n in gemma_files for _, n in final])
    print(f"FINAL 1,000 photos:  {right.mean():.1%} right, calibration error "
          f"{calibration_error(chances, right):.3f}")
    print(f"Gemma's 100 photos:  {right[gemma].mean():.1%}  (Gemma 3 itself: 29%)")
    out = ROOT / "experiments" / f"head_eurosat_{backbone}_n{len(train) // len(CLASSES)}_final.csv"
    save_answers(out, final, head, x_final, float("nan"))
    print(f"Saved {out.relative_to(ROOT)}")
    return head


def test_replay(final, head, x_final, rates: list[float]) -> None:
    """Test 3: the satellite replay, now with real Sentinel-2 photos."""
    print("\n=== Test 3: satellite replay with Sentinel-2 photos ===")
    root = dataset_dir()
    chances = head.predict_proba(x_final)
    views = {}
    for (label, name), row in zip(final, chances):
        views[name] = ModelView(guess="", unsure=False, head_guess=head.classes_[row.argmax()],
                                p_head=float(row.max()))
    with open(GEMMA) as f:  # Gemma's answers exist only for its 100 photos
        for r in csv.DictReader(f):
            views[r["file"]].guess = r["guess"]
            views[r["file"]].unsure = len(labels_mentioned(r["answer"], LABELS)) != 1

    sat = load_satellite()
    start = datetime(2026, 10, 19, tzinfo=timezone.utc)  # same day as the main results
    gemma_pool = [(root / l / n, l) for l, n in final if views[n].guess]
    full_pool = [(root / l / n, l) for l, n in final]
    runs = [("Gemma's 100 photos", gemma_pool, ["send all raw", "caption all", "agent v1", "agent v4"]),
            ("all 1,000 final photos", full_pool, ["send all raw", "agent v4"])]

    rows = []
    for pool_name, pool, policies in runs:
        print(f"\nPhotos drawn from {pool_name}. Each counts as {PHOTO_BYTES // 1000} KB. "
              "Cell = correct info / wrong labels.")
        print(f"{'Policy':<14}" + "".join(f"{r:>8g} Mbps" for r in rates))
        results = {p: [] for p in policies}
        for rate in rates:
            windows, _ = build_replay(sat, "delhi", start, 24, pool, 30, rate, 42,
                                      photo_bytes=PHOTO_BYTES)
            photos = sum(len(w.photos) for w in windows)
            for name in policies:
                s = score([c for w in windows for c in POLICIES[name](w, views)])
                results[name].append(f"{s['correct'] / photos:.0%} / {s['wrong']}")
                rows.append({"pool": pool_name, "rate_mbps": rate, "policy": name, "photos": photos,
                             "mb_sent": round(s["mb"], 2), "correct": s["correct"],
                             "wrong_labels": s["wrong"], "unsure_labels": s["unsure"]})
        for name in policies:
            print(f"{name:<14}" + "".join(f"{cell:>13}" for cell in results[name]))

    out = ROOT / "experiments" / f"sentinel_test3_replay_{datetime.now():%Y%m%d_%H%M}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent v4 on real Sentinel-2 photos (EuroSAT)")
    parser.add_argument("--backbone", default="siglip-base")
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4])
    args = parser.parse_args()

    final, tuning, train = photo_sets(args.train_per_class)
    gemma_files = {p.name for p, _ in sample(10, seed=42)}
    print(f"EuroSAT photos: {len(final):,} final test (incl. Gemma's 100), {len(tuning)} tuning, "
          f"{len(train):,} training")
    root = dataset_dir()
    x_final, ms = embed(final, args.backbone, root=root, dataset="eurosat")
    x_tuning, _ = embed(tuning, args.backbone, root=root, dataset="eurosat")
    x_train, _ = embed(train, args.backbone, root=root, dataset="eurosat")
    if not np.isnan(ms):
        print(f"Speed: {ms:.0f} ms per photo")

    test_aid_head(args.backbone, final, x_final, gemma_files)
    head = test_new_head(args.backbone, final, x_final, tuning, x_tuning, train, x_train, gemma_files)
    test_replay(final, head, x_final, args.rates)


if __name__ == "__main__":
    main()
