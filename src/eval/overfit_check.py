"""Are we overfitting? Four checks on agent v4's small decision head (frozen SigLIP-base + head).

1. Train vs tuning vs final accuracy: settings tuned to the test photos would show a gap.
2. 95% confidence range (bootstrap): how much the final accuracy could move with other photos.
3. Near-duplicates: does a final test photo have an almost identical twin among the training photos?
4. Fresh test set: photos that NO model, setting or decision ever touched. The heads are trained
   exactly as before and tested once on these.

Run from the orbit-agent folder:
    python -m src.eval.overfit_check        (downloads 900 new AID photos the first time)
"""
from __future__ import annotations

import random
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.agent.decide import POLICIES, ModelView
from src.data.aid import AID_DIR, pick
from src.data.eurosat import CLASSES as EUROSAT_CLASSES, dataset_dir, photos_in, sample
from src.eval.run_agent import FINAL, score
from src.eval.sentinel_test import PHOTO_BYTES, photo_sets as eurosat_sets
from src.model.head import download_all, embed, fit_head
from src.orbit.contact_windows import load_satellite
from src.orbit.replay import build_replay

ROOT = Path(__file__).resolve().parents[2]
TWIN = 0.95  # cosine similarity above which two photos count as near-duplicates


def eurosat_fresh(per_class: int = 100, seed: int = 42):
    """EuroSAT photos after every set used so far (same shuffle as src/eval/sentinel_test.py)."""
    gemma = {p.name for p, _ in sample(10, seed=seed)}
    fresh = []
    for label in EUROSAT_CLASSES:
        names = sorted(p.name for p in photos_in(dataset_dir() / label))
        random.Random(f"{seed}-{label}").shuffle(names)
        rest = [n for n in names if n not in gemma]
        fresh += [(label, n) for n in rest[240:240 + per_class]]  # 0-89 final, 90-139 tuning, 140-239 train
    return fresh


def bootstrap(right: np.ndarray, times: int = 2000) -> tuple[float, float]:
    """95% range of accuracy if we had drawn other photos like these."""
    rng = np.random.default_rng(0)
    means = [right[rng.integers(0, len(right), len(right))].mean() for _ in range(times)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def nearest(x_from: np.ndarray, x_to: np.ndarray) -> np.ndarray:
    """For each row of x_from: cosine similarity to its closest row in x_to (rows are unit length)."""
    return (x_from @ x_to.T).max(1)


def check(name, train, tuning, final, fresh, folder, dataset):
    print(f"\n==================== {name} ====================")
    x_train, _ = embed(train, "siglip-base", root=folder, dataset=dataset)
    x_tuning, _ = embed(tuning, "siglip-base", root=folder, dataset=dataset)
    x_final, _ = embed(final, "siglip-base", root=folder, dataset=dataset)
    x_fresh, _ = embed(fresh, "siglip-base", root=folder, dataset=dataset)
    head = fit_head(x_train, [l for l, _ in train], x_tuning, [l for l, _ in tuning], verbose=False)

    def right(x, photos):
        return head.predict(x) == np.array([l for l, _ in photos])

    print("1. Accuracy on each set (a big tuning -> final drop would mean overfitting to tuning):")
    results = {}
    for set_name, x, photos in [("training", x_train, train), ("tuning", x_tuning, tuning),
                                ("FINAL", x_final, final), ("FRESH (never used)", x_fresh, fresh)]:
        r = right(x, photos)
        low, high = bootstrap(r)
        results[set_name] = r
        print(f"   {set_name:<19} {r.mean():6.1%}  (95% range {low:.1%} - {high:.1%}, {len(r):,} photos)")

    print(f"3. Near-duplicates (cosine similarity > {TWIN}) between test and training photos:")
    sim_final = nearest(x_final, x_train)
    within = x_train @ x_train.T
    np.fill_diagonal(within, -1)
    print(f"   final photos with a near-twin in training: {(sim_final > TWIN).sum()} of {len(final)}")
    print(f"   typical closest-training similarity: final {np.median(sim_final):.3f}, "
          f"training to other training {np.median(within.max(1)):.3f}")
    keep = sim_final <= TWIN
    print(f"   FINAL accuracy without the photos that have a twin: {results['FINAL'][keep].mean():.1%}")
    top = np.argsort(-sim_final)[:3]
    print("   closest pairs: " + ", ".join(f"{final[i][1]}~{train[int((x_final[i] @ x_train.T).argmax())][1]} "
                                          f"({sim_final[i]:.3f})" for i in top))
    return head, x_fresh


def fresh_replay(head, x_fresh, fresh, folder, photo_bytes: int, name: str) -> None:
    """The replay over Delhi, 7 days x 3 photo orders, with ONLY fresh photos."""
    chances = head.predict_proba(x_fresh)
    views = {n: ModelView("", False, head_guess=head.classes_[c.argmax()], p_head=float(c.max()))
             for (_, n), c in zip(fresh, chances)}
    pool = [(folder / l / n, l) for l, n in fresh]
    sat = load_satellite()
    print(f"4b. Replay with fresh {name} photos (Delhi, 7 days x 3 photo orders): correct / wrong labels")
    for rate in [0.25, 1]:
        cells = {"send all raw": [], "agent v4": []}
        for day in range(7):
            for seed in [42, 43, 44]:
                start = datetime(2026, 10, 19, tzinfo=timezone.utc) + timedelta(days=day)
                windows, _ = build_replay(sat, "delhi", start, 24, pool, 30, rate, seed, photo_bytes=photo_bytes)
                photos = sum(len(w.photos) for w in windows)
                for policy in cells:
                    s = score([c for w in windows for c in POLICIES[policy](w, views)])
                    cells[policy].append((100 * s["correct"] / photos, s["wrong"]))
        print(f"   {rate:g} Mbps: " + " | ".join(
            f"{p}: {statistics.mean(c for c, _ in v):.1f}% / {statistics.mean(w for _, w in v):.0f} wrong"
            for p, v in cells.items()))


def main() -> None:
    import csv

    # AID: same sets as agent v4's small head; fresh = photos 131-160 per class (never used)
    train, tuning, final = pick(20, skip=30), pick(10, skip=10), pick(10, skip=20)
    fresh = pick(30, skip=130)
    download_all(fresh)
    head, x_fresh = check("AID (head trained on 20 photos/class)", train, tuning, final, fresh, AID_DIR, "aid")
    rows = list(csv.DictReader(open(FINAL)))
    gemma = np.array([r["guess"] == r["truth"] for r in rows])
    low, high = bootstrap(gemma)
    print(f"2. For comparison, Gemma 3 on the same final photos: {gemma.mean():.1%} (95% range {low:.1%} - {high:.1%})")
    size = sum((AID_DIR / l / n).stat().st_size for l, n in final) // len(final)
    fresh_replay(head, x_fresh, fresh, AID_DIR, size, "AID")

    # EuroSAT (real Sentinel-2): same sets as before; fresh = the next 100 photos per class
    e_final, e_tuning, e_train = eurosat_sets(100)
    e_fresh = eurosat_fresh()
    head, x_fresh = check("Sentinel-2 / EuroSAT (head trained on 100 photos/class)",
                          e_train, e_tuning, e_final, e_fresh, dataset_dir(), "eurosat")
    fresh_replay(head, x_fresh, e_fresh, dataset_dir(), PHOTO_BYTES, "Sentinel-2")


if __name__ == "__main__":
    main()
