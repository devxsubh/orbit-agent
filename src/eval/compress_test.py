"""Experiment: when does onboard AI beat plain compression? The R sweep.

R = data the camera takes / data the passes can send. The paper's replay sits at R = 1.7-6.7.
For each R (Delhi, 7 days x 3 photo orders, AID FINAL photos, 280 KB each, same bytes for all):

- raw random: full photos at random until the pass is full
- compress to fit: every photo re-saved smaller (the JPEG ladder below) so the whole pass fits.
  On the ground, the same small head labels the shrunk photos. If even the smallest readable size
  doesn't fit, a random part of the photos goes down at that size.
- triage (agent v4) and audit (floor 0.5): onboard labels + full photos (src/agent/audit.py)
- audit with thumbnails: the same audit, but each checked photo goes down as a 160 px thumbnail
  (6.4 KB, enough for a person to read the class) instead of the full 280 KB photo: ~45x more checks

Two scores:
1. Per photo (the paper's score): a photo counts if its label reached the ground right, or the
   photo came down. "strict": a shrunk photo only counts if the ground's model labels it right.
2. The day's class shares with an honest 90% error bar ("scaled" bars, src/eval/estimate.py):
   raw random; compress-to-fit + a person checking as many shrunk photos as the thumbnail audit
   sends (same bytes AND same human effort); the audit with full photos; the thumbnail audit.
ASSUMPTION (favours compression): a person can tell the class of any photo shrunk to >= 160 px.

Run from the orbit-agent folder (about 3 min the first time: 2,700 shrunk photos to embed):
    python -m src.eval.compress_test
"""
from __future__ import annotations

import argparse
import csv
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from src.agent.audit import POLICIES, Sent, label_bytes, n_full
from src.eval.estimate import Estimate, combine, estimate, interval
from src.paths import result

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "processed" / "compress_aid_siglip-base.npz"
AID_TILE = 280_000  # the average AID photo (600 x 600 px JPEG), as in the paper's replay
FIRST_DAY = datetime(2026, 10, 19, tzinfo=timezone.utc)
SEEDS = [42, 43, 44]
RATES = [1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025]
LADDER = [("original", None, None), ("600px q50", 600, 50), ("600px q20", 600, 20),
          ("448px q75", 448, 75), ("336px q75", 336, 75), ("224px q90", 224, 90),
          ("224px q75", 224, 75), ("224px q40", 224, 40), ("160px q75", 160, 75),
          ("112px q75", 112, 75)]
READABLE_PX = 160


def shrink(path: Path, side, quality) -> tuple[Image.Image, int]:
    """(decoded photo, bytes) after resizing to side x side and saving as JPEG."""
    if side is None:
        raw = path.read_bytes()
    else:
        buf = io.BytesIO()
        Image.open(path).convert("RGB").resize((side, side), Image.BICUBIC).save(buf, "JPEG",
                                                                                 quality=quality)
        raw = buf.getvalue()
    return Image.open(io.BytesIO(raw)).convert("RGB"), len(raw)


def ladder(final) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For each ladder step: (bytes per photo, embedding per photo). Saved after the first run."""
    from src.data.aid import AID_DIR
    from src.model.head import embed_images

    saved = dict(np.load(CACHE)) if CACHE.exists() else {}
    for label, side, quality in LADDER:
        if f"{label}|x" in saved:
            continue
        print(f"Shrinking and embedding {len(final)} photos: {label}...")
        pairs = [shrink(AID_DIR / cls / n, side, quality) for cls, n in final]
        saved[f"{label}|bytes"] = np.array([b for _, b in pairs])
        saved[f"{label}|x"] = embed_images([img for img, _ in pairs])
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, **saved)
    return {label: (saved[f"{label}|bytes"], saved[f"{label}|x"]) for label, _, _ in LADDER}


def two_phase(P_got: np.ndarray, Y_checked: np.ndarray, checked: np.ndarray, N: int) -> Estimate:
    """Class shares when a random part of the photos came down (shrunk, labelled by the ground's
    model) and a person checked a random part of those: model average + checked mistakes.
    P_got: the ground model's chances for the photos that came down; checked: positions in P_got."""
    n1, h = len(P_got), len(checked)
    K = P_got.shape[1]
    if h < 2:
        return Estimate(P_got.mean(0), np.full(K, np.inf), np.full(K, np.inf), True, h, photos=N)
    d = Y_checked - P_got[checked]
    value = P_got.mean(0) + d.mean(0)
    var = (1 - n1 / N) * Y_checked.var(0, ddof=1) / n1 + (1 - h / n1) * d.var(0, ddof=1) / h
    var_model = (1 - n1 / N) * np.maximum(Y_checked.mean(0) * (1 - Y_checked.mean(0)), 1 / (h + 2)) / n1 \
        + (1 - h / n1) * (P_got * (1 - P_got)).mean(0) / h
    return Estimate(value, var, var_model, True, h, photos=N)


def run(rates, days: int, station: str) -> list[dict]:
    from src.data.aid import pick
    from src.model.faults import chances
    from src.model.head import aid_head, embed
    from src.orbit.contact_windows import load_satellite
    from src.orbit.replay import build_replay

    final = pick(10, skip=20)
    head = aid_head("siglip-base", 20)
    classes = list(head.classes_)
    K = len(classes)
    truth = np.array([classes.index(label) for label, _ in final])
    Y = np.eye(K)[truth]
    P_onboard = chances(head, embed(final, "siglip-base")[0])
    steps = ladder(final)
    ground = {s: chances(head, x) for s, (_, x) in steps.items()}
    right = {s: P.argmax(1) == truth for s, P in ground.items()}
    p95 = {s: np.percentile(b, 95) for s, (b, _) in steps.items()}
    mean_bytes = {s: b.mean() for s, (b, _) in steps.items()}
    readable = [s for s, side, _ in LADDER if side is None or side >= READABLE_PX]
    smallest = min(readable, key=lambda s: mean_bytes[s])
    print("\nLadder (photo size and how often the ground's head is right on the shrunk photo):")
    for s, _, _ in LADDER:
        print(f"  {s:<12} {mean_bytes[s] / 1e3:>7.1f} KB  {right[s].mean():6.1%}"
              + ("" if s in readable else "  (too small for a person to read)"))

    index = {n: i for i, (_, n) in enumerate(final)}
    pool = [(Path(n), cls) for cls, n in final]
    sat = load_satellite(tle_path=str(ROOT / "data" / "raw" / "tle" / "40697.tle"))
    rows = []
    for rate in rates:
        for seed in SEEDS:
            rng = np.random.default_rng([seed, int(rate * 10_000)])
            for day in range(days):
                start = FIRST_DAY + timedelta(days=day)
                windows, _ = build_replay(sat, station, start, 24, pool, 30, rate, seed + 1000 * day,
                                          photo_bytes=AID_TILE)
                passes = [(w.budget_bytes, np.array([index[p.path.name] for p in w.photos]))
                          for w in windows if w.photos]
                idx_all = np.concatenate([i for _, i in passes])
                N_day, truth_day = len(idx_all), Y[idx_all].mean(0)
                raw_bytes = N_day * AID_TILE
                budget_day = sum(b for b, _ in passes)
                score = {m: {"strict": 0, "lenient": 0, "wrong": 0, "parts": []} for m in
                         ["raw random", "compress to fit", "triage (v4)", "audit (floor 0.5)",
                          "audit, thumbnails"]}
                for budget, idx in passes:
                    N = len(idx)
                    # compress to fit: best readable step that fits every photo, else a random part
                    fits = [s for s in readable if p95[s] * N <= budget]
                    if fits:
                        step = max(fits, key=lambda s: right[s].mean())
                        got = np.arange(N)
                    else:
                        step = smallest
                        got = np.sort(rng.choice(N, min(N, int(budget // mean_bytes[step])), replace=False))
                    sc = score["compress to fit"]
                    sc["strict"] += right[step][idx[got]].sum()
                    sc["lenient"] += len(got)
                    # same human effort as the thumbnail audit: a person checks as many photos as it sends
                    h = n_full(budget, N, int(mean_bytes[smallest]), N * label_bytes(K))
                    checked = np.sort(rng.choice(len(got), min(h, len(got)), replace=False)) if len(got) else np.array([], int)
                    sc["parts"].append(two_phase(ground[step][idx[got]], Y[idx[got]][checked], checked, N))
                    # onboard policies
                    for name in ["raw random", "triage (v4)", "audit (floor 0.5)", "audit, thumbnails"]:
                        policy = "audit (floor 0.5)" if name == "audit, thumbnails" else name
                        tile = int(mean_bytes[smallest]) if name == "audit, thumbnails" else AID_TILE
                        sent = POLICIES[policy](P_onboard[idx], budget, tile, rng)
                        sc = score[name]
                        full = np.zeros(N, bool)
                        full[sent.full] = True
                        sc["strict"] += full.sum()
                        sc["lenient"] += full.sum()
                        if sent.labels:
                            P = P_onboard[idx]
                            plain = (~full) & (P.max(1) >= 0.5)  # labels under 50% sure go as "unsure"
                            ok = P.argmax(1) == truth[idx]
                            sc["strict"] += (plain & ok).sum()
                            sc["lenient"] += (plain & ok).sum()
                            sc["wrong"] += (plain & ~ok).sum()
                        how = {"raw random": "raw", "triage (v4)": "count", "audit (floor 0.5)": "difference",
                               "audit, thumbnails": "difference"}[name]
                        sc["parts"].append(estimate(P_onboard[idx], sent, Y[idx][sent.full], how, rng))
                sizes = [len(i) for _, i in passes]
                for name, sc in score.items():
                    e = combine(sc["parts"], sizes)
                    lo, hi = interval(e, "scaled")
                    rows.append({"rate_mbps": rate, "R": round(raw_bytes / budget_day, 1), "seed": seed,
                                 "day": f"{start:%Y-%m-%d}", "method": name, "photos": N_day,
                                 "strict_pct": round(100 * sc["strict"] / N_day, 2),
                                 "lenient_pct": round(100 * sc["lenient"] / N_day, 2),
                                 "wrong_labels": int(sc["wrong"]),
                                 "halfwidth_pts": round(100 * float(((hi - lo) / 2).mean()), 3),
                                 "holds": round(float(((lo - 1e-9 <= truth_day) & (truth_day <= hi + 1e-9)).mean()), 4),
                                 "rmse_pts": round(100 * float(np.sqrt(((e.value - truth_day) ** 2).mean())), 3)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="R sweep: onboard AI vs plain compression")
    parser.add_argument("--rates", type=float, nargs="+", default=RATES)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--station", default="delhi")
    args = parser.parse_args()

    rows = run(args.rates, args.days, args.station)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    print(f"\nMean of {len(SEEDS) * args.days} days per cell ({args.station.title()}). "
          "Per photo: strict % / with shrunk photos counted as seen; class shares: ± points (holds %)")
    print(f"{'R':>7} {'Mbps':>7}" + "".join(f"{m:>30}" for m in methods))
    for rate in args.rates:
        mine = [r for r in rows if r["rate_mbps"] == rate]
        R = np.mean([r["R"] for r in mine])
        cells = []
        for m in methods:
            rs = [r for r in mine if r["method"] == m]
            cells.append(f"{np.mean([r['strict_pct'] for r in rs]):5.1f}/{np.mean([r['lenient_pct'] for r in rs]):5.1f}  "
                         f"±{np.mean([r['halfwidth_pts'] for r in rs]):4.1f} ({np.mean([r['holds'] for r in rs]):.0%})")
        print(f"{R:>7.1f} {rate:>7g}" + "".join(f"{c:>30}" for c in cells))
    out = result(f"compress_sweep_{datetime.now():%Y%m%d_%H%M}.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
