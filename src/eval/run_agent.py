"""Experiment: the agent vs. simple policies, on the photo replay.

Run from the orbit-agent folder:
    python -m src.eval.run_agent
    python -m src.eval.run_agent --rates 0.5 1 2

Uses saved model answers (no model calls, so it takes seconds):
- the "not sure" test on the 300 NEW photos, if it has been run (needed for agent v2), else
- the AID baseline on the first 300 photos.

Scoring, per photo:
- full photo sent      -> correct (people on the ground can see what it is)
- label sent, right    -> correct
- label sent, wrong    -> wrong info (worse than nothing: the ground is misled)
- "unsure: X?" label   -> no info (the ground was warned, so not misled)
- dropped              -> nothing
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from src.agent.decide import INTERESTING, POLICIES, Decision, ModelView
from src.data.aid import AID_DIR, LABELS
from src.eval.metrics import useful_per_mb
from src.model.confidence import is_unsure
from src.model.vlm import labels_mentioned
from src.orbit.contact_windows import load_satellite
from src.orbit.replay import build_replay

ROOT = Path(__file__).resolve().parents[2]


def newest_aid_answers() -> Path:
    """The AID baseline CSV with the most photos (newest if tied)."""
    files = sorted((ROOT / "experiments").glob("baseline_aid_*.csv"))
    if not files:
        raise SystemExit("No AID answers yet. Run: python -m src.eval.baseline_eurosat --dataset aid")
    return max(files, key=lambda f: (sum(1 for _ in open(f)), f.name))


def default_answers() -> Path:
    """The full "not sure" test on new photos if it exists, else the first-300 baseline."""
    tests = sorted((ROOT / "experiments").glob("confidence_aid_test_*_n10.csv"))
    return tests[-1] if tests else newest_aid_answers()


def load_views(path: Path) -> tuple[dict[str, ModelView], list[tuple[Path, str]]]:
    """What the model said about each photo, from a baseline or "not sure" test CSV.

    Returns (views by file name, photo pool for the replay).
    """
    views, pool = {}, []
    with open(path) as f:
        for row in csv.DictReader(f):
            unsure = len(labels_mentioned(row["answer"], LABELS)) != 1
            doubt = unsure
            if "agree" in row:  # "not sure" test file: use the full agent v2 rule
                repeats = 1 + len(row["other_guesses"].split("|")) if row["other_guesses"] else 1
                doubt = is_unsure(unsure, int(row["agree"]), repeats, float(row["margin"]))
            views[row["file"]] = ModelView(row["guess"], unsure, doubt)
            pool.append((AID_DIR / row["truth"] / row["file"], row["truth"]))
    return views, pool


def score(choices) -> dict:
    """Add up one policy's results over all its choices."""
    sent = correct = wrong = interesting_ok = full = unsure = 0
    for c in choices:
        sent += c.bytes_sent
        ok = c.decision == Decision.SEND_FULL or (
            c.decision == Decision.DESCRIBE and c.label == c.photo.truth)
        correct += ok
        wrong += c.decision == Decision.DESCRIBE and not ok and c.label != "unknown"
        unsure += c.decision == Decision.DESCRIBE_UNSURE  # no info, not wrong
        interesting_ok += ok and c.photo.truth in INTERESTING
        full += c.decision == Decision.SEND_FULL
    return {"mb": sent / 1e6, "correct": correct, "wrong": wrong, "full": full, "unsure": unsure,
            "interesting_ok": interesting_ok, "per_mb": useful_per_mb(correct, sent)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent vs. simple policies on the photo replay")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4],
                        help="downlink speeds to try, in Mbps")
    parser.add_argument("--station", default="delhi")
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--every", type=int, default=30, help="seconds between photos")
    parser.add_argument("--start", default="2026-10-19T00:00", help="start time in UTC")
    parser.add_argument("--answers", help="CSV with model answers (default: 'not sure' test "
                                          "on new photos if done, else the AID baseline)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tle", help="use a saved TLE file (for repeatable runs)")
    args = parser.parse_args()

    answers = Path(args.answers) if args.answers else default_answers()
    views, pool = load_views(answers)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    sat = load_satellite(tle_path=args.tle)

    unsure = sum(v.unsure for v in views.values())
    doubt = sum(v.doubt for v in views.values())
    print(f"\nModel answers: {answers.name} ({len(views)} photos; not sure: "
          f"{unsure} by v1 rule, {doubt} by v2 rule)")
    print(f"Replay: {sat.name} over {args.station.title()}, {args.hours:g} h from {args.start} UTC, "
          f"photo every {args.every} s in daylight. Unsent photos are dropped.")

    rows = []
    for rate in args.rates:
        windows, _ = build_replay(sat, args.station, start, args.hours, pool,
                                  args.every, rate, args.seed)
        photos = sum(len(w.photos) for w in windows)
        need = sum(w.waiting_bytes for w in windows) / 1e6
        budget = sum(w.budget_bytes for w in windows) / 1e6
        interesting = sum(p.truth in INTERESTING for w in windows for p in w.photos)
        print(f"\n=== Downlink {rate:g} Mbps: {photos:,} photos ({interesting} interesting), "
              f"raw needs {need:,.0f} MB, passes allow {budget:,.0f} MB ===")
        print(f"{'Policy':<14} {'MB sent':>8} {'Full photos':>12} {'Correct info':>14} "
              f"{'Wrong labels':>13} {'Unsure labels':>14} {'Interesting OK':>15} {'Correct/MB':>11}")

        for name, plan in POLICIES.items():
            choices = [c for w in windows for c in plan(w, views)]
            s = score(choices)
            print(f"{name:<14} {s['mb']:>8,.2f} {s['full']:>12,} "
                  f"{s['correct']:>7,} ({s['correct'] / photos:>4.0%}) {s['wrong']:>13,} "
                  f"{s['unsure']:>14,} "
                  f"{s['interesting_ok']:>7,} ({s['interesting_ok'] / interesting:>4.0%}) "
                  f"{s['per_mb']:>11,.1f}")
            rows.append({"rate_mbps": rate, "policy": name, "photos": photos,
                         "mb_sent": round(s["mb"], 2), "full_photos": s["full"],
                         "correct": s["correct"], "wrong_labels": s["wrong"],
                         "unsure_labels": s["unsure"],
                         "interesting_correct": s["interesting_ok"],
                         "interesting_total": interesting,
                         "correct_per_mb": round(s["per_mb"], 2)})

    out = ROOT / "experiments" / f"agents_{args.station}_{datetime.now():%Y%m%d_%H%M}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
