"""Experiment: the agent vs. simple policies, on the photo replay.

Run from the orbit-agent folder:
    python -m src.eval.run_agent
    python -m src.eval.run_agent --rates 0.5 1 2

Uses saved model answers (no model calls, so it takes seconds). Default, best first:
- the FINAL test photos (21-30 per class), once that run is complete
- the tuning photos (11-20), used to learn agent v3's chances
- the AID baseline on the first 300 photos
Agent v4 runs too if the decision head's answers for the same photos exist
(python -m src.model.head).

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
from src.model.confidence import class_trust, fit_right_chance, is_unsure, right_chance
from src.model.vlm import labels_mentioned
from src.orbit.contact_windows import load_satellite
from src.orbit.replay import build_replay
from src.paths import find, result

ROOT = Path(__file__).resolve().parents[2]


def newest_aid_answers() -> Path:
    """The AID baseline CSV with the most photos (newest if tied)."""
    files = sorted(find("baseline_aid_*.csv"))
    if not files:
        raise SystemExit("No AID answers yet. Run: python -m src.eval.baseline_eurosat --dataset aid")
    return max(files, key=lambda f: (sum(1 for _ in open(f)), f.name))


TUNING = result("confidence_aid_test_gemma3-4b_r3_n10.csv")       # photos 11-20
FINAL = result("confidence_aid_test_gemma3-4b_r3_n10_skip20.csv")  # photos 21-30


def _rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def default_answers() -> Path:
    """Final test photos if that run is complete, else tuning photos, else the first baseline."""
    if FINAL.exists() and len(_rows(FINAL)) >= 300:
        return FINAL
    return TUNING if TUNING.exists() else newest_aid_answers()


def photo_set_name(path: Path) -> str:
    return {FINAL: "FINAL TEST photos 21-30", TUNING: "tuning photos 11-20 (v3 learned here)"}.get(
        path.resolve(), "first 300 photos")


def head_answers(path: Path) -> Path | None:
    """The newest decision-head CSV for the same photo set (agent v4), if there is one."""
    which = {FINAL: "final", TUNING: "tuning"}.get(path.resolve())
    files = sorted(find(f"head_aid_*_{which}.csv"),
                   key=lambda f: f.stat().st_mtime) if which else []
    return files[-1] if files else None


def load_views(path: Path, head: Path | None = None) -> tuple[dict[str, ModelView],
                                                              list[tuple[Path, str]]]:
    """What the model said about each photo, from a baseline or "not sure" test CSV.

    Returns (views by file name, photo pool for the replay).
    """
    # Agent v3's chances: B learned on the first 300 photos, weights learned on tuning photos
    trust = class_trust(newest_aid_answers())
    w_lite = w_full = None
    if TUNING.exists():
        tuning = _rows(TUNING)
        w_lite = fit_right_chance(tuning, trust, use_agreement=False)
        w_full = fit_right_chance(tuning, trust, use_agreement=True)

    views, pool = {}, []
    for row in _rows(path):
        unsure = len(labels_mentioned(row["answer"], LABELS)) != 1
        view = ModelView(row["guess"], unsure, doubt=unsure)
        if w_lite:
            view.p_lite = view.p_full = right_chance(row, trust, w_lite)
        if "agree" in row:  # "not sure" test file: full v2 rule, and v3 with agreement
            repeats = 1 + len(row["other_guesses"].split("|")) if row["other_guesses"] else 1
            view.doubt = is_unsure(unsure, int(row["agree"]), repeats, float(row["margin"]))
            if w_full:
                view.p_full = right_chance(row, trust, w_full)
        views[row["file"]] = view
        pool.append((AID_DIR / row["truth"] / row["file"], row["truth"]))

    if head:  # agent v4: the decision head's label and chance for each photo
        for row in _rows(head):
            if row["file"] in views:
                views[row["file"]].head_guess = row["guess"]
                views[row["file"]].p_head = float(row["p"])
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
    parser.add_argument("--head", help="decision-head CSV for agent v4 (default: newest for "
                                       "the same photos)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tle", help="use a saved TLE file (for repeatable runs)")
    parser.add_argument("--sizes", choices=["equal", "jpeg"], default="equal",
                        help="equal = every photo the average size, like raw satellite tiles "
                             "(fair, default); jpeg = real file sizes")
    args = parser.parse_args()

    answers = Path(args.answers) if args.answers else default_answers()
    head = Path(args.head) if args.head else head_answers(answers)
    views, pool = load_views(answers, head)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    sat = load_satellite(tle_path=args.tle)
    photo_bytes = None
    if args.sizes == "equal":
        photo_bytes = sum(p.stat().st_size for p, _ in pool) // len(pool)

    unsure = sum(v.unsure for v in views.values())
    doubt = sum(v.doubt for v in views.values())
    v3_unsure = sum(v.p_full < 0.5 for v in views.values())
    print(f"\nModel answers: {answers.name} = {photo_set_name(answers)}")
    print(f"{len(views)} photos. Not sure: {unsure} by v1 rule, {doubt} by v2 rule, "
          f"{v3_unsure} by v3 (chance right < 50%).")
    print(f"Agent v4 head: {head.name}" if head else
          "Agent v4 skipped: no decision-head answers for these photos (python -m src.model.head)")
    print(f"Replay: {sat.name} over {args.station.title()}, {args.hours:g} h from {args.start} UTC, "
          f"photo every {args.every} s in daylight. Unsent photos are dropped.")
    print("Photo sizes: " + (f"all {photo_bytes / 1000:.0f} KB (equal, like raw tiles)"
                             if photo_bytes else "real JPEG sizes"))

    rows = []
    for rate in args.rates:
        windows, _ = build_replay(sat, args.station, start, args.hours, pool,
                                  args.every, rate, args.seed, photo_bytes=photo_bytes)
        photos = sum(len(w.photos) for w in windows)
        need = sum(w.waiting_bytes for w in windows) / 1e6
        budget = sum(w.budget_bytes for w in windows) / 1e6
        interesting = sum(p.truth in INTERESTING for w in windows for p in w.photos)
        print(f"\n=== Downlink {rate:g} Mbps: {photos:,} photos ({interesting} interesting), "
              f"raw needs {need:,.0f} MB, passes allow {budget:,.0f} MB ===")
        print(f"{'Policy':<14} {'MB sent':>8} {'Full photos':>12} {'Correct info':>14} "
              f"{'Wrong labels':>13} {'Unsure labels':>14} {'Interesting OK':>15} {'Correct/MB':>11}")

        for name, plan in POLICIES.items():
            if name == "agent v4" and not head:
                continue
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

    which = "final" if answers.resolve() == FINAL else "tuning" if answers.resolve() == TUNING else "first"
    out = result(f"agents_{args.station}_{which}_{args.sizes}_"
                                  f"{datetime.now():%Y%m%d_%H%M}.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
