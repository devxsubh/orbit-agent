"""Which "not sure" signal catches the model's wrong answers best?

Run from the orbit-agent folder:
    python -m src.eval.confidence_test --per-class 1    (quick test: 30 photos, ~4 min)
    python -m src.eval.confidence_test                  (test set: 300 NEW photos, ~2 hours)
    python -m src.eval.confidence_test --evaluate-only  (redo the table + chart, no model)
    python -m src.eval.confidence_test --skip 20        (final test set: photos 21-30)

Photo sets (per class): 1-10 = first baseline, 11-20 = tuning (--skip 10, default),
21-30 = final test (--skip 20, never looked at while building the agent).
Safe to stop and restart: photos already done are kept in the CSV and skipped.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from tqdm import tqdm

from src.data import aid
from src.eval.run_agent import newest_aid_answers
from src.model.confidence import MARGIN_SURE, check_photo, class_trust, is_unsure
from src.model.vlm import MODEL, check_ready

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["file", "truth", "guess", "correct", "answer", "seconds", "photo_bytes", "label_bytes",
          "several", "margin", "agree", "majority", "other_guesses"]


def run(model: str, per_class: int, repeats: int, out: Path, skip: int = 10) -> None:
    """Ask the model about every test photo; append rows to `out` (skips photos already done)."""
    done = set()
    if out.exists():
        with open(out) as f:
            done = {row["file"] for row in csv.DictReader(f)}
    photos = [(p, t) for p, t in aid.sample(per_class, skip=skip) if p.name not in done]
    if done:
        print(f"{len(done)} photos already done, {len(photos)} to go.")
    if not photos:
        return

    new_file = not out.exists()
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for path, truth in tqdm(photos, desc=f"Asking {repeats}x per photo"):
            r = check_photo(path, aid.LABELS, aid.INTRO, model, repeats=repeats)
            writer.writerow({"file": path.name, "truth": truth, "correct": r["guess"] == truth,
                             "photo_bytes": path.stat().st_size,
                             "label_bytes": len(r["guess"].encode()),
                             **{k: r[k] for k in FIELDS if k in r}})
            f.flush()  # so a stop/crash keeps everything done so far


def rank_quality(trust: list[float], wrong: list[bool]) -> float:
    """AUROC: chance a wrong answer gets LOWER trust than a right one. 0.5 = useless, 1 = perfect."""
    bad = [t for t, w in zip(trust, wrong) if w]
    good = [t for t, w in zip(trust, wrong) if not w]
    if not bad or not good:
        return float("nan")
    wins = sum(1.0 if b < g else 0.5 if b == g else 0.0 for b in bad for g in good)
    return wins / (len(bad) * len(good))


def catch_curve(trust: list[float], wrong: list[bool]) -> list[float]:
    """Share of wrong answers caught when flagging the k least-trusted photos, for k = 0..N.

    Ties are broken in a fixed random order, so a yes/no signal isn't helped by file order.
    """
    order = list(range(len(trust)))
    random.Random(0).shuffle(order)
    order.sort(key=lambda i: trust[i])
    total_wrong = sum(wrong) or 1
    curve, caught = [0.0], 0
    for i in order:
        caught += wrong[i]
        curve.append(caught / total_wrong)
    return curve


def save_chart(curves: dict[str, list[float]], wrong_share: float, out: Path) -> None:
    """Line chart: photos flagged (x) vs wrong answers caught (y), one line per signal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink_2, grid, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2", "#a3a29a"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated, fixed order
    markers = ["o", "s", "^", "D", "v"]  # second cue, so identity isn't color alone

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    n = len(next(iter(curves.values()))) - 1
    xs = [k / n * 100 for k in range(n + 1)]

    ax.plot([0, 100], [0, 100], color=muted, linewidth=1.2, linestyle="--", label="Random flagging")
    perfect = [min(1.0, (k / n) / wrong_share) * 100 for k in range(n + 1)]
    ax.plot(xs, perfect, color=muted, linewidth=1.2, linestyle=":", label="Perfect signal")
    marks_at = [round(n * r) for r in (0.1, 0.2, 0.3)]
    for (name, curve), color, marker in zip(curves.items(), colors, markers):
        ys = [c * 100 for c in curve]
        ax.plot(xs, ys, color=color, linewidth=2, label=name, zorder=3)
        ax.plot([xs[k] for k in marks_at], [ys[k] for k in marks_at], linestyle="none",
                marker=marker, markersize=7, color=color, markeredgecolor=surface,
                markeredgewidth=1.5, zorder=4)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Photos flagged as \"not sure\" (%)", color=ink_2)
    ax.set_ylabel("Wrong answers caught (%)", color=ink_2)
    ax.set_title("Which signal catches the model's mistakes?", color=ink, fontsize=11, loc="left")
    ax.tick_params(colors=ink_2, labelsize=8)
    ax.grid(color=grid, linewidth=0.6, zorder=0)
    for side in ax.spines.values():
        side.set_color(grid)
    legend = ax.legend(loc="lower right", fontsize=8, frameon=True)
    legend.get_frame().set_facecolor(surface)
    legend.get_frame().set_edgecolor(grid)
    for text in legend.get_texts():
        text.set_color(ink)
    fig.tight_layout()
    fig.savefig(out, facecolor=surface)
    plt.close(fig)


def evaluate(out: Path, repeats: int) -> None:
    with open(out) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    wrong = [r["guess"] != r["truth"] for r in rows]
    several = [r["several"] == "True" for r in rows]
    margin = [float(r["margin"]) for r in rows]
    agree = [int(r["agree"]) for r in rows]
    calibration = newest_aid_answers()
    trust_b = class_trust(calibration)

    signals = {
        "A. Several answers": [0.0 if s else 1.0 for s in several],
        "B. Confused classes": [trust_b[r["guess"]] for r in rows],
        "C. Answer margin": margin,
        f"D. Agreement ({repeats} asks)": [float(a) for a in agree],
        "A + D + C combined": [(0 if s else 1) * 1e6 + a * 1e3 + min(m, 999)
                               for s, a, m in zip(several, agree, margin)],
    }

    right = n - sum(wrong)
    majority_right = sum(r["majority"] == r["truth"] for r in rows)
    print(f"\nTest set: {n} new AID photos. First answer right: {right} ({right / n:.0%}), "
          f"wrong: {sum(wrong)}.")
    print(f"Majority vote of {repeats} asks: {majority_right} right ({majority_right / n:.0%}).")
    print(f"Signal B learned from: {calibration.name}\n")

    print(f"{'Signal':<24} {'Ranking (AUROC)':>16}   Wrong answers caught when flagging")
    print(f"{'':<24} {'0.5=useless 1=best':>16}   {'10%':>6} {'20%':>6} {'30%':>6}")
    curves = {}
    for name, trust in signals.items():
        curve = catch_curve(trust, wrong)
        curves[name] = curve
        caught = [curve[round(n * r)] for r in (0.1, 0.2, 0.3)]
        print(f"{name:<24} {rank_quality(trust, wrong):>16.2f}   "
              + " ".join(f"{c:>6.0%}" for c in caught))

    unsure = [is_unsure(s, a, repeats, m) for s, a, m in zip(several, agree, margin)]
    flagged, caught = sum(unsure), sum(u and w for u, w in zip(unsure, wrong))
    print(f"\nAgent v2 rule (A or D or margin < {MARGIN_SURE:g}, fixed before testing): flags "
          f"{flagged} photos ({flagged / n:.0%}), catches {caught} of {sum(wrong)} wrong answers "
          f"({caught / max(sum(wrong), 1):.0%}).")
    print(f"Of the photos it calls sure, {sum(not u and not w for u, w in zip(unsure, wrong))} of "
          f"{n - flagged} are right ({sum(not u and not w for u, w in zip(unsure, wrong)) / max(n - flagged, 1):.0%}).")

    chart = out.with_suffix(".png")
    save_chart(curves, sum(wrong) / n, chart)
    print(f"\nAnswers: {out}\nChart:   {chart}")


def results_path(model: str = MODEL, repeats: int = 3, per_class: int = 10, skip: int = 10) -> Path:
    """Where a run's answers are saved. (Photos 11-20 keep the original file name.)"""
    tag = f"{model.replace(':', '-')}_r{repeats}_n{per_class}" + (f"_skip{skip}" if skip != 10 else "")
    return ROOT / "experiments" / f"confidence_aid_test_{tag}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the 'not sure' signals on new AID photos")
    parser.add_argument("--per-class", type=int, default=10, help="test photos per class")
    parser.add_argument("--repeats", type=int, default=3, help="asks per photo (signal D)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--skip", type=int, default=10,
                        help="10 = tuning photos 11-20, 20 = final test photos 21-30")
    parser.add_argument("--evaluate-only", action="store_true", help="skip the model, redo results")
    args = parser.parse_args()

    out = results_path(args.model, args.repeats, args.per_class, args.skip)
    if not args.evaluate_only:
        run(check_ready(args.model), args.per_class, args.repeats, out, args.skip)
    if not out.exists():
        raise SystemExit(f"No results yet: {out}")
    evaluate(out, args.repeats)


if __name__ == "__main__":
    main()
