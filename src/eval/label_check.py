"""Check the real replay's truth labels by eye.

The real replay's labels come from maps (the image's own cloud map, else ESA WorldCover 2021),
not from a person. If the maps are often wrong, every real-replay number is off. So: look at a
fixed random sample of photos and write down whether the label matches what you see.

Sample: about 20 photos per class from the TEST days (13-19 Oct; all of a class if it has fewer),
seed 0, picked before looking. 150 photos on 6 contact sheets of 25.

Run from the orbit-agent folder:
    python -m src.eval.label_check            # makes the sheets + an empty verdict column
    python -m src.eval.label_check --summary  # after filling in the verdicts

Sheets: data/processed/label_check/sheet_<n>.png (not in git). Verdicts: experiments/label_check_real.csv,
column `by_eye` = agree / disagree / unsure, `seen_as` = what it looks like, `checked_by` = who looked.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.data import s2_real

ROOT = Path(__file__).resolve().parents[2]
SHEETS = ROOT / "data" / "processed" / "label_check"
OUT = ROOT / "experiments" / "label_check_real.csv"
TEST_DAYS = [f"2026-10-{d}" for d in range(13, 20)]
PER_CLASS, TOTAL, PER_SHEET, TILE = 20, 150, 25, 224


def share(row: dict) -> float:
    """How much of the chip the label's map says is that class."""
    if row["label"] == "Cloud":
        return float(row["cloud"])
    if row["label"] == "SnowIce" and float(row["snow"]) > 0:
        return float(row["snow"])
    return float(row[f"wc_{row['label']}"])


def pick() -> list[dict]:
    rows = [r for r in s2_real.load_chips() if r["day"] in TEST_DAYS]
    by_class = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["name"]):
        by_class[r["label"]].append(r)
    rng = random.Random(0)
    picked = []
    for c in s2_real.CLASSES:
        picked += rng.sample(by_class[c], min(PER_CLASS, len(by_class[c])))
    rest = [r for r in rows if r not in picked]
    picked += rng.sample(rest, TOTAL - len(picked))  # top up to 150 with random photos (any class)
    rng.shuffle(picked)  # mixed order, so the sheets don't give the label away by position
    return picked


def make_sheets(picked: list[dict]) -> None:
    SHEETS.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=16)
    for s in range(0, len(picked), PER_SHEET):
        part = picked[s:s + PER_SHEET]
        sheet = Image.new("RGB", (5 * TILE, 5 * (TILE + 24)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(part):
            x, y = (i % 5) * TILE, (i // 5) * (TILE + 24)
            chip = Image.open(s2_real.OUT_DIR / r["day"] / r["name"]).convert("RGB").resize((TILE - 4, TILE - 4))
            sheet.paste(chip, (x + 2, y + 2))
            draw.text((x + 4, y + TILE), f"#{s + i + 1} {r['label']} ({share(r):.0%})", fill="black", font=font)
        sheet.save(SHEETS / f"sheet_{s // PER_SHEET + 1}.png")
    print(f"Saved {len(range(0, len(picked), PER_SHEET))} sheets in {SHEETS.relative_to(ROOT)}")


def summary() -> None:
    rows = list(csv.DictReader(open(OUT)))
    done = [r for r in rows if r["by_eye"]]
    print(f"{len(done)} of {len(rows)} photos checked (by: {', '.join(sorted({r['checked_by'] for r in done}))})")
    print(f"{'label':<11} {'photos':>6} {'agree':>6} {'disagree':>9} {'unsure':>7}")
    for c in s2_real.CLASSES + ["all"]:
        mine = [r for r in done if c in ("all", r["label"])]
        if mine:
            n = Counter(r["by_eye"] for r in mine)
            print(f"{c:<11} {len(mine):>6} {n['agree']:>6} {n['disagree']:>9} {n['unsure']:>7}")
    sure = [r for r in done if r["by_eye"] != "unsure"]
    if sure:
        ok = sum(r["by_eye"] == "agree" for r in sure)
        print(f"\nLabel right where the eye could tell: {ok}/{len(sure)} = {ok / len(sure):.0%}")
        # weighted back to the real class mix of the test days (the sample has ~20 per class)
        counts = Counter(r["label"] for r in s2_real.load_chips() if r["day"] in TEST_DAYS)
        weighted = sum(counts[c] * sum(r["by_eye"] == "agree" for r in sure if r["label"] == c)
                       / max(1, sum(r["label"] == c for r in sure)) for c in counts) / sum(counts.values())
        print(f"Weighted to the test days' class mix: {weighted:.0%}")
    wrong = Counter((r["label"], r["seen_as"]) for r in done if r["by_eye"] == "disagree")
    if wrong:
        print("Disagreements (label -> looks like): " + ", ".join(f"{a}->{b} {n}x" for (a, b), n in wrong.most_common()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the real replay's labels by eye")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summary()
        return
    picked = pick()
    make_sheets(picked)
    if OUT.exists():
        print(f"{OUT.relative_to(ROOT)} exists already: not overwritten (verdicts may be in it)")
        return
    with open(OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "day", "lat", "lon", "label", "label_share", "by_eye", "seen_as",
                         "checked_by", "note"])
        for i, r in enumerate(picked, 1):
            writer.writerow([i, r["name"], r["day"], r["lat"], r["lon"], r["label"], round(share(r), 2),
                             "", "", "", ""])
    print(f"Saved {OUT.relative_to(ROOT)}: fill in by_eye / seen_as / checked_by, then --summary")


if __name__ == "__main__":
    main()
