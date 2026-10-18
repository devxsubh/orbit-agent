"""Baseline: how well does Gemma 3 recognise satellite photos, with no training?

Run from the orbit-agent folder:
    python -m src.eval.baseline_eurosat                 (10 photos per class = 100 photos)
    python -m src.eval.baseline_eurosat --per-class 3   (quick test: 30 photos)
    python -m src.eval.baseline_eurosat --prompt v1     (try another prompt: v0, v1, v2)
    python -m src.eval.baseline_eurosat --size 896      (try another image size)
    python -m src.eval.baseline_eurosat --dataset aid   (AID: sharper photos, 30 classes, as in NAVI)

Saves every answer to experiments/baseline_<dataset>_<model>_<prompt>_<size>px_<date>.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.data import aid, eurosat
from src.eval.metrics import accuracy
from src.model.vlm import MODEL, PROMPTS, check_ready, classify_image

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {"eurosat": eurosat, "aid": aid}


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot Gemma 3 on EuroSAT photos")
    parser.add_argument("--per-class", type=int, default=10, help="photos per class")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seed", type=int, default=42, help="change to test different photos")
    parser.add_argument("--prompt", choices=PROMPTS, default="v1",
                        help="v1 = names shuffled (best, 29%%), v0 = numbered list (first run, 26%%), "
                             "v2 = names + descriptions (22%%)")
    parser.add_argument("--size", type=int, default=448,
                        help="upscale smaller photos to this many pixels (EuroSAT photos are 64)")
    parser.add_argument("--dataset", choices=list(DATASETS), default="eurosat",
                        help="eurosat (64 px, 10 classes) or aid (600 px, 30 classes, as in NAVI)")
    args = parser.parse_args()

    data = DATASETS[args.dataset]
    if args.prompt == "v2" and not data.DETAILS:
        raise SystemExit(f"Prompt v2 isn't set up for {args.dataset}. Use v0 or v1.")

    model = check_ready(args.model)
    photos = data.sample(args.per_class, args.seed)
    print(f"Settings: {args.dataset}, model {model}, prompt {args.prompt}, "
          f"size {args.size} px, {len(photos)} photos, seed {args.seed}")

    rows = []
    for path, truth in tqdm(photos, desc="Asking the model"):
        guess, answer, seconds = classify_image(path, data.LABELS, model, prompt=args.prompt,
                                                size=args.size, details=data.DETAILS,
                                                seed=args.seed, intro=data.INTRO)
        rows.append({
            "file": path.name,
            "truth": truth,
            "guess": guess,
            "correct": guess == truth,
            "answer": answer,
            "seconds": round(seconds, 2),
            "photo_bytes": path.stat().st_size,
            "label_bytes": len(guess.encode()),
        })

    # Settings go in the file name so runs don't get mixed up (":" isn't safe in file names)
    tag = f"{model.replace(':', '-')}_{args.prompt}_{args.size}px"
    out = ROOT / "experiments" / f"baseline_{args.dataset}_{tag}_{datetime.now():%Y%m%d_%H%M}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    avg_seconds = sum(r["seconds"] for r in rows) / total
    photo_kb = sum(r["photo_bytes"] for r in rows) / total / 1000
    label_bytes = sum(r["label_bytes"] for r in rows) / total

    print(f"\nResults: {model}, prompt {args.prompt}, {args.size} px, "
          f"no training, {total} {args.dataset} photos\n")
    print(f"Accuracy: {accuracy(correct, total):.0%}   "
          f"(random guessing = {1 / len(data.CLASSES):.0%})")
    print(f"Speed:    {avg_seconds:.1f} seconds per photo")
    print(f"Data:     photo {photo_kb:.1f} KB vs label {label_bytes:.0f} bytes "
          f"({photo_kb * 1000 / label_bytes:,.0f}x smaller)")

    print("\nPer class:")
    for label in data.CLASSES:
        mine = [r for r in rows if r["truth"] == label]
        hits = sum(r["correct"] for r in mine)
        print(f"  {label:<22} {hits:>2}/{len(mine)}  {'#' * hits}")

    mixups = Counter((r["truth"], r["guess"]) for r in rows if not r["correct"])
    if mixups:
        print("\nMost common mix-ups (real -> model said):")
        for (truth, guess), n in mixups.most_common(5):
            print(f"  {truth} -> {guess}  ({n})")

    print(f"\nEvery answer saved to: {out}")


if __name__ == "__main__":
    main()
