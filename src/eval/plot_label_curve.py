"""Chart: accuracy vs number of labelled training photos, frozen head vs tiny CNN.

Run:  python -m src.eval.plot_label_curve       (uses the newest results/label_curve_*.csv)
Saves docs/figures/label_curve.png (dark, like docs/figures/results.png).
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window, just save the file
import matplotlib.pyplot as plt

from src.paths import find

ROOT = Path(__file__).resolve().parents[2]

# Model name in the CSV -> (label, colour). Head = same yellow as agent v4 in docs/figures/results.png.
# Colours checked with the dataviz palette validator on the black background.
MODELS = {
    "head (frozen SigLIP-base)": ("Frozen SigLIP-base + head", "#c98500"),
    "tiny CNN (MobileNetV3-Small)": ("Tiny CNN (MobileNetV3-Small)", "#d55181"),
}
# Gemma 3 4B with no training, for reference (final test photos; EuroSAT: its 100 photos)
GEMMA = {"aid": 63, "eurosat": 29}
TITLES = {"aid": "AID (sharp aerial photos, 30 classes)", "eurosat": "Sentinel-2 / EuroSAT (64 px, 10 classes)"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Label-curve chart")
    parser.add_argument("--csv", help="label_curve_*.csv (default: newest)")
    parser.add_argument("--out", default=str(ROOT / "docs" / "figures" / "label_curve.png"))
    args = parser.parse_args()

    path = Path(args.csv) if args.csv else max(find("label_curve_2*.csv"))
    rows = list(csv.DictReader(open(path)))
    plt.rcParams.update({
        "figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
        "text.color": "white", "axes.labelcolor": "#bdbdbd", "axes.edgecolor": "#444",
        "xtick.color": "#9a9a9a", "ytick.color": "#9a9a9a", "grid.color": "#222", "font.size": 11,
    })
    datasets = [d for d in ("aid", "eurosat") if any(r["dataset"] == d for r in rows)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.6), squeeze=False)

    for ax, dataset in zip(axes[0], datasets):
        ends = []  # (last x, last mean, colour) of each line, labelled after both are drawn
        for model, (name, colour) in MODELS.items():
            mine = [r for r in rows if r["dataset"] == dataset and r["model"] == model]
            sizes = sorted({int(r["per_class"]) for r in mine})
            runs = [[100 * float(r["accuracy"]) for r in mine if int(r["per_class"]) == n] for n in sizes]
            mean = [sum(v) / len(v) for v in runs]
            ax.fill_between(sizes, [min(v) for v in runs], [max(v) for v in runs], color=colour, alpha=0.18,
                            linewidth=0)  # band = lowest to highest of the 3 picks of photos
            ax.plot(sizes, mean, "-o", color=colour, linewidth=2.2, markersize=6, label=name)
            ends.append((sizes[-1], mean[-1], colour))
        # End labels (white text, next to their own line); nudge apart if the two ends are close
        close = len(ends) == 2 and abs(ends[0][1] - ends[1][1]) < 4
        for i, (x, y, _) in enumerate(sorted(ends, key=lambda e: -e[1])):
            ax.annotate(f"{y:.0f}%", (x, y), xytext=(6, (7 if i == 0 else -7) if close else 0),
                        textcoords="offset points", va="center", color="white", fontsize=10,
                        fontweight="bold")
        ax.axhline(GEMMA[dataset], color="#8b8b94", linestyle="--", linewidth=1.4,
                   label=f"Gemma 3 4B, no training ({GEMMA[dataset]}%)")
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.minorticks_off()
        ax.set_ylim(0, 105)
        ax.set_xlabel("Labelled training photos per class")
        ax.set_ylabel("Right on the final test photos (%)")
        ax.set_title(TITLES[dataset], loc="left", fontweight="bold")
        ax.grid(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Chart saved: {args.out} (from {path.name})")


if __name__ == "__main__":
    main()
