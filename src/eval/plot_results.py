"""Draws the main result chart for the README (dark, same colours as the demo page).

Left: share of photos whose information reached the ground correctly.
Right: wrong labels sent down. Both at each downlink speed.

Run:  python -m src.eval.plot_results
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window, just save the file
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "experiments" / "agents_delhi_final_equal_20261019_1229.csv"
DEFAULT_OUT = ROOT / "docs" / "results.png"

# Policy name in the CSV -> (label, line style, marker, colour)
POLICIES = {
    "send all raw": ("Send all raw", ":", "s", "#8b8b94"),
    "caption all": ("Caption every photo", "--", "^", "#a78bfa"),
    "agent v1": ("Agent v1 (rules)", "-", "o", "#4da3ff"),
    "agent v3": ("Agent v3 (chance-based)", "-", "o", "#34d399"),
}


def load(path: Path) -> dict:
    """Reads the agents CSV into {policy: [(rate, correct %, wrong labels), ...]}."""
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["policy"] in POLICIES:
                share = 100 * int(r["correct"]) / int(r["photos"])
                rows.setdefault(r["policy"], []).append(
                    (float(r["rate_mbps"]), share, int(r["wrong_labels"])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Main result chart for the README")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="agents_*.csv from run_agent")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    rows = load(Path(args.csv))
    plt.rcParams.update({
        "figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
        "text.color": "white", "axes.labelcolor": "#bdbdbd", "axes.edgecolor": "#444",
        "xtick.color": "#9a9a9a", "ytick.color": "#9a9a9a", "grid.color": "#222",
        "font.size": 11,
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6))

    rates = sorted({rate for points in rows.values() for rate, _, _ in points})
    for policy, (name, style, marker, colour) in POLICIES.items():
        points = sorted(rows.get(policy, []))
        x = [p[0] for p in points]
        left.plot(x, [p[1] for p in points], style, marker=marker, color=colour, label=name,
                  linewidth=2.6 if policy == "agent v3" else 1.6, markersize=6)

    left.set_xscale("log", base=2)
    left.set_xticks(rates, [f"{r:g}" for r in rates])
    left.set_ylim(0, 105)
    left.set_xlabel("Downlink speed (Mbps)")
    left.set_ylabel("Correct info reached the ground (%)")
    left.set_title("More correct info, same downlink", loc="left", fontweight="bold")
    left.grid(True)
    left.legend(frameon=False, loc="lower right", fontsize=10)

    # Wrong labels: caption-all vs agent v1 vs agent v3 (raw sends none)
    bars = [(policy, POLICIES[policy][3]) for policy in ("caption all", "agent v1", "agent v3")]
    width = 0.26
    for i, (policy, colour) in enumerate(bars):
        wrong = dict((p[0], p[2]) for p in rows.get(policy, []))
        xs = [j + (i - 1) * width for j in range(len(rates))]
        right.bar(xs, [wrong.get(r, 0) for r in rates], width, color=colour, label=POLICIES[policy][0])
    right.set_xticks(range(len(rates)), [f"{r:g}" for r in rates])
    right.set_xlabel("Downlink speed (Mbps)")
    right.set_ylabel("Wrong labels sent down")
    right.set_title("Fewer wrong labels", loc="left", fontweight="bold")
    right.set_ylim(0, 1.5 * max(b.get_height() for b in right.patches))  # room for the legend
    right.set_axisbelow(True)
    right.grid(True, axis="y")
    right.legend(frameon=False, fontsize=10, loc="upper right")

    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Chart saved: {out}")


if __name__ == "__main__":
    main()
