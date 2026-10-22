"""Error bars for the replay: the same policies over many days, ground stations and photo orders.

The main results use ONE day (19 Oct), ONE station (Delhi) and ONE photo order (seed 42).
Here every combination of 7 days x 3 stations x 3 seeds is replayed (63 runs per policy and
rate), so we can say how much the numbers move. Uses saved answers: no model calls, seconds.

Run from the orbit-agent folder:
    python -m src.eval.replay_bars             # table + results/replay_bars_<date>.csv
    python -m src.eval.replay_bars --plot      # also draws docs/figures/replay_bars.png

Photo sets: AID final test photos (21-30) and EuroSAT (Sentinel-2) final test photos.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.agent.decide import POLICIES
from src.demo.export import sentinel_views
from src.eval.run_agent import FINAL, load_views, score
from src.eval.sentinel_test import PHOTO_BYTES
from src.orbit.contact_windows import GROUND_STATIONS, load_satellite
from src.orbit.replay import build_replay
from src.paths import result

ROOT = Path(__file__).resolve().parents[2]
FIRST_DAY = datetime(2026, 10, 19, tzinfo=timezone.utc)  # the day of the main results
SEEDS = [42, 43, 44]                                       # 42 = the main results' photo order
HEADS = {  # agent v4's answers: (decision head, tiny CNN), same training photos
    "aid": (result("head_aid_siglip-base_n20_final.csv"), result("cnn_aid_mobilenet_v3_small_n20_final.csv")),
    "eurosat": (result("head_eurosat_siglip-base_n100_final.csv"),
                result("cnn_eurosat_mobilenet_v3_small_n100_final.csv")),
}


def policies(dataset: str):
    """[(name, plan, views)], and the photo pool and photo size for this dataset."""
    head_csv, cnn_csv = HEADS[dataset]
    if dataset == "aid":
        views, pool = load_views(FINAL, head_csv)
        cnn_views, _ = load_views(FINAL, cnn_csv)
        size = sum(p.stat().st_size for p, _ in pool) // len(pool)  # equal sizes, like run_agent
        gemma = ["send all raw", "caption all", "agent v1", "agent v2", "agent v3-lite", "agent v3"]
        plans = [(name, POLICIES[name], views) for name in gemma]
    else:  # Gemma only answered 100 of these photos, so no Gemma agents here
        views, pool = sentinel_views(head_csv)
        cnn_views, _ = sentinel_views(cnn_csv)
        size = PHOTO_BYTES
        plans = [("send all raw", POLICIES["send all raw"], views)]
    plans += [("agent v4 (head)", POLICIES["agent v4"], views),
              ("agent v4 (CNN)", POLICIES["agent v4"], cnn_views)]
    return plans, pool, size


def run(datasets, days: int, rates: list[float]) -> list[dict]:
    sat = load_satellite()
    rows = []
    for dataset in datasets:
        plans, pool, size = policies(dataset)
        for station in GROUND_STATIONS:
            for day in range(days):
                start = FIRST_DAY + timedelta(days=day)
                for seed in SEEDS:
                    for rate in rates:
                        windows, _ = build_replay(sat, station, start, 24, pool, 30, rate, seed,
                                                  photo_bytes=size)
                        photos = sum(len(w.photos) for w in windows)
                        for name, plan, views in plans:
                            s = score([c for w in windows for c in plan(w, views)])
                            rows.append({"dataset": dataset, "station": station,
                                         "day": f"{start:%Y-%m-%d}", "seed": seed, "rate_mbps": rate,
                                         "policy": name, "photos": photos,
                                         "correct_pct": round(100 * s["correct"] / photos, 2),
                                         "wrong_labels": s["wrong"], "unsure_labels": s["unsure"],
                                         "mb_sent": round(s["mb"], 2)})
    return rows


def spread(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def print_tables(rows: list[dict], rates: list[float]) -> None:
    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        names = list(dict.fromkeys(r["policy"] for r in rows if r["dataset"] == dataset))
        for rate in rates:
            print(f"\n=== {dataset}, {rate:g} Mbps: correct info (mean ± spread) / wrong labels (mean ± spread),"
                  f" over {len(SEEDS) * len({r['day'] for r in rows})} runs per station ===")
            print(f"{'Policy':<17}" + "".join(f"{s.title():>28}" for s in GROUND_STATIONS))
            for name in names:
                cells = []
                for station in GROUND_STATIONS:
                    mine = [r for r in rows if r["dataset"] == dataset and r["rate_mbps"] == rate
                            and r["policy"] == name and r["station"] == station]
                    c, cs = spread([r["correct_pct"] for r in mine])
                    w, ws = spread([r["wrong_labels"] for r in mine])
                    cells.append(f"{c:5.1f} ± {cs:3.1f}% / {w:4.0f} ± {ws:3.0f}")
                print(f"{name:<17}" + "".join(f"{cell:>28}" for cell in cells))


def plot(rows: list[dict], out: Path) -> None:
    """Correct info and wrong labels vs downlink, per station (AID photos), with spread bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Same colours as docs/figures/results.png and docs/figures/label_curve.png (checked with the palette validator)
    lines = {"send all raw": ("Send all raw", "#8b8b94", ":"), "agent v3": ("Agent v3 (Gemma)", "#199e70", "-"),
             "agent v4 (head)": ("Agent v4, frozen head", "#c98500", "-"),
             "agent v4 (CNN)": ("Agent v4, tiny CNN", "#d55181", "-")}
    plt.rcParams.update({
        "figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
        "text.color": "white", "axes.labelcolor": "#bdbdbd", "axes.edgecolor": "#444",
        "xtick.color": "#9a9a9a", "ytick.color": "#9a9a9a", "grid.color": "#222", "font.size": 10.5,
    })
    stations = list(GROUND_STATIONS)
    fig, axes = plt.subplots(2, len(stations), figsize=(5 * len(stations), 7), sharex=True)
    aid = [r for r in rows if r["dataset"] == "aid"]
    rates = sorted({r["rate_mbps"] for r in aid})
    for col, station in enumerate(stations):
        for row, (key, ylabel) in enumerate([("correct_pct", "Correct info reached the ground (%)"),
                                              ("wrong_labels", "Wrong labels sent down")]):
            ax = axes[row][col]
            for name, (label, colour, style) in lines.items():
                stats = [spread([r[key] for r in aid if r["station"] == station and r["policy"] == name
                                 and r["rate_mbps"] == rate]) for rate in rates]
                mean = [m for m, _ in stats]
                ax.fill_between(rates, [max(0, m - s) for m, s in stats], [m + s for m, s in stats],
                                color=colour, alpha=0.18, linewidth=0)
                ax.plot(rates, mean, style, marker="o", color=colour, label=label, linewidth=2, markersize=5)
            ax.set_xscale("log", base=2)
            ax.set_xticks(rates, [f"{r:g}" for r in rates])
            ax.minorticks_off()
            ax.grid(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylim(bottom=0)  # can't send fewer than 0 wrong labels
            if row == 0:
                ax.set_ylim(0, 105)
                ax.set_title(f"{station.title()} ground station", loc="left", fontweight="bold")
            else:
                ax.set_xlabel("Downlink speed (Mbps)")
            if col == 0:
                ax.set_ylabel(ylabel)
    axes[0][0].legend(frameon=False, loc="lower right", fontsize=9.5)
    fig.suptitle("AID final test photos: 7 days x 3 photo orders per station (line = mean, band = ± spread)",
                 x=0.01, ha="left", fontsize=11, color="#bdbdbd")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Chart saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay error bars over days, stations and photo orders")
    parser.add_argument("--datasets", nargs="+", default=["aid", "eurosat"], choices=list(HEADS))
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.25, 0.5, 1, 2])
    parser.add_argument("--plot", action="store_true", help="also draw docs/figures/replay_bars.png")
    args = parser.parse_args()

    rows = run(args.datasets, args.days, args.rates)
    print_tables(rows, args.rates)
    out = result(f"replay_bars_{datetime.now():%Y%m%d_%H%M%S}.csv")  # seconds: no clash between quick runs
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {out.relative_to(ROOT)}")
    if args.plot:
        plot(rows, ROOT / "docs" / "figures" / "replay_bars.png")


if __name__ == "__main__":
    main()
