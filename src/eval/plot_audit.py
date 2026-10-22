"""Charts for the audited downlink (dark, like the other README charts). Reads the newest CSVs.

    python -m src.eval.plot_audit

Draws (in docs/):
- audit_honesty.png  every onboard model, healthy or broken: how far off each policy's numbers are,
                     and how often its 90% error bar held (src/eval/audit_test.py)
- audit_floor.png    how much to spend on checking photos the model is sure about (src/eval/audit_floor.py)
- audit_rsweep.png   when onboard AI beats plain compression (src/eval/compress_test.py)
- audit_loop.png     a new camera learning from its own audits (src/eval/audit_loop.py)

Colours (checked with the dataviz palette validator on black, all pairs: blue, yellow, magenta
pass; grey = no onboard AI, as in the other charts): magenta = the audit, yellow = agent v4's
triage, blue = the other onboard option in each chart, grey = raw photos.
"""
from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
GREY, BLUE, YELLOW, MAGENTA = "#8b8b94", "#3987e5", "#c98500", "#d55181"
INK, MUTED = "#e6e6e6", "#9a9a9a"


def newest(pattern: str) -> Path:
    files = sorted(glob.glob(str(ROOT / "experiments" / pattern)), key=os.path.getmtime)
    if not files:
        raise SystemExit(f"No {pattern} yet: run the experiment first.")
    return Path(files[-1])


def rows_of(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
        "text.color": INK, "axes.labelcolor": "#bdbdbd", "axes.edgecolor": "#444",
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#222", "font.size": 10.5,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })
    return plt


def honesty(out: Path, rate: float = 0.25, data: str = "eurosat") -> None:
    """Dot plot: one row per onboard model; left = typical error, right = error bar held."""
    plt = style()
    path = newest(f"audit_summary_{data}_*.csv")
    rows = [r for r in rows_of(path) if float(r["rate_mbps"]) == rate]
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    series = [("raw random", "Raw photos, random (no AI)", GREY, "o"),
              ("labels only", "Labels only (NAVI-style)", BLUE, "D"),
              ("triage (v4)", "Triage: least sure first (agent v4)", YELLOW, "s"),
              ("audit, self-check", "Self-checking audit: random checks with known chances", MAGENTA, "o")]

    def mean(scen, method, key):
        return np.mean([float(r[key]) for r in rows if r["scenario"] == scen and r["method"] == method])

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True,
                                      gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.08})
    y = np.arange(len(scenarios))[::-1]
    offsets = np.linspace(-0.24, 0.24, len(series))
    for (method, label, colour, marker), dy in zip(series, offsets):
        err = [mean(s, method, "rmse_pts") for s in scenarios]
        held = [100 * mean(s, method, "holds_scaled+1") for s in scenarios]
        left.scatter(err, y + dy, s=52, color=colour, marker=marker, label=label, zorder=3,
                     edgecolors="black", linewidths=1.5)
        right.scatter(held, y + dy, s=52, color=colour, marker=marker, zorder=3,
                      edgecolors="black", linewidths=1.5)
    left.set_xscale("log")
    left.set_xticks([0.3, 1, 3, 10, 30], ["0.3", "1", "3", "10", "30"])
    left.minorticks_off()
    left.set_xlabel("Typical error in the day's class shares\n(percentage points, log scale)")
    left.set_yticks(y, [s.replace("damaged camera: ", "damaged camera:\n").replace(" (", "\n(")
                        for s in scenarios])
    left.set_title("How far off the numbers are")
    right.axvline(90, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    right.text(88, y[0] + 0.45, "90% target", color=MUTED, fontsize=9, ha="right")
    right.set_xlim(-3, 103)
    right.set_xlabel("How often the 90% error bar contained the truth (%)\n")
    right.set_title("Whether its error bar can be trusted")
    for ax in (left, right):
        ax.grid(True, axis="x")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.55, 0.935), ncol=2,
               fontsize=9.5)
    what = ("real Sentinel-2 photos (EuroSAT) in the stand-in replay" if data == "eurosat"
            else "what Sentinel-2A really photographed, 13-19 Oct 2026")
    fig.suptitle(f"Healthy or broken onboard model: {what}, {rate:g} Mbps, "
                 "3 stations x 7 days x 3 photo orders", x=0.01, y=0.985, ha="left", fontsize=11,
                 color="#bdbdbd")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.79, bottom=0.14, wspace=0.08)
    fig.savefig(out, dpi=150)
    print(f"Chart saved: {out.relative_to(ROOT)}  (from {path.name})")


def spread_labels(values: list[float], gap: float) -> list[float]:
    """Label heights near the given values, at least `gap` apart (for labels at line ends)."""
    order = np.argsort(values)
    placed = list(values)
    for lower, upper in zip(order[:-1], order[1:]):
        placed[upper] = max(placed[upper], placed[lower] + gap)
    return placed


def floor_chart(out: Path) -> None:
    """Two panels sharing x = floor: typical error, and whether the error bar held. One line per
    onboard model (all the audit, so one colour; line style + direct labels tell them apart)."""
    plt = style()
    path = newest("audit_floor_*.csv")
    rows = [r for r in rows_of(path) if "tuned" not in r["method"]]
    floors = sorted({float(r["method"].split()[1]) for r in rows})
    lines = [("healthy head", "healthy head", "-", "o"),
             ("silent swap", "silent swap", "--", "s"),
             ("bit flips", "radiation bit flips", ":", "D")]

    def stat(scen, key):
        return [np.mean([float(r[key]) for r in rows if r["method"] == f"floor {f:g}" and r["scenario"] == scen])
                for f in floors]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    ends = {"a": [], "b": []}
    for scen, label, ls, marker in lines:
        err = stat(scen, "rmse_pts")
        held = [100 * v for v in stat(scen, "holds_scaled+1")]
        a.plot(floors, err, ls, marker=marker, color=MAGENTA, linewidth=2, markersize=6)
        b.plot(floors, held, ls, marker=marker, color=MAGENTA, linewidth=2, markersize=6)
        ends["a"].append(err[-1])
        ends["b"].append(held[-1])
    for ax, key, gap in ((a, "a", 0.13), (b, "b", 0.9)):  # labels at the line ends, kept apart
        for (_, label, _, _), y in zip(lines, spread_labels(ends[key], gap)):
            ax.text(floors[-1] + 0.04, y, label, color=INK, fontsize=9, va="center")
    a.set_title("Typical error in the class shares (lower is better)")
    a.set_ylabel("Percentage points")
    a.set_ylim(bottom=0)
    b.set_title("How often the 90% error bar held")
    b.set_ylabel("Held (%)")
    b.axhline(90, color=MUTED, linewidth=1, linestyle="--")
    b.set_ylim(80, 100)
    for ax in (a, b):
        ax.set_xlim(0, 1.32)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Floor = every photo's smallest chance of being checked, as a share of the average chance "
                  "(0 = check only unsure photos, 1 = check at random)", fontsize=9.5, color="#bdbdbd")
    fig.suptitle("How much should a satellite check itself? Audit at 0.25 Mbps, EuroSAT replay, 63 days per point",
                 x=0.01, ha="left", fontsize=11, color="#bdbdbd")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Chart saved: {out.relative_to(ROOT)}  (from {path.name})")


def rsweep(out: Path) -> None:
    """Two panels sharing x = R: photos right on the ground, and honest error-bar width."""
    plt = style()
    path = newest("compress_sweep_*.csv")
    rows = rows_of(path)
    by = defaultdict(list)
    for r in rows:
        by[(r["method"], float(r["rate_mbps"]))].append(r)
    rates = sorted({float(r["rate_mbps"]) for r in rows}, reverse=True)
    EXACT = 0.03  # where an exact answer (± 0) is drawn on the log scale

    def line(method, key):
        R = [np.mean([float(r["R"]) for r in by[(method, rate)]]) for rate in rates]
        return R, [np.mean([float(r[key]) for r in by[(method, rate)]]) for rate in rates]

    looks = {"raw random": ("Raw photos, random (no AI)", GREY, ":", "s"),
             "compress to fit": ("Compress every photo to fit (no AI); a person checks some", BLUE, "-", "D"),
             "triage (v4)": ("Triage (agent v4)", YELLOW, "-", "s"),
             "audit (floor 0.5)": ("Audit, checks sent as full photos", MAGENTA, "--", "o"),
             "audit, thumbnails": ("Audit, checks sent as 160 px thumbnails", MAGENTA, "-", "o")}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    for method in ["raw random", "compress to fit", "triage (v4)", "audit, thumbnails"]:
        label, colour, ls, marker = looks[method]
        R, v = line(method, "strict_pct")
        a.plot(R, v, ls, marker=marker, color=colour, label=label, linewidth=2, markersize=5)
    for method in ["raw random", "compress to fit", "audit (floor 0.5)", "audit, thumbnails"]:
        label, colour, ls, marker = looks[method]
        R, v = line(method, "halfwidth_pts")
        b.plot(R, np.maximum(v, EXACT), ls, marker=marker, color=colour, label=label, linewidth=2,
               markersize=5)
    a.set_title("Photos whose class reached the ground right")
    a.set_ylabel("% of the day's photos")
    a.set_ylim(0, 102)
    b.set_title("Honest 90% error bar on the day's class shares")
    b.set_ylabel("± percentage points (log)")
    b.set_yscale("log")
    b.set_yticks([EXACT, 0.1, 0.3, 1, 3, 10, 50], ["exact", "0.1", "0.3", "1", "3", "10", "50"])
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xticks([1.4, 3, 10, 30, 100, 300], ["1.4", "3", "10", "30", "100", "300"])
        ax.minorticks_off()
        ax.axvspan(1.4, 5.6, color="#1d1d1d", zorder=0)
        ax.grid(True)
        ax.spines[["top", "right"]].set_visible(False)
    a.text(1.55, 66, "the paper's\nreplay so far", color=MUTED, fontsize=8.5)
    fig.supxlabel("R = data the camera takes ÷ data the passes can send (log scale)", fontsize=10,
                  color="#bdbdbd", y=0.13)
    handles, labels = [], []
    for ax in (a, b):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("When does onboard AI beat plain compression? AID photos, Delhi, 21 days per point; "
                 "same bytes and same number of human checks", x=0.01, ha="left", fontsize=11, color="#bdbdbd")
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out, dpi=150)
    print(f"Chart saved: {out.relative_to(ROOT)}  (from {path.name})")


def loop_chart(out: Path, data: str = "eurosat") -> None:
    """Typical error and honest error bar per mission day, and the closed loop's head accuracy."""
    plt = style()
    path = newest(f"audit_loop_{data}_*.csv")
    rows = rows_of(path)
    days = list(dict.fromkeys(r["day"] for r in rows))
    x = np.arange(1, len(days) + 1)

    def per_day(method, key):
        return [np.mean([float(r[key]) for r in rows if r["day"] == d and r["method"] == method]) for d in days]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True)
    for method, label, colour, ls in [
            ("raw random", "Raw photos, random", GREY, ":"),
            ("launch head + triage", "Launch head + triage (v4)", YELLOW, "-"),
            ("launch head + audit", "Launch head + audit", MAGENTA, "--"),
            ("closed loop", "Closed loop: audit + retrain after every pass", MAGENTA, "-"),
            ("head trained before launch + audit", "Head trained before launch (best case)", BLUE, "-")]:
        a.plot(x, per_day(method, "rmse_pts"), ls, marker="o", color=colour, label=label,
               linewidth=2, markersize=5)
        if method != "launch head + triage":  # triage has no honest error bar
            b.plot(x, per_day(method, "halfwidth_pts"), ls, marker="o", color=colour, linewidth=2,
                   markersize=5)
    c.plot(x, [100 * v for v in per_day("closed loop", "head_right")], "-", marker="o", color=MAGENTA,
           linewidth=2, markersize=5)
    for xi, n in zip(x, per_day("closed loop", "labels_so_far")):
        c.annotate(f"{n:.0f}", (xi, 3), ha="center", color=MUTED, fontsize=8)
    c.text(x[0], 12, "labels learned so far:", color=MUTED, fontsize=8)
    a.set_title("Typical error in the class shares")
    a.set_ylabel("Percentage points")
    b.set_title("Honest 90% error bar (±)")
    b.set_ylabel("Percentage points")
    c.set_title("Closed loop: head right")
    c.set_ylabel("% of the day's photos")
    c.set_ylim(0, 100)
    for ax in (a, b, c):
        ax.set_xlabel("Mission day")
        ax.set_xticks(x)
        ax.grid(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=5, fontsize=9)
    where = ("EuroSAT photos, AID head at launch" if data == "eurosat"
             else "real Sentinel-2A photos, EuroSAT head at launch")
    rate = rows[0]["rate_mbps"]
    fig.suptitle(f"A new camera with zero labels learns from its own audits ({where}; Delhi, {rate} Mbps, "
                 "3 photo orders)", x=0.01, ha="left", fontsize=11, color="#bdbdbd")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out, dpi=150)
    print(f"Chart saved: {out.relative_to(ROOT)}  (from {path.name})")


def main() -> None:
    for name, draw in [("audit_honesty.png", honesty),
                       ("audit_honesty_real.png", lambda out: honesty(out, 0.05, "real")),
                       ("audit_floor.png", floor_chart),
                       ("audit_rsweep.png", rsweep), ("audit_loop.png", loop_chart),
                       ("audit_loop_real.png", lambda out: loop_chart(out, "real"))]:
        try:
            draw(DOCS / name)
        except SystemExit as missing:
            print(f"Skipped {name}: {missing}")


if __name__ == "__main__":
    main()
