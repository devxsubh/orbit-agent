"""Experiment: can the ground trust the numbers? Audited downlink vs today's policies, healthy or broken.

Task for the ground: after each day, say what share of the photos were each class (cropland,
forest, water...) and give a 90% error bar. Same orbit replay and budgets as the other experiments
(7 days x 3 stations x 3 photo orders, budget per pass), every policy spends the same bytes.

Onboard models ("scenarios"): a healthy head, and the failure suite of src/model/faults.py
(new camera, silent swap, radiation bit flips, damaged camera).
Policies and ground methods: src/agent/audit.py and src/eval/estimate.py.

Data:
- eurosat: real Sentinel-2 photos (EuroSAT FINAL + FRESH, 2,000), drawn with ASSUMED India-like
  class shares (the photo slots are stand-ins, as in the other replays).
- real: what Sentinel-2A really photographed on 13-19 Oct 2026 (src/data/s2_real.py), with real
  clouds and real places. The head is trained on 6-12 Oct.

Run from the orbit-agent folder:
    python -m src.eval.audit_test                    # eurosat, about 5 min the first time
    python -m src.eval.audit_test --data real        # needs python -m src.data.s2_real first

Saves experiments/audit_summary_<data>_<time>.csv (the tables) and every single estimate in
data/processed/audit_runs_<data>_<time>.csv.gz (big, not in git).
"""
from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.agent.audit import POLICIES, audit, label_bytes
from src.agent.self_check import SelfCheck
from src.eval.estimate import combine, estimate, interval

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
TILE = 300_000  # one full photo: 224 x 224 px x 4 bands (red, green, blue, near-infrared) x 12 bits
THUMB = 6_400   # a 160 px JPEG thumbnail (measured on AID photos, src/eval/compress_test.py)
FIRST_DAY = datetime(2026, 10, 19, tzinfo=timezone.utc)
SEEDS = [42, 43, 44]
LEVEL = 0.9
INTERVALS = ["normal", "t", "safe", "scaled", "scaled+1", "betting"]
# ASSUMPTION (not measured): an India-like mix of EuroSAT classes; three are rare (2-4%)
SHARES = {"AnnualCrop": .30, "Forest": .12, "HerbaceousVegetation": .12, "Highway": .02,
          "Industrial": .02, "Pasture": .06, "PermanentCrop": .06, "Residential": .08,
          "River": .04, "SeaLake": .18}
# Ground methods: (policy that decides what is sent, how the ground turns it into numbers)
METHODS = {"raw in order": ("raw in order", "raw"), "raw random": ("raw random", "raw"),
           "labels only": ("labels only", "count"), "triage (v4)": ("triage (v4)", "count"),
           "stratified": ("stratified", "stratified"),
           "audit, floor 1": ("audit (floor 1)", "difference"),
           "audit, floor 0.5": ("audit (floor 0.5)", "difference"),
           "audit, floor 0.1": ("audit (floor 0.1)", "difference"),
           "audit, floor 0.5, tuned": ("audit (floor 0.5)", "tuned"),
           # checks sent as 160 px thumbnails (6.4 KB, enough for a person to read the class: ~47x more
           # checks per byte; see src/eval/compress_test.py)
           "audit, thumbnails": (lambda P, b, t, r: audit(P, b, THUMB, r, floor=0.5), "difference"),
           # the audit + the ground's self-check: safe mode once checked mistakes are 1.5x what the model
           # expected (src/agent/self_check.py); keeps memory across the days of a mission
           "audit, self-check": ("self-check", "self-check")}
EUROSAT_TO_REAL = {"AnnualCrop": "Cropland", "PermanentCrop": "Cropland", "Forest": "Trees",
                   "Pasture": "ShrubGrass", "HerbaceousVegetation": "ShrubGrass",
                   "Residential": "BuiltUp", "Industrial": "BuiltUp", "Highway": "BuiltUp",
                   "River": "Water", "SeaLake": "Water"}


class Population:
    """All photos one experiment can use: names, true classes, and each onboard model's chances."""

    def __init__(self, names, truth, classes, scenarios, rare, notes, x=None, launch=None):
        self.x = x                       # embeddings (for retraining in src/eval/audit_loop.py)
        self.launch = launch             # scenario name of the mismatched head a new camera starts with
        self.names = names               # photo file names
        self.truth = np.asarray(truth)   # true class index per photo
        self.classes = classes
        self.scenarios = scenarios       # scenario name -> chances (photos x classes)
        self.rare = rare                 # True for classes counted as rare
        self.notes = notes               # one line per scenario for the printout
        self.index = {n: i for i, n in enumerate(names)}
        self.Y = np.eye(len(classes))[self.truth]


def _head_chances(head, x, classes):
    """The head's chances in the order of `classes` (0 for classes it never saw in training)."""
    from src.model.faults import chances

    P_head = chances(head, x)
    P = np.zeros((len(x), len(classes)))
    for j, c in enumerate(head.classes_):
        P[:, classes.index(c)] = P_head[:, j]
    return P


def eurosat_population() -> Population:
    """EuroSAT FINAL + FRESH photos (2,000) and every scenario's chances. Embeddings are cached."""
    from src.data.eurosat import CLASSES, dataset_dir
    from src.eval.overfit_check import eurosat_fresh
    from src.eval.sentinel_test import MATCH, photo_sets
    from src.model import faults
    from src.model.head import aid_head, embed, fit_head

    root = dataset_dir()
    final, tuning, train = photo_sets(100)
    pool = final + eurosat_fresh()
    x = {k: embed(v, "siglip-base", root=root, dataset="eurosat")[0]
         for k, v in (("pool", pool), ("tuning", tuning), ("train", train))}
    classes = list(CLASSES)
    truth = np.array([classes.index(label) for label, _ in pool])
    head = fit_head(x["train"], [l for l, _ in train], x["tuning"], [l for l, _ in tuning], verbose=False)
    good = _head_chances(head, x["pool"], classes)

    aid = aid_head("siglip-base", 20)
    camera = faults.mapped_chances(faults.chances(aid, x["pool"]), list(aid.classes_), MATCH, classes)
    swap = faults.silent_swap(good, truth, classes.index("AnnualCrop"), classes.index("PermanentCrop"),
                              0.3, np.random.default_rng(7))
    labels = np.array([l for l, _ in pool])
    flipped, seed = faults.harmful_flips(head, x["pool"], labels, n_flips=8)
    scenarios = {"healthy head": good, "new camera (AID head)": camera, "silent swap": swap,
                 "bit flips": _head_chances(flipped, x["pool"], classes)}
    for kind in faults.DAMAGE:
        scenarios[f"damaged camera: {kind}"] = _head_chances(
            head, faults.damaged_embeddings(pool, root, "eurosat", kind), classes)
    notes = {name: f"{np.mean(P.argmax(1) == truth):.1%} right" for name, P in scenarios.items()}
    notes["bit flips"] += f" (8 random bits flipped, seed {seed}: first seed that hurt 5-30 points)"
    notes["silent swap"] += " (30% of AnnualCrop labelled PermanentCrop, as sure as right answers)"
    rare = np.array([SHARES[c] <= 0.04 for c in classes])
    return Population([n for _, n in pool], truth, classes, scenarios, rare, notes, x=x["pool"],
                      launch="new camera (AID head)")


def eurosat_days(pop: Population, station: str, rate: float, days: int, seed: int, sat):
    """Stand-in replay: each photo slot gets a pool photo, drawn with the assumed class shares."""
    from src.orbit.replay import build_replay

    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(pop.truth == k) for k in range(len(pop.classes))]
    shares = np.array([SHARES[c] for c in pop.classes])
    picks = [rng.choice(by_class[k]) for k in rng.choice(len(shares), 20_000, p=shares)]
    weighted = [(Path(pop.names[i]), pop.classes[pop.truth[i]]) for i in picks]
    for day in range(days):
        start = FIRST_DAY + timedelta(days=day)
        windows, _ = build_replay(sat, station, start, 24, weighted, 30, rate, seed + 1000 * day,
                                  photo_bytes=TILE)
        passes = [(w.budget_bytes, np.array([pop.index[p.path.name] for p in w.photos]))
                  for w in windows if w.photos]
        yield f"{start:%Y-%m-%d}", passes


def real_population(train_days: list[str], test_days: list[str], keep=None) -> Population:
    """Real Sentinel-2A photos: head trained on train_days, every scenario on test_days."""
    from src.data import s2_real
    from src.model import faults
    from src.model.head import embed, fit_head

    classes = s2_real.CLASSES
    rows = [r for r in s2_real.load_chips() if keep is None or keep(r)]  # keep: optional photo filter (EDA)
    train = [r for r in rows if r["day"] in train_days[:-1]]
    tuning = [r for r in rows if r["day"] == train_days[-1]]
    test = [r for r in rows if r["day"] in test_days]

    def xs(part):
        return embed([(r["day"], r["name"]) for r in part], "siglip-base", root=s2_real.OUT_DIR,
                     dataset="s2real")[0]

    x_train, x_tuning, x_test = xs(train), xs(tuning), xs(test)
    head = fit_head(x_train, [r["label"] for r in train], x_tuning, [r["label"] for r in tuning],
                    verbose=False)
    truth = np.array([classes.index(r["label"]) for r in test])
    good = _head_chances(head, x_test, classes)

    from src.data.eurosat import dataset_dir
    from src.eval.sentinel_test import photo_sets
    e_final, e_tuning, e_train = photo_sets(100)
    xe = {k: embed(v, "siglip-base", root=dataset_dir(), dataset="eurosat")[0]
          for k, v in (("train", e_train), ("tuning", e_tuning))}
    eurosat_head = fit_head(xe["train"], [l for l, _ in e_train], xe["tuning"],
                            [l for l, _ in e_tuning], verbose=False)
    camera = faults.mapped_chances(faults.chances(eurosat_head, x_test), list(eurosat_head.classes_),
                                   EUROSAT_TO_REAL, classes)
    labels = np.array([r["label"] for r in test])
    flipped, seed = faults.harmful_flips(head, x_test, labels, n_flips=8)
    swap = faults.silent_swap(good, truth, classes.index("Cropland"), classes.index("ShrubGrass"),
                              0.3, np.random.default_rng(7))
    scenarios = {"healthy head": good, "new place (EuroSAT head)": camera, "silent swap": swap,
                 "bit flips": _head_chances(flipped, x_test, classes)}
    test_photos = [(r["day"], r["name"]) for r in test]
    for kind in ("blur", "noise"):
        scenarios[f"damaged camera: {kind}"] = _head_chances(
            head, faults.damaged_embeddings(test_photos, s2_real.OUT_DIR, "s2real", kind), classes)
    notes = {name: f"{np.mean(P.argmax(1) == truth):.1%} right" for name, P in scenarios.items()}
    notes["healthy head"] += f" (trained on {len(train):,} real photos of {train_days[0]}..{train_days[-2]})"
    notes["new place (EuroSAT head)"] += " (trained on EuroSAT, Europe; has no cloud/bare/snow class)"
    notes["silent swap"] += " (30% of Cropland labelled ShrubGrass, as sure as right answers)"
    notes["bit flips"] += f" (8 random bits flipped, seed {seed})"
    shares = np.bincount(truth, minlength=len(classes)) / len(truth)
    return Population([r["name"] for r in test], truth, classes, scenarios, shares < 0.05, notes,
                      x=x_test, launch="new place (EuroSAT head)")


def real_days(pop: Population, station: str, rate: float, days: list[str], sat):
    """The real photos in time order, each waiting for the next pass over the station."""
    from src.data import s2_real
    from src.orbit.contact_windows import contact_windows

    rows = {r["name"]: r for r in s2_real.load_chips(days)}
    for day in days:
        start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        passes = contact_windows(sat, station, start, start + timedelta(days=1))
        taken = sorted((datetime.fromisoformat(rows[n]["slot_utc"].replace("Z", "+00:00")), n)
                       for n in pop.names if rows[n]["day"] == day)
        waiting = defaultdict(list)
        for when, name in taken:
            if any(p.start <= when <= p.end for p in passes):
                continue  # busy talking to the ground
            nxt = next((i for i, p in enumerate(passes) if p.start > when), None)
            if nxt is not None:
                waiting[nxt].append(pop.index[name])
        yield day, [(int(passes[i].data_mb(rate) * 1e6), np.array(idx)) for i, idx in sorted(waiting.items())]


def run_day(pop: Population, P_all: np.ndarray, passes, rng, methods=None, check=None) -> dict:
    """Every method on one day: {method: (estimate, {interval kind: (low, high)}, bytes, safe share)}.
    methods: {name: (policy, ground method)}; a policy can also be a function (default: METHODS).
    check: the mission's SelfCheck (src/agent/self_check.py) for the "audit, self-check" method."""
    out = {}
    K = len(pop.classes)
    for method, (policy, how) in (methods or METHODS).items():
        if policy == "self-check" and check is None:
            continue
        send = policy if callable(policy) or policy == "self-check" else POLICIES[policy]
        parts, sizes, used, safe = [], [], 0, []
        for budget, idx in passes:
            P = P_all[idx]
            if policy not in ("raw in order", "raw random") and len(idx) * label_bytes(K) > budget:
                break  # this pass can't even carry the labels
            if policy == "self-check":
                sent, e, was_safe = check.run_pass(P, budget, TILE, lambda full, idx=idx: pop.Y[idx][full], rng)
                safe.append(was_safe)
            else:
                sent = send(P, budget, TILE, rng)
                e = estimate(P, sent, pop.Y[idx][sent.full], how, rng)
            parts.append(e)
            sizes.append(len(idx))
            used += sent.bytes_used
        if len(parts) < len(passes):
            continue
        day = combine(parts, sizes)
        out[method] = (day.value, {k: interval(day, k, LEVEL) for k in INTERVALS}, used,
                       float(np.mean(safe)) if safe else "")
    return out


def run(pop: Population, data: str, stations, rates, days, seeds, scenarios, methods=None) -> list[dict]:
    from src.orbit.contact_windows import load_satellite

    sat = load_satellite(tle_path=str(ROOT / "data" / "raw" / "tle" / "40697.tle"))
    rows = []
    for station in stations:
        for rate in rates:
            for seed in seeds:
                day_iter = (eurosat_days(pop, station, rate, days, seed, sat) if data == "eurosat"
                            else real_days(pop, station, rate, days, sat))
                # one mission per station x rate x seed: the self-check remembers all its checks
                checks = {s: SelfCheck(len(pop.classes)) for s in scenarios}
                for day, passes in day_iter:
                    idx_all = np.concatenate([idx for _, idx in passes])
                    truth = pop.Y[idx_all].mean(0)
                    for scenario in scenarios:
                        rng = np.random.default_rng([seed, rates.index(rate), stations.index(station),
                                                     int(day.replace("-", ""))])
                        for method, (value, bars, used, safe) in run_day(
                                pop, pop.scenarios[scenario], passes, rng, methods, checks[scenario]).items():
                            for k, c in enumerate(pop.classes):
                                row = {"data": data, "scenario": scenario, "station": station,
                                       "rate_mbps": rate, "seed": seed, "day": day, "method": method,
                                       "cls": c, "rare": bool(pop.rare[k]), "photos": len(idx_all),
                                       "mb": round(used / 1e6, 3), "truth": round(float(truth[k]), 7),
                                       "est": round(float(value[k]), 7), "safe_share": safe}
                                for kind, (lo, hi) in bars.items():
                                    row[f"lo_{kind}"] = round(float(lo[k]), 7)
                                    row[f"hi_{kind}"] = round(float(hi[k]), 7)
                                rows.append(row)
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    """One line per scenario x station x rate x method: error, bias, coverage, error-bar width."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scenario"], r["station"], r["rate_mbps"], r["method"])].append(r)
    out = []
    for (scenario, station, rate, method), rs in groups.items():
        err = np.array([r["est"] - r["truth"] for r in rs])
        by_class = defaultdict(list)
        for r, e in zip(rs, err):
            by_class[r["cls"]].append(e)
        line = {"scenario": scenario, "station": station, "rate_mbps": rate, "method": method,
                "runs": len(rs) // len(by_class), "mb_per_day": round(np.mean([r["mb"] for r in rs]), 2),
                "rmse_pts": round(100 * np.sqrt((err ** 2).mean()), 3),
                "worst_bias_pts": round(100 * max(abs(np.mean(v)) for v in by_class.values()), 3)}
        rare = np.array([r["rare"] for r in rs])
        for kind in INTERVALS:
            held = np.array([r[f"lo_{kind}"] - 1e-9 <= r["truth"] <= r[f"hi_{kind}"] + 1e-9 for r in rs])
            half = np.array([(r[f"hi_{kind}"] - r[f"lo_{kind}"]) / 2 for r in rs])
            line[f"holds_{kind}"] = round(held.mean(), 4)
            line[f"holds_{kind}_rare"] = round(held[rare].mean(), 4) if rare.any() else ""
            line[f"holds_{kind}_worst_class"] = round(min(np.mean([r[f"lo_{kind}"] - 1e-9 <= r["truth"] <= r[f"hi_{kind}"] + 1e-9
                                                                   for r in rs if r["cls"] == c])
                                                          for c in by_class), 4)
            line[f"halfwidth_{kind}_pts"] = round(100 * half.mean(), 3)
        safe = [r["safe_share"] for r in rs if r.get("safe_share", "") != ""]
        line["safe_share"] = round(float(np.mean(safe)), 3) if safe else ""  # self-check: passes in safe mode
        out.append(line)
    return out


def print_tables(summary: list[dict], pop: Population, rate: float, station: str) -> None:
    print(f"\n### {station.title()}, {rate:g} Mbps. Error in percentage points. 'holds' = how often "
          f"the {LEVEL:.0%} error bar contained the truth (safe bars; labels/triage: their pretend bars).")
    for scenario in pop.scenarios:
        mine = [s for s in summary if s["scenario"] == scenario and s["station"] == station
                and s["rate_mbps"] == rate]
        if not mine:
            continue
        print(f"\n== {scenario}: {pop.notes[scenario]} ==")
        print(f"{'method':<26} {'MB/day':>7} {'RMSE':>6} {'worst bias':>10} {'holds':>6} {'rare':>6} "
              f"{'worst class':>11} {'± pts':>6}")
        for s in mine:
            print(f"{s['method']:<26} {s['mb_per_day']:>7.1f} {s['rmse_pts']:>6.2f} {s['worst_bias_pts']:>10.2f} "
                  f"{s['holds_safe']:>6.0%} {s['holds_safe_rare'] if s['holds_safe_rare'] == '' else format(s['holds_safe_rare'], '.0%'):>6} "
                  f"{s['holds_safe_worst_class']:>11.0%} {s['halfwidth_safe_pts']:>6.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audited downlink vs today's policies, healthy or broken")
    parser.add_argument("--data", choices=["eurosat", "real"], default="eurosat")
    parser.add_argument("--stations", nargs="+", default=["delhi", "bengaluru", "svalbard"])
    parser.add_argument("--rates", type=float, nargs="+", help="downlink speeds in Mbps")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--scenarios", nargs="+", help="default: all")
    args = parser.parse_args()

    if args.data == "eurosat":
        pop = eurosat_population()
        rates = args.rates or [0.05, 0.1, 0.25, 0.5, 1, 2]
        days, seeds = args.days, SEEDS
    else:
        from src.data import s2_real
        all_days = sorted({r["day"] for r in s2_real.load_chips()})
        train_days, test_days = all_days[:-args.days], all_days[-args.days:]
        pop = real_population(train_days, test_days)
        rates = args.rates or [0.01, 0.02, 0.05, 0.1, 0.25]
        days, seeds = test_days, SEEDS  # same real photos; the seed only changes the random picks
    scenarios = args.scenarios or list(pop.scenarios)
    print(f"\n{len(pop.names):,} photos, classes: {', '.join(pop.classes)}")
    for name in scenarios:
        print(f"  {name:<28} {pop.notes[name]}")

    rows = run(pop, args.data, args.stations, rates, days, seeds, scenarios)
    summary = summarise(rows)
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    raw_out = ROOT / "data" / "processed" / f"audit_runs_{args.data}_{stamp}.csv.gz"
    with gzip.open(raw_out, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = EXP / f"audit_summary_{args.data}_{stamp}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    show_rate = 0.25 if 0.25 in rates else rates[len(rates) // 2]
    for station in args.stations[:1]:
        print_tables(summary, pop, show_rate, station)
    print(f"\nSaved {out.relative_to(ROOT)} and {raw_out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
