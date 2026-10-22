"""Exploratory data analysis: what is in the photos, how clean are the labels, and how much do the numbers move?

Three data sets:
- real:    what Sentinel-2A really photographed, 6-19 Oct 2026 (src/data/s2_real.py), 8 classes
- eurosat: EuroSAT RGB (Sentinel-2, 64 px), 27,000 photos, 10 classes
- aid:     the AID photos we downloaded (aerial, 600 px), 30 classes

For the real replay (the one the main results use) it also:
1. flags every photo's data problems (no-data strips, partly cloudy, mixed land types, blown-out white),
2. checks each label against its 10 most similar photos (nearest neighbours in SigLIP space),
3. compares training days (6-12 Oct) with test days (13-19 Oct): classes, places, near-duplicates,
4. measures where the healthy head's mistakes are (clean photos vs flagged ones),
5. re-runs the head and its scores with clean data and other day splits, with bootstrap ranges:
   how much do the numbers move?
6. re-runs the main audit experiment on clean photos only: do its conclusions change?

Run from the orbit-agent folder (uses saved photos and embeddings; about 2 min):
    python -m src.eval.eda

Saves results/eda_real_chips.csv (one row per real photo with its flags), eda_summary_<time>.csv
(every number printed), eda_sensitivity_real_<time>.csv, eda_audit_clean_<time>.csv (the audit on all vs
clean photos), and charts docs/figures/eda_{classes,quality,sensitivity,map}.png.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from src.paths import result

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "figures"
BLUE, YELLOW, GREY = "#3987e5", "#c98500", "#8b8b94"  # validated set used by the other charts
INK, MUTED = "#e6e6e6", "#9a9a9a"

# Data-problem flags (fixed before looking at any model result)
NODATA = 0.05        # more than 5% of the photo is "no data" (black strips at the image edge)
PARTLY_CLOUDY = 0.2  # a land-labelled photo with 20-50% of its visible part under cloud
BORDER_CLOUD = 0.7   # a cloud-labelled photo that is only 50-70% cloud
MIXED = 0.5          # no land type covers at least half of a land-labelled photo
BLOWN_OUT = 0.5      # a land-labelled photo with more than half its pixels pure white

found: list[dict] = []  # every number, also saved as CSV


def note(section: str, what: str, value, detail: str = "") -> None:
    found.append({"section": section, "what": what, "value": value, "detail": detail})
    print(f"  {what}: {value}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------- real replay

def real_table():
    """One row per real photo with its numbers and data-problem flags."""
    from src.data import s2_real

    rows = s2_real.load_chips(s2_real.MAIN_DAYS)  # the main replay, 6-19 Oct
    C = s2_real.CLASSES
    out = []
    for r in rows:
        a = np.asarray(Image.open(s2_real.OUT_DIR / r["day"] / r["name"]).convert("RGB")).astype(float)
        nodata, cloud, snow = float(r["nodata"]), float(r["cloud"]), float(r["snow"])
        seen = max(1 - nodata, 1e-9)
        wc = {c: float(r[f"wc_{c}"]) for c in C[1:]}
        label = r["label"]
        cloud_seen = cloud / seen
        land = label not in ("Cloud", "SnowIce")
        purity = cloud_seen if label == "Cloud" else (snow / seen if label == "SnowIce" else wc[label])
        flags = {
            "nodata": nodata > NODATA,
            "partly_cloudy": land and cloud_seen >= PARTLY_CLOUDY,
            "border_cloud": label == "Cloud" and cloud_seen < BORDER_CLOUD,
            "mixed": land and wc[label] < MIXED,
            "blown_out": land and (a.min(2) >= 250).mean() > BLOWN_OUT,
        }
        out.append({"name": r["name"], "day": r["day"], "lat": float(r["lat"]), "lon": float(r["lon"]),
                     "label": label, "cloud_seen": round(cloud_seen, 3), "purity": round(purity, 3),
                     "nodata": nodata, "brightness": round(a.mean(), 1),
                     "white": round(float((a.min(2) >= 250).mean()), 3),
                     "black": round(float((a.max(2) == 0).mean()), 3),
                     **flags, "clean": not any(flags.values())})
    return out, C


def knn_agreement(x: np.ndarray, labels: np.ndarray, k: int = 10) -> np.ndarray:
    """For each photo: share of its k most similar OTHER photos with the same label (1 = all agree)."""
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    nearest = np.argsort(-sim, axis=1)[:, :k]
    return (labels[nearest] == labels[:, None]).mean(1)


def km_between(a, b) -> np.ndarray:
    """Great-circle distance (km) from each point in a to its nearest point in b. a, b: (n, 2) lat/lon."""
    la1, lo1 = np.radians(a[:, :1]), np.radians(a[:, 1:])
    la2, lo2 = np.radians(b[:, 0])[None], np.radians(b[:, 1])[None]
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return (2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))).min(1)


MIN_PHOTOS = 5  # a class needs at least this many test photos to count in balanced accuracy / macro-F1


def scores(truth, guess, classes):
    """Accuracy, and balanced accuracy + macro-F1 over the classes with at least MIN_PHOTOS test photos
    (BuiltUp has 0-1, so it is left out of every variant: otherwise one photo moves the score by 10 points)."""
    from sklearn.metrics import f1_score

    truth, guess = np.asarray(truth), np.asarray(guess)
    counted = [c for c in classes if (truth == c).sum() >= MIN_PHOTOS]
    return {"accuracy": float((truth == guess).mean()),
            "balanced_accuracy": float(np.mean([(guess[truth == c] == c).mean() for c in counted])),
            "macro_f1": float(f1_score(truth, guess, labels=counted, average="macro", zero_division=0))}


def bootstrap(truth, guess, classes, times=1000, seed=0):
    """95% range of each score if we had other photos like these."""
    rng = np.random.default_rng(seed)
    truth, guess = np.asarray(truth), np.asarray(guess)
    draws = [scores(truth[i], guess[i], classes) for i in (rng.integers(0, len(truth), len(truth)) for _ in range(times))]
    return {k: (np.percentile([d[k] for d in draws], 2.5), np.percentile([d[k] for d in draws], 97.5)) for k in draws[0]}


def train_and_score(table, x, classes, train_days, tuning_day, test_days, keep_train=None, keep_test=None):
    """Train the 30 KB head on some days, test on others (same steps as src/eval/audit_test.py)."""
    from src.model.head import fit_head

    idx = lambda days, keep: [i for i, r in enumerate(table) if r["day"] in days and (keep is None or keep(r))]
    tr, tu, te = idx(train_days, keep_train), idx([tuning_day], keep_train), idx(test_days, keep_test)
    head = fit_head(x[tr], [table[i]["label"] for i in tr], x[tu], [table[i]["label"] for i in tu], verbose=False)
    truth = [table[i]["label"] for i in te]
    guess = list(head.predict(x[te]))
    return truth, guess, len(tr), len(te)


def real_eda(stamp: str) -> dict:
    from src.model.head import embed
    from src.data import s2_real

    print("\n================ Real Sentinel-2A replay (6-19 Oct 2026) ================")
    table, C = real_table()
    days = s2_real.MAIN_DAYS
    train_days, tuning_day, test_days = days[:6], days[6], days[7:]
    x = embed([(r["day"], r["name"]) for r in table], "siglip-base", root=s2_real.OUT_DIR, dataset="s2real")[0]
    labels = np.array([r["label"] for r in table])
    is_test = np.array([r["day"] in test_days for r in table])
    is_train = np.array([r["day"] in days[:7] for r in table])

    print("1. Size and classes")
    note("real", "photos", len(table), f"{len(days)} days, {days[0]}..{days[-1]}")
    for c in C:
        tr, te = (labels[is_train] == c).mean(), (labels[is_test] == c).mean()
        note("real", f"share {c} (train days / test days)", f"{tr:.1%} / {te:.1%}",
             f"{(labels[is_train] == c).sum()} / {(labels[is_test] == c).sum()} photos")
    shift = 0.5 * sum(abs((labels[is_train] == c).mean() - (labels[is_test] == c).mean()) for c in C)
    note("real", "class-mix difference train vs test (total variation, 0 = same)", f"{shift:.3f}")

    print("2. Data problems (flags fixed before looking at results)")
    for flag, text in [("nodata", f"more than {NODATA:.0%} no-data (black strips)"),
                       ("partly_cloudy", f"land label but {PARTLY_CLOUDY:.0%}-50% cloud"),
                       ("border_cloud", f"cloud label but only 50-{BORDER_CLOUD:.0%} cloud"),
                       ("mixed", f"land label covers under {MIXED:.0%} of the photo"),
                       ("blown_out", f"land label but over {BLOWN_OUT:.0%} pure white pixels")]:
        n = sum(r[flag] for r in table)
        by = Counter(r["label"] for r in table if r[flag])
        note("real", f"flag {flag}: {text}", f"{n} ({n / len(table):.1%})", ", ".join(f"{k} {v}" for k, v in by.most_common(4)))
    clean = np.array([r["clean"] for r in table])
    note("real", "clean photos (no flag)", f"{clean.sum()} ({clean.mean():.1%})")
    white = np.array([r["white"] for r in table])
    for c in ("Cloud", "SnowIce", "Bare"):
        m = labels == c
        note("real", f"{c} photos more than half pure white", f"{(white[m] > 0.5).sum()} of {m.sum()}",
             "blown-out: no texture left to tell snow from cloud" if c != "Bare" else "")
    note("real", "brightness by class (0-255)", ", ".join(f"{c} {np.mean([r['brightness'] for r in table if r['label'] == c]):.0f}" for c in C))
    lat = np.array([r["lat"] for r in table])
    note("real", "latitude range", f"{lat.min():.0f} to {lat.max():.0f}", f"{(abs(lat) > 60).sum()} photos beyond 60 degrees")

    print("3. Do labels agree with similar photos? (10 nearest neighbours, all 14 days)")
    agree = knn_agreement(x, labels)
    for c in C:
        m = labels == c
        note("real", f"neighbour agreement {c}", f"{agree[m].mean():.0%}", f"{m.sum()} photos")
    suspect = agree <= 0.2
    note("real", "photos whose label at most 2 of 10 neighbours share (suspect)", f"{suspect.sum()} ({suspect.mean():.1%})",
         f"{(suspect & ~clean).sum()} of them are flagged")
    eye = {r["name"]: r for r in csv.DictReader(open(result("label_check_real.csv")))} if (result("label_check_real.csv")).exists() else {}
    if eye:
        by_name = {r["name"]: i for i, r in enumerate(table)}
        for verdict in ("agree", "unsure", "disagree"):
            ids = [by_name[n] for n, r in eye.items() if r["by_eye"] == verdict and n in by_name]
            note("real", f"neighbour agreement of eye-check '{verdict}' photos", f"{agree[ids].mean():.0%}", f"{len(ids)} photos")

    print("4. Training days vs test days")
    pos = np.array([[r["lat"], r["lon"]] for r in table])
    near = km_between(pos[is_test], pos[is_train])
    note("real", "test photos within 10 km of a training photo", f"{(near < 10).sum()} of {is_test.sum()}",
         f"median distance to the nearest training photo {np.median(near):.0f} km")
    twin = (x[is_test] @ x[is_train].T).max(1) > 0.95
    note("real", "test photos with a near-twin (cosine > 0.95) in training", f"{twin.sum()} ({twin.mean():.0%})")

    print("5. Where are the healthy head's mistakes?")
    truth, guess, n_tr, n_te = train_and_score(table, x, C, train_days, tuning_day, test_days)
    test_rows = [r for r in table if r["day"] in test_days]
    wrong = np.array([t != g for t, g in zip(truth, guess)])
    groups = [("clean", lambda r: r["clean"])] + [(f, lambda r, f=f: r[f]) for f in
                                                   ("partly_cloudy", "border_cloud", "mixed", "nodata", "blown_out")]
    quality = []
    for name, test in groups:
        m = np.array([test(r) for r in test_rows])
        if m.any():
            quality.append((name, m.sum(), wrong[m].mean()))
            note("real", f"healthy head error rate on {name} test photos", f"{wrong[m].mean():.1%}", f"{m.sum()} photos")
    flagged = ~np.array([r["clean"] for r in test_rows])
    note("real", "share of all mistakes that are on flagged photos", f"{(wrong & flagged).sum() / wrong.sum():.0%}",
         f"flagged photos are {flagged.mean():.0%} of the test days")
    snow_wrong = [(t, g) for t, g in zip(truth, guess) if t in ("SnowIce", "BuiltUp")]
    note("real", "SnowIce / BuiltUp test photos and what they were called", len(snow_wrong), str(Counter(snow_wrong).most_common(3)))

    print("6. How much do the numbers move?")
    variants = []

    def run(name, train_d, tuning_d, test_d, keep_train=None, keep_test=None):
        t, g, ntr, nte = train_and_score(table, x, C, train_d, tuning_d, test_d, keep_train, keep_test)
        s, ci = scores(t, g, C), bootstrap(t, g, C)
        variants.append({"variant": name, "train_photos": ntr, "test_photos": nte, **{k: round(v, 4) for k, v in s.items()},
                         **{f"{k}_low": round(v[0], 4) for k, v in ci.items()}, **{f"{k}_high": round(v[1], 4) for k, v in ci.items()}})
        print(f"  {name:<46} acc {s['accuracy']:.1%} ({ci['accuracy'][0]:.1%}-{ci['accuracy'][1]:.1%}) | "
              f"balanced {s['balanced_accuracy']:.1%} ({ci['balanced_accuracy'][0]:.1%}-{ci['balanced_accuracy'][1]:.1%}) | "
              f"macro-F1 {s['macro_f1']:.1%} | {ntr} train / {nte} test")

    is_clean = lambda r: r["clean"]
    run("as used: train 6-11, tune 12, test 13-19", train_days, tuning_day, test_days)
    run("same head, clean test photos only", train_days, tuning_day, test_days, keep_test=is_clean)
    run("clean training + clean test photos", train_days, tuning_day, test_days, is_clean, is_clean)
    run("reversed: train 13-18, tune 19, test 6-12", days[7:13], days[13], days[:7])
    even, odd = days[0::2], days[1::2]
    run("interleaved: train even days, test odd days", even[:-1], even[-1], odd)
    run("interleaved: train odd days, test even days", odd[:-1], odd[-1], even)

    print("7. Does the audit still hold on clean photos? (Delhi, Bengaluru, Svalbard x 7 days x 3 orders)")
    print("   (bit flips: the flips are re-picked on the clean photos, so that broken model differs between the two)")
    from src.eval import audit_test as at

    methods = {k: at.METHODS[k] for k in ("labels only", "triage (v4)", "raw random", "audit, self-check")}
    clean_names = {r["name"] for r in table if r["clean"]}
    audit_rows = []
    for photos, keep in (("all photos", None), ("clean photos", lambda r: r["name"] in clean_names)):
        pop = at.real_population(days[:7], days[7:], keep=keep)
        scenarios = ["healthy head", "new place (EuroSAT head)", "bit flips"]
        # same speed list as the main run, so the random checks are drawn exactly as there
        summary = at.summarise(at.run(pop, "real", ["delhi", "bengaluru", "svalbard"], [0.01, 0.02, 0.05, 0.1, 0.25],
                                      days[7:], at.SEEDS, scenarios))  # all methods: they share one random stream
        for scenario in scenarios:
            for rate in (0.05, 0.25):
                mine = [r for r in summary if r["scenario"] == scenario and r["rate_mbps"] == rate]
                for method in methods:
                    m = [r for r in mine if r["method"] == method]
                    error = np.mean([r["rmse_pts"] for r in m])
                    held = np.mean([r["holds_scaled+1"] for r in m])
                    audit_rows.append({"photos": photos, "scenario": scenario, "rate_mbps": rate, "method": method,
                                       "rmse_pts": round(float(error), 2), "bar_held": round(float(held), 3)})
    for scenario in ("healthy head", "new place (EuroSAT head)", "bit flips"):
        for rate in (0.05, 0.25):
            cells = []
            for method in methods:
                a = [r for r in audit_rows if r["scenario"] == scenario and r["rate_mbps"] == rate and r["method"] == method]
                cells.append(f"{method}: " + " -> ".join(f"{r['rmse_pts']:.1f}/{r['bar_held']:.0%}" for r in a))
            print(f"  {scenario}, {rate:g} Mbps (all -> clean, error pts / bar held): " + " | ".join(cells))
    with open(result(f"eda_audit_clean_{stamp}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0]))
        w.writeheader()
        w.writerows(audit_rows)

    # charts
    charts(table, C, labels, is_train, is_test, quality, variants)
    with open(result("eda_real_chips.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]) + ["neighbour_agreement"])
        w.writeheader()
        for r, a in zip(table, agree):
            w.writerow({**r, "neighbour_agreement": round(float(a), 2)})
    with open(result(f"eda_sensitivity_real_{stamp}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(variants[0]))
        w.writeheader()
        w.writerows(variants)
    return {"variants": variants}


# ---------------------------------------------------------------- EuroSAT and AID

def image_stats(paths):
    """(md5 of the pixels, brightness, width, height, file bytes) for each photo."""
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        a = np.asarray(img)
        out.append((hashlib.md5(a.tobytes()).hexdigest(), float(a.mean()), img.width, img.height, p.stat().st_size))
    return out


def eurosat_eda() -> None:
    from src.data.eurosat import CLASSES, dataset_dir, photos_in

    print("\n================ EuroSAT (Sentinel-2 RGB, 64 px) ================")
    root = dataset_dir()
    paths = {c: photos_in(root / c) for c in CLASSES}
    note("eurosat", "photos", sum(len(v) for v in paths.values()),
         ", ".join(f"{c} {len(v)}" for c, v in paths.items()))
    stats = {c: image_stats(v) for c, v in paths.items()}
    hashes = Counter(h for v in stats.values() for h, *_ in v)
    dup = sum(n - 1 for n in hashes.values() if n > 1)
    note("eurosat", "exact duplicate photos (same pixels)", dup)
    cross = defaultdict(set)
    for c, v in stats.items():
        for h, *_ in v:
            cross[h].add(c)
    note("eurosat", "identical photos filed under two classes", sum(len(s) > 1 for s in cross.values()))
    sizes = Counter((w, h) for v in stats.values() for _, _, w, h, _ in v)
    note("eurosat", "photo sizes", dict(sizes))
    note("eurosat", "brightness by class (0-255)", ", ".join(f"{c} {np.mean([b for _, b, *_ in v]):.0f}" for c, v in stats.items()))
    dark = {c: sum(b < 15 for _, b, *_ in v) for c, v in stats.items()}
    note("eurosat", "very dark photos (mean < 15)", sum(dark.values()), ", ".join(f"{c} {n}" for c, n in dark.items() if n))
    coords = ROOT / "data" / "processed" / "eurosat_coords.csv"
    if coords.exists():
        rows = list(csv.DictReader(open(coords)))
        lat, lon = np.array([float(r["lat"]) for r in rows]), np.array([float(r["lon"]) for r in rows])
        note("eurosat", "where", f"latitude {lat.min():.0f} to {lat.max():.0f}, longitude {lon.min():.0f} to {lon.max():.0f}",
             f"{len({(r['epsg'], int(float(r['x']) // 10000), int(float(r['y']) // 10000)) for r in rows}):,} different 10 km blocks")
        same = Counter((round(float(r["lat"]), 4), round(float(r["lon"]), 4)) for r in rows)
        note("eurosat", "photos at exactly the same spot as another photo", sum(n - 1 for n in same.values() if n > 1))


def aid_eda() -> None:
    from src.data.aid import AID_DIR, FILE_LIST, LABELS

    print("\n================ AID (aerial, Google Earth) ================")
    files = json.loads(FILE_LIST.read_text())
    if isinstance(files, dict):
        full = {c: len(v) for c, v in files.items()}
        note("aid", "full dataset (all 10,000)", sum(full.values()),
             f"smallest class {min(full, key=full.get)} {min(full.values())}, biggest {max(full, key=full.get)} {max(full.values())}")
    paths = {c: sorted((AID_DIR / c).glob("*.jpg")) for c in LABELS if (AID_DIR / c).exists()}
    note("aid", "photos downloaded here", sum(len(v) for v in paths.values()))
    stats = {c: image_stats(v) for c, v in paths.items()}
    hashes = Counter(h for v in stats.values() for h, *_ in v)
    note("aid", "exact duplicate photos (same pixels)", sum(n - 1 for n in hashes.values() if n > 1))
    note("aid", "photo sizes", dict(Counter((w, h) for v in stats.values() for _, _, w, h, _ in v)))
    kb = {c: np.mean([s for *_, s in v]) / 1000 for c, v in stats.items()}
    note("aid", "JPEG size by class (KB): smallest / biggest",
         f"{min(kb, key=kb.get)} {min(kb.values()):.0f} / {max(kb, key=kb.get)} {max(kb.values()):.0f}",
         "why the replay gives every photo the same size: small files are easy scenes")


# ---------------------------------------------------------------- charts

def charts(table, C, labels, is_train, is_test, quality, variants) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
                         "text.color": INK, "axes.labelcolor": "#bdbdbd", "axes.edgecolor": "#444",
                         "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#222", "font.size": 10.5,
                         "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})

    def finish(fig, ax, out, note_text):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.text(0.01, 0.01, note_text, fontsize=8.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(out, dpi=150)
        plt.close(fig)

    # 1. class mix, training days vs test days
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    order = sorted(C, key=lambda c: (labels == c).mean())
    y = np.arange(len(order))
    tr = [(labels[is_train] == c).mean() * 100 for c in order]
    te = [(labels[is_test] == c).mean() * 100 for c in order]
    ax.barh(y + 0.2, tr, 0.38, color=BLUE, label="training days (6-12 Oct)")
    ax.barh(y - 0.2, te, 0.38, color=YELLOW, label="test days (13-19 Oct)")
    for yi, a, b in zip(y, tr, te):
        ax.text(max(a, b) + 0.8, yi, f"{a:.1f}% / {b:.1f}%", va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(y, order)
    ax.set_xlabel("share of the photos (%)")
    ax.set_xlim(0, max(tr + te) * 1.25)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("What Sentinel-2A really photographed: class mix, training vs test days")
    finish(fig, ax, DOCS / "eda_classes.png", "Labels from the image's cloud map and ESA WorldCover 2021. "
           "Cloud dominates; BuiltUp and SnowIce are rare.")

    # 2. error rate by data problem
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    names = {"clean": "clean photos", "partly_cloudy": "land label, 20-50% cloud", "border_cloud": "cloud label, only 50-70% cloud",
             "mixed": "mixed land types", "nodata": "no-data strips", "blown_out": "land label, mostly pure white"}
    q = sorted(quality, key=lambda t: t[2])
    y = np.arange(len(q))
    ax.barh(y, [e * 100 for _, _, e in q], 0.6, color=[GREY if n == "clean" else BLUE for n, _, _ in q])
    for yi, (n, cnt, e) in zip(y, q):
        ax.text(e * 100 + 1, yi, f"{e:.0%} wrong ({cnt} photos)", va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(y, [names[n] for n, _, _ in q])
    ax.set_xlabel("healthy head: share of photos labelled wrong (%)")
    ax.set_xlim(0, max(e for *_, e in q) * 140)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_title("Mistakes pile up on messy photos (test days 13-19 Oct)")
    finish(fig, ax, DOCS / "eda_quality.png", "A photo can have more than one problem. Grey = photos with no problem flag.")

    # 3. sensitivity: accuracy and balanced accuracy per variant, with 95% ranges
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(variants))[::-1]
    for key, colour, label, dy in (("accuracy", BLUE, "accuracy", 0.12), ("balanced_accuracy", YELLOW, "balanced accuracy", -0.12)):
        v = np.array([r[key] for r in variants]) * 100
        lo = v - np.array([r[f"{key}_low"] for r in variants]) * 100
        hi = np.array([r[f"{key}_high"] for r in variants]) * 100 - v
        ax.errorbar(v, y + dy, xerr=[lo, hi], fmt="o", color=colour, ms=7, lw=2, capsize=0, label=label)
    ax.set_yticks(y, [r["variant"] for r in variants])
    ax.set_xlabel("healthy head on real photos (%), dot = score, line = 95% range")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_title("How much do the numbers move?", pad=24)
    finish(fig, ax, DOCS / "eda_sensitivity.png", "Same head and steps each time; only the photos and the day split change. "
           "Balanced accuracy over classes with 5+ test photos (BuiltUp left out).")

    # 4. map
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for m, colour, label in ((is_train, BLUE, "training days"), (is_test, YELLOW, "test days")):
        pts = np.array([[r["lon"], r["lat"]] for r, k in zip(table, m) if k])
        ax.scatter(pts[:, 0], pts[:, 1], s=9, color=colour, label=label, linewidths=0.4, edgecolors="black")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", markerscale=2)
    ax.set_title("Where the real photos are: training days vs test days")
    finish(fig, ax, DOCS / "eda_map.png", "Sentinel-2A photographs land and coasts in daylight; open ocean has no images.")


def main() -> None:
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    real_eda(stamp)
    eurosat_eda()
    aid_eda()
    out = result(f"eda_summary_{stamp}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "what", "value", "detail"])
        w.writeheader()
        w.writerows(found)
    print(f"\nSaved {out.relative_to(ROOT)}, results/eda_real_chips.csv, results/eda_sensitivity_real_{stamp}.csv, "
          "docs/figures/eda_{classes,quality,sensitivity,map}.png")


if __name__ == "__main__":
    main()
