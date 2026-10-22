"""Experiment 12: retrain the onboard head using what the data analysis found (notebooks/01_data.ipynb).

What the data showed, and the fix tried for each:
- Rare classes are drowned by cloud (46% of photos), so snow and built-up are never predicted
  -> "balanced": every class counts the same in training. That makes the head expect rare classes far too
     often (dishonest chances), so "corrected" then scales the chances back to the real class mix
     (prior correction: multiply by each class's share of the training photos, renormalise).
- Snow and cloud are both blown-out white in the colour photos (42 of 43 snow photos)
  -> "infrared": six numbers from the green, red, near-infrared and short-wave infrared bands
     (src/data/s2_bands.py: snow index NDSI, vegetation index NDVI, SWIR brightness).
- Only 15 snow and 3 built-up photos in the training days
  -> "more days": 3 more weeks of what Sentinel-2A photographed (15 Sep - 5 Oct) for training.
- Is the 90M image model the limit?
  -> "big model": SigLIP 400M embeddings (made on a Kaggle GPU: python -m src.model.retrain --embed siglip).

The test is the same everywhere: what Sentinel-2A really photographed on 13-19 Oct 2026. Tuning (settings,
infrared weight) uses 12 Oct only. Recipes change one thing at a time, so each fix can be judged.

Run from the orbit-agent folder:
    python -m src.model.retrain               # every recipe (a few minutes; embeds new photos first)
    python -m src.model.retrain --embed siglip   # only make embeddings (what the Kaggle run does)
    python -m src.model.retrain --audit          # the main audit experiment, baseline head vs retrained head

Saves results/12_retraining/retrain_real_<time>.csv (one row per recipe), retrain_per_class_<time>.csv,
retrain_confusion_*.csv, and charts docs/figures/retrain.png and docs/figures/confusion_retrained.png.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data import s2_bands, s2_real
from src.model.head import calibration_error, embed, fit_head
from src.paths import result

ROOT = Path(__file__).resolve().parents[2]
TEST_DAYS, TUNING_DAY, TRAIN_DAYS = s2_real.TEST_DAYS, s2_real.TUNING_DAY, s2_real.TRAIN_DAYS
MIN_PHOTOS = 5  # a class needs this many test photos to count in balanced accuracy / macro-F1
IR_WEIGHTS = [0.03, 0.1, 0.3, 1.0]  # how strongly the six infrared numbers count; picked on the tuning day

BASE = dict(backbone="siglip-base", balanced=False, corrected=False, infrared=False, more_days=False)
RECIPES = {
    "1 baseline (as in experiments 9-11)": BASE,
    "2 + balanced classes": {**BASE, "balanced": True},
    "3 + chances corrected to the real mix": {**BASE, "balanced": True, "corrected": True},
    "4 + infrared bands": {**BASE, "balanced": True, "corrected": True, "infrared": True},
    "5 + 3 more weeks of training photos": {**BASE, "balanced": True, "corrected": True, "infrared": True, "more_days": True},
    "6 + bigger image model (SigLIP 400M)": {**BASE, "backbone": "siglip", "balanced": True, "corrected": True,
                                             "infrared": True, "more_days": True},
    # one change on its own, to see what each brings by itself
    "check: baseline + infrared only": {**BASE, "infrared": True},
    "check: baseline + more days only": {**BASE, "more_days": True},
}


def split(more_days: bool):
    """(training, tuning, test) rows of the real replay."""
    rows = s2_real.load_chips()
    extra = set(s2_real.EXTRA_DAYS) if more_days else set()
    train = [r for r in rows if r["day"] in TRAIN_DAYS or r["day"] in extra]
    tuning = [r for r in rows if r["day"] == TUNING_DAY]
    test = [r for r in rows if r["day"] in TEST_DAYS]
    return train, tuning, test


def embeddings(rows, backbone: str) -> np.ndarray:
    return embed([(r["day"], r["name"]) for r in rows], backbone, root=s2_real.OUT_DIR, dataset="s2real")[0]


def infrared(rows) -> np.ndarray:
    return np.stack([s2_bands.features(r) for r in rows])


class Recipe:
    """Turns photos into the head's input for one recipe, and trains the head."""

    def __init__(self, backbone="siglip-base", balanced=False, corrected=False, infrared=False, more_days=False):
        self.backbone, self.balanced, self.corrected = backbone, balanced, corrected
        self.use_ir, self.more_days = infrared, more_days
        self.ir_mean = self.ir_sd = None
        self.ir_weight = 0.0
        self.prior = None  # each class's share of the training photos (for the correction)

    def inputs(self, rows, x=None) -> np.ndarray:
        """Embedding (+ scaled infrared numbers). x: ready-made embeddings (e.g. of damaged photos)."""
        x = embeddings(rows, self.backbone) if x is None else x
        if not self.use_ir:
            return x
        ir = (infrared(rows) - self.ir_mean) / self.ir_sd
        return np.hstack([x, self.ir_weight * ir])

    def chances(self, head, x: np.ndarray) -> np.ndarray:
        """The head's chance for every class (scaled back to the real class mix if "corrected")."""
        P = head.predict_proba(x)
        if self.corrected:
            P = P * self.prior[None, :]
            P /= P.sum(1, keepdims=True)
        return P

    def fit(self, train, tuning, verbose=False):
        from sklearn.metrics import log_loss

        weight = "balanced" if self.balanced else None
        y_train, y_tuning = [r["label"] for r in train], [r["label"] for r in tuning]
        labels = sorted(set(y_train))
        self.prior = np.array([y_train.count(c) for c in labels]) / len(y_train)
        if not self.use_ir:
            return fit_head(self.inputs(train), y_train, self.inputs(tuning), y_tuning, verbose, weight)
        ir = infrared(train)
        self.ir_mean, self.ir_sd = ir.mean(0), ir.std(0) + 1e-9
        best = None
        for w in IR_WEIGHTS:  # how much the infrared numbers count: picked on the tuning day, never the test
            self.ir_weight = w
            head = fit_head(self.inputs(train), y_train, self.inputs(tuning), y_tuning, False, weight)
            loss = log_loss(y_tuning, self.chances(head, self.inputs(tuning)), labels=head.classes_)
            if best is None or loss < best[0]:
                best = (loss, w, head)
        self.ir_weight = best[1]
        if verbose:
            print(f"  infrared weight {best[1]} (lowest tuning log-loss)")
        return best[2]


def scores(truth, guess, P=None, classes=None, top_p=None) -> dict:
    from sklearn.metrics import f1_score

    truth, guess = np.asarray(truth), np.asarray(guess)
    counted = [c for c in s2_real.CLASSES if (truth == c).sum() >= MIN_PHOTOS]
    out = {"accuracy": (truth == guess).mean(),
           "balanced_accuracy": np.mean([(guess[truth == c] == c).mean() for c in counted]),
           "macro_f1": f1_score(truth, guess, labels=counted, average="macro", zero_division=0)}
    if top_p is not None:
        out["calibration_error"] = calibration_error(np.asarray(top_p), truth == guess)
    if P is not None:
        Y = (np.array(classes)[None, :] == truth[:, None]).astype(float)
        out["brier"] = ((P - Y) ** 2).sum(1).mean()
    for c in ("SnowIce", "Cropland", "Trees", "ShrubGrass", "Cloud", "Water", "Bare"):
        out[f"recall_{c}"] = (guess[truth == c] == c).mean() if (truth == c).any() else np.nan
    return out


def bootstrap(truth, guess, key="balanced_accuracy", times=1000):
    rng = np.random.default_rng(0)
    truth, guess = np.asarray(truth), np.asarray(guess)
    vals = [scores(truth[i], guess[i])[key] for i in (rng.integers(0, len(truth), len(truth)) for _ in range(times))]
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def has_embeddings(backbone: str, rows) -> bool:
    cache = ROOT / "data" / "processed" / f"embeddings_s2real_{backbone}.npz"
    if not cache.exists():
        return False
    saved = set(np.load(cache).files)
    return all(r["name"] in saved for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 12: retrain the head using what the data showed")
    parser.add_argument("--embed", help="only make embeddings of every real photo with this backbone (e.g. siglip)")
    parser.add_argument("--audit", action="store_true", help="rerun the audit with the baseline vs the retrained head")
    args = parser.parse_args()
    if args.audit:
        audit()
        return
    if args.embed:
        rows = s2_real.load_chips()
        print(f"Embedding {len(rows):,} real photos with {args.embed}...")
        embeddings(rows, args.embed)
        return

    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    summary, per_class, kept = [], [], {}
    for name, recipe in RECIPES.items():
        train, tuning, test = split(recipe["more_days"])
        if not has_embeddings(recipe["backbone"], train + tuning + test) and recipe["backbone"] != "siglip-base":
            print(f"{name}: skipped (no {recipe['backbone']} embeddings yet; run it on Kaggle first)")
            continue
        if recipe["infrared"] and sum(s2_bands.load(r) is None for r in train + tuning + test):
            missing = sum(s2_bands.load(r) is None for r in train + tuning + test)
            print(f"{name}: note, {missing} photos have no infrared bands yet (they get zeros)")
        r = Recipe(**recipe)
        head = r.fit(train, tuning, verbose=True)
        P = r.chances(head, r.inputs(test))
        classes = list(head.classes_)
        truth = [x["label"] for x in test]
        guess = [classes[i] for i in P.argmax(1)]
        s = scores(truth, guess, P, classes, P.max(1))
        low, high = bootstrap(truth, guess)
        summary.append({"recipe": name, "training_photos": len(train), "test_photos": len(test),
                        "infrared_weight": r.ir_weight if r.use_ir else "",
                        **{k: round(float(v), 4) for k, v in s.items()},
                        "balanced_accuracy_low": round(low, 4), "balanced_accuracy_high": round(high, 4)})
        print(f"{name:<40} acc {s['accuracy']:.1%} | balanced {s['balanced_accuracy']:.1%} ({low:.1%}-{high:.1%}) | "
              f"macro-F1 {s['macro_f1']:.1%} | snow recall {s['recall_SnowIce']:.0%} | ECE {s['calibration_error']:.3f} | "
              f"{len(train):,} training photos")
        kept[name] = (truth, guess)
        from src.eval.metrics_report import per_class as per_class_rows
        per_class += [{"recipe": name, **row} for row in per_class_rows(truth, guess, s2_real.CLASSES)]

    out = result(f"retrain_real_{stamp}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(result(f"retrain_per_class_{stamp}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class[0]))
        w.writeheader()
        w.writerows(per_class)

    # confusion matrices: baseline vs the best main recipe (by balanced accuracy)
    from src.eval.metrics_report import DOCS, confusion, draw

    main_recipes = [s for s in summary if not s["recipe"].startswith("check")]
    best = max(main_recipes, key=lambda s: s["balanced_accuracy"])["recipe"]
    panels = []
    for name in (summary[0]["recipe"], best):
        M, cols = confusion(*kept[name], s2_real.CLASSES)
        with open(result(f"retrain_confusion_{name.split()[0]}_{stamp}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true \\ given"] + cols)
            for c, row in zip(s2_real.CLASSES, M):
                w.writerow([c] + list(row))
        acc = next(s for s in summary if s["recipe"] == name)
        panels.append((f"{name[:38]} ({acc['balanced_accuracy']:.0%} balanced)", M, s2_real.CLASSES, cols))
    draw(panels, DOCS / "confusion_retrained.png", "Real Sentinel-2A photos, 13-19 Oct: before and after retraining")
    chart(summary, DOCS / "retrain.png")
    print(f"\nBest: {best}\nSaved {out.relative_to(ROOT)} and docs/figures/retrain.png, confusion_retrained.png")


# Chosen BEFORE seeing any retraining result: every data fix, and it still allows the damaged-camera cases.
AUDIT_RECIPE = "5 + 3 more weeks of training photos"


def audit() -> None:
    """The main experiment (src/eval/audit_test.py, real replay) with the baseline head and the retrained head.
    Same photos, same random checks (all methods and speeds run, so the random draws match the main run)."""
    from src.eval import audit_test as at

    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    rows = []
    for label, recipe in (("baseline head", None), (f"retrained head ({AUDIT_RECIPE})", RECIPES[AUDIT_RECIPE])):
        pop = at.real_population(s2_real.MAIN_DAYS[:7], s2_real.MAIN_DAYS[7:], recipe=recipe)
        print(f"\n{label}: " + "; ".join(f"{k} {v.split(' (')[0]}" for k, v in pop.notes.items()))
        summary = at.summarise(at.run(pop, "real", ["delhi", "bengaluru", "svalbard"], [0.01, 0.02, 0.05, 0.1, 0.25],
                                      s2_real.MAIN_DAYS[7:], at.SEEDS, list(pop.scenarios)))
        rows += [{"head": label, "head_right": pop.notes[r["scenario"]].split("%")[0] + "%", **r} for r in summary]
    out = result(f"retrain_audit_{stamp}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import pandas as pd
    t = pd.DataFrame(rows)
    t = t[t.method.isin(["labels only", "triage (v4)", "raw random", "audit, self-check"]) & t.rate_mbps.isin([0.05, 0.25])]
    t = t.groupby(["scenario", "rate_mbps", "method", "head"])[["rmse_pts", "holds_scaled+1"]].mean().round(3)
    print(t.to_string())
    print(f"\nSaved {out.relative_to(ROOT)}")


def chart(summary, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.eval.eda import BLUE, INK, MUTED, YELLOW

    plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
                         "text.color": INK, "axes.labelcolor": "#bdbdbd", "axes.edgecolor": "#444",
                         "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#222", "font.size": 10.5,
                         "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    fig, ax = plt.subplots(figsize=(8.8, 0.55 * len(summary) + 1.6))
    y = np.arange(len(summary))[::-1]
    acc = np.array([s["accuracy"] for s in summary]) * 100
    bal = np.array([s["balanced_accuracy"] for s in summary]) * 100
    lo = bal - np.array([s["balanced_accuracy_low"] for s in summary]) * 100
    hi = np.array([s["balanced_accuracy_high"] for s in summary]) * 100 - bal
    ax.plot(acc, y + 0.12, "o", color=BLUE, ms=7, label="accuracy")
    ax.errorbar(bal, y - 0.12, xerr=[lo, hi], fmt="o", color=YELLOW, ms=7, lw=2, label="balanced accuracy (95% range)")
    for yi, a, b in zip(y, acc, bal):
        ax.text(max(a, b) + 1.2, yi, f"{a:.0f}% / {b:.0f}%", va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(y, [s["recipe"] for s in summary])
    ax.set_xlabel("head on real Sentinel-2A photos, 13-19 Oct (%)")
    ax.set_xlim(min(bal.min() - lo.max(), acc.min()) - 3, 100)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_title("Retraining with what the data showed", pad=26)
    fig.text(0.01, 0.01, "Same test days for every recipe. Balanced accuracy over classes with 5+ test photos.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
