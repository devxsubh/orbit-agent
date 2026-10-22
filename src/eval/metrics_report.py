"""More than accuracy: confusion matrices and per-class scores for every onboard labeller.

Accuracy alone hides three things:
- WHICH classes get mixed up (the confusion matrix),
- how the rare classes do (balanced accuracy and macro-F1 give every class the same weight;
  on the real replay 46% of photos are cloud and 0.1% built-up, so accuracy is mostly "cloud"),
- whether the chances are honest (calibration error, Brier score, log-loss).

Test sets (the same final photos as everywhere else, never used for building):
- aid:     300 AID final photos (21-30 per class), 30 classes
- eurosat: 1,000 EuroSAT final photos (Sentinel-2), 10 classes
- real:    what Sentinel-2A really photographed on 13-19 Oct 2026 (head trained on 6-12 Oct),
           8 classes, the healthy head and every broken one from src/model/faults.py

Metrics (all from sklearn, the usual definitions):
- accuracy: share of photos labelled right
- balanced accuracy: the average of each class's recall (every class counts the same)
- precision of a class: of the photos labelled X, how many really are X
- recall of a class: of the photos that really are X, how many were labelled X
- F1: one number from precision and recall; macro = plain average over classes,
  weighted = average weighted by how many photos each class has
- Cohen's kappa: agreement with the truth beyond what lucky guessing would give (1 = perfect, 0 = chance)
- calibration error (ECE): how far "says 80% sure" is from "is right 80% of the time" (0 = perfect)
- Brier score and log-loss: how good ALL the chances are, not only the top one (lower = better);
  only where every class's chance is known (Gemma gives no chances; the CSVs keep only the top two)

Run from the orbit-agent folder (uses saved answers and embeddings; about 1 min):
    python -m src.eval.metrics_report

Saves (in experiments/): metrics_summary_<time>.csv (one row per test set x model),
metrics_per_class_<time>.csv, and confusion/<test set>__<model>.csv (rows = truth, columns = label given).
Charts: docs/confusion_{real,eurosat,aid}.png.
"""
from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"

# (test set, model name, answers CSV, where its full chances come from or None)
SAVED = [
    ("aid", "small head (SigLIP-base + 30 KB head)", "head_aid_siglip-base_n20_final.csv", "head"),
    ("aid", "big head (SigLIP 400M + head)", "head_aid_siglip_final.csv", None),
    ("aid", "tiny CNN (MobileNetV3-Small)", "cnn_aid_mobilenet_v3_small_n20_final.csv", None),
    ("aid", "SigLIP zero-shot", "zeroshot_aid_siglip-base_final.csv", "zeroshot"),
    ("aid", "Gemma 3 4B (no training)", "confidence_aid_test_gemma3-4b_r3_n10_skip20.csv", None),
    ("eurosat", "small head (SigLIP-base + 30 KB head)", "head_eurosat_siglip-base_n100_final.csv", "head"),
    ("eurosat", "big head (SigLIP 400M + head)", "head_eurosat_siglip_n100_final.csv", None),
    ("eurosat", "tiny CNN (MobileNetV3-Small)", "cnn_eurosat_mobilenet_v3_small_n100_final.csv", None),
    ("eurosat", "SigLIP zero-shot", "zeroshot_eurosat_siglip-base_final.csv", "zeroshot"),
    ("eurosat", "Gemma 3 4B (no training, its 100 photos)", "baseline_eurosat_gemma3-4b_v1_448px_20261018_1528.csv", None),
]
UNKNOWN = "(no valid answer)"  # Gemma sometimes answers with no class at all


def full_chances(test_set: str, how: str):
    """(classes, chances, file names) recomputed from the saved embeddings, same steps as the
    script that made the answers CSV. Checked against that CSV before use (see main)."""
    if test_set == "aid":
        from src.data import aid
        from src.data.aid import AID_DIR, pick
        from src.model.head import aid_head, embed

        final = pick(10, skip=20)
        if how == "head":
            head = aid_head("siglip-base", 20)
            return list(head.classes_), head.predict_proba(embed(final, "siglip-base")[0]), [n for _, n in final]
        from src.model.zeroshot import chances
        classes, P = chances(aid.LABELS, pick(10, skip=10), final, AID_DIR, "aid")
        return classes, P, [n for _, n in final]

    from src.data import eurosat
    from src.eval.sentinel_test import photo_sets
    from src.model.head import embed, fit_head

    root = eurosat.dataset_dir()
    final, tuning, train = photo_sets(100)
    if how == "head":
        x = {k: embed(v, "siglip-base", root=root, dataset="eurosat")[0]
             for k, v in (("final", final), ("tuning", tuning), ("train", train))}
        head = fit_head(x["train"], [l for l, _ in train], x["tuning"], [l for l, _ in tuning], verbose=False)
        return list(head.classes_), head.predict_proba(x["final"]), [n for _, n in final]
    from src.model.zeroshot import chances
    classes, P = chances(eurosat.LABELS, tuning, final, root, "eurosat")
    return classes, P, [n for _, n in final]


def scores(truth: list[str], guess: list[str], classes: list[str], top_p=None, P=None) -> dict:
    """Every metric for one model on one test set."""
    from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, f1_score,
                                 precision_recall_fscore_support)

    from src.model.head import calibration_error

    truth, guess = np.array(truth), np.array(guess)
    present = [c for c in classes if (truth == c).any()]  # classes that occur in this test set
    p, r, f, _ = precision_recall_fscore_support(truth, guess, labels=present, average="macro", zero_division=0)
    out = {"photos": len(truth), "accuracy": (truth == guess).mean(),
           "balanced_accuracy": balanced_accuracy_score(truth, guess),
           "macro_precision": p, "macro_recall": r, "macro_f1": f,
           "weighted_f1": f1_score(truth, guess, labels=present, average="weighted", zero_division=0),
           "kappa": cohen_kappa_score(truth, guess), "ece": "", "brier": "", "log_loss": ""}
    if top_p is not None:
        out["ece"] = calibration_error(np.asarray(top_p), truth == guess)
    if P is not None:
        Y = (np.array(classes)[None, :] == truth[:, None]).astype(float)
        out["brier"] = ((P - Y) ** 2).sum(1).mean()
        out["log_loss"] = -np.log(np.clip((P * Y).sum(1), 1e-12, 1)).mean()
    return {k: (round(float(v), 4) if v != "" and not isinstance(v, int) else v) for k, v in out.items()}


def per_class(truth, guess, classes) -> list[dict]:
    from sklearn.metrics import precision_recall_fscore_support

    truth, guess = np.array(truth), np.array(guess)
    p, r, f, support = precision_recall_fscore_support(truth, guess, labels=classes, zero_division=0)
    return [{"class": c, "photos": int(n), "labelled_as_this": int((guess == c).sum()),
             "precision": round(float(pi), 4), "recall": round(float(ri), 4), "f1": round(float(fi), 4)}
            for c, pi, ri, fi, n in zip(classes, p, r, f, support)]


def confusion(truth, guess, classes) -> tuple[np.ndarray, list[str]]:
    """Counts: rows = true class, columns = label given (plus "no valid answer" if any)."""
    columns = classes + ([UNKNOWN] if any(g not in classes for g in guess) else [])
    M = np.zeros((len(classes), len(columns)), dtype=int)
    for t, g in zip(truth, guess):
        M[classes.index(t), columns.index(g if g in classes else UNKNOWN)] += 1
    return M, columns


def top_mixups(M, classes, columns, n=3) -> str:
    off = [(M[i, j], classes[i], columns[j]) for i in range(len(classes)) for j in range(len(columns))
           if classes[i] != columns[j] and M[i, j] > 0]
    return ", ".join(f"{t} -> {g} {k}x" for k, t, g in sorted(off, reverse=True)[:n])


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


# ---- charts ----

# Sequential blue ramp (dataviz reference palette, steps 700 -> 100). On a dark chart the
# anchor flips: "few photos" sits near the black surface, "many" is the lightest step.
RAMP = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7",
        "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]
INK, MUTED, DARK_INK = "#e6e6e6", "#9a9a9a", "#0b0b0b"


def draw(panels, out: Path, title: str, cell_text: bool = True) -> None:
    """panels: list of (subtitle, counts M, row classes, column labels). Row-normalised colour
    (= recall on the diagonal), counts written in the cells, empty cells left black."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
                         "text.color": INK, "axes.labelcolor": "#bdbdbd", "axes.edgecolor": "#444",
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9.5,
                         "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    k = max(len(p[3]) for p in panels)
    size = 0.3 * k + 2.6 if k > 12 else 0.42 * k + 2.2
    fig, axes = plt.subplots(1, len(panels), figsize=(size * len(panels) + 0.8, size + 0.9))
    axes = np.atleast_1d(axes)
    for ax, (subtitle, M, rows, cols) in zip(axes, panels):
        share = M / np.maximum(M.sum(1, keepdims=True), 1)
        shown = np.ma.masked_where(M == 0, share)  # empty cells stay black
        image = ax.imshow(shown, cmap=cmap, vmin=0, vmax=1, aspect="equal", interpolation="nearest")
        # 1px black grid between cells, so neighbouring cells of similar colour stay apart
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
        ax.grid(which="minor", color="black", linewidth=1.2)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=8 if k > 12 else 9)
        ax.set_yticks(range(len(rows)), rows, fontsize=8 if k > 12 else 9)
        ax.set_xlabel("label the model gave")
        ax.set_ylabel("true class")
        ax.set_title(subtitle, pad=8)
        for side in ax.spines.values():
            side.set_visible(False)
        if cell_text:
            for i in range(len(rows)):
                for j in range(len(cols)):
                    if M[i, j]:
                        light_cell = share[i, j] > 0.55
                        ax.text(j, i, str(M[i, j]), ha="center", va="center",
                                fontsize=7 if k > 12 else 8.5, color=DARK_INK if light_cell else INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.012, "Each row adds up to 100% of that class's photos: the diagonal is the class's recall. "
             "Numbers = photos. Black = none.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.07 if k > 12 else 0.05, 0.92, 0.95))
    key = fig.add_axes((0.94, 0.25, 0.012, 0.5))  # colour key: share of the true class's photos
    bar = fig.colorbar(image, cax=key, ticks=[0, 0.5, 1])
    bar.ax.set_yticklabels(["0%", "50%", "100%"], color=MUTED, fontsize=8.5)
    bar.outline.set_visible(False)
    bar.ax.set_title("share of\nthe row", fontsize=8, color=MUTED, loc="center", pad=6)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    (EXP / "confusion").mkdir(exist_ok=True)
    summary, classes_rows, matrices = [], [], {}

    def record(test_set, model, truth, guess, classes, top_p=None, P=None):
        s = scores(truth, guess, classes, top_p, P)
        summary.append({"test_set": test_set, "model": model, **s})
        classes_rows.extend({"test_set": test_set, "model": model, **r} for r in per_class(truth, guess, classes))
        M, cols = confusion(truth, guess, classes)
        matrices[(test_set, model)] = (M, classes, cols)
        with open(EXP / "confusion" / f"{test_set}__{slug(model)}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true \\ given"] + cols)
            for c, row in zip(classes, M):
                w.writerow([c] + list(row))

    # 1. AID and EuroSAT: saved answers (+ full chances where they can be remade exactly)
    for test_set, model, name, how in SAVED:
        rows = list(csv.DictReader(open(EXP / name)))
        if test_set == "aid":
            from src.data.aid import LABELS
        else:
            from src.data.eurosat import LABELS
        classes = list(LABELS)
        truth, guess = [r["truth"] for r in rows], [r["guess"] for r in rows]
        top_p = [float(r["p"]) for r in rows] if "p" in rows[0] else None
        P = None
        if how:
            cls, P_all, files = full_chances(test_set, how)
            order = {f: i for i, f in enumerate(files)}
            P_all = P_all[[order[r["file"]] for r in rows]][:, [cls.index(c) for c in classes]]
            same = np.mean(np.array(classes)[P_all.argmax(1)] == np.array(guess))
            if same == 1.0:
                P = P_all
            else:  # the remade chances don't give the same answers: don't mix them
                print(f"  note: {test_set} {model}: remade chances agree on only {same:.1%} of answers, "
                      "so Brier and log-loss are left out")
        record(test_set, model, truth, guess, classes, top_p, P)

    # 2. The real replay: the healthy head and every broken one
    from src.data import s2_real
    from src.eval.audit_test import real_population

    days = sorted({r["day"] for r in s2_real.load_chips()})
    pop = real_population(days[:-7], days[-7:])
    truth = [pop.classes[t] for t in pop.truth]
    for scenario, P in pop.scenarios.items():
        guess = [pop.classes[i] for i in P.argmax(1)]
        record("real", scenario, truth, guess, list(pop.classes), P.max(1), P)

    # ---- save ----
    out = EXP / f"metrics_summary_{stamp}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    out2 = EXP / f"metrics_per_class_{stamp}.csv"
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(classes_rows[0]))
        w.writeheader()
        w.writerows(classes_rows)

    # ---- print ----
    names = {"aid": "AID final photos (300, 30 classes, 10 per class)",
             "eurosat": "Sentinel-2 / EuroSAT final photos (1,000, 10 classes, 100 per class)",
             "real": f"Real Sentinel-2A replay, 13-19 Oct ({len(truth):,} photos, 8 classes, very uneven)"}
    for test_set in ("aid", "eurosat", "real"):
        print(f"\n=== {names[test_set]} ===")
        print(f"{'model':<52} {'acc':>6} {'bal acc':>7} {'macroF1':>7} {'kappa':>6} {'ECE':>6} {'Brier':>6}")
        for s in summary:
            if s["test_set"] == test_set:
                fmt = lambda v, p=True: (f"{v:.1%}" if p else f"{v:.3f}") if v != "" else "-"
                print(f"{s['model'][:52]:<52} {fmt(s['accuracy']):>6} {fmt(s['balanced_accuracy']):>7} "
                      f"{fmt(s['macro_f1']):>7} {fmt(s['kappa'], False):>6} {fmt(s['ece'], False):>6} "
                      f"{fmt(s['brier'], False):>6}")
        print("Most mixed-up pairs (true -> label given):")
        for (ts, model), (M, rows, cols) in matrices.items():
            if ts == test_set:
                print(f"  {model[:50]:<50} {top_mixups(M, rows, cols)}")
    if pop.rare.any():
        rare = [c for c, r in zip(pop.classes, pop.rare) if r]
        print(f"\nReal replay, rare classes ({', '.join(rare)}): recall per model")
        for r in classes_rows:
            if r["test_set"] == "real" and r["class"] in rare:
                print(f"  {r['model'][:50]:<50} {r['class']:<10} {r['photos']:>4} photos, recall {r['recall']:.0%}, "
                      f"precision {r['precision']:.0%}")

    # ---- charts ----
    healthy = next(k for k in matrices if k[0] == "real" and k[1].startswith("healthy"))
    new_place = next(k for k in matrices if k[0] == "real" and k[1].startswith("new place"))
    draw([(f"Healthy head ({summary[[s['model'] for s in summary].index(healthy[1])]['accuracy']:.0%} right)",
           *matrices[healthy]),
          ("New place: EuroSAT head, no cloud class", *matrices[new_place])],
         DOCS / "confusion_real.png", "What Sentinel-2A really photographed, 13-19 Oct 2026: which classes get mixed up")
    small = ("eurosat", SAVED[5][1])
    cnn = ("eurosat", SAVED[7][1])
    draw([("Small head (frozen SigLIP-base + 30 KB head)", *matrices[small]),
          ("Tiny CNN (MobileNetV3-Small)", *matrices[cnn])],
         DOCS / "confusion_eurosat.png", "Sentinel-2 / EuroSAT final photos (1,000): which classes get mixed up")
    draw([("Small head (trained, 95.7% right)", *matrices[("aid", SAVED[0][1])]),
          ("Gemma 3 4B (no training)", *matrices[("aid", SAVED[4][1])])],
         DOCS / "confusion_aid.png", "AID final photos (300, 30 classes): which classes get mixed up")
    print(f"\nSaved {out.relative_to(ROOT)}, {out2.relative_to(ROOT)}, experiments/confusion/*.csv, "
          "docs/confusion_{real,eurosat,aid}.png")


if __name__ == "__main__":
    main()
