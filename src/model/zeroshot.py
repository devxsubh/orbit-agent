"""A FAIR zero-shot rival for Gemma 3: SigLIP with class names only, no training photos.

Gemma 3 never sees a labelled photo; the decision head learns from 20-100 per class. That's not a
like-for-like contest. SigLIP can also label photos with no training: it compares each photo with a
short text per class ("a satellite photo of forest.") and picks the closest.

Same frozen SigLIP-base as the head, and the SAME saved photo embeddings, so this costs only the
text side (a few seconds). Prompt fixed before testing: "a satellite photo of {words}." with the
same class words Gemma's prompt uses (LABELS in src/data/aid.py and src/data/eurosat.py).

Two versions:
- plain: chances = softmax of SigLIP's own scores. No labelled photos at all.
- temperature: one number (how sharp the chances are) fitted on the TUNING photos. Changes the
  chances, never the answer. Uses labels, but only to calibrate, not to learn classes.

Run from the orbit-agent folder (needs the saved embeddings from src.model.head / sentinel_test):
    python -m src.model.zeroshot

Saves experiments/zeroshot_{aid,eurosat}_siglip-base_final.csv in the head format, so it plugs in:
    python -m src.eval.run_agent --head experiments/zeroshot_aid_siglip-base_final.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import aid, eurosat
from src.data.aid import AID_DIR, pick
from src.eval.sentinel_test import photo_sets
from src.model.head import BACKBONES, calibration_error, embed, write_answers

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = "a satellite photo of {}."


def text_side(words: list[str], backbone: str = "siglip-base") -> tuple[np.ndarray, float, float]:
    """Unit-length text embeddings for the class words, plus SigLIP's score scale and bias."""
    import torch
    from transformers import AutoTokenizer, SiglipModel
    from transformers.utils import logging

    logging.set_verbosity_error()
    name = BACKBONES[backbone]
    model = SiglipModel.from_pretrained(name).eval()
    tokens = AutoTokenizer.from_pretrained(name)([TEMPLATE.format(w) for w in words],
                                                 padding="max_length", return_tensors="pt")
    with torch.no_grad():
        text = model.get_text_features(**tokens)
        text = text.pooler_output if hasattr(text, "pooler_output") else text
    text = text.float().numpy()
    return (text / np.linalg.norm(text, axis=1, keepdims=True),
            float(model.logit_scale.exp()), float(model.logit_bias))


def softmax(scores: np.ndarray) -> np.ndarray:
    e = np.exp(scores - scores.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fit_temperature(scores: np.ndarray, truth: np.ndarray) -> float:
    """The T that gives the lowest log-loss on these photos (simple grid search)."""
    best = None
    for t in np.geomspace(0.05, 20, 200):
        p = softmax(scores / t)[np.arange(len(truth)), truth]
        loss = -np.log(np.clip(p, 1e-12, 1)).mean()
        if best is None or loss < best[0]:
            best = (loss, t)
    return float(best[1])


def scores(labels: dict[str, str], tuning, final, root: Path, dataset: str):
    """SigLIP's scores for every class: (classes, tuning scores, final scores, temperature)."""
    classes = list(labels)
    text, scale, bias = text_side(list(labels.values()))
    x_tuning, _ = embed(tuning, "siglip-base", root=root, dataset=dataset)
    x_final, _ = embed(final, "siglip-base", root=root, dataset=dataset)
    tuning_scores = x_tuning @ text.T * scale + bias
    final_scores = x_final @ text.T * scale + bias
    t = fit_temperature(tuning_scores, np.array([classes.index(l) for l, _ in tuning]))
    return classes, tuning_scores, final_scores, t


def chances(labels: dict[str, str], tuning, final, root: Path, dataset: str):
    """(classes, chance of every class for every final photo), temperature version."""
    classes, _, final_scores, t = scores(labels, tuning, final, root, dataset)
    return classes, softmax(final_scores / t)


def run(name: str, labels: dict[str, str], tuning, final, root: Path, dataset: str) -> None:
    classes, _, final_scores, t = scores(labels, tuning, final, root, dataset)
    truth_final = np.array([classes.index(l) for l, _ in final])
    print(f"\n=== {name}: SigLIP-base zero-shot, {len(final):,} final photos ===")
    for version, chances in [("plain", softmax(final_scores)), (f"temperature {t:.2f}", softmax(final_scores / t))]:
        right = chances.argmax(1) == truth_final
        print(f"  {version:<18} {right.mean():.1%} right, calibration error "
              f"{calibration_error(chances.max(1), right):.3f}")
    out = ROOT / "experiments" / f"zeroshot_{dataset}_siglip-base_final.csv"
    write_answers(out, final, classes, softmax(final_scores / t), float("nan"))
    print(f"  saved {out.relative_to(ROOT)} (temperature version)")


def main() -> None:
    run("AID (30 classes)", aid.LABELS, pick(10, skip=10), pick(10, skip=20), AID_DIR, "aid")
    final, tuning, _ = photo_sets(100)
    run("Sentinel-2 / EuroSAT (10 classes)", eurosat.LABELS, tuning, final,
        eurosat.dataset_dir(), "eurosat")


if __name__ == "__main__":
    main()
