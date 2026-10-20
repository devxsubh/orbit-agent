"""Agent v4's "decision head": a frozen image model + a tiny trained classifier.

The idea comes from "decision models" like TypeSafe's Jev: the answers are fixed in advance
(the 30 AID classes), and ONE quick pass gives a probability for every answer.
Nothing is generated, so the model can't reply with an answer that isn't on the list.

1. A frozen image model turns each photo into a list of numbers (an "embedding").
   Its weights are never changed. Default: SigLIP, the same family as Gemma 3's image encoder.
2. A small classifier (logistic regression) learns to map embeddings to the 30 classes.
   It gives a chance for every class, so "how sure" comes for free.

Photo sets (same shuffle as src/data/aid.py):
    1-10   first Gemma baseline (not used here)
    11-20  tuning: picks the classifier setting
    21-30  FINAL test: the paper numbers. Don't change settings after looking at it.
    31+    training photos for the head (Gemma never saw them)

Run from the orbit-agent folder:
    python -m src.model.head --backbone siglip-base --train-per-class 20   # quick Mac test
    python -m src.model.head                                               # full run (Kaggle GPU)

Saves experiments/head_aid_<backbone>_tuning.csv and ..._final.csv. run_agent uses them for agent v4.
"""
from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.aid import AID_DIR, _download, pick

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "processed"  # saved embeddings (gitignored)

# Frozen image models we can try (all free on Hugging Face, no login needed)
BACKBONES = {
    "siglip": "google/siglip-so400m-patch14-384",    # 400M, same family as Gemma 3's image encoder
    "siglip-base": "google/siglip-base-patch16-224",  # 90M, small and quick (good for a Mac test)
    "dinov2": "facebook/dinov2-base",                 # 90M, learned from photos only (no text)
}


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"   # Kaggle / Colab GPU
    if torch.backends.mps.is_available():
        return "mps"    # Apple Silicon
    return "cpu"


def download_all(photos: list[tuple[str, str]]) -> None:
    """Download photos we don't have yet, 8 at a time."""
    missing = [(label, name) for label, name in photos if not (AID_DIR / label / name).exists()]
    if not missing:
        return
    print(f"Downloading {len(missing):,} AID photos from Hugging Face...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(lambda p: _download(*p), missing), 1):
            if i % 500 == 0 or i == len(missing):
                print(f"  {i:,}/{len(missing):,}")


def embed(photos: list[tuple[str, str]], backbone: str, batch: int = 32,
          device: str | None = None, half: bool | None = None,
          root: Path = AID_DIR, dataset: str = "aid") -> tuple[np.ndarray, float]:
    """Embeddings for these photos (saved to disk, so each photo is only done once).

    photos: (label, file name) pairs; the photo is at root/label/name.
    Returns (one row per photo, milliseconds per photo on this computer).
    """
    cache_file = CACHE / f"embeddings_{dataset}_{backbone}.npz"
    saved = dict(np.load(cache_file)) if cache_file.exists() else {}
    todo = [(label, name) for label, name in photos if name not in saved]

    ms_per_photo = float("nan")
    if todo:
        import torch
        from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel
        from transformers.utils import logging

        logging.set_verbosity_error()  # hide the long "unused text-model weights" list (harmless)

        device = device or best_device()
        if half is None:
            half = device == "cuda"  # half precision: an Nvidia GPU is several times faster
        dtype = torch.float16 if half else torch.float32
        name = BACKBONES[backbone]
        print(f"Loading the frozen image model {name} on {device} ({dtype})...")
        model_class = SiglipVisionModel if backbone.startswith("siglip") else AutoModel
        model = model_class.from_pretrained(name, dtype=dtype).to(device).eval()  # eval = frozen
        processor = AutoImageProcessor.from_pretrained(name)

        print(f"Making embeddings for {len(todo):,} photos...")
        start = time.perf_counter()
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            images = [Image.open(root / label / n).convert("RGB") for label, n in chunk]
            inputs = processor(images=images, return_tensors="pt").to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            with torch.no_grad():  # no_grad = we never change the model's weights
                vectors = model(**inputs).pooler_output.float().cpu().numpy()
            for (_, n), v in zip(chunk, vectors):
                saved[n] = v
            done = i + len(chunk)
            if done % (batch * 20) == 0 or done == len(todo):
                print(f"  {done:,}/{len(todo):,}")
                CACHE.mkdir(parents=True, exist_ok=True)
                np.savez(cache_file, **saved)  # save as we go, so a stop doesn't lose work
        ms_per_photo = (time.perf_counter() - start) / len(todo) * 1000

    rows = np.stack([saved[name] for _, name in photos])
    return rows / np.linalg.norm(rows, axis=1, keepdims=True), ms_per_photo  # same length for all


def calibration_error(chances: np.ndarray, right: np.ndarray, bins: int = 10) -> float:
    """ECE: on average, how far "says 80% sure" is from "is right 80% of the time". 0 = perfect."""
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (chances > low) & (chances <= high)
        if inside.any():
            error += inside.mean() * abs(chances[inside].mean() - right[inside].mean())
    return float(error)


def save_answers(path: Path, photos, classifier, x: np.ndarray, ms_per_photo: float) -> None:
    """Same first columns as the Gemma CSVs, so run_agent can read both."""
    chances = classifier.predict_proba(x)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "truth", "guess", "correct", "p", "second_guess", "p_second",
                         "ms_per_photo"])
        for (truth, name), row in zip(photos, chances):
            first, second = np.argsort(row)[::-1][:2]
            guess = classifier.classes_[first]
            writer.writerow([name, truth, guess, guess == truth, round(float(row[first]), 4),
                             classifier.classes_[second], round(float(row[second]), 4),
                             round(ms_per_photo, 1)])


def fit_head(x_train: np.ndarray, y_train: list[str], x_tuning: np.ndarray, y_tuning: list[str]):
    """Train the small classifier. The setting C (how strongly to hold it back) is picked on the
    TUNING photos by log-loss, which rewards honest chances, not just right answers."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    print(f"\nTraining the head on {len(y_train):,} photos...")
    y_tuning = np.array(y_tuning)
    best = None
    for c in [1, 10, 100, 300, 1000, 3000]:
        classifier = LogisticRegression(C=c, max_iter=3000).fit(x_train, y_train)
        loss = log_loss(y_tuning, classifier.predict_proba(x_tuning), labels=classifier.classes_)
        accuracy = (classifier.predict(x_tuning) == y_tuning).mean()
        print(f"  C={c:<5} tuning accuracy {accuracy:.1%}, log-loss {loss:.3f}")
        if best is None or loss < best[0]:
            best = (loss, c, classifier)
    print(f"Chosen: C={best[1]} (lowest tuning log-loss)")
    return best[2]


def aid_head(backbone: str, train_per_class: int):
    """The AID head again, from saved embeddings (same photos and steps as `main`)."""
    train, tuning = pick(train_per_class or None, skip=30), pick(10, skip=10)
    x_train, _ = embed(train, backbone)
    x_tuning, _ = embed(tuning, backbone)
    return fit_head(x_train, [l for l, _ in train], x_tuning, [l for l, _ in tuning])


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent v4: frozen image model + small classifier")
    parser.add_argument("--backbone", default="siglip", choices=BACKBONES)
    parser.add_argument("--train-per-class", type=int, default=0,
                        help="training photos per class (0 = all, about 9,100 photos)")
    args = parser.parse_args()

    train = pick(args.train_per_class or None, skip=30)
    tuning = pick(10, skip=10)
    final = pick(10, skip=20)
    download_all(train + tuning + final)

    x_train, _ = embed(train, args.backbone)
    x_tuning, _ = embed(tuning, args.backbone)
    x_final, ms_final = embed(final, args.backbone)
    classifier = fit_head(x_train, [label for label, _ in train],
                          x_tuning, [label for label, _ in tuning])

    tag = f"{args.backbone}_n{args.train_per_class}" if args.train_per_class else args.backbone
    for name, photos, x in [("tuning", tuning, x_tuning), ("final", final, x_final)]:
        chances = classifier.predict_proba(x).max(axis=1)
        right = classifier.predict(x) == np.array([label for label, _ in photos])
        print(f"{name:>6} photos: accuracy {right.mean():.1%} ({right.sum()}/{len(right)}), "
              f"calibration error {calibration_error(chances, right):.3f}")
        out = ROOT / "experiments" / f"head_aid_{tag}_{name}.csv"
        save_answers(out, photos, classifier, x, ms_final)
        print(f"        saved {out.relative_to(ROOT)}")
    if not np.isnan(ms_final):
        print(f"Speed: {ms_final:.0f} ms per photo for the image model on {best_device()} "
              f"(Gemma 3 4B: about 7,500 ms on the Mac)")


if __name__ == "__main__":
    main()
