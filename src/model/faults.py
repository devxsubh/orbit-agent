"""Ways an onboard model can go wrong without anyone noticing: the failure suite for the audit tests.

1. New camera: a head trained on one kind of photo labels another kind (the AID head on
   Sentinel-2 photos: 33% right in src/eval/sentinel_test.py).
2. Silent swap: part of one class gets another class's label, AS CONFIDENTLY as the right answers
   (like the 400M head's mistakes, which were all more than 96% sure).
3. Bit flips: radiation flips bits of the 30 KB head in the satellite's memory (single-event
   upsets). Most flips do nothing; a flip in a number's exponent can make a weight huge.
4. Damaged camera: blur (focus drift), sensor noise, stripes (uneven detector columns).
   The damaged photos go through the same frozen image model and head as before.

Every function returns the model's chances (one row per photo, one column per class), so the
audit experiments can swap a healthy model for a broken one.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "processed"


def mapped_chances(P_src: np.ndarray, src_classes: list[str], mapping: dict[str, str],
                   dst_classes: list[str]) -> np.ndarray:
    """Chances over another set of classes: each new class gets the sum of the old classes mapped
    to it. Old classes with no match are dropped; each row is scaled to add up to 1 again."""
    P = np.zeros((len(P_src), len(dst_classes)))
    for j, c in enumerate(src_classes):
        if c in mapping:
            P[:, dst_classes.index(mapping[c])] += P_src[:, j]
    total = P.sum(1, keepdims=True)
    return np.where(total > 0, P / np.where(total > 0, total, 1), 1 / len(dst_classes))


def silent_swap(P: np.ndarray, truth: np.ndarray, src: int, dst: int, share: float,
                rng: np.random.Generator) -> np.ndarray:
    """A share of the photos whose true class is `src` get label `dst`, with a confidence drawn
    from the model's confidence on its RIGHT answers (so the mistakes look exactly as sure)."""
    out = P.copy()
    hit = (truth == src) & (rng.random(len(truth)) < share)
    right = P.max(1)[P.argmax(1) == truth]
    sure = rng.choice(right, hit.sum())
    out[hit] = ((1 - sure) / (P.shape[1] - 1))[:, None]
    out[hit, dst] = sure
    return out


def flip_bits(head, n_flips: int, rng: np.random.Generator):
    """A copy of a trained head after radiation flipped `n_flips` random bits of its weights,
    stored as 32-bit floats (768 x classes weights + biases, about 30 KB)."""
    broken = copy.deepcopy(head)
    weights = np.concatenate([head.coef_.ravel(), head.intercept_]).astype(np.float32)
    bits = weights.view(np.uint32)
    for i in rng.choice(len(bits), n_flips):
        bits[i] ^= np.uint32(1) << np.uint32(rng.integers(0, 32))
    with np.errstate(all="ignore"):  # a flipped exponent can give inf or nan: keep them huge, not broken
        weights = np.nan_to_num(bits.view(np.float32).astype(np.float64), nan=0.0,
                                posinf=3e38, neginf=-3e38)
    broken.coef_ = weights[:head.coef_.size].reshape(head.coef_.shape)
    broken.intercept_ = weights[head.coef_.size:]
    return broken


def harmful_flips(head, x: np.ndarray, truth_labels: np.ndarray, n_flips: int = 8,
                  drop: tuple[float, float] = (0.05, 0.30), seed: int = 0, tries: int = 500):
    """Flip bits with seed, seed+1, ... until accuracy drops by between drop[0] and drop[1].
    Most random flips do nothing and some wreck the head (easy to notice); this finds a SUBTLE
    bad day in between. Returns (broken head, seed used)."""
    healthy = (head.predict(x) == truth_labels).mean()
    for s in range(seed, seed + tries):
        broken = flip_bits(head, n_flips, np.random.default_rng(s))
        with np.errstate(all="ignore"):
            loss = healthy - (broken.predict(x) == truth_labels).mean()
        if drop[0] <= loss <= drop[1]:
            return broken, s
    raise RuntimeError("no subtle harmful flips found")


def chances(head, x: np.ndarray) -> np.ndarray:
    """predict_proba that survives the huge numbers a bit flip can create (in 64-bit floats: heads
    trained on 32-bit embeddings keep 32-bit weights, and 32-bit rounding blurs exact totals)."""
    with np.errstate(all="ignore"):
        logits = x.astype(np.float64) @ head.coef_.T.astype(np.float64) + head.intercept_.astype(np.float64)
    logits = np.nan_to_num(logits, nan=0.0, posinf=1e300, neginf=-1e300)
    logits -= logits.max(1, keepdims=True)
    e = np.exp(np.clip(logits, -700, 0))
    return e / e.sum(1, keepdims=True)


# ---- damaged camera ----

DAMAGE = ["blur", "noise", "stripes"]


def damage(img: Image.Image, kind: str, rng: np.random.Generator) -> Image.Image:
    """The same photo as a degraded camera would take it (moderate damage, easy to miss by eye)."""
    if kind == "blur":       # focus drift: about 1 pixel of blur
        return img.filter(ImageFilter.GaussianBlur(radius=1.0))
    a = np.asarray(img, dtype=np.float32)
    if kind == "noise":      # sensor noise
        a = a + rng.normal(0, 10, a.shape)
    elif kind == "stripes":  # uneven detector columns (pushbroom striping)
        gain = rng.normal(1, 0.08, (1, a.shape[1], 1))
        offset = rng.normal(0, 6, (1, a.shape[1], 1))
        a = a * gain + offset
    else:
        raise ValueError(kind)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def damaged_embeddings(photos: list[tuple[str, str]], root: Path, dataset: str, kind: str,
                       backbone: str = "siglip-base") -> np.ndarray:
    """Embeddings of damaged copies of these photos (saved, so each is only made once)."""
    cache_file = CACHE / f"embeddings_{dataset}-{kind}_{backbone}.npz"
    saved = dict(np.load(cache_file)) if cache_file.exists() else {}
    todo = [(label, name) for label, name in photos if name not in saved]
    if todo:
        from src.model.head import embed_images

        rng = np.random.default_rng(0)
        print(f"Embedding {len(todo):,} photos with '{kind}' damage...")
        images = [damage(Image.open(root / label / n).convert("RGB"), kind, rng) for label, n in todo]
        for (_, n), v in zip(todo, embed_images(images, backbone)):
            saved[n] = v
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, **saved)
    rows = np.stack([saved[name] for _, name in photos])
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)
