"""Pilot A (20 Oct 2026, exploratory, not part of the paper pipeline):
is "shrink every photo until the pass fits" already as good as agent v4?

Run from the orbit-agent folder (about 2 min on the Mac):
    python notes/pilots/pilot_compress.py

For the 300 AID FINAL photos: re-save each photo at a smaller size / JPEG quality,
measure its bytes, and let the SAME small head (frozen SigLIP-base + logistic regression,
trained exactly like the paper's) label the shrunk photo "on the ground".
Then, for each pass of the Delhi replay, pick the best setting that fits the pass budget.
"""
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))  # append (not insert): stdlib always wins

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel
from transformers.utils import logging

from src.data.aid import AID_DIR, pick
from src.model.head import aid_head
from src.orbit.contact_windows import load_satellite
from src.orbit.replay import build_replay

logging.set_verbosity_error()

head = aid_head("siglip-base", 20)  # same photos, same C search as the paper's small head
final = pick(10, skip=20)
truth = np.array([label for label, _ in final])

name = "google/siglip-base-patch16-224"
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SiglipVisionModel.from_pretrained(name).to(device).eval()
processor = AutoImageProcessor.from_pretrained(name)


def shrink(path: Path, side, quality):
    """Return (decoded image, bytes) after resizing to side x side and saving as JPEG."""
    if side is None:
        raw = path.read_bytes()
        return Image.open(io.BytesIO(raw)).convert("RGB"), len(raw)
    img = Image.open(path).convert("RGB").resize((side, side), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    raw = buf.getvalue()
    return Image.open(io.BytesIO(raw)).convert("RGB"), len(raw)


def accuracy(images):
    vectors = []
    for i in range(0, len(images), 32):
        inputs = processor(images=images[i:i + 32], return_tensors="pt").to(device)
        with torch.no_grad():
            vectors.append(model(**inputs).pooler_output.float().cpu().numpy())
    x = np.concatenate(vectors)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return float((head.predict(x) == truth).mean())


settings = [("original file", None, None), ("600px q50", 600, 50), ("600px q20", 600, 20),
            ("448px q75", 448, 75), ("336px q75", 336, 75), ("224px q90", 224, 90),
            ("224px q75", 224, 75), ("224px q40", 224, 40), ("160px q75", 160, 75),
            ("112px q75", 112, 75)]
results = []
print(f"\n{'Setting':<14} {'mean KB':>8} {'p95 KB':>7} {'head right on ground':>21}")
for label, side, quality in settings:
    pairs = [shrink(AID_DIR / cls / n, side, quality) for cls, n in final]
    sizes = np.array([b for _, b in pairs])
    acc = accuracy([img for img, _ in pairs])
    results.append((label, sizes.mean(), np.percentile(sizes, 95), acc))
    print(f"{label:<14} {sizes.mean() / 1e3:>8.1f} {np.percentile(sizes, 95) / 1e3:>7.1f} {acc:>20.1%}")

# Pass budgets from the same replay as the paper (Delhi, 24 h from 19 Oct, photo every 30 s)
sat = load_satellite()
pool = [(AID_DIR / cls / n, cls) for cls, n in final]
start = datetime(2026, 10, 19, tzinfo=timezone.utc)
print("\nCompress-to-fit per pass (setting = best one whose 95th-percentile size fits):")
for rate in [0.25, 0.5, 1, 2]:
    windows, _ = build_replay(sat, "delhi", start, 24, pool, 30, rate, 42, photo_bytes=283_000)
    got, total, parts = 0.0, 0, []
    for w in windows:
        if not w.photos:
            continue
        per_photo = w.budget_bytes / len(w.photos)
        fits = [r for r in results if r[2] <= per_photo]
        best = max(fits, key=lambda r: r[3]) if fits else None
        acc = best[3] if best else 0.0
        got += acc * len(w.photos)
        total += len(w.photos)
        parts.append(f"{len(w.photos)} photos, {per_photo / 1e3:.0f} KB each -> "
                     f"{best[0] if best else 'nothing fits'}")
    print(f"  {rate:>4} Mbps: ground labels right on {got / total:.1%} of {total} photos "
          f"(every photo also visible to people) | " + "; ".join(parts))
