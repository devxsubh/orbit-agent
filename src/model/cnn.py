"""Baseline: a tiny CNN, trained end to end (how earlier onboard papers did it).

Reviewers will ask: "why a frozen image model + head, and not just a small CNN?"
This trains MobileNetV3-Small (about 1.5M weights, made for phones) on the SAME photos as
agent v4's heads, and saves its answers in the same CSV format, so the agents can use them.

Differences from the decision head:
- every weight of the CNN is trained (nothing frozen)
- training stops at the epoch with the best TUNING log-loss
- its chances are then adjusted with one number ("temperature"), also learned on tuning photos

Run from the orbit-agent folder:
    python -m src.model.cnn                          # AID (20/class) and Sentinel-2 (100/class)
    python -m src.model.cnn --datasets eurosat --epochs 2 --train-per-class 5    # quick check
    python -m src.cloud.kaggle_run --command "src.model.cnn"                     # on a Kaggle GPU

Saves results/cnn_<dataset>_<arch>_n<N>_final.csv. Then, for the satellite replay:
    python -m src.eval.run_agent --head results/cnn_aid_..._final.csv
    python -m src.eval.sentinel_test --head-csv results/cnn_eurosat_..._final.csv
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.aid import AID_DIR, pick
from src.model.head import best_device, calibration_error, download_all, write_answers
from src.paths import result

ROOT = Path(__file__).resolve().parents[2]
SIZE = 224  # input size (ImageNet-trained CNNs expect this)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet colour normalisation


def photo_sets(dataset: str, train_per_class: int):
    """(train, tuning, final, folder): the same photos the decision heads used."""
    if dataset == "aid":
        train, tuning, final = pick(train_per_class, skip=30), pick(10, skip=10), pick(10, skip=20)
        download_all(train + tuning + final)
        return train, tuning, final, AID_DIR
    from src.data.eurosat import dataset_dir, download
    from src.eval.sentinel_test import photo_sets as sentinel_sets

    final, tuning, train = sentinel_sets(train_per_class)
    return train, tuning, final, dataset_dir() or download()


def load_images(photos, folder: Path):
    """All photos as one uint8 tensor (N, 3, 224, 224). Small sets, so they fit in memory."""
    import torch

    rows = []
    for label, name in photos:
        with Image.open(folder / label / name) as img:
            img = img.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
            rows.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1))
    return torch.stack(rows)


def build(arch: str, classes: int):
    """An ImageNet-trained CNN with a new last layer for our classes."""
    from torch import nn
    from torchvision import models

    model = getattr(models, arch)(weights="DEFAULT")
    if arch.startswith("mobilenet") or arch.startswith("efficientnet"):
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, classes)
    else:  # resnet
        model.fc = nn.Linear(model.fc.in_features, classes)
    return model


def logits_of(model, images, device: str, batch: int = 64):
    """Raw scores for every photo (no training)."""
    import torch

    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = images[i:i + batch].to(device).float() / 255
            out.append(model((x - mean) / std).float().cpu())
    return torch.cat(out)


def augment(x):
    """Random flips and quarter turns: an aerial photo has no "up"."""
    import torch

    if torch.rand(1) < 0.5:
        x = x.flip(3)
    if torch.rand(1) < 0.5:
        x = x.flip(2)
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(2, 3))


def train(arch: str, classes: list[str], x_train, y_train, x_tuning, y_tuning, epochs: int,
          device: str, verbose: bool = True, seed: int = 0):
    """Train the whole CNN. Keep the epoch with the best tuning log-loss."""
    import torch
    from torch.nn import functional as F

    torch.manual_seed(seed)
    say = print if verbose else (lambda *a, **k: None)
    model = build(arch, len(classes)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    best = (math.inf, None, 0)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train))
        for i in range(0, len(order), 32):
            idx = order[i:i + 32]
            x = augment(x_train[idx].to(device).float() / 255)
            loss = F.cross_entropy(model((x - mean) / std), y_train[idx].to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        tuning_logits = logits_of(model, x_tuning, device)
        tuning_loss = F.cross_entropy(tuning_logits, y_tuning).item()
        accuracy = (tuning_logits.argmax(1) == y_tuning).float().mean().item()
        say(f"  epoch {epoch:>2}: tuning accuracy {accuracy:.1%}, log-loss {tuning_loss:.3f}")
        if tuning_loss < best[0]:
            best = (tuning_loss, copy.deepcopy(model.state_dict()), epoch)
    model.load_state_dict(best[1])
    say(f"Kept epoch {best[2]} (lowest tuning log-loss)")
    return model


def best_temperature(logits, labels) -> float:
    """One number that makes the chances honest: chances = softmax(scores / T). Picked on tuning."""
    from torch.nn import functional as F

    return min(np.arange(0.5, 5.01, 0.05),
               key=lambda t: F.cross_entropy(logits / float(t), labels).item())


def run(dataset: str, arch: str, train_per_class: int, epochs: int, device: str) -> None:
    import torch

    train_set, tuning_set, final_set, folder = photo_sets(dataset, train_per_class)
    classes = sorted({label for label, _ in train_set})
    index = {c: i for i, c in enumerate(classes)}
    print(f"\n=== {dataset}: {arch}, {len(train_set):,} training photos, {epochs} epochs, on {device} ===")
    x_train, x_tuning = load_images(train_set, folder), load_images(tuning_set, folder)
    y_train = torch.tensor([index[l] for l, _ in train_set])
    y_tuning = torch.tensor([index[l] for l, _ in tuning_set])
    model = train(arch, classes, x_train, y_train, x_tuning, y_tuning, epochs, device)
    temperature = float(best_temperature(logits_of(model, x_tuning, device), y_tuning))

    # Final test: time the whole job (open the photo, resize, run the CNN), like the head's timing
    start = time.perf_counter()
    x_final = load_images(final_set, folder)
    logits = logits_of(model, x_final, device)
    ms = (time.perf_counter() - start) / len(final_set) * 1000

    truths = np.array([l for l, _ in final_set])
    raw = torch.softmax(logits, 1).numpy()
    chances = torch.softmax(logits / temperature, 1).numpy()
    right = np.array(classes)[chances.argmax(1)] == truths
    weights = sum(p.numel() for p in model.parameters())
    print(f"FINAL {len(final_set):,} photos: {right.mean():.1%} right")
    print(f"  calibration error: {calibration_error(raw.max(1), right):.3f} as trained, "
          f"{calibration_error(chances.max(1), right):.3f} after temperature {temperature:.2f}")
    print(f"  size: {weights / 1e6:.2f}M weights ({weights * 2 / 1e6:.1f} MB in half precision); "
          f"speed: {ms:.1f} ms per photo on {device}")
    out = result(f"cnn_{dataset}_{arch}_n{train_per_class}_final.csv")
    write_answers(out, final_set, classes, chances, ms)
    print(f"Saved {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny CNN baseline, trained end to end")
    parser.add_argument("--datasets", nargs="+", default=["aid", "eurosat"], choices=["aid", "eurosat"])
    parser.add_argument("--arch", default="mobilenet_v3_small",
                        choices=["mobilenet_v3_small", "efficientnet_b0", "resnet18"])
    parser.add_argument("--train-per-class", type=int, help="default: 20 for AID, 100 for EuroSAT "
                                                            "(the same as the decision heads)")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    device = best_device()
    for dataset in args.datasets:
        n = args.train_per_class or {"aid": 20, "eurosat": 100}[dataset]
        run(dataset, args.arch, n, args.epochs, device)


if __name__ == "__main__":
    main()
