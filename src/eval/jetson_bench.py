"""Time (and on a Jetson, energy) of everything the satellite runs: image model + head + audit sampler.

Reviewers will ask: "can this run on flight-like hardware, and what does it cost in power?"
This measures the three onboard steps on whatever computer runs it:
1. the frozen image model (SigLIP-base, 90M weights) turning a 224 px photo into an embedding,
2. the small head (30 KB: one matrix) turning the embedding into a chance per class,
3. the audit sampler (chances to check + Pareto sampling) for one pass of photos.

On an Nvidia Jetson (Orin Nano / AGX) it also reads the board's power meter (`tegrastats`) and
reports energy per photo. Elsewhere (the Mac) power is "n/a": the times are still useful.

Run from the orbit-agent folder:
    python -m src.eval.jetson_bench                  # 200 photos, batch 1 and 8
    python -m src.eval.jetson_bench --photos 50      # quicker

Saves experiments/bench_<computer>_<time>.csv.
"""
from __future__ import annotations

import argparse
import csv
import platform
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.agent.audit import audit, label_bytes
from src.model.head import BACKBONES, best_device

ROOT = Path(__file__).resolve().parents[2]


class PowerMeter:
    """Reads the Jetson's total board power (VDD_IN, milliwatts) from tegrastats, 10 times a second."""

    def __init__(self):
        self.ok = shutil.which("tegrastats") is not None
        self.readings: list[float] = []
        self._proc = None

    def start(self) -> None:
        self.readings = []
        if not self.ok:
            return
        self._proc = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self._proc.stdout:
            match = re.search(r"(?:VDD_IN|POM_5V_IN) (\d+)mW", line)
            if match:
                self.readings.append(float(match.group(1)))

    def stop(self) -> float:
        """Average power in watts while running (nan if there is no power meter)."""
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
            self._proc = None
        return float(np.mean(self.readings)) / 1000 if self.readings else float("nan")


def board_name() -> str:
    """e.g. "NVIDIA Jetson Orin Nano Developer Kit" on a Jetson, else "Darwin arm64" (no host name)."""
    model = Path("/proc/device-tree/model")
    if model.exists():
        return model.read_text().strip("\x00 \n")
    return f"{platform.system()} {platform.machine()}"


def timed(step, repeats: int, meter: PowerMeter) -> tuple[float, float]:
    """(milliseconds per call, average watts) for calling step() `repeats` times."""
    step()  # warm-up (first call loads kernels)
    meter.start()
    start = time.perf_counter()
    for _ in range(repeats):
        step()
    seconds = time.perf_counter() - start
    return seconds / repeats * 1000, meter.stop()


def main() -> None:
    import torch
    from transformers import SiglipVisionModel
    from transformers.utils import logging

    parser = argparse.ArgumentParser(description="Time and energy of the onboard steps")
    parser.add_argument("--photos", type=int, default=200)
    parser.add_argument("--classes", type=int, default=8, help="8 = the real replay's classes")
    parser.add_argument("--pass-photos", type=int, default=500, help="photos waiting for one pass")
    args = parser.parse_args()

    logging.set_verbosity_error()
    device = best_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = SiglipVisionModel.from_pretrained(BACKBONES["siglip-base"], dtype=dtype).to(device).eval()
    meter = PowerMeter()
    computer = board_name()
    print(f"Computer: {computer} ({platform.machine()}), device {device}, {dtype}, "
          f"power meter: {'tegrastats' if meter.ok else 'n/a (not a Jetson)'}")

    meter.start()
    time.sleep(3)
    idle = meter.stop()
    rows = []

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    for batch in (1, 8):
        pixels = torch.rand(batch, 3, 224, 224, dtype=dtype, device=device)  # same cost as a real photo

        def encode():
            with torch.no_grad():
                model(pixel_values=pixels)
            sync()

        ms, watts = timed(encode, max(1, args.photos // batch), meter)
        rows.append({"step": f"image model, batch {batch}", "ms_per_photo": round(ms / batch, 2),
                     "watts": round(watts, 2), "joules_per_photo": round(watts * ms / batch / 1000, 4)})

    rng = np.random.default_rng(0)
    weights = rng.normal(size=(768, args.classes)).astype(np.float32)  # same size as the real head (~30 KB)
    embedding = rng.normal(size=(1, 768)).astype(np.float32)

    def head():
        scores = embedding @ weights
        np.exp(scores - scores.max())

    ms, watts = timed(head, 2000, meter)
    rows.append({"step": "small head (CPU)", "ms_per_photo": round(ms, 4), "watts": round(watts, 2),
                 "joules_per_photo": round(watts * ms / 1000, 6)})

    P = rng.dirichlet(np.ones(args.classes), size=args.pass_photos)
    budget = args.pass_photos * label_bytes(args.classes) + 40 * 300_000  # labels + 40 full photos

    def sampler():
        audit(P, budget, 300_000, rng, floor=0.5)

    ms, watts = timed(sampler, 50, meter)
    rows.append({"step": f"audit sampler, one pass of {args.pass_photos} photos", "ms_per_photo":
                 round(ms / args.pass_photos, 4), "watts": round(watts, 2),
                 "joules_per_photo": round(watts * ms / args.pass_photos / 1000, 6)})

    print(f"\nIdle power: {idle:.2f} W" if not np.isnan(idle) else "\nIdle power: n/a")
    print(f"{'step':<42} {'ms/photo':>9} {'watts':>6} {'J/photo':>8}")
    for r in rows:
        print(f"{r['step']:<42} {r['ms_per_photo']:>9} {r['watts']:>6} {r['joules_per_photo']:>8}")
    total = rows[1]["ms_per_photo"] + rows[2]["ms_per_photo"] + rows[3]["ms_per_photo"]
    print(f"\nAll onboard steps (batch 8): {total:.1f} ms per photo; a photo is taken every 30 s in the replay.")

    out = ROOT / "experiments" / f"bench_{re.sub(r'[^A-Za-z0-9]+', '-', computer)}_{datetime.now():%Y%m%d_%H%M}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["computer", "device", "idle_watts"] + list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({"computer": computer, "device": device, "idle_watts": round(idle, 2), **r})
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
