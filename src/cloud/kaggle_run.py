"""Run a project command (default: decision-head training) on a free Kaggle GPU, from this terminal.

Run from the orbit-agent folder:
    python -m src.cloud.kaggle_run                                   # AID head: SigLIP, all photos
    python -m src.cloud.kaggle_run --command "src.model.head --backbone dinov2"
    python -m src.cloud.kaggle_run --command "src.eval.sentinel_test --backbone siglip"
    python -m src.cloud.kaggle_run --command "src.model.retrain --embed siglip" --dataset s2real
        (experiment 12: big-model embeddings of the real photos; upload them first with
         python -m src.cloud.kaggle_upload)

What it does:
1. Writes kaggle/orbit-agent-head.ipynb, a normal notebook you can also open and run on kaggle.com.
   It downloads this project from GitHub, so push your code first.
2. Uploads it to Kaggle and starts it on a GPU.
3. Waits, then saves the new CSVs into results/ (and the embeddings into data/processed/).

One-time setup:
- kaggle.com -> Settings -> verify your phone number (needed for GPU + internet in notebooks)
- kaggle.com -> Settings -> API -> "Create New Token" -> save it in ~/.kaggle/access_token
  (older keys: ~/.kaggle/kaggle.json). Set KAGGLE_USERNAME if you have no kaggle.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from src.paths import RESULTS, result

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "kaggle"           # what gets uploaded
NOTEBOOK = "orbit-agent-head.ipynb"
SLUG = "orbit-agent-head"          # the notebook's name on Kaggle
REPO = "https://github.com/devxsubh/orbit-agent.git"
AID_ON_KAGGLE = "jiayuanchengala/aid-scene-classification-datasets"  # same photos as Hugging Face
S2_REAL_ON_KAGGLE = "orbit-agent-s2-real"  # your private copy of the real replay (src/cloud/kaggle_upload.py)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def kaggle(*args: str) -> str:
    """Run a kaggle command. Stops with a plain message if it fails."""
    done = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if done.returncode != 0 or "Authentication required" in done.stdout:
        raise SystemExit(f"Kaggle command failed: kaggle {' '.join(args)}\n"
                         f"{(done.stdout + done.stderr).strip()[-600:]}\n"
                         "If it says 'Authentication': make a new API token (see the top of this file).")
    return done.stdout.strip()


def kaggle_username() -> str:
    """Your Kaggle name: from KAGGLE_USERNAME, or from kaggle.json (we never print the key)."""
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    path = Path.home() / ".kaggle" / "kaggle.json"
    if not path.exists():
        raise SystemExit("Don't know your Kaggle username. Run: export KAGGLE_USERNAME=your_name")
    return json.loads(path.read_text())["username"]


def check_pushed(branch: str) -> None:
    """Kaggle downloads the code from GitHub, so warn if GitHub doesn't have what we have."""
    git("fetch", "origin", branch)
    if git("rev-parse", "HEAD") != git("rev-parse", f"origin/{branch}"):
        raise SystemExit(f"GitHub's '{branch}' is not the same as your code. Commit and "
                         f"'git push' first, so Kaggle runs the code you have.")
    if git("status", "--porcelain", "src"):
        print("Note: you have uncommitted changes in src/. Kaggle will NOT see them.")


def cell(kind: str, text: str) -> dict:
    lines = text.strip("\n").splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": lines}


def write_notebook(branch: str, command: str, dataset: str = "aid") -> None:
    """The notebook Kaggle runs. Plain steps, so it's easy to read on kaggle.com too."""
    cells = [
        cell("markdown", f"""
# orbit-agent on a GPU

Runs `python -m {command}`.
Code: {REPO} (branch `{branch}`). Photos: {"the Kaggle copy of AID (" + AID_ON_KAGGLE + ")" if dataset == "aid" else "the real Sentinel-2A replay (private dataset)"}.
Results are saved in `results/` (Output tab).
"""),
        cell("code", f"""
# 1. Get the project code (into /tmp, so it isn't copied into the output)
import glob, os, shutil, subprocess, time

def run(command):
    \"\"\"Run a shell command and STOP the notebook if it fails.\"\"\"
    print("$", command, flush=True)
    subprocess.run(command, shell=True, check=True)

run("rm -rf /tmp/orbit-agent && git clone --depth 1 -b {branch} {REPO} /tmp/orbit-agent")
os.chdir("/tmp/orbit-agent")
run("git log --oneline -1")
run("pip install -q skyfield ollama")  # not on Kaggle (it has torch, transformers, sklearn)
time.sleep(1)
started = time.time()  # results = files written after this (not the ones that came with the code)
"""),
        cell("code", """
# 2. Which GPU did we get? Where are the photos?
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
found = glob.glob("/kaggle/input/**/AID/Airport", recursive=True)
if found:
    os.environ["AID_DIR"] = os.path.dirname(found[0])
    print("AID photos in", os.environ["AID_DIR"])
found = glob.glob("/kaggle/input/**/chips.csv", recursive=True)
if found:
    os.environ["S2_REAL_DIR"] = os.path.dirname(found[0])
    print("Real Sentinel-2A photos in", os.environ["S2_REAL_DIR"])
"""),
        cell("code", f"""
# 3. The actual work
run("python -u -m {command}")
"""),
        cell("code", """
# 4. Keep the results (Kaggle saves everything in /kaggle/working)
os.makedirs("/kaggle/working/results", exist_ok=True)
files = [f for f in glob.glob("results/*/*.csv") + glob.glob("data/processed/embeddings_*.npz")
         if os.path.getmtime(f) > started]  # only what THIS run made
assert files, "No results were made"
for f in files:
    shutil.copy(f, "/kaggle/working/results/")
run("ls -lh /kaggle/working/results")
"""),
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"name": "python3",
                "display_name": "Python 3", "language": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    FOLDER.mkdir(exist_ok=True)
    (FOLDER / NOTEBOOK).write_text(json.dumps(notebook, indent=1) + "\n")


def write_metadata(user: str, dataset: str = "aid") -> str:
    kernel_id = f"{user}/{SLUG}"
    source = AID_ON_KAGGLE if dataset == "aid" else f"{user}/{S2_REAL_ON_KAGGLE}"
    metadata = {"id": kernel_id, "title": SLUG, "code_file": NOTEBOOK, "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
                "enable_internet": True, "dataset_sources": [source], "competition_sources": [],
                "kernel_sources": []}
    (FOLDER / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1) + "\n")
    return kernel_id


def wait_until_done(kernel_id: str, check_every: int = 60) -> str:
    started = time.time()
    while True:
        status = kaggle("kernels", "status", kernel_id).lower()
        minutes = (time.time() - started) / 60
        for word in ("complete", "error", "cancel"):
            if word in status:
                print(f"  {minutes:.0f} min: {word}")
                return word
        print(f"  {minutes:.0f} min: still running...")
        time.sleep(check_every)


def merge_embeddings(new: Path, saved: Path) -> None:
    """Add Kaggle's embeddings to the saved ones (never throw saved photos away)."""
    import numpy as np

    rows = dict(np.load(saved)) if saved.exists() else {}
    before = len(rows)
    rows.update(dict(np.load(new)))
    saved.parent.mkdir(parents=True, exist_ok=True)
    np.savez(saved, **rows)
    print(f"  saved data/processed/{saved.name} ({before:,} -> {len(rows):,} photos)")


def fetch_results(kernel_id: str) -> None:
    out = RESULTS / "kaggle_downloads" / f"{SLUG}_{datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kernel_id, "-p", str(out))
    for f in sorted(out.rglob("*")):
        if f.suffix == ".csv":
            shutil.copy(f, result(f.name))
            print(f"  saved {result(f.name).relative_to(ROOT)}")
        elif f.name.startswith("embeddings_") and f.suffix == ".npz":
            merge_embeddings(f, ROOT / "data" / "processed" / f.name)  # so the Mac can reuse them
    print(f"Everything Kaggle made (including the run log) is in {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a project command on a Kaggle GPU")
    parser.add_argument("--branch", default=git("branch", "--show-current"),
                        help="GitHub branch Kaggle downloads (default: your current branch)")
    parser.add_argument("--command", default="src.model.head --backbone siglip",
                        help='module and options to run, e.g. "src.eval.sentinel_test --backbone siglip"')
    parser.add_argument("--accelerator", default="NvidiaTeslaT4", help="Kaggle GPU type")
    parser.add_argument("--dataset", default="aid", choices=["aid", "s2real"], help="which photos to attach")
    parser.add_argument("--no-wait", action="store_true", help="start it and return straight away")
    args = parser.parse_args()

    user = kaggle_username()
    check_pushed(args.branch)
    write_notebook(args.branch, args.command, args.dataset)
    kernel_id = write_metadata(user, args.dataset)
    print(f"Uploading {FOLDER.relative_to(ROOT)}/{NOTEBOOK} to Kaggle as {kernel_id}...")
    print(kaggle("kernels", "push", "-p", str(FOLDER), "--accelerator", args.accelerator))
    print(f"Watch it live: https://www.kaggle.com/code/{kernel_id}")
    if args.no_wait:
        print(f"Later, get the results with: kaggle kernels output {kernel_id} -p results/kaggle_downloads")
        return

    print("Waiting for Kaggle (checks every minute; the full run takes about 20-40 min)...")
    if wait_until_done(kernel_id) == "complete":
        fetch_results(kernel_id)
    else:
        fetch_results(kernel_id)  # the log helps find what went wrong
        raise SystemExit("The Kaggle run did not finish. Check the log above or on kaggle.com.")


if __name__ == "__main__":
    main()
