"""Run the decision-head training (src/model/head.py) on a free Kaggle GPU, from this terminal.

Run from the orbit-agent folder:
    python -m src.cloud.kaggle_run                                  # full run: SigLIP, all photos
    python -m src.cloud.kaggle_run --head-args "--backbone dinov2"  # other options for head.py

What it does:
1. Writes kaggle/orbit-agent-head.ipynb, a normal notebook you can also open and run on kaggle.com.
   It downloads this project from GitHub, so push your code first.
2. Uploads it to Kaggle and starts it on a GPU.
3. Waits, then saves the results into experiments/ (and the embeddings into data/processed/).

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

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "kaggle"           # what gets uploaded
NOTEBOOK = "orbit-agent-head.ipynb"
SLUG = "orbit-agent-head"          # the notebook's name on Kaggle
REPO = "https://github.com/devxsubh/orbit-agent.git"
AID_ON_KAGGLE = "jiayuanchengala/aid-scene-classification-datasets"  # same photos as Hugging Face


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def kaggle(*args: str) -> str:
    """Run a kaggle command. Stops with a plain message if it fails."""
    result = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if result.returncode != 0 or "Authentication required" in result.stdout:
        raise SystemExit(f"Kaggle command failed: kaggle {' '.join(args)}\n"
                         f"{(result.stdout + result.stderr).strip()[-600:]}\n"
                         "If it says 'Authentication': make a new API token (see the top of this file).")
    return result.stdout.strip()


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


def write_notebook(branch: str, head_args: str) -> None:
    """The notebook Kaggle runs. Plain steps, so it's easy to read on kaggle.com too."""
    cells = [
        cell("markdown", f"""
# orbit-agent: agent v4 decision head

Trains the small classifier on top of a frozen image model (`src/model/head.py`) on a GPU.
Code: {REPO} (branch `{branch}`). Photos: the Kaggle copy of AID ({AID_ON_KAGGLE}).
Results are saved in `results/` (Output tab).
"""),
        cell("code", f"""
# 1. Get the project code (into /tmp, so it isn't copied into the output)
import glob, os, shutil, subprocess

def run(command):
    \"\"\"Run a shell command and STOP the notebook if it fails.\"\"\"
    print("$", command, flush=True)
    subprocess.run(command, shell=True, check=True)

run("rm -rf /tmp/orbit-agent && git clone --depth 1 -b {branch} {REPO} /tmp/orbit-agent")
os.chdir("/tmp/orbit-agent")
run("git log --oneline -1")
run("rm -f experiments/head_aid_*.csv")  # so only results made in THIS run are kept
"""),
        cell("code", """
# 2. Which GPU did we get? Where are the AID photos?
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
found = glob.glob("/kaggle/input/**/AID/Airport", recursive=True)
assert found, "AID dataset not attached (add it in the notebook's Input panel)"
os.environ["AID_DIR"] = os.path.dirname(found[0])
print("AID photos in", os.environ["AID_DIR"])
"""),
        cell("code", f"""
# 3. Make embeddings with the frozen model, train the head
run("python -u -m src.model.head {head_args}")
"""),
        cell("code", """
# 4. Keep the results (Kaggle saves everything in /kaggle/working)
os.makedirs("/kaggle/working/results", exist_ok=True)
files = glob.glob("experiments/head_aid_*.csv") + glob.glob("data/processed/embeddings_aid_*.npz")
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


def write_metadata(user: str) -> str:
    kernel_id = f"{user}/{SLUG}"
    metadata = {"id": kernel_id, "title": SLUG, "code_file": NOTEBOOK, "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
                "enable_internet": True, "dataset_sources": [AID_ON_KAGGLE], "competition_sources": [],
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


def fetch_results(kernel_id: str) -> None:
    out = ROOT / "experiments" / "kaggle" / f"{SLUG}_{datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kernel_id, "-p", str(out))
    for f in sorted(out.rglob("*")):
        if f.name.startswith("head_aid_") and f.suffix == ".csv":
            shutil.copy(f, ROOT / "experiments" / f.name)
            print(f"  saved experiments/{f.name}")
        elif f.name.startswith("embeddings_aid_") and f.suffix == ".npz":
            (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
            shutil.copy(f, ROOT / "data" / "processed" / f.name)  # so the Mac can reuse them
            print(f"  saved data/processed/{f.name}")
    print(f"Everything Kaggle made (including the run log) is in {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the decision head on a Kaggle GPU")
    parser.add_argument("--branch", default=git("branch", "--show-current"),
                        help="GitHub branch Kaggle downloads (default: your current branch)")
    parser.add_argument("--head-args", default="--backbone siglip",
                        help='options passed to src.model.head, e.g. "--backbone dinov2"')
    parser.add_argument("--accelerator", default="NvidiaTeslaT4", help="Kaggle GPU type")
    parser.add_argument("--no-wait", action="store_true", help="start it and return straight away")
    args = parser.parse_args()

    user = kaggle_username()
    check_pushed(args.branch)
    write_notebook(args.branch, args.head_args)
    kernel_id = write_metadata(user)
    print(f"Uploading {FOLDER.relative_to(ROOT)}/{NOTEBOOK} to Kaggle as {kernel_id}...")
    print(kaggle("kernels", "push", "-p", str(FOLDER), "--accelerator", args.accelerator))
    print(f"Watch it live: https://www.kaggle.com/code/{kernel_id}")
    if args.no_wait:
        print(f"Later, get the results with: kaggle kernels output {kernel_id} -p experiments/kaggle")
        return

    print("Waiting for Kaggle (checks every minute; the full run takes about 20-40 min)...")
    if wait_until_done(kernel_id) == "complete":
        fetch_results(kernel_id)
    else:
        fetch_results(kernel_id)  # the log helps find what went wrong
        raise SystemExit("The Kaggle run did not finish. Check the log above or on kaggle.com.")


if __name__ == "__main__":
    main()
