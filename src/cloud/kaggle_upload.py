"""Upload the real Sentinel-2A replay photos to Kaggle as a PRIVATE dataset (for GPU runs on them).

Experiment 12 embeds every real photo with the big SigLIP (400M) on a free Kaggle GPU. Kaggle can't see
this computer, so the photos (PNG chips + chips.csv, about 0.5 GB) go up once as a private dataset in your
account. Only the photos and their table: no band files, no backups.

Run from the orbit-agent folder:
    python -m src.cloud.kaggle_upload            # first time: creates the dataset
    python -m src.cloud.kaggle_upload --update   # after downloading more days: new version
Then: python -m src.cloud.kaggle_run --command "src.model.retrain --embed siglip" --dataset s2real
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from src.cloud.kaggle_run import S2_REAL_ON_KAGGLE, kaggle, kaggle_username
from src.data import s2_real
from src.paths import ROOT

STAGE = ROOT / "data" / "processed" / "kaggle_s2real"  # hard links, so nothing is copied twice on disk


def stage() -> int:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    rows = s2_real.load_chips()
    shutil.copy(s2_real.CHIPS_CSV, STAGE / "chips.csv")
    for r in rows:
        (STAGE / r["day"]).mkdir(exist_ok=True)
        os.link(s2_real.OUT_DIR / r["day"] / r["name"], STAGE / r["day"] / r["name"])
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the real replay photos to Kaggle (private)")
    parser.add_argument("--update", action="store_true", help="add a new version instead of creating it")
    args = parser.parse_args()
    user = kaggle_username()
    n = stage()
    (STAGE / "dataset-metadata.json").write_text(json.dumps({
        "title": "orbit-agent s2 real", "id": f"{user}/{S2_REAL_ON_KAGGLE}",
        "licenses": [{"name": "other"}]}, indent=1))
    print(f"Uploading {n:,} real photos to Kaggle as {user}/{S2_REAL_ON_KAGGLE} (private)...")
    if args.update:
        print(kaggle("datasets", "version", "-p", str(STAGE), "-m", f"{n} photos", "--dir-mode", "zip"))
    else:
        print(kaggle("datasets", "create", "-p", str(STAGE), "--dir-mode", "zip"))


if __name__ == "__main__":
    main()
