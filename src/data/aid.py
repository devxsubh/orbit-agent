"""AID: 10,000 aerial photos (600x600 pixels, 30 scene types). The dataset NAVI-Orbital used.

The full dataset is 2.6 GB, so we only download the photos we actually test
(10 per class = 300 photos, about 100 MB).
Source: Hugging Face copy at huggingface.co/datasets/blanchon/AID

Download 10 photos per class:
    python -m src.data.aid
Download the test set (photos 11-20 of each class, never used for tuning):
    python -m src.data.aid --skip 10
"""
from __future__ import annotations

import json
import random
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AID_DIR = ROOT / "data" / "raw" / "aid"
FILE_LIST = AID_DIR / "file_list.json"
REPO = "https://huggingface.co/datasets/blanchon/AID"
API = "https://huggingface.co/api/datasets/blanchon/AID/tree/main/data"
USER_AGENT = "orbit-agent/0.1 (student research project)"

# Opening line of the prompt. AID photos come from Google Earth at different zoom levels.
INTRO = "This is an aerial photo of Earth taken from above."

# The 30 classes, and the words we use when asking the model about them
LABELS = {
    "Airport": "airport",
    "BareLand": "bare land",
    "BaseballField": "baseball field",
    "Beach": "beach",
    "Bridge": "bridge",
    "Center": "city center",
    "Church": "church",
    "Commercial": "commercial area",
    "DenseResidential": "dense residential area",
    "Desert": "desert",
    "Farmland": "farmland",
    "Forest": "forest",
    "Industrial": "industrial area",
    "Meadow": "meadow",
    "MediumResidential": "medium residential area",
    "Mountain": "mountain",
    "Park": "park",
    "Parking": "parking lot",
    "Playground": "playground",
    "Pond": "pond",
    "Port": "port",
    "RailwayStation": "railway station",
    "Resort": "resort",
    "River": "river",
    "School": "school",
    "SparseResidential": "sparse residential area",
    "Square": "square",
    "Stadium": "stadium",
    "StorageTanks": "storage tanks",
    "Viaduct": "viaduct",
}
CLASSES = list(LABELS)
DETAILS = None  # no prompt v2 for AID (v2 didn't help on EuroSAT)


def _get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def file_list() -> dict[str, list[str]]:
    """Names of all photos in each class. Asked from Hugging Face once, then saved."""
    if FILE_LIST.exists():
        return json.loads(FILE_LIST.read_text())
    print("Getting the AID file list from Hugging Face...")
    files = {}
    for label in CLASSES:
        entries = json.loads(_get(f"{API}/{label}"))
        files[label] = sorted(e["path"].split("/")[-1] for e in entries if e["type"] == "file")
    total = sum(len(v) for v in files.values())
    if total != 10_000:
        print(f"Warning: expected 10,000 photos, found {total:,}.")
    AID_DIR.mkdir(parents=True, exist_ok=True)
    FILE_LIST.write_text(json.dumps(files, indent=1))
    return files


def _download(label: str, name: str) -> Path:
    """Download one photo (skips it if we already have it). Tries 3 times."""
    dest = AID_DIR / label / name
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    for attempt in range(1, 4):  # Hugging Face sometimes drops a connection
        try:
            part.write_bytes(_get(f"{REPO}/resolve/main/data/{label}/{name}"))
            break
        except OSError as error:
            if attempt == 3:
                raise
            print(f"  {name}: download failed ({error}), trying again...")
            time.sleep(5 * attempt)
    part.replace(dest)
    return dest


def pick(per_class: int | None = 10, seed: int = 42, skip: int = 0) -> list[tuple[str, str]]:
    """Choose photos without downloading them. Returns (label, file name).

    Each class is shuffled with its own seed, so the same photos come out every time.
    per_class=None means "all photos after the first `skip`".
    """
    picked = []
    for label, names in file_list().items():
        names = list(names)
        random.Random(f"{seed}-{label}").shuffle(names)
        end = None if per_class is None else skip + per_class
        picked += [(label, n) for n in names[skip:end]]
    return picked


def sample(per_class: int = 10, seed: int = 42, skip: int = 0) -> list[tuple[Path, str]]:
    """Pick `per_class` random photos from each class, download them, return (path, label).

    Each class is shuffled with its own seed, so a small run (--per-class 3)
    uses the same first photos as a big run (--per-class 10).
    skip: leave out the first photos of each class. skip=10 gives a NEW test set
          (photos 11-20) that never overlaps the first 10.
    """
    from tqdm import tqdm

    photos = []
    for label, name in tqdm(pick(per_class, seed, skip), desc="AID photos", unit="photo"):
        photos.append((_download(label, name), label))
    return photos


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download AID photos")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--skip", type=int, default=0, help="10 = the test set (photos 11-20)")
    args = parser.parse_args()
    photos = sample(args.per_class, skip=args.skip)
    size_mb = sum(p.stat().st_size for p, _ in photos) / 1e6
    print(f"\nReady: {len(photos)} AID photos ({size_mb:.0f} MB) in {AID_DIR}")
