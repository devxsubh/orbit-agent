"""EuroSAT: 27,000 labeled Sentinel-2 photos (64x64 pixels, 10 land types).

Download once (about 95 MB):
    python -m src.data.eurosat
"""
from __future__ import annotations

import random
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ZIP_PATH = RAW / "EuroSAT_RGB.zip"
EXTRACT_DIR = RAW / "eurosat"
URLS = ["https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"]
USER_AGENT = "orbit-agent/0.1 (student research project)"
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Opening line of the prompt (64 pixels x 10 m per pixel = 640 m)
INTRO = "This is a satellite photo of Earth, about 640 metres across."

# The 10 classes, and the words we use when asking the model about them
LABELS = {
    "AnnualCrop": "farmland with seasonal crops",
    "Forest": "forest",
    "HerbaceousVegetation": "grassland or shrubs",
    "Highway": "highway or road",
    "Industrial": "industrial buildings",
    "Pasture": "pasture for animals",
    "PermanentCrop": "orchards or vineyards",
    "Residential": "residential houses",
    "River": "river",
    "SeaLake": "sea or lake",
}
CLASSES = list(LABELS)

# Longer hints for prompt v2: what each class looks like from space
DETAILS = {
    "AnnualCrop": "patchwork of rectangular fields, often brown or light green, with straight edges",
    "Forest": "dense, dark green tree cover with a rough texture, no fields or buildings",
    "HerbaceousVegetation": "natural grass or shrubs, mixed green and brown, irregular patches",
    "Highway": "a long straight or curved grey line crossing the photo, fields around it",
    "Industrial": "large grey or white flat roofs, big buildings, parking lots",
    "Pasture": "smooth, even, bright green grassland with few features",
    "PermanentCrop": "fields with fine regular rows or dots (trees or vines in lines)",
    "Residential": "many small houses and streets packed together, mixed colours",
    "River": "a winding band of dark water crossing the photo, land on both sides",
    "SeaLake": "mostly or fully water, dark blue or black, very little texture",
}


def _download(url: str, dest: Path) -> None:
    from tqdm import tqdm

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    part = dest.with_suffix(".part")
    with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        with tqdm(total=total, unit="B", unit_scale=True, desc="EuroSAT") as bar:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                bar.update(len(chunk))
    part.replace(dest)


def dataset_dir() -> Path | None:
    """The folder holding the 10 class folders, or None if not downloaded yet."""
    if not EXTRACT_DIR.exists():
        return None
    for folder in EXTRACT_DIR.rglob("Forest"):
        if folder.is_dir():
            return folder.parent
    return None


def photos_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_TYPES)


def download() -> Path:
    """Download and unzip EuroSAT. Skips whatever is already done."""
    found = dataset_dir()
    if found:
        print(f"EuroSAT is already downloaded: {found}")
        return found

    RAW.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        for url in URLS:
            try:
                print("Downloading EuroSAT (about 95 MB)...")
                _download(url, ZIP_PATH)
                break
            except Exception as error:
                print(f"  Download failed ({error}).")
        else:
            raise SystemExit("Could not download EuroSAT. Get EuroSAT_RGB.zip from "
                             "https://zenodo.org/records/7711810 and put it in data/raw/")

    print("Unzipping...")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(EXTRACT_DIR)

    found = dataset_dir()
    if not found:
        raise SystemExit("Unzipped, but couldn't find the class folders in data/raw/eurosat/")
    print(f"Done: {found}")
    return found


def sample(per_class: int = 10, seed: int = 42) -> list[tuple[Path, str]]:
    """Pick `per_class` random photos from each class. Same seed = same photos every time."""
    root = dataset_dir() or download()
    rng = random.Random(seed)
    picked = []
    for label in CLASSES:
        files = photos_in(root / label)
        picked += [(f, label) for f in rng.sample(files, min(per_class, len(files)))]
    return picked


if __name__ == "__main__":
    root = download()
    print()
    for label in CLASSES:
        print(f"  {label:<22} {len(photos_in(root / label)):>6,} photos")
