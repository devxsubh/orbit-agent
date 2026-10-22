"""Extra bands for every real Sentinel-2A photo: green, red, near-infrared, short-wave infrared.

Why: the data analysis (notebooks/01_data.ipynb) found that snow and cloud are both blown-out white in the
colour photos (42 of 43 snow photos), so no colour model can tell them apart. In short-wave infrared
(SWIR, band 11) they look very different: snow is dark, cloud stays bright. The standard snow index
NDSI = (green - SWIR) / (green + SWIR) is high for snow and low for cloud. Red and near-infrared give
NDVI (how green the plants are), which may help with Trees vs ShrubGrass vs Cropland.

For each photo in data/raw/s2_real/chips.csv this reads the same 2.24 km window from the original image's
band files (Element 84 Earth Search, AWS open data; reflectance x 10,000) at 56 x 56 pixels, and saves
data/raw/s2_real/bands/<day>/<name>.npz with arrays green, red, nir, swir16. Resumable.

Run from the orbit-agent folder (about 1 h for 6,000 photos):
    python -m src.data.s2_bands
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.data import s2_real  # sets the GDAL settings for reading small windows over the network

ITEM_URL = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/{}"
BANDS = ["green", "red", "nir", "swir16"]
SIZE = 56          # pixels per side (40 m per pixel over the 2.24 km chip)
OUT = s2_real.OUT_DIR / "bands"


def get_item(item_id: str, tries: int = 4) -> dict:
    request = urllib.request.Request(ITEM_URL.format(item_id), headers={"User-Agent": s2_real.USER_AGENT})
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except OSError:
            if attempt == tries:
                raise
            time.sleep(3 * attempt)
    raise RuntimeError("unreachable")


def read_bands(item: dict, lat: float, lon: float) -> dict[str, np.ndarray]:
    """The chip window of each band, SIZE x SIZE, reflectance x 10,000 (0 = no data)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import from_bounds

    out = {}
    for band in BANDS:
        with rasterio.open(item["assets"][band]["href"]) as src:
            xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
            x, y, half = xs[0], ys[0], s2_real.HALF_M
            window = from_bounds(x - half, y - half, x + half, y + half, src.transform)
            out[band] = src.read(1, window=window, out_shape=(SIZE, SIZE), boundless=True, fill_value=0,
                                 resampling=Resampling.average).astype(np.uint16)
    return out


def fetch(row: dict) -> str:
    path = OUT / row["day"] / row["name"].replace(".png", ".npz")
    if path.exists():
        return "done"
    try:
        bands = read_bands(get_item(row["item"]), float(row["lat"]), float(row["lon"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **bands)
        return "saved"
    except Exception as error:  # network trouble: a rerun tries again
        print(f"  {row['name']}: {type(error).__name__}: {str(error)[:100]}")
        return "error"


def load(row: dict) -> dict[str, np.ndarray] | None:
    path = OUT / row["day"] / row["name"].replace(".png", ".npz")
    return dict(np.load(path)) if path.exists() else None


def features(row: dict) -> np.ndarray:
    """Six numbers per photo from the extra bands (NaN-free; zeros if the bands are missing):
    mean NDSI, share of snow-like pixels (NDSI > 0.4), mean SWIR reflectance, share of SWIR-bright
    pixels (> 0.2, cloud-like), mean NDVI, share of green-plant pixels (NDVI > 0.4)."""
    b = load(row)
    if b is None:
        return np.zeros(6)
    g, r, n, s = (b[k].astype(float) / 10000 for k in BANDS)
    valid = (g > 0) & (s > 0)
    if not valid.any():
        return np.zeros(6)
    ndsi = (g - s) / np.maximum(g + s, 1e-6)
    ndvi = (n - r) / np.maximum(n + r, 1e-6)
    v = valid
    return np.array([ndsi[v].mean(), (ndsi[v] > 0.4).mean(), s[v].mean(), (s[v] > 0.2).mean(),
                     ndvi[v].mean(), (ndvi[v] > 0.4).mean()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extra bands (green, red, NIR, SWIR) for every real photo")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = s2_real.load_chips()
    todo = [r for r in rows if not (OUT / r["day"] / r["name"].replace(".png", ".npz")).exists()]
    print(f"{len(rows):,} real photos, {len(rows) - len(todo):,} already have their bands, {len(todo):,} to go")
    counts: dict[str, int] = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, status in enumerate(pool.map(fetch, todo), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 200 == 0 or i == len(todo):
                print(f"  {i:,}/{len(todo):,} ({i / (time.time() - started):.1f}/s): "
                      + ", ".join(f"{k} {v:,}" for k, v in sorted(counts.items())), flush=True)
    if counts.get("error"):
        print(f"{counts['error']} photos failed (network). Run the same command again to retry them.")


if __name__ == "__main__":
    main()
