"""Real Sentinel-2 photos at the places Sentinel-2A really was: the "satellite on your desk", for real.

The replay (src/orbit/replay.py) puts a random dataset photo in every photo slot. This module
finds what Sentinel-2A REALLY photographed at that moment instead. For every daylight photo slot
(same orbit, same 30 s steps), it asks the free Sentinel-2 catalogue (Element 84 Earth Search,
AWS open data) for a Sentinel-2A image taken within a few minutes at that spot, and cuts out a
224 x 224 px chip (2.24 km, 10 m per pixel). Clouds, deserts, snow, cities: whatever was there.
No real image at a slot (open ocean: Sentinel-2 only photographs land and coasts) = no photo.

The label ("truth") of each chip comes from two maps:
- the image's own cloud map (Sentinel-2 L2A scene classification): mostly cloud -> "Cloud",
  mostly snow -> "SnowIce"
- else ESA WorldCover 2021 (10 m land cover): the most common land type in the chip
WorldCover is a map from 2021 with its own mistakes, so the labels are a good reference, not perfect.

Download 14 days (6-19 Oct 2026, a few thousand chips, about an hour; safe to stop and restart):
    python -m src.data.s2_real
    python -m src.data.s2_real --start 2026-10-13 --days 1      # quick look: one day

Saves data/raw/s2_real/<day>/<name>.png, plus data/raw/s2_real/slots.csv (every photo slot and
what was found there) and data/raw/s2_real/chips.csv (one row per real photo).
"""
from __future__ import annotations

import os

# GDAL settings for reading small windows from big cloud files (must be set before rasterio loads)
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_USERAGENT", "orbit-agent/0.1")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")         # don't let one stuck download block a worker
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "10")

import argparse
import csv
import json
import math
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "s2_real"
SLOTS_CSV = OUT_DIR / "slots.csv"
CHIPS_CSV = OUT_DIR / "chips.csv"
TLE_FILE = ROOT / "data" / "raw" / "tle" / "40697.tle"  # Sentinel-2A, saved copy (repeatable runs)
STAC = "https://earth-search.aws.element84.com/v1/search"
WORLDCOVER = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
              "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
USER_AGENT = "orbit-agent/0.1 (student research project)"
CHIP_PX = 224            # chip size in pixels (the frozen image model's own input size)
HALF_M = CHIP_PX * 10 / 2  # half the chip width in metres (10 m per pixel)
SEARCH_MINUTES = 5       # how far from the slot time a real image may be
FIRST_DAY = "2026-10-06"

# The 8 classes of the real replay, and plain words for printing
CLASSES = ["Cloud", "Trees", "ShrubGrass", "Cropland", "BuiltUp", "Bare", "SnowIce", "Water"]
WORDS = {"Cloud": "cloud", "Trees": "trees", "ShrubGrass": "shrubs and grass", "Cropland": "cropland",
         "BuiltUp": "built-up", "Bare": "bare ground", "SnowIce": "snow and ice", "Water": "water"}
# ESA WorldCover code -> our class
WORLDCOVER_CLASS = {10: "Trees", 20: "ShrubGrass", 30: "ShrubGrass", 40: "Cropland", 50: "BuiltUp",
                    60: "Bare", 70: "SnowIce", 80: "Water", 90: "Water", 95: "Trees", 100: "ShrubGrass"}
# Sentinel-2 scene classification codes (SCL band)
SCL_CLOUD = (8, 9, 10)   # cloud (medium, high probability) and thin cirrus
SCL_SNOW = 11
SCL_NODATA = (0, 1)      # no data, saturated or broken pixels

SLOT_FIELDS = ["slot", "day", "slot_utc", "lat", "lon", "status", "item", "gap_s"]
CHIP_FIELDS = (["name", "day", "slot_utc", "lat", "lon", "item", "item_utc", "gap_s", "label",
                "cloud", "snow", "nodata"] + [f"wc_{c}" for c in CLASSES[1:]])
_lock = threading.Lock()


def _post_json(url: str, body: dict, tries: int = 4) -> dict:
    data = json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT,
                                                              "Content-Type": "application/json"})
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except (OSError, urllib.error.HTTPError) as error:
            if attempt == tries:
                raise
            time.sleep(3 * attempt)  # the catalogue sometimes asks us to slow down
    raise RuntimeError("unreachable")


def photo_slots(start: datetime, days: int, every: int = 30) -> list[tuple[str, datetime, float, float]]:
    """Every daylight photo slot along Sentinel-2A's real path: (slot name, time, lat, lon)."""
    from src.orbit.contact_windows import ground_track, load_satellite
    from src.orbit.replay import is_daytime

    sat = load_satellite("sentinel-2a", str(TLE_FILE) if TLE_FILE.exists() else None)
    slots = []
    for when, lat, lon in ground_track(sat, start, start + timedelta(days=days), every):
        if is_daytime(when, float(lon)):
            slots.append((f"{when:%Y%m%dT%H%M%S}", when, float(lat), float(lon)))
    return slots


def find_image(when: datetime, lat: float, lon: float) -> dict | None:
    """The Sentinel-2A image (catalogue item) taken closest in time at this spot, or None."""
    window = timedelta(minutes=SEARCH_MINUTES)
    body = {"collections": ["sentinel-2-l2a"], "limit": 10,
            "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{when - window:%Y-%m-%dT%H:%M:%SZ}/{when + window:%Y-%m-%dT%H:%M:%SZ}"}
    items = [it for it in _post_json(STAC, body)["features"]
             if it["properties"].get("platform") == "sentinel-2a"]
    if not items:
        return None
    return min(items, key=lambda it: abs(_time(it) - when))


def _time(item: dict) -> datetime:
    return datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00"))


def worldcover_tile(lat: float, lon: float) -> str:
    """WorldCover 3x3 degree tile name for this point, e.g. N27E075."""
    la, lo = math.floor(lat / 3) * 3, math.floor(lon / 3) * 3
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def read_chip(item: dict, lat: float, lon: float):
    """(RGB chip 224x224x3, SCL codes) cut around the point from the item's cloud-optimised files."""
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import from_bounds

    with rasterio.open(item["assets"]["visual"]["href"]) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        x, y = xs[0], ys[0]
        window = from_bounds(x - HALF_M, y - HALF_M, x + HALF_M, y + HALF_M, src.transform)
        rgb = src.read(window=window, out_shape=(3, CHIP_PX, CHIP_PX), boundless=True, fill_value=0)
    with rasterio.open(item["assets"]["scl"]["href"]) as src:
        window = from_bounds(x - HALF_M, y - HALF_M, x + HALF_M, y + HALF_M, src.transform)
        scl = src.read(1, window=window, out_shape=(CHIP_PX // 2, CHIP_PX // 2), boundless=True,
                       fill_value=0)
    return np.moveaxis(rgb, 0, -1), scl


def read_worldcover(lat: float, lon: float) -> np.ndarray | None:
    """WorldCover codes for the chip area, or None where WorldCover has no tile (open sea)."""
    import rasterio
    from rasterio.errors import RasterioIOError
    from rasterio.windows import from_bounds

    dlat = HALF_M / 111_320
    dlon = HALF_M / (111_320 * max(0.05, math.cos(math.radians(lat))))
    try:
        with rasterio.open(WORLDCOVER.format(tile=worldcover_tile(lat, lon))) as src:
            window = from_bounds(lon - dlon, lat - dlat, lon + dlon, lat + dlat, src.transform)
            return src.read(1, window=window, out_shape=(CHIP_PX, CHIP_PX), boundless=True,
                            fill_value=0)
    except RasterioIOError:
        return None  # no WorldCover tile = open water


def label_chip(scl: np.ndarray, worldcover: np.ndarray | None) -> tuple[str, dict]:
    """The chip's class and the shares behind it (cloud, snow, no data, WorldCover classes)."""
    total = scl.size
    shares = {"cloud": np.isin(scl, SCL_CLOUD).sum() / total, "snow": (scl == SCL_SNOW).sum() / total,
              "nodata": np.isin(scl, SCL_NODATA).sum() / total}
    land = {c: 0.0 for c in CLASSES[1:]}
    if worldcover is None:
        land["Water"] = 1.0
    else:
        valid = worldcover[worldcover > 0]
        for code, n in zip(*np.unique(valid, return_counts=True)):
            land[WORLDCOVER_CLASS.get(int(code), "Bare")] += n / max(1, valid.size)
        if valid.size == 0:
            land["Water"] = 1.0
    shares.update({f"wc_{c}": v for c, v in land.items()})
    seen = 1 - shares["nodata"]
    if seen > 0 and shares["cloud"] / seen >= 0.5:
        return "Cloud", shares
    if seen > 0 and shares["snow"] / seen >= 0.5:
        return "SnowIce", shares
    return max(land, key=land.get), shares


def _done_slots() -> set[str]:
    if not SLOTS_CSV.exists():
        return set()
    with open(SLOTS_CSV) as f:
        return {row["slot"] for row in csv.DictReader(f) if row["status"] != "error"}


def _append(path: Path, fields: list[str], row: dict) -> None:
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerow(row)


def fetch_slot(slot: tuple[str, datetime, float, float]) -> str:
    """Find, cut, label and save the real photo for one slot. Returns the status."""
    from PIL import Image

    name, when, lat, lon = slot
    day = f"{when:%Y-%m-%d}"
    record = {"slot": name, "day": day, "slot_utc": f"{when:%Y-%m-%dT%H:%M:%SZ}",
              "lat": round(lat, 4), "lon": round(lon, 4), "status": "", "item": "", "gap_s": ""}
    try:
        item = find_image(when, lat, lon)
        if item is None:
            record["status"] = "no_image"
        else:
            gap = abs((_time(item) - when).total_seconds())
            record.update(item=item["id"], gap_s=round(gap, 1))
            rgb, scl = read_chip(item, lat, lon)
            if np.isin(scl, SCL_NODATA).mean() > 0.5 or rgb.max() == 0:
                record["status"] = "edge"  # mostly outside the image (edge of the swath)
            else:
                label, shares = label_chip(scl, read_worldcover(lat, lon))
                file = f"{name}_{item['id'].split('_')[1]}.png"
                (OUT_DIR / day).mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(OUT_DIR / day / file)
                _append(CHIPS_CSV, CHIP_FIELDS, {
                    "name": file, "day": day, "slot_utc": record["slot_utc"], "lat": record["lat"],
                    "lon": record["lon"], "item": item["id"],
                    "item_utc": f"{_time(item):%Y-%m-%dT%H:%M:%SZ}", "gap_s": record["gap_s"],
                    "label": label, **{k: round(float(v), 4) for k, v in shares.items()}})
                record["status"] = "photo"
    except Exception as error:  # network trouble: mark it, a rerun tries this slot again
        record["status"] = "error"
        print(f"  {name}: {type(error).__name__}: {str(error)[:120]}")
    _append(SLOTS_CSV, SLOT_FIELDS, record)
    return record["status"]


def load_chips(days: list[str] | None = None) -> list[dict]:
    """The saved real photos (rows of chips.csv), optionally only some days, in time order."""
    with open(CHIPS_CSV) as f:
        rows = {r["name"]: r for r in csv.DictReader(f)}  # by name: a retried slot counts once
    return sorted((r for r in rows.values() if days is None or r["day"] in days),
                  key=lambda r: r["slot_utc"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download what Sentinel-2A really saw, slot by slot")
    parser.add_argument("--start", default=FIRST_DAY, help="first day (UTC), e.g. 2026-10-06")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--workers", type=int, default=8, help="downloads at the same time")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    slots = photo_slots(start, args.days)
    done = _done_slots()
    todo = [s for s in slots if s[0] not in done]
    print(f"{len(slots):,} daylight photo slots in {args.days} days from {args.start}; "
          f"{len(slots) - len(todo):,} already done, {len(todo):,} to go.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, status in enumerate(pool.map(fetch_slot, todo), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 200 == 0 or i == len(todo):
                rate = i / (time.time() - started)
                print(f"  {i:,}/{len(todo):,} slots ({rate:.1f}/s): "
                      + ", ".join(f"{k} {v:,}" for k, v in sorted(counts.items())))

    if CHIPS_CSV.exists():
        rows = load_chips()
        print(f"\n{len(rows):,} real photos saved in {OUT_DIR.relative_to(ROOT)}. Classes:")
        for c in CLASSES:
            n = sum(r["label"] == c for r in rows)
            print(f"  {c:<11} {n:>6,}  ({n / len(rows):.0%})")
    if counts.get("error"):
        print(f"{counts['error']} slots failed (network). Run the same command again to retry them.")


if __name__ == "__main__":
    main()
