"""Where on Earth is each EuroSAT photo? (for a test split by LOCATION)

EuroSAT cuts neighbouring 64x64 tiles from the same Sentinel-2 scenes. A random split can put a
tile in the test set and its neighbour in the training set, so test accuracy looks too good.
Fix: group photos by place and keep whole groups on one side.

The RGB download (JPEGs) has no coordinates. The GeoTIFF version (EuroSAT_MS.zip, 2 GB) has them
in the first ~800 bytes of each file. We stream the zip from Zenodo ONCE and keep only those bytes,
so nothing big is saved to disk. (One small request per file would be less data, but Zenodo
answers "429 too many requests" after about a hundred.)

Run from the orbit-agent folder (one 2 GB download, ~5-20 min depending on the network):
    python -m src.data.eurosat_geo

Saves data/processed/eurosat_coords.csv: file (as in the RGB set, e.g. Forest_1.jpg), label, lat, lon.
"""
from __future__ import annotations

import csv
import struct
import time
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "eurosat_coords.csv"
URL = "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1"
USER_AGENT = "orbit-agent/0.1 (student research project)"
HEAD_BYTES = 4096  # compressed bytes to fetch per file; the geo tags are in the first ~800 bytes


def get(start: int, end: int, tries: int = 4) -> bytes:
    """Bytes start..end-1 of the zip on Zenodo (retries on network errors)."""
    for attempt in range(tries):
        try:
            request = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT,
                                                           "Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(30 * (attempt + 1))  # Zenodo's "too many requests" lasts about a minute
    raise RuntimeError("unreachable")


def zip_size() -> int:
    request = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def list_tifs() -> dict[str, tuple[int, int]]:
    """{'Forest/Forest_1.tif': (compression method, where the file starts in the zip)}.
    Read from the zip's table of contents at its end (about 3 MB)."""
    size = zip_size()
    tail = get(size - 70_000, size)
    at = tail.rfind(b"PK\x05\x06")  # "end of table of contents" marker
    _, toc_size, toc_start = struct.unpack("<HII", tail[at + 10:at + 20])
    toc = get(toc_start, toc_start + toc_size)
    found, p = {}, 0
    while p < len(toc):
        method = struct.unpack("<H", toc[p + 10:p + 12])[0]
        name_len, extra_len, comment_len = struct.unpack("<HHH", toc[p + 28:p + 34])
        start = struct.unpack("<I", toc[p + 42:p + 46])[0]
        name = toc[p + 46:p + 46 + name_len].decode()
        if name.endswith(".tif"):
            found[name.split("/", 1)[1]] = (method, start)
        p += 46 + name_len + extra_len + comment_len
    return found


def geo_tags(head: bytes) -> tuple[int, float, float]:
    """(EPSG code, x, y of the top-left corner) from the start of a GeoTIFF."""
    order = "<" if head[:2] == b"II" else ">"
    ifd = struct.unpack(order + "I", head[4:8])[0]
    tags = {}
    for k in range(struct.unpack(order + "H", head[ifd:ifd + 2])[0]):
        tag, kind, count, value = struct.unpack(order + "HHII", head[ifd + 2 + 12 * k:ifd + 14 + 12 * k])
        tags[tag] = (kind, count, value)
    _, count, at = tags[33922]  # ModelTiepoint: (i, j, k, x, y, z) as doubles
    tie = struct.unpack(order + "6d", head[at:at + 48])
    _, count, at = tags[34735]  # GeoKeyDirectory: 4 shorts per key
    keys = struct.unpack(order + f"{count}H", head[at:at + 2 * count])
    epsg = 0
    for i in range(4, len(keys), 4):
        if keys[i] in (3072, 2048):  # projected CRS, or geographic CRS
            epsg = keys[i + 3]
            if keys[i] == 3072:
                break
    return epsg, tie[3], tie[4]


def read_one(name: str, method: int, start: int) -> tuple[int, float, float]:
    raw = get(start, start + 30 + 200 + HEAD_BYTES)
    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    data = raw[30 + name_len + extra_len:]
    head = zlib.decompressobj(-15).decompress(data) if method == 8 else data
    return geo_tags(head)


def to_lat_lon(epsg: int, x: float, y: float) -> tuple[float, float]:
    from rasterio.warp import transform

    if epsg == 4326:
        return y, x
    lons, lats = transform(f"EPSG:{epsg}", "EPSG:4326", [x + 320], [y - 320])  # tile centre (640 m tile)
    return lats[0], lons[0]


def stream_heads(tifs: dict[str, tuple[int, int]]):
    """Download the zip once, front to back, and yield (name, epsg, x, y) for every GeoTIFF."""
    order = sorted(tifs.items(), key=lambda item: item[1][1])  # in the order they sit in the zip
    first = order[0][1][1]
    request = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT, "Range": f"bytes={first}-"})
    with urllib.request.urlopen(request, timeout=120) as response:
        position = first
        for (name, (method, start)), following in zip(order, order[1:] + [(None, (0, None))]):
            response.read(start - position)  # skip to this file (normally 0 bytes)
            end = following[1][1]
            raw = response.read(end - start) if end else response.read(30 + 200 + HEAD_BYTES)
            position = start + len(raw)
            name_len, extra_len = struct.unpack("<HH", raw[26:30])
            data = raw[30 + name_len + extra_len:30 + name_len + extra_len + HEAD_BYTES]
            head = zlib.decompressobj(-15).decompress(data) if method == 8 else data
            yield (name, *geo_tags(head))


def main() -> None:
    if OUT.exists() and sum(1 for _ in open(OUT)) > 27_000:
        print(f"Already done: {OUT.relative_to(ROOT)}")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Reading the table of contents of EuroSAT_MS.zip on Zenodo...")
    tifs = list_tifs()
    print(f"{len(tifs):,} GeoTIFFs. Streaming the 2 GB zip once (only the first bytes of each are kept)...")
    part = OUT.with_suffix(".part")
    started = time.time()
    with open(part, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "epsg", "x", "y", "lat", "lon"])
        for i, (name, epsg, x, y) in enumerate(stream_heads(tifs), 1):
            lat, lon = to_lat_lon(epsg, x, y)
            label, tif = name.split("/")
            writer.writerow([Path(tif).with_suffix(".jpg").name, label, epsg, round(x, 1), round(y, 1),
                             round(lat, 5), round(lon, 5)])
            if i % 1000 == 0 or i == len(tifs):
                print(f"  {i:,}/{len(tifs):,} ({time.time() - started:.0f} s)")
    part.replace(OUT)  # only a finished file gets the real name
    print(f"Saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
