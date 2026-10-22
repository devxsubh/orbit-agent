"""Make the data for the demo page (demo/index.html).

Run from the orbit-agent folder:
    python -m src.demo.export                 (real Sentinel-2 photos, agent v4, 1 Mbps)
    python -m src.demo.export --rate 0.5
    python -m src.demo.export --photos aid    (AID stand-in photos, agent v4 with the AID head)
    python -m src.demo.export --photos aid --agent "agent v3"   (Gemma 3 labels instead)

It runs the photo replay and the agents once, then saves:
- demo/data.js     everything the page plays back (orbit, passes, photos, decisions)
- demo/thumbs/     small copies of the photos
Then open demo/index.html in a browser. No server needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image

from src.agent.decide import POLICIES, Decision, ModelView
from src.data import eurosat
from src.data.aid import LABELS
from src.eval.run_agent import default_answers, load_views, photo_set_name, score
from src.eval.sentinel_test import PHOTO_BYTES
from src.orbit.contact_windows import GROUND_STATIONS, ground_track, load_satellite
from src.orbit.replay import build_replay
from src.paths import result

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo"
LAND_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-110m.json"
LAND_FILE = ROOT / "data" / "raw" / "land-110m.json"
USER_AGENT = "orbit-agent/0.1 (student research project)"

# Agent v4's heads for the demo: the small ones (SigLIP-base, 90M). They suit a satellite better and
# let fewer wrong labels through than the big head (see notebooks/02_experiments.ipynb).
SMALL_HEAD = result("head_aid_siglip-base_n20_final.csv")            # AID photos
SENTINEL_HEAD = result("head_eurosat_siglip-base_n100_final.csv")     # Sentinel-2

# Which model makes the labels, as shown on the page: (short, longer)
MODEL_TEXT = {
    "agent v4": ("SigLIP-base + small head, on board",
                 "a frozen SigLIP-base image model (90M) with a small classifier trained on other "
                 "AID photos, on a laptop"),
}
GEMMA_TEXT = ("Gemma 3 (4B), on board", "gemma3:4b on a laptop")
SENTINEL_TEXT = ("SigLIP-base + Sentinel-2 head, on board",
                 "a frozen SigLIP-base image model (90M) with a small classifier trained on 1,000 "
                 "other Sentinel-2 photos, on a laptop")


def sentinel_views(head_csv: Path) -> tuple[dict[str, ModelView], list[tuple[Path, str]]]:
    """Agent v4's answers for the 1,000 final EuroSAT (Sentinel-2) photos, and the photo pool."""
    root = eurosat.dataset_dir() or eurosat.download()
    views, pool = {}, []
    with open(head_csv) as f:
        for row in csv.DictReader(f):
            views[row["file"]] = ModelView(guess="", unsure=False, head_guess=row["guess"],
                                           p_head=float(row["p"]))
            pool.append((root / row["truth"] / row["file"], row["truth"]))
    return views, pool

# One letter per decision, to keep data.js small
CODES = {Decision.SEND_FULL: "F", Decision.DESCRIBE: "L", Decision.DESCRIBE_UNSURE: "U",
         Decision.DROPPED: "D", Decision.SKIP: "S", Decision.SEND_CROP: "C"}


def land_outlines() -> list[list[list[float]]]:
    """World land shapes as lists of [lon, lat] points (downloaded once, then cached)."""
    if not LAND_FILE.exists():
        request = urllib.request.Request(LAND_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            LAND_FILE.parent.mkdir(parents=True, exist_ok=True)
            LAND_FILE.write_bytes(response.read())
    topo = json.loads(LAND_FILE.read_text())

    # TopoJSON stores shared borders ("arcs") as small steps; turn them back into points
    (sx, sy), (tx, ty) = topo["transform"]["scale"], topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        points = []
        for dx, dy in arc:
            x, y = x + dx, y + dy
            points.append([round(x * sx + tx, 2), round(y * sy + ty, 2)])
        arcs.append(points)

    def ring(indices: list[int]) -> list[list[float]]:
        points: list[list[float]] = []
        for i in indices:
            arc = arcs[i] if i >= 0 else arcs[~i][::-1]  # negative index = arc backwards
            points.extend(arc if not points else arc[1:])
        return points

    shapes = []
    for geometry in topo["objects"]["land"]["geometries"]:
        polygons = geometry["arcs"] if geometry["type"] == "MultiPolygon" else [geometry["arcs"]]
        for polygon in polygons:
            shapes.append(ring(polygon[0]))  # outer edge only; lakes don't matter at this size
    return shapes


def save_thumbnail(path: Path) -> str:
    """Small copy of a photo for the page. Returns its file name."""
    out = DEMO / "thumbs" / path.name
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as img:
            img.convert("RGB").resize((160, 160), Image.Resampling.LANCZOS).save(out, quality=82)
    return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Make the data for the demo page")
    parser.add_argument("--rate", type=float, default=1.0, help="downlink speed in Mbps")
    parser.add_argument("--agent", default="agent v4", choices=[p for p in POLICIES if "agent" in p])
    parser.add_argument("--photos", default="sentinel", choices=["sentinel", "aid"],
                        help="sentinel = real Sentinel-2 photos (EuroSAT); aid = AID stand-ins")
    parser.add_argument("--head", help="decision-head CSV for agent v4 (default: the small head "
                                       "for the chosen photos)")
    parser.add_argument("--station", default="delhi", choices=GROUND_STATIONS)
    parser.add_argument("--start", default="2026-10-19T00:00", help="start time in UTC")
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--answers", help="CSV with model answers (default: same as run_agent)")
    args = parser.parse_args()

    if args.photos == "sentinel":
        if args.agent != "agent v4":
            raise SystemExit("Gemma's answers exist for only 100 Sentinel-2 photos, so the "
                             "Sentinel-2 demo shows agent v4. For agents v1-v3 add: --photos aid")
        views, pool = sentinel_views(Path(args.head or SENTINEL_HEAD))
        photo_bytes = PHOTO_BYTES  # same as the Sentinel-2 replay test (a full tile)
        photo_set = "FINAL TEST Sentinel-2 photos"
        labels, model_text = eurosat.LABELS, SENTINEL_TEXT
        photo_about = ("each photo is a real Sentinel-2 photo from EuroSAT (64x64 px, 10 m per pixel; "
                       f"counted as {photo_bytes // 1000} KB, like a full tile), but not the view at "
                       "that exact spot")
    else:
        answers = Path(args.answers) if args.answers else default_answers()
        views, pool = load_views(answers, Path(args.head or SMALL_HEAD)
                                 if args.agent == "agent v4" else None)
        photo_bytes = sum(p.stat().st_size for p, _ in pool) // len(pool)  # equal sizes, like run_agent
        photo_set = photo_set_name(answers)
        labels, model_text = LABELS, MODEL_TEXT.get(args.agent, GEMMA_TEXT)
        photo_about = f"each photo is an AID aerial photo ({photo_bytes // 1000} KB each), not the real view at that spot"

    def said(view) -> tuple[str, int]:
        """The label and chance (%) of the chosen agent's model."""
        if args.agent == "agent v4":
            return view.head_guess, round(view.p_head * 100)
        return view.guess, round((view.p_lite if args.agent == "agent v3-lite" else view.p_full) * 100)

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    sat = load_satellite()
    windows, left_over = build_replay(sat, args.station, start, args.hours, pool,
                                      rate_mbps=args.rate, photo_bytes=photo_bytes)

    def seconds(when: datetime) -> int:
        return round((when - start).total_seconds())

    # Decisions of the agent and of "send all raw", per photo and per pass
    images: dict[str, int] = {}
    photos, passes = [], []
    for w_index, window in enumerate(windows):
        agent = {id(c.photo): c for c in POLICIES[args.agent](window, views)}
        raw = {id(c.photo): c for c in POLICIES["send all raw"](window, views)}
        for photo in window.photos:
            view = views[photo.path.name]
            name = save_thumbnail(photo.path)
            images.setdefault(name, len(images))
            photos.append([seconds(photo.taken), round(photo.lat, 2), round(photo.lon, 2),
                           images[name], photo.truth, *said(view),
                           w_index, CODES[agent[id(photo)].decision], CODES[raw[id(photo)].decision]])
        results = {name: score(list(by_photo.values()))
                   for name, by_photo in (("agent", agent), ("raw", raw))}
        passes.append({"start": seconds(window.downlink.start), "end": seconds(window.downlink.end),
                       "budget_mb": round(window.budget_bytes / 1e6, 1),
                       "waiting_mb": round(window.waiting_bytes / 1e6, 1),
                       **{f"{k}_{m}": round(v[m], 1) if m == "mb" else v[m]
                          for k, v in results.items() for m in ("correct", "wrong", "unsure", "full", "mb")}})
    for photo in left_over:  # taken after the last pass: still waiting at the end of the day
        view = views[photo.path.name]
        name = save_thumbnail(photo.path)
        images.setdefault(name, len(images))
        photos.append([seconds(photo.taken), round(photo.lat, 2), round(photo.lon, 2), images[name],
                       photo.truth, *said(view), -1, "", ""])

    track = [[seconds(t), round(float(lat), 2), round(float(lon), 2)]
             for t, lat, lon in ground_track(sat, start, start + timedelta(hours=args.hours), 30)]
    lat, lon = GROUND_STATIONS[args.station]
    data = {
        "satellite": sat.name, "station": args.station.title(), "station_lat": lat, "station_lon": lon,
        "start_utc": start.isoformat(), "hours": args.hours, "rate_mbps": args.rate,
        "photo_kb": round(photo_bytes / 1000), "agent": args.agent,
        "model": model_text[0], "model_about": model_text[1], "photo_about": photo_about,
        "photo_set": photo_set, "made": datetime.now(timezone.utc).isoformat(),
        "labels": labels, "images": list(images), "track": track, "passes": passes,
        "photo_fields": ["t", "lat", "lon", "img", "truth", "guess", "chance", "window",
                         "agent", "raw"],
        "photos": photos, "land": land_outlines(),
    }
    out = DEMO / "data.js"
    out.write_text("// Made by: python -m src.demo.export  (don't edit by hand)\n"
                   "window.DEMO = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"Demo data: {len(photos):,} photos, {len(passes)} passes, {len(images)} thumbnails, "
          f"{args.agent} at {args.rate:g} Mbps, answers from {photo_set}.")
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB). Open demo/index.html in a browser.")


if __name__ == "__main__":
    main()
