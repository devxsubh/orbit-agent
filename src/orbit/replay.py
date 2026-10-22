"""A satellite on your desk: replay photos along the satellite's real path.

The satellite takes a photo every few seconds (in daylight). Photos wait onboard
until the next pass over the ground station. Each pass can only send so many MB.
That gap (photos taken vs. MB that fit) is what the agent has to manage.

Run from the orbit-agent folder:
    python -m src.orbit.replay
    python -m src.orbit.replay --hours 48 --every 60 --rate 2

Note: each photo is a random AID (or EuroSAT) photo. It is NOT the real view at that
place yet. Time, position and the data budget are real; the photo content is a stand-in.
"""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.orbit.contact_windows import (GROUND_STATIONS, IST, SATELLITES, Pass, contact_windows,
                                       ground_track, load_satellite)
from src.paths import result

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Photo:
    """One photo taken by the satellite."""

    taken: datetime      # when (UTC)
    lat: float           # where the satellite was
    lon: float
    path: Path           # stand-in photo from the dataset
    truth: str           # its real class (for scoring later)
    size_bytes: int      # size if we send the whole photo


@dataclass
class Window:
    """Photos taken since the last pass, and the pass that can send them down."""

    downlink: Pass
    budget_bytes: int
    photos: list[Photo] = field(default_factory=list)

    @property
    def waiting_bytes(self) -> int:
        return sum(p.size_bytes for p in self.photos)

    @property
    def share_sendable(self) -> float:
        """Fraction of the waiting data that fits in this pass (1.0 = everything fits)."""
        return min(1.0, self.budget_bytes / self.waiting_bytes) if self.photos else 1.0


def is_daytime(when: datetime, lon: float) -> bool:
    """Rough check: local sun time between 6 AM and 6 PM. (Optical cameras need daylight.)"""
    sun_hour = (when.hour + when.minute / 60 + lon / 15) % 24
    return 6 <= sun_hour < 18


def build_replay(sat, station: str, start: datetime, hours: float,
                 photo_pool: list[tuple[Path, str]], every_seconds: int = 30,
                 rate_mbps: float = 1.0, seed: int = 42,
                 daytime_only: bool = True,
                 photo_bytes: int | None = None) -> tuple[list[Window], list[Photo]]:
    """Simulate `hours` of flying. Returns (windows, photos left over after the last pass).

    photo_pool: (path, true class) pairs; each new photo picks one at random.
    photo_bytes: give every photo this size (like raw satellite tiles, which are all the same
                 size). None = use each JPEG's real size (simple scenes compress smaller).
    """
    end = start + timedelta(hours=hours)
    passes = contact_windows(sat, station, start, end)
    windows = [Window(p, int(p.data_mb(rate_mbps) * 1_000_000)) for p in passes]
    left_over: list[Photo] = []
    rng = random.Random(seed)

    for when, lat, lon in ground_track(sat, start, end, every_seconds):
        if daytime_only and not is_daytime(when, lon):
            continue
        if any(p.start <= when <= p.end for p in passes):
            continue  # busy talking to the ground during a pass
        path, truth = rng.choice(photo_pool)
        size = photo_bytes or path.stat().st_size
        photo = Photo(when, float(lat), float(lon), path, truth, size)
        # The photo waits for the first pass that starts after it was taken
        target = next((w for w in windows if w.downlink.start > when), None)
        (target.photos if target else left_over).append(photo)

    return windows, left_over


def save_csv(windows: list[Window], out: Path) -> None:
    """One row per photo, so you can check the replay in a spreadsheet."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window", "taken_utc", "lat", "lon", "file", "truth", "bytes",
                         "pass_start_utc", "pass_budget_bytes"])
        for i, w in enumerate(windows, 1):
            for p in w.photos:
                writer.writerow([i, f"{p.taken:%Y-%m-%d %H:%M:%S}", round(p.lat, 3),
                                 round(p.lon, 3), p.path.name, p.truth, p.size_bytes,
                                 f"{w.downlink.start:%Y-%m-%d %H:%M:%S}", w.budget_bytes])


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay photos along the satellite's real path")
    parser.add_argument("--satellite", default="sentinel-2a", choices=SATELLITES)
    parser.add_argument("--station", default="delhi", choices=GROUND_STATIONS)
    parser.add_argument("--hours", type=float, default=24, help="how long to fly")
    parser.add_argument("--every", type=int, default=30, help="seconds between photos")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="downlink speed in Mbps (small: our photos are small JPEGs)")
    parser.add_argument("--dataset", choices=["aid", "eurosat"], default="aid")
    parser.add_argument("--per-class", type=int, default=10, help="photos per class in the pool")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", help="start time in UTC, e.g. 2026-10-19T00:00 (default: now)")
    parser.add_argument("--tle", help="use a saved TLE file (for repeatable runs)")
    parser.add_argument("--all-day", action="store_true", help="take photos at night too")
    args = parser.parse_args()

    if args.dataset == "aid":
        from src.data.aid import sample
    else:
        from src.data.eurosat import sample
    pool = sample(args.per_class, args.seed)

    if args.start:
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    else:
        start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    sat = load_satellite(args.satellite, args.tle)
    windows, left_over = build_replay(sat, args.station, start, args.hours, pool, args.every,
                                      args.rate, args.seed, not args.all_day)

    print(f"\nReplay: {sat.name} over {args.station.title()}, {args.hours:g} h "
          f"from {start.astimezone(IST):%d %b %H:%M} IST (start UTC: {start:%Y-%m-%dT%H:%M})")
    print(f"Photo every {args.every} s{'' if args.all_day else ' in daylight'}, "
          f"stand-in photos from {args.dataset} ({len(pool)} in the pool). "
          f"Downlink {args.rate:g} Mbps.\n")

    if not windows:
        print("No passes in this time, so nothing can be sent. Try --hours 48.")
        return

    print(f"{'#':>2}  {'Pass (IST)':<13} {'Minutes':>7} {'Photos':>7} {'Waiting':>9} "
          f"{'Can send':>9} {'Fits':>5}")
    for i, w in enumerate(windows, 1):
        print(f"{i:>2}  {w.downlink.start.astimezone(IST):%d %b %H:%M}  {w.downlink.minutes:>7.1f} "
              f"{len(w.photos):>7,} {w.waiting_bytes / 1e6:>7.0f} MB {w.budget_bytes / 1e6:>6.0f} MB "
              f"{w.share_sendable:>5.0%}")

    taken = sum(len(w.photos) for w in windows)
    waiting = sum(w.waiting_bytes for w in windows)
    sendable = sum(min(w.budget_bytes, w.waiting_bytes) for w in windows)
    print(f"\nTotal: {taken:,} photos ({waiting / 1e6:,.0f} MB) waiting for {len(windows)} passes.")
    if waiting:
        print(f"Sending raw, only {sendable / waiting:.0%} of the data fits. "
              "The agent has to choose what goes down.")
    if left_over:
        print(f"{len(left_over):,} photos were taken after the last pass (not counted).")

    out = result(f"replay_{args.station}_{start:%Y%m%d_%H%M}.csv")
    save_csv(windows, out)
    print(f"\nEvery photo saved to: {out}")


if __name__ == "__main__":
    main()
