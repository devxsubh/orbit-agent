"""Where is the satellite, and when can it talk to Earth?

Run from the orbit-agent folder:
    python -m src.orbit.contact_windows
    python -m src.orbit.contact_windows --station svalbard --hours 48 --rate 200

What it does:
1. Downloads the latest orbit data (TLE) for the satellite from CelesTrak
2. Finds every pass over a ground station (the only time they can "talk")
3. Works out how much data can be sent down in each pass
4. Saves a map of the satellite's path to experiments/orbit_map.png
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from skyfield.api import EarthSatellite, load, wgs84

ROOT = Path(__file__).resolve().parents[2]
TLE_DIR = ROOT / "data" / "raw" / "tle"
BACKUP_TLES = Path(__file__).with_name("backup_tles.txt")
USER_AGENT = "orbit-agent/0.1 (student research project)"
IST = ZoneInfo("Asia/Kolkata")

# Satellites we can simulate (number = NORAD catalog ID)
SATELLITES = {
    "sentinel-2a": 40697,
    "sentinel-2b": 42063,
}

# Ground stations: (latitude, longitude). Locations are approximate.
GROUND_STATIONS = {
    "delhi": (28.61, 77.21),
    "bengaluru": (13.03, 77.51),
    "svalbard": (78.23, 15.40),  # polar station used by many Earth-observation satellites
}

ts = load.timescale()


@dataclass
class Pass:
    """One window when the satellite can talk to the ground station."""

    start: datetime
    end: datetime
    max_elevation_deg: float

    @property
    def minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60

    def data_mb(self, rate_mbps: float) -> float:
        """Megabytes that fit in this pass at a downlink speed in megabits/second."""
        return rate_mbps / 8 * (self.end - self.start).total_seconds()


def _checksum_ok(line: str) -> bool:
    """Each TLE line ends with a check digit. This catches broken downloads."""
    line = line.strip()
    if len(line) != 69 or not line[68].isdigit():
        return False
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
    return total % 10 == int(line[68])


def _fetch(url: str, accept: str = "*/*") -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8")


def _from_celestrak(catnr: int) -> str:
    text = _fetch(f"https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT=TLE")
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(lines[:3]) + "\n"


def _from_tle_api(catnr: int) -> str:
    data = json.loads(_fetch(f"https://tle.ivanstanojevic.me/api/tle/{catnr}", "application/json"))
    return f"{data['name']}\n{data['line1']}\n{data['line2']}\n"


# Where to get orbit data, tried in this order
SOURCES = [("CelesTrak", _from_celestrak), ("TLE API mirror", _from_tle_api)]


def load_satellite(name: str = "sentinel-2a", tle_path: str | None = None,
                   max_age_days: float = 2.0) -> EarthSatellite:
    """Load the satellite's orbit (TLE).

    Tries, in order: a fresh saved copy -> CelesTrak -> TLE API mirror
    -> an older saved copy -> the built-in backup in backup_tles.txt.
    """
    if tle_path:
        return load.tle_file(str(Path(tle_path).resolve()))[0]

    catnr = SATELLITES[name.lower()]
    TLE_DIR.mkdir(parents=True, exist_ok=True)
    saved = TLE_DIR / f"{catnr}.tle"
    is_fresh = saved.exists() and (time.time() - saved.stat().st_mtime) < max_age_days * 86400

    if not is_fresh:
        for source, fetch in SOURCES:
            try:
                text = fetch(catnr)
                lines = text.strip().splitlines()
                if len(lines) < 3 or not (_checksum_ok(lines[1]) and _checksum_ok(lines[2])):
                    raise ValueError("data looks broken")
                saved.write_text(text)
                print(f"Orbit data downloaded from {source}.")
                break
            except Exception as error:
                print(f"  {source} didn't work ({error}). Trying the next option...")
        else:
            if saved.exists():
                print("Using the orbit data saved earlier.")

    sats = load.tle_file(str(saved)) if saved.exists() else []
    if not sats:
        print("Using the built-in backup orbit data.")
        sats = [s for s in load.tle_file(str(BACKUP_TLES)) if s.model.satnum == catnr]
    if not sats:
        raise SystemExit("No orbit data available. Check your internet and try again.")
    return sats[0]


def contact_windows(sat: EarthSatellite, station: str, start: datetime, end: datetime,
                    min_elevation_deg: float = 10.0) -> list[Pass]:
    """Every pass where the satellite is at least `min_elevation_deg` above the horizon."""
    lat, lon = GROUND_STATIONS[station]
    ground = wgs84.latlon(lat, lon)
    view = sat - ground
    t0, t1 = ts.from_datetime(start), ts.from_datetime(end)

    times, events = sat.find_events(ground, t0, t1, altitude_degrees=min_elevation_deg)

    passes: list[Pass] = []
    rise, peak = None, 0.0
    for t, event in zip(times, events):
        if rise is None:  # pass was already happening when we started
            rise = t if event == 0 else t0
            peak = view.at(rise).altaz()[0].degrees
        if event == 1:  # highest point of the pass
            peak = max(peak, view.at(t).altaz()[0].degrees)
        elif event == 2:  # satellite sets
            passes.append(Pass(rise.utc_datetime(), t.utc_datetime(), round(peak, 1)))
            rise, peak = None, 0.0
    if rise is not None:  # pass still going at the end
        passes.append(Pass(rise.utc_datetime(), end, round(peak, 1)))
    return passes


def ground_track(sat: EarthSatellite, start: datetime, end: datetime,
                 step_seconds: int = 60) -> list[tuple[datetime, float, float]]:
    """Points on Earth directly under the satellite: (time, lat, lon)."""
    steps = int((end - start).total_seconds() // step_seconds) + 1
    times = ts.from_datetimes([start + timedelta(seconds=i * step_seconds) for i in range(steps)])
    lat, lon = wgs84.latlon_of(sat.at(times))
    return list(zip(times.utc_datetime(), lat.degrees, lon.degrees))


def save_map(track, passes: list[Pass], station: str, out_path: Path, title: str) -> None:
    """Map of the satellite path. Blue = can talk to the station, grey = can't."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink_2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
    no_contact, in_contact, station_color = "#c3c2b7", "#2a78d6", "#eb6834"

    talking = [any(p.start <= t <= p.end for p in passes) for t, _, _ in track]
    lons = [lo for _, _, lo in track]
    lats = [la for _, la, _ in track]

    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    ax.scatter([x for x, c in zip(lons, talking) if not c], [y for y, c in zip(lats, talking) if not c],
               s=4, color=no_contact, label="Path (no contact)", zorder=2)
    ax.scatter([x for x, c in zip(lons, talking) if c], [y for y, c in zip(lats, talking) if c],
               s=10, color=in_contact, label=f"Path (talking to {station.title()})", zorder=3)

    lat, lon = GROUND_STATIONS[station]
    ax.scatter([lon], [lat], s=180, marker="*", color=station_color, edgecolors=surface,
               linewidths=1.5, label="Ground station", zorder=4)
    ax.annotate(station.title(), (lon, lat), xytext=(12, -4), textcoords="offset points",
                color=ink, fontsize=9, zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=surface, edgecolor="none"))

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_xlabel("Longitude", color=ink_2)
    ax.set_ylabel("Latitude", color=ink_2)
    ax.tick_params(colors=ink_2, labelsize=8)
    ax.grid(color=grid, linewidth=0.6, zorder=0)
    for side in ax.spines.values():
        side.set_color(grid)
    ax.set_title(title, color=ink, fontsize=11, loc="left")
    legend = ax.legend(loc="lower left", fontsize=8, frameon=True, markerscale=1.5)
    legend.get_frame().set_facecolor(surface)
    legend.get_frame().set_edgecolor(grid)
    for text in legend.get_texts():
        text.set_color(ink)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=surface)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="When can the satellite talk to Earth?")
    parser.add_argument("--satellite", default="sentinel-2a", choices=SATELLITES)
    parser.add_argument("--station", default="delhi", choices=GROUND_STATIONS)
    parser.add_argument("--hours", type=float, default=24, help="how far ahead to look")
    parser.add_argument("--rate", type=float, default=100, help="downlink speed in Mbps")
    parser.add_argument("--min-elevation", type=float, default=10, help="degrees above horizon")
    parser.add_argument("--tle", help="use a saved TLE file instead of downloading")
    parser.add_argument("--map", default=str(ROOT / "experiments" / "orbit_map.png"))
    parser.add_argument("--no-map", action="store_true")
    args = parser.parse_args()

    sat = load_satellite(args.satellite, args.tle)
    start = datetime.now(timezone.utc).replace(microsecond=0)
    end = start + timedelta(hours=args.hours)
    passes = contact_windows(sat, args.station, start, end, args.min_elevation)

    age_days = (ts.from_datetime(start) - sat.epoch)
    print(f"\n{sat.name} over {args.station.title()}: next {args.hours:g} hours (times in IST)")
    print(f"Orbit data is {age_days:.1f} days old.")
    if age_days > 14:
        print("That's old, so pass times may be off by a few minutes. Fine for testing.")
    print()

    if not passes:
        print("No passes in this time. Try --hours 48 or --station svalbard.")
    else:
        print(f"{'#':>2}  {'Start':<16} {'End':<9} {'Minutes':>7} {'Max elev':>9} {'Data (MB)':>10}")
        for i, p in enumerate(passes, 1):
            s, e = p.start.astimezone(IST), p.end.astimezone(IST)
            print(f"{i:>2}  {s:%d %b %H:%M:%S}  {e:%H:%M:%S} {p.minutes:>7.1f} "
                  f"{p.max_elevation_deg:>8.1f}° {p.data_mb(args.rate):>10,.0f}")

        total_min = sum(p.minutes for p in passes)
        total_mb = sum(p.data_mb(args.rate) for p in passes)
        share = total_min / (args.hours * 60) * 100
        print(f"\nTotal: {len(passes)} passes, {total_min:.1f} minutes of contact, "
              f"{total_mb:,.0f} MB at {args.rate:g} Mbps.")
        print(f"The satellite can talk to {args.station.title()} only {share:.1f}% of the time.")
        print("The rest of the time, it has to decide on its own. That's the agent's job.")

    if not args.no_map:
        track = ground_track(sat, start, end)
        out = Path(args.map)
        save_map(track, passes, args.station, out,
                 f"{sat.name}: path for the next {args.hours:g} h")
        print(f"\nMap saved: {out}")


if __name__ == "__main__":
    main()
