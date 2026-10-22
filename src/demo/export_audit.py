"""Data for the second demo page, demo/audit.html: "the satellite reports its own error".

For every onboard model (healthy, or broken in one of the ways in src/model/faults.py) and every
day of a week at Delhi, it runs labels only, triage (agent v4) and the self-checking audit
(src/agent/self_check.py) on the same passes and bytes, and saves each day's class shares: the
truth, each policy's estimate, the audit's honest 90% error bar, and for every pass whether the
self-check had switched to safe mode. For real Sentinel-2A photos it also saves thumbnails of the
photos the audit checked, so the page can show them.

Run from the orbit-agent folder:
    python -m src.demo.export_audit              # real Sentinel-2A photos if downloaded, else EuroSAT
    python -m src.demo.export_audit --data eurosat
    open demo/audit.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.agent.audit import POLICIES
from src.agent.self_check import SWITCH, SelfCheck
from src.eval.audit_test import eurosat_days, eurosat_population, real_days, real_population
from src.eval.estimate import combine, estimate, interval

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo"
THUMBS = DEMO / "audit_thumbs"
TILE = 300_000
SHOWN = 12  # checked photos shown per day
WORDS = {"Cloud": "cloud", "Trees": "trees", "ShrubGrass": "shrubs & grass", "Cropland": "cropland",
         "BuiltUp": "built-up", "Bare": "bare ground", "SnowIce": "snow & ice", "Water": "water",
         "AnnualCrop": "seasonal crops", "Forest": "forest", "HerbaceousVegetation": "grass & shrubs",
         "Highway": "highway", "Industrial": "industrial", "Pasture": "pasture",
         "PermanentCrop": "orchards", "Residential": "houses", "River": "river", "SeaLake": "sea & lake"}


def r4(a) -> list[float]:
    return [round(float(v), 4) for v in a]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export data for demo/audit.html")
    parser.add_argument("--data", choices=["real", "eurosat"])
    parser.add_argument("--station", default="delhi")
    args = parser.parse_args()

    from src.data import s2_real
    from src.orbit.contact_windows import load_satellite

    data = args.data or ("real" if s2_real.CHIPS_CSV.exists() else "eurosat")
    sat = load_satellite(tle_path=str(ROOT / "data" / "raw" / "tle" / "40697.tle"))
    if data == "real":
        days_all = sorted({r["day"] for r in s2_real.load_chips()})
        train, test = days_all[:-7], days_all[-7:]
        pop = real_population(train, test)
        rate, day_iter = 0.1, lambda: real_days(pop, args.station, 0.1, test, sat)
        rows = {r["name"]: r for r in s2_real.load_chips(test)}
    else:
        pop = eurosat_population()
        rate, day_iter = 0.25, lambda: eurosat_days(pop, args.station, 0.25, 7, 42, sat)
        rows = {}

    methods = {"labels": ("labels only", "count"), "triage": ("triage (v4)", "count"),
               "audit": ("self-check", "self-check")}  # the self-checking audit (src/agent/self_check.py)
    out = {"data": data, "station": args.station.title(), "rate_mbps": rate, "switch": SWITCH,
           "classes": [{"name": c, "words": WORDS.get(c, c)} for c in pop.classes],
           "scenarios": [{"name": s, "note": pop.notes[s]} for s in pop.scenarios], "days": {}}
    THUMBS.mkdir(parents=True, exist_ok=True)
    days = list(day_iter())
    for scenario, P_all in pop.scenarios.items():
        out["days"][scenario] = []
        check = SelfCheck(len(pop.classes))  # one mission: the ground remembers every check
        for d, (day, passes) in enumerate(days):
            rng = np.random.default_rng([42, d, len(scenario)])
            idx_all = np.concatenate([idx for _, idx in passes])
            entry = {"day": day, "photos": len(idx_all), "passes": len(passes),
                     "truth": r4(pop.Y[idx_all].mean(0)),
                     "budget_mb": round(sum(b for b, _ in passes) / 1e6, 1)}
            for key, (policy, how) in methods.items():
                parts, checked, modes = [], [], []
                for budget, idx in passes:
                    if policy == "self-check":
                        ratio = check.ratio if check.expected > 0 else None  # what decided this pass's mode
                        sent, e, safe = check.run_pass(P_all[idx], budget, TILE,
                                                       lambda full, idx=idx: pop.Y[idx][full], rng)
                        modes.append({"safe": bool(safe), "ratio": None if ratio is None else round(ratio, 2)})
                        parts.append(e)
                    else:
                        sent = POLICIES[policy](P_all[idx], budget, TILE, rng)
                        parts.append(estimate(P_all[idx], sent, pop.Y[idx][sent.full], how, rng))
                    checked += [int(i) for i in idx[sent.full]]
                if modes:
                    entry["modes"] = modes
                e = combine(parts, [len(i) for _, i in passes])
                lo, hi = interval(e, "scaled+1")
                entry[key] = {"est": r4(e.value), "lo": r4(lo), "hi": r4(hi), "checked": len(checked)}
                if key == "audit" and data == "real":
                    shown = []
                    for i in checked[:SHOWN]:
                        name = pop.names[i]
                        thumb = THUMBS / name.replace(".png", ".jpg")
                        if not thumb.exists():
                            img = Image.open(s2_real.OUT_DIR / rows[name]["day"] / name).convert("RGB")
                            img.resize((96, 96), Image.Resampling.LANCZOS).save(thumb, quality=80)
                        shown.append({"file": thumb.name, "truth": pop.classes[pop.truth[i]],
                                      "said": pop.classes[int(np.argmax(P_all[i]))],
                                      "where": f"{float(rows[name]['lat']):.1f}, {float(rows[name]['lon']):.1f}"})
                    entry["shown"] = shown
            out["days"][scenario].append(entry)
    (DEMO / "audit_data.js").write_text("window.AUDIT = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print(f"Saved demo/audit_data.js ({data}, {len(days)} days x {len(pop.scenarios)} onboard models). "
          "Open demo/audit.html")


if __name__ == "__main__":
    main()
