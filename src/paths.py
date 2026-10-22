"""Where results live: one folder per experiment, in the order we did them (results/NN_name/).

Every script asks this module for a result file's path, so it is always the same folder:
    from src.paths import result, find
    out = result("audit_summary_real_20261022_1318.csv")   # results/09_audit/...
    newest = find("audit_summary_real_*.csv")[-1]          # newest match in any experiment folder

The experiments, in order (notebooks/02_experiments.ipynb tells each one's story):
    00_data             what is in the data: EDA, label checks, split by location
    01_orbit            orbit simulator and photo replay
    02_gemma            Gemma 3 with no training (EuroSAT, AID)
    03_agents_v1_v3     "not sure" signals, agents v1-v3 on the replay
    04_decision_head    frozen SigLIP + 30 KB head (AID)
    05_sentinel2        heads on real Sentinel-2 photos (EuroSAT)
    06_cnn_label_curve  tiny CNN, and how many labelled photos each model needs
    07_robustness       replay error bars over days, stations and photo orders
    08_compression      the turning point: plain compression as a baseline
    09_audit            audited downlink and the self-check (EuroSAT and real replay)
    10_rivals           fair zero-shot rival, Jetson timing
    11_metrics          beyond accuracy: confusion matrices, per-class scores
    12_retraining       retraining based on what the data showed
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Start of a result file's name -> its experiment folder (first match wins, so longer starts come first)
FOLDERS = [
    ("eda_", "00_data"), ("label_check", "00_data"), ("spatial_split", "00_data"),
    ("orbit_map", "01_orbit"), ("replay_bars", "07_robustness"), ("replay_", "01_orbit"),
    ("baseline_", "02_gemma"),
    ("confidence_", "03_agents_v1_v3"), ("agent_v1", "03_agents_v1_v3"),
    ("agents_delhi_final_equal_zeroshot", "10_rivals"), ("agents_", "03_agents_v1_v3"),
    ("head_aid", "04_decision_head"),
    ("head_eurosat", "05_sentinel2"), ("sentinel_test", "05_sentinel2"),
    ("cnn_", "06_cnn_label_curve"), ("label_curve", "06_cnn_label_curve"),
    ("compress_", "08_compression"),
    ("audit_", "09_audit"),
    ("zeroshot_", "10_rivals"), ("bench_", "10_rivals"),
    ("metrics_", "11_metrics"), ("confusion", "11_metrics"),
    ("retrain", "12_retraining"),
]


def folder(name: str) -> Path:
    """The experiment folder a result file belongs to (results/other/ if no rule matches)."""
    for start, sub in FOLDERS:
        if name.startswith(start):
            return RESULTS / sub
    return RESULTS / "other"


def result(name: str) -> Path:
    """Path of a result file: where it already is, else where it belongs (folder created)."""
    found = sorted(RESULTS.glob(f"*/{name}"))
    if found:
        return found[0]
    path = folder(name) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def find(pattern: str) -> list[Path]:
    """Every result file matching a pattern (e.g. "audit_summary_real_*.csv"), oldest first."""
    return sorted(RESULTS.glob(f"*/{pattern}"), key=lambda p: p.stat().st_mtime)
