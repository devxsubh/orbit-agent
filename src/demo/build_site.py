"""Build the demo website in site/ (used by GitHub Pages, works on your Mac too).

Run from the orbit-agent folder:
    python -m src.demo.build_site
    python -m http.server -d site     (then open http://localhost:8000)

It does NOT run the model or the replay. It uses the saved demo/data.js and only
remakes the photo thumbnails (demo/thumbs/ is not in git). It downloads what it needs:
AID photos from Hugging Face, or EuroSAT (Sentinel-2, about 95 MB) from Zenodo.

site/
    index.html          sends visitors on to demo/
    demo/index.html     the demo page
    demo/data.js
    demo/thumbs/*.jpg
    docs/logo.svg       the page uses ../docs/logo.svg
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from src.data import eurosat
from src.data.aid import CLASSES, _download

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo"
SITE = ROOT / "site"

# Root page: just forwards to the demo
REDIRECT = """<!doctype html>
<meta charset="utf-8">
<title>orbit-agent demo</title>
<meta http-equiv="refresh" content="0; url=demo/">
<a href="demo/">Open the demo</a>
"""


def photo_names() -> list[str]:
    """Photo file names the demo shows, read from demo/data.js."""
    text = (DEMO / "data.js").read_text()
    data = json.loads(text[text.index("{"):text.rindex("}") + 1])
    return data["images"]


def photo_path(name: str) -> Path:
    """Where a demo photo is, downloading it if needed.

    EuroSAT names start with a capital ("Forest_36.jpg"), AID names don't ("forest_36.jpg").
    """
    prefix = name.rsplit("_", 1)[0]
    if prefix in eurosat.CLASSES:
        return (eurosat.dataset_dir() or eurosat.download()) / prefix / name
    by_lower = {label.lower(): label for label in CLASSES}  # "bareland" -> "BareLand"
    return _download(by_lower[prefix], name)                # _download retries 3 times


def make_thumbnail(name: str, out: Path) -> None:
    """Save a small copy of one photo (same size as src/demo/export.py)."""
    with Image.open(photo_path(name)) as img:
        img.convert("RGB").resize((160, 160), Image.Resampling.LANCZOS).save(out, quality=82)


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    thumbs = SITE / "demo" / "thumbs"
    thumbs.mkdir(parents=True)
    (SITE / "docs").mkdir()

    shutil.copy(DEMO / "index.html", SITE / "demo" / "index.html")
    shutil.copy(DEMO / "data.js", SITE / "demo" / "data.js")
    shutil.copy(ROOT / "docs" / "logo.svg", SITE / "docs" / "logo.svg")
    (SITE / "index.html").write_text(REDIRECT)

    # Second page (the audited downlink), if its data was exported (python -m src.demo.export_audit).
    # Its thumbnails are small JPEGs of real Sentinel-2 photos in demo/audit_thumbs/ (commit them to publish).
    if (DEMO / "audit_data.js").exists():
        shutil.copy(DEMO / "audit.html", SITE / "demo" / "audit.html")
        shutil.copy(DEMO / "audit_data.js", SITE / "demo" / "audit_data.js")
        if (DEMO / "audit_thumbs").exists():
            shutil.copytree(DEMO / "audit_thumbs", SITE / "demo" / "audit_thumbs")

    names = photo_names()
    print(f"Making {len(names)} thumbnails (downloads photos you don't have yet)...")
    for i, name in enumerate(names, 1):
        make_thumbnail(name, thumbs / name)
        if i % 100 == 0:
            print(f"  {i}/{len(names)}")

    size_mb = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file()) / 1e6
    print(f"Site ready in {SITE} ({size_mb:.1f} MB). Try: python -m http.server -d site")


if __name__ == "__main__":
    main()
