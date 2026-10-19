"""Build the demo website in site/ (used by GitHub Pages, works on your Mac too).

Run from the orbit-agent folder:
    python -m src.demo.build_site
    python -m http.server -d site     (then open http://localhost:8000)

It does NOT run the model or the replay. It uses the saved demo/data.js and only
remakes the photo thumbnails (demo/thumbs/ is not in git), downloading the AID
photos it needs from Hugging Face.

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
import time
from pathlib import Path

from PIL import Image

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


def make_thumbnail(name: str, out: Path) -> None:
    """Download one AID photo and save a small copy (same size as src/demo/export.py)."""
    by_lower = {label.lower(): label for label in CLASSES}  # "bareland" -> "BareLand"
    label = by_lower[name.rsplit("_", 1)[0]]                # "bareland_29.jpg" -> "BareLand"
    for attempt in range(1, 4):  # Hugging Face sometimes drops a connection: try 3 times
        try:
            path = _download(label, name)
            break
        except OSError as error:
            if attempt == 3:
                raise
            print(f"  {name}: download failed ({error}), trying again...")
            time.sleep(5 * attempt)
    with Image.open(path) as img:
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
