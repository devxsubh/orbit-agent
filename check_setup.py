"""Checks that orbit-agent is ready to go.

Run:  python check_setup.py
"""
import importlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
problems = 0


def ok(msg):
    print(f"  OK    {msg}")


def bad(msg, fix=""):
    global problems
    problems += 1
    print(f"  FIX   {msg}" + (f"\n        -> {fix}" if fix else ""))


print("\nPython")
if sys.version_info >= (3, 10):
    ok(f"Python {sys.version.split()[0]}")
else:
    bad(f"Python {sys.version.split()[0]} is too old", "Use Python 3.10 or newer")

print("\nPackages")
packages = {
    "skyfield": "skyfield",
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "pillow",
    "rasterio": "rasterio",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "ollama": "ollama",
    "langgraph": "langgraph",
}
for module, pip_name in packages.items():
    try:
        importlib.import_module(module)
        ok(pip_name)
    except ImportError:
        bad(f"{pip_name} not installed", "Run ./setup.sh (or: pip install -r requirements.txt)")

print("\nAI model (Ollama)")
try:
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
        models = [m["name"] for m in json.load(r).get("models", [])]
    ok("Ollama is running")
    gemma = [m for m in models if m.startswith("gemma3")]
    if gemma:
        ok(f"Gemma 3 found: {', '.join(gemma)}")
    else:
        bad("Gemma 3 not downloaded", "Run: ollama pull gemma3:4b")
except Exception:
    bad("Ollama is not running", "Open the Ollama app (or run: ollama serve)")

print("\nFolders")
for folder in ["papers", "data/raw", "data/processed",
               "src/orbit", "src/model", "src/agent", "src/eval", "results", "notebooks"]:
    if (ROOT / folder).is_dir():
        ok(folder)
    else:
        bad(f"{folder} is missing")

print("\nPapers")
pdfs = sorted((ROOT / "papers").rglob("*.pdf"))
if len(pdfs) >= 9:
    ok(f"{len(pdfs)} papers downloaded")
else:
    bad(f"{len(pdfs)} of 9 papers downloaded", "Run: bash papers/download_papers.sh")

print()
if problems == 0:
    print("All good. Start with papers/README.md")
else:
    print(f"{problems} thing(s) to fix (see FIX lines above).")
