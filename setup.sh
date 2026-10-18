#!/usr/bin/env bash
# One-time setup for orbit-agent. Run on your Mac:  ./setup.sh
set -e
cd "$(dirname "$0")"

# Is this Python 3.10 or newer?
is_new_enough() {
  "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

# Look for a Python 3.10+ anywhere common on a Mac
find_python() {
  local c
  for c in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && is_new_enough "$(command -v "$c")"; then
      command -v "$c"; return 0
    fi
  done
  for c in "$HOME"/.pyenv/versions/3.12*/bin/python3 \
           "$HOME"/.pyenv/versions/3.11*/bin/python3 \
           "$HOME"/.pyenv/versions/3.13*/bin/python3 \
           "$HOME"/.pyenv/versions/3.10*/bin/python3 \
           /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3 \
           /usr/local/bin/python3.12 /usr/local/bin/python3.11 /usr/local/bin/python3; do
    if [ -x "$c" ] && is_new_enough "$c"; then
      echo "$c"; return 0
    fi
  done
  return 1
}

echo "== 1/5 Finding Python 3.10+ =="
PY="$(find_python || true)"

if [ -z "$PY" ]; then
  echo "No Python 3.10+ found (your default is $(python3 --version 2>&1))."
  if command -v pyenv >/dev/null 2>&1 || [ -x "$HOME/.pyenv/bin/pyenv" ]; then
    PYENV="$(command -v pyenv || echo "$HOME/.pyenv/bin/pyenv")"
    read -r -p "Install Python 3.11 with pyenv now? (takes a few minutes) [y/N] " ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then
      "$PYENV" install -s 3.11
      PY="$(find_python || true)"
    fi
  else
    echo "Install it with Homebrew:  brew install python@3.11"
  fi
fi

if [ -z "$PY" ]; then
  echo "Still no Python 3.10+. Install one (see above) and run ./setup.sh again."
  exit 1
fi
echo "Using $("$PY" --version) at $PY"

echo "== 2/5 Creating Python environment (.venv) =="
# Remove an old environment made with a too-old Python
if [ -x .venv/bin/python ] && ! is_new_enough .venv/bin/python; then
  echo "Old .venv uses an old Python. Recreating it."
  rm -rf .venv
fi
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip -q

echo "== 3/5 Installing packages =="
pip install -r requirements.txt -q

echo "== 4/5 Downloading papers =="
bash papers/download_papers.sh || true

echo "== 5/5 AI model (Gemma 3, about 3 GB) =="
if command -v ollama >/dev/null 2>&1; then
  if ollama list 2>/dev/null | grep -q "gemma3"; then
    echo "Gemma 3 is already downloaded."
  else
    read -r -p "Download Gemma 3 4B now? (about 3 GB) [y/N] " ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then
      ollama pull gemma3:4b
    else
      echo "Skipped. Run later:  ollama pull gemma3:4b"
    fi
  fi
else
  echo "Ollama not found. Install it from https://ollama.com then run:  ollama pull gemma3:4b"
fi

echo
python check_setup.py
echo
echo "Next time, start with:  source .venv/bin/activate"
