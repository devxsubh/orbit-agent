#!/usr/bin/env bash
# Downloads the free starter papers into the right folders.
# Run from anywhere:  bash papers/download_papers.sh
# Already-downloaded papers are skipped, so it's safe to run again.

cd "$(dirname "$0")"

ok=0
failed=()

get() {
  local folder="$1" file="$2" url="$3"
  local dest="$folder/$file"
  if [ -s "$dest" ]; then
    echo "  ✓ already have  $file"
    ok=$((ok+1))
    return
  fi
  echo "  ↓ downloading   $file"
  local tmp
  tmp="$(mktemp)"
  if curl -sSL -f --retry 2 --max-time 120 -A "Mozilla/5.0 (orbit-agent setup)" -o "$tmp" "$url" \
     && head -c 4 "$tmp" | grep -q "%PDF"; then
    mv "$tmp" "$dest"
    ok=$((ok+1))
  else
    rm -f "$tmp"
    failed+=("$file  ->  $url")
  fi
}

echo "Downloading starter papers..."

# 1. Basics: why put AI on satellites
get 1-basics "01_orbital-edge-computing_denby-lucia_2020.pdf" "https://classpages.cselabs.umn.edu/Fall-2021/csci8980-ec/papers/Orbital.pdf"
get 1-basics "02_kodan_denby-et-al_2023.pdf"                  "https://par.nsf.gov/servlets/purl/10424445"

# 2. Latest work
get 2-latest-work "04_navi-orbital_nasa-jpl_2026.pdf"            "https://arxiv.org/pdf/2606.18271"
get 2-latest-work "05_fool-feature-compression_2025.pdf"         "https://arxiv.org/pdf/2403.16677"
get 2-latest-work "06_on-orbit-space-ai-survey_2026.pdf"         "https://arxiv.org/pdf/2604.16518"
get 2-latest-work "07_ml-in-embedded-space-systems_review_2026.pdf" "https://strathprints.strath.ac.uk/96859/1/Eshaq-etal-IEEEA-2026-A-review-of-machine-learning-in-embedded-space-systems.pdf"

# 3. Tools and data
get 3-tools-and-data "08_gemma-3-technical-report_2025.pdf"      "https://arxiv.org/pdf/2503.19786"
get 3-tools-and-data "09_aid-dataset_xia_2017.pdf"               "https://arxiv.org/pdf/1608.05167"
get 3-tools-and-data "10_eurosat-dataset_helber_2019.pdf"        "https://arxiv.org/pdf/1709.00029"

echo
echo "Done: $ok of 9 papers ready."
echo "Paper 03 (Φ-Sat-1) is behind a paywall. See papers/README.md for the link."
if [ ${#failed[@]} -gt 0 ]; then
  echo
  echo "These didn't download. Open the link in your browser and save the PDF into papers/:"
  for f in "${failed[@]}"; do echo "  - $f"; done
fi
