# Reading list

Read in this order. ⭐ = read closely. Others: read the abstract, intro and conclusion first.

To download the free PDFs into these folders, run:
```bash
bash papers/download_papers.sh
```

## 1. Basics: why put AI on satellites
- [ ] ⭐ **01 · Orbital Edge Computing** (Denby & Lucia, ASPLOS 2020)
  The paper that started this field. Explains why "send everything down" breaks as satellites multiply.
  [Abstract](https://asplos.hosting2.acm.org/wp/wp-content/uploads/2020/abstracts/paper_9_5.html)
- [ ] **02 · Kodan** (Denby et al., ASPLOS 2023)
  Shows onboard processing can cut the data sent down by about 90%.
  [PDF](https://par.nsf.gov/servlets/purl/10424445)
- [ ] **03 · The Φ-Sat-1 Mission** (Giuffrida et al., IEEE TGRS 2022)
  The first deep learning model run on a satellite (cloud detection). Paywalled: get it via your college library or search the title.
  [Mission page](https://www.eoportal.org/satellite-missions/phisat-1)

## 2. Latest work
- [ ] ⭐ **04 · NAVI-Orbital** (NASA JPL, 2026)
  First vision-language model (Gemma 3) running in space. **Most important paper.** Your project builds on this.
  [arXiv](https://arxiv.org/abs/2606.18271)
- [ ] **05 · FOOL** (Furutanpey et al., IEEE TMC 2025)
  Smart compression before sending data down. A strong baseline to compare against.
  [arXiv](https://arxiv.org/abs/2403.16677)
- [ ] ⭐ **06 · On-Orbit Space AI survey** (2026)
  Lists the open problems. Section 6 is where your paper idea comes from.
  [arXiv](https://arxiv.org/abs/2604.16518)
- [ ] **07 · Review of ML in embedded space systems** (Eshaq et al., IEEE Access 2026)
  What has actually flown on small satellites, and what went wrong.
  [PDF](https://strathprints.strath.ac.uk/96859/1/Eshaq-etal-IEEEA-2026-A-review-of-machine-learning-in-embedded-space-systems.pdf)

## 3. Tools and data
- [ ] **08 · Gemma 3 Technical Report** (Google, 2025)
  The model you'll run. Skim the vision and model size sections.
  [arXiv](https://arxiv.org/abs/2503.19786)
- [ ] **09 · AID dataset** (Xia et al., 2017)
  NASA tested on this (88.16% accuracy), so you can compare directly.
  [arXiv](https://arxiv.org/abs/1608.05167)
- [ ] **10 · EuroSAT dataset** (Helber et al., 2019)
  Free labeled Sentinel-2 images: the same satellite data you'll use.
  [arXiv](https://arxiv.org/abs/1709.00029)

## Tip
After each paper, copy `notes/_template.md` to `notes/04_navi-orbital.md` (etc.) and fill it in.
Five minutes of notes now saves hours when you write your own paper.
