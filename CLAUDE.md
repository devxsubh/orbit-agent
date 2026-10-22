# orbit-agent: context for Claude

Last updated: 22 Oct 2026 (restructured; the old long version is `archive/CLAUDE_before_restructure.md`).

## How to work with me (Subham)
- **Write simple, short, plain language.** Dense answers are hard for me to read.
- **Plan before big changes.** Show the plan, then build.
- Test code before handing it over. If the model or network isn't reachable, test with fakes and say so.
- Keep code beginner-friendly: clear names, short comments, plain-English terminal output.
- **Keep the repo clean:** results go in `results/NN_name/` (via `src/paths.py`), charts in `docs/figures/`,
  superseded things in `archive/`. Tell the story in order: data first, then experiments by date.
- Don't commit PDFs (the report PDF is built locally: `python report/build.py`).

About me: final-year CS (AI) student at Newton School of Technology (Rishihood University), Delhi.
Good at Python, LangGraph + Claude agents, RAG, full-stack.
Goals for this project:
1. A **research paper** (helps with M.Tech / foreign university applications)
2. A **demo** for internships at Indian spacetech companies (target: strong internship by end of Nov 2026)
3. Maybe a startup later, but research + career first

## The project in one line
An AI on a satellite decides what to send down to Earth, and the ground can check how much to trust it.

- **Problem:** satellites take far more photos than they can send (Sentinel-2A talks to Delhi ~1.5% of the day).
- **Product:** the share of each day's photos in each class (cloud, cropland, water...) with an honest 90% error bar.
- **Method (the audited downlink):** label every photo onboard (frozen SigLIP + 30 KB head), then send full photos
  picked at random with known chances (floor 0.5); the ground corrects the labels with them (Horvitz-Thompson).
  **Self-check:** checked mistakes > 1.5x what the model expected -> safe mode (random checks + recalibration).
- **Main finding:** with a broken onboard model (new place, bit flips, damaged camera), labels-only and triage fail
  silently (bars hold 9-27%); the self-checking audit stays honest (97-98%) on what Sentinel-2A really photographed.
- **Method: "a satellite on your desk":** laptop now, Jetson Orin later; real orbit data (TLE) for the passes.

## Why this topic (research done Oct 2026)
- **NAVI-Orbital (NASA JPL, arXiv 2606.18271):** on 16 Apr 2026, the first vision-language model
  ran fully in orbit. Gemma 3 on Loft Orbital's YAM-9 satellite, Nvidia Jetson Orin AGX, 4-bit GGUF.
  88.16% zero-shot on AID cut to 18 classes. **Our project builds on this.**
- Others: DPhi Space ran a Liquid AI VLM in orbit (Apr 2026). ImageSat ran Gemma-3n on a Jetson Orin
  satellite processor at under 2 s per image (SmallSat 2026).
- **Downlink bottleneck:** Kodan (ASPLOS 2023) cut downlink by about 89-97% with onboard processing.
- **Open problems** (On-Orbit Space AI survey, arXiv 2604.16518): trust and verification in orbit,
  what to send down, multi-satellite tasking, power limits, conservative fallback modes.
- **Indian companies doing this:** SkyServe (Bengaluru, STORM), Pixxel, GalaxEye (Jetson Orin onboard), Digantara, SatSure.
- **Honest business read:** few buyers (satellite operators, government, defence). Research + career first.

## My setup (MacBook, Apple Silicon, 8 GB RAM)
- Project folder: `~/Downloads/SubhamPortfolio/orbit-agent`, GitHub `devxsubh/orbit-agent`, Pages from `main`.
- **Python:** system Python is 3.9 (too old). The `.venv` uses pyenv Python 3.12.0. Always `source .venv/bin/activate`.
  Notebook kernel: `orbit-agent` (installed with ipykernel).
- **Ollama** with Gemma 3 4B (auto-picks any `gemma3*`). 8 GB RAM, so gemma3:12b doesn't fit.
- **My network blocks some sites** (connection reset): `celestrak.org`, `par.nsf.gov`.
  Works: arXiv, Zenodo, PyPI, Hugging Face, Element 84 Earth Search + AWS COGs, `tle.ivanstanojevic.me`
  (rejects a "Mozilla" User-Agent: use `orbit-agent/0.1`). Zenodo returns 429 after ~100 small range requests.
- **Kaggle:** username `subhammahapatra004`, token in `~/.kaggle/access_token`. Private dataset
  `orbit-agent-s2-real` = the real replay photos (`python -m src.cloud.kaggle_upload`). Kaggle clones the
  GitHub branch, so push before `python -m src.cloud.kaggle_run`.
- No Jetson yet.

## Repo map
| Path | What |
|---|---|
| `notebooks/01_data.ipynb` | The data first (EDA of EuroSAT, AID, real replay; infrared; extra days) |
| `notebooks/02_experiments.ipynb` | Experiments 1-12 in order; loads every number from `results/` |
| `report/report.md` + `report/build.py` | The report (PDF built locally, not in git) |
| `results/NN_name/` | Result files per experiment; `src/paths.py`: `result(name)` finds/places a file, `find(pattern)` searches |
| `docs/figures/` | Every chart (`docs/logo.svg` for the demo) |
| `src/orbit/` | `contact_windows.py` (passes), `replay.py` (stand-in photo replay) |
| `src/data/` | `eurosat.py`, `eurosat_geo.py` (coords), `aid.py`, `s2_real.py` (real replay + **fixed day split**), `s2_bands.py` (green/red/NIR/SWIR + 6 features) |
| `src/model/` | `vlm.py`, `confidence.py` (Gemma), `head.py` (frozen SigLIP + head; `fit_head(class_weight=)`), `cnn.py`, `zeroshot.py`, `faults.py`, `retrain.py` (experiment 12) |
| `src/agent/` | `decide.py` (agents v1-v4), `audit.py` (audited downlink), `self_check.py` |
| `src/eval/` | `estimate.py` (ground estimators, error bars incl. betting), `audit_test.py` (main experiment; `real_population(..., keep=, recipe=)`), `eda.py`, `metrics_report.py`, and one script per older experiment |
| `src/cloud/` | `kaggle_run.py` (`--dataset aid|s2real`), `kaggle_upload.py` |
| `src/demo/`, `demo/` | Demo pages (`index.html`: one day, no score card; `audit.html`: the audit) |
| `paper/outline.md` | Paper outline, IGARSS 2027 (audited downlink) |
| `archive/` | Old experiment log (full history), old README / CLAUDE.md, notes, pilots, outline v1, superseded results, old PDFs |

**Day split (never by position):** `s2_real.TRAIN_DAYS` 6-11 Oct, `TUNING_DAY` 12 Oct, `TEST_DAYS` 13-19 Oct,
`EXTRA_DAYS` 15 Sep - 5 Oct (training only, experiment 12). `MAIN_DAYS` = 6-19 Oct.

## Experiments (details in notebooks/02_experiments.ipynb; full history in archive/experiment_log.md)
0. **Data (22 Oct):** EuroSAT/AID clean; no EuroSAT leak (location split 93.4% vs 92.7%). Real replay: 46% cloud,
   43 snow, 4 built-up; 14% messy photos carry 30% of mistakes; snow + cloud blown-out white (42/43); labels 97%
   right by eye (AI first pass); day splits move acc ~1 pt; audit conclusions hold on clean photos.
1. **Orbit (18 Oct):** Delhi 2-3 passes/day, ~1.5%; Svalbard ~8%.
2. **Gemma 3 zero-shot (18 Oct):** EuroSAT 22-32%, AID 61-65% (NAVI-style 18 classes: 76.5%), 7.5 s/photo.
3. **Agents v1-v3 (19 Oct):** best "not sure" signal = confusable classes (AUROC 0.77); old per-photo score.
4. **Decision head (20 Oct):** SigLIP-base + 30 KB head 95.7% (~95 ± 2.5); 400M head 98.3% but 5 mistakes >96% sure.
5. **Sentinel-2 / EuroSAT (20 Oct):** AID head 33% -> retrained head 92.8% (400M 94.8%).
6. **Tiny CNN + label curve (20 Oct):** head wins with few labels; CNN catches up ~50/class.
7. **Robustness (20 Oct):** order stable; head 100.0 ± 0.0% on the old score = ceiling.
8. **Compression (20 Oct):** compress-to-fit sends everything at R <= 14 -> old score can't fail. **Question changed.**
9. **Audit + self-check (20 Oct):** real replay; labels/triage fail silently with broken models; audit honest;
   self-check beats raw random (5.3-5.5 vs 5.9 pts); floor 0.5; closed loop 46% -> 83% in 4 days.
10. **Rivals (22 Oct):** SigLIP zero-shot AID 66.3% / EuroSAT 40.8% (> Gemma); betting bars hold for every class
    (1.1-2.4x wider); 19 ms/photo on the Mac.
11. **Beyond accuracy (22 Oct):** real healthy head 82.3% acc, 67.0% balanced (classes >= 5 test photos); snow never predicted.
12. **Retraining (22 Oct):** recipes on real replay (test 13-19 Oct): baseline 82.3% / 67.0% bal; + balanced 63.9 / 69.1
    (dishonest, ECE 0.17); + prior correction 81.1 / 64.8; + infrared (6 features, weight 0.1 by CV) 82.2 / 67.0;
    **+ 3 more weeks (7,158 train, snow 120) 84.7 / 73.1, snow 32%, ECE 0.012**; + SigLIP 400M (Kaggle) 85.6 / 74.7.
    Ice clouds look snow-like in SWIR (301/1,615 cloud photos), latitude didn't help. Audit with recipe 5 (fixed
    beforehand): self-check 0.25 Mbps 1.10 -> 0.83 pts, bars 96%; labels still 87%; broken models still fail.

**Old per-photo score:** removed from README/demo (it saturates; compression ties it). Don't bring it back as a headline.

## How to run (key commands; each experiment's command is in its notebook section)
```bash
cd ~/Downloads/SubhamPortfolio/orbit-agent && source .venv/bin/activate
python -m src.data.s2_real                    # real replay (add --start 2026-09-15 --days 21 for the extra days)
python -m src.data.s2_bands                   # infrared bands for every real photo
python -m src.eval.eda                        # data analysis (results/00_data, docs/figures/eda_*)
python -m src.eval.audit_test --data real     # main experiment (results/09_audit)
python -m src.model.retrain                   # experiment 12 recipes (results/12_retraining)
python -m src.model.retrain --audit           # audit with the retrained head
python -m src.eval.metrics_report             # confusion matrices, per-class scores
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=orbit-agent notebooks/02_experiments.ipynb
python report/build.py                        # report PDF (local only)
python -m src.demo.export_audit && open demo/audit.html
```

## Key decisions
- Satellites: Sentinel-2A (NORAD 40697). Default ground station: Delhi. Times in IST.
- Orbit data order: fresh saved copy, then CelesTrak, then TLE mirror, then old saved copy, then built-in backup.
  The audit experiments always use the saved `data/raw/tle/40697.tle` (repeatable).
- Real-replay labels: cloud map >= 50% cloud -> Cloud, >= 50% snow -> SnowIce, else most common WorldCover class.
- Fair testing: settings picked on tuning data only; test photos used once; rules for flags fixed before results.
- Balanced accuracy / macro-F1 over classes with >= 5 test photos (BuiltUp has 0-1).
- Big files stay out of git: `data/raw/*`, `data/processed/*`, PDFs.

## Next steps
1. Subham re-checks the eye-check labels (`results/00_data/label_check_real.csv`) and the 395 low-agreement photos.
2. Jetson Orin Nano: `python -m src.eval.jetson_bench` on the board.
3. Demo video of `demo/audit.html`.
4. Paper sections 3-4 (`paper/outline.md`, IGARSS 2027, deadline ~10 Jan 2027).

## Papers (in `papers/`, reading order in `papers/README.md`)
01 Orbital Edge Computing (Denby & Lucia 2020) · 02 Kodan (2023, **missing**: my network blocks NSF,
download by hand from https://par.nsf.gov/servlets/purl/10424445) · 03 Φ-Sat-1 (**paywalled**, IEEE TGRS 2022)
· 04 NAVI-Orbital ⭐ · 05 FOOL · 06 On-Orbit Space AI survey ⭐ · 07 ML in embedded space systems review
· 08 Gemma 3 report · 09 AID dataset · 10 EuroSAT dataset.
Read first: 01, 04, 06.
