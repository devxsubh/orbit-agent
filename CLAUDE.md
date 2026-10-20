# orbit-agent: context for Claude

Last updated: 18 Oct 2026 (handed over from a claude.ai chat).

## How to work with me (Subham)
- **Write simple, short, plain language.** Dense answers are hard for me to read.
- **Plan before big changes.** Show the plan, then build.
- Test code before handing it over. If the model or network isn't reachable, test with fakes and say so.
- Keep code beginner-friendly: clear names, short comments, plain-English terminal output.

About me: final-year CS (AI) student at Newton School of Technology (Rishihood University), Delhi.
Good at Python, LangGraph + Claude agents, RAG, full-stack.
Goals for this project:
1. A **research paper** (helps with M.Tech / foreign university applications)
2. A **demo** for internships at Indian spacetech companies (target: strong internship by end of Nov 2026)
3. Maybe a startup later, but research + career first

## The project in one line
An AI agent that runs on a satellite and decides what is worth sending down to Earth.

### The problem
Satellites take far more photos than they can send down. They only get a few minutes of
connection per ground-station pass. Most data is wasted or arrives late.

### The idea
A small vision-language model (Gemma 3) on the satellite looks at each photo. An agent decides:
- **Skip** (clouds, nothing useful)
- **Describe** in short text
- **Send** a crop or the full image
- **Not sure?** Send the full image instead of guessing (abstain)

### Research question
With a fixed downlink budget, how much useful information can an onboard agent deliver,
compared with sending everything raw?

**Main metric:** correct information reaching the ground per MB sent (`src/eval/metrics.py`).
**Baselines to beat:** send everything raw / cloud filter only / caption every photo (NAVI-style) /
smart compression (FOOL paper).

### Method: "a satellite on your desk"
- Laptop now, Jetson Orin Nano later (same chip family as flight hardware)
- Free Sentinel-2 photos replayed as if the satellite is flying over Earth
- Real orbit data (TLE) decides when the satellite can talk to the ground

## Why this topic (research done Oct 2026)
- **NAVI-Orbital (NASA JPL, arXiv 2606.18271):** on 16 Apr 2026, the first vision-language model
  ran fully in orbit. Gemma 3 on Loft Orbital's YAM-9 satellite, Nvidia Jetson Orin AGX, 4-bit GGUF.
  88.16% zero-shot accuracy on the AID dataset. **Our project builds on this.**
- Others: DPhi Space ran a Liquid AI VLM in orbit (Apr 2026). ImageSat ran Gemma-3n on a Jetson Orin
  satellite processor at under 2 s per image (SmallSat 2026).
- **Downlink bottleneck:** Kodan (ASPLOS 2023) cut downlink by about 89-97% with onboard processing.
- **Open problems** (On-Orbit Space AI survey, arXiv 2604.16518): trust and verification in orbit,
  what to send down, multi-satellite tasking, power limits, conservative fallback modes.
- **Indian companies doing this:** SkyServe (Bengaluru, STORM onboard AI platform, tested with NASA JPL),
  Pixxel ($100M round, hyperspectral, Pathfinder orbital data centre with Sarvam),
  GalaxEye (Jetson Orin onboard), Digantara, SatSure.
- **Honest business read:** few buyers (satellite operators, government, defence). Treat this as
  research + career first.
- Alternative considered and set aside: Lean theorem-proving agents (Meta's AutoformBench, Oct 2026).

## My setup (MacBook, Apple Silicon)
- Project folder: `~/Downloads/SubhamPortfolio/orbit-agent` (not a git repo yet)
- **Python:** system Python is 3.9 (too old). The `.venv` uses pyenv Python 3.12.0.
  Always run `source .venv/bin/activate` first.
- **Ollama** is installed and a Gemma 3 model is downloaded. Check the exact name with `ollama list`.
  The code auto-picks any `gemma3*` model if `gemma3:4b` isn't found.
- **My network blocks some sites** (connection reset): `celestrak.org`, `par.nsf.gov`.
  These work: arXiv, Zenodo, PyPI, Hugging Face, `tle.ivanstanojevic.me`.
  Note: `tle.ivanstanojevic.me` rejects a "Mozilla" User-Agent (406). Use `orbit-agent/0.1`.
- No Jetson yet. Laptop only for now.

## Files
| Path | What it is | Status |
|---|---|---|
| `README.md` | Project overview, 10-week plan, status checklist | done |
| `papers/README.md` | Reading list in order, with links | done |
| `papers/download_papers.sh` | Downloads the free PDFs | done (8 of 10 downloaded) |
| `notes/_template.md` | Template for paper notes | done |
| `setup.sh` | One-time setup: finds Python 3.10+, makes `.venv`, installs packages | done, works |
| `check_setup.py` | Health check | done |
| `src/orbit/contact_windows.py` | Orbit simulator: passes over a ground station, data budget per pass, map | **done, works on my Mac** |
| `src/orbit/backup_tles.txt` | Built-in orbit data (17 Oct 2026) for Sentinel-2A/2B, offline fallback | done |
| `src/data/eurosat.py` | Downloads EuroSAT (Zenodo, about 95 MB), picks random photos per class | written, tested with fakes only |
| `src/orbit/replay.py` | Photo replay: photo every 30 s in daylight along `ground_track()`, queued per Delhi pass, budget = pass time x `--rate`. `build_replay()` returns windows for the agent | **done, works** |
| `src/data/aid.py` | AID (NAVI's dataset) from Hugging Face `blanchon/AID`. Downloads only the sampled photos (10/class = 85 MB) | **done, works** |
| `src/model/vlm.py` | Talks to Gemma 3 via Ollama: `ask`, `describe_image`, `classify_image`, `check_ready` | written, tested with fakes only |
| `src/eval/baseline_eurosat.py` | Zero-shot accuracy test on EuroSAT, saves CSV to `experiments/` | written, tested with fakes only |
| `src/eval/metrics.py` | `accuracy`, `useful_per_mb` | done |
| `src/agent/decide.py` | Agent v1 + v2 (plain rules): label every photo, then full photos by priority. v2 uses the better "not sure" rule and sends "unsure: X?" labels (DESCRIBE_UNSURE). Also baselines `plan_raw`, `plan_caption` | **done, works** (SKIP / SEND_CROP not used yet) |
| `src/eval/run_agent.py` | Runs all 4 policies (raw, caption, agent v1, agent v2) on the replay at several `--rates`, uses saved answers (no model calls). Prefers the full "not sure" test CSV if it exists. Saves `experiments/agents_*.csv` | **done, works** |
| `src/model/confidence.py` | "Not sure" signals: A several answers, B confused classes (from first 300), C answer margin (logprobs), D agreement over 3 shuffled asks. `is_unsure()` = agent v2 rule (A or D or margin < 8), fixed before testing | **done, works** |
| `src/demo/export.py` + `demo/index.html` | Demo: export runs replay + agent once → `demo/data.js` + `demo/thumbs/` (gitignored); index.html plays it back (map with night side, timeline, stacked score bars, photo card, queue). Rebuilt minimal on 19 Oct (plain words, IBM Plex fonts, `docs/logo.svg`); tested with puppeteer (all controls, phone width). 20 Oct: default is now **real Sentinel-2 photos** (`--photos sentinel`, EuroSAT final 1,000, head `head_eurosat_siglip-base_n100_final.csv`, agent v4 only, photos counted as 280 KB); `--photos aid` for AID (any agent). Page text comes from `D.model`, `D.model_about`, `D.photo_about`. `build_site.py` gets EuroSAT thumbs from Zenodo (names with a capital = EuroSAT) | **done, works** |
| `src/demo/build_site.py` + `.github/workflows/pages.yml` | Website for GitHub Pages: copies demo page + saved `data.js` + logo into `site/` (gitignored), remakes the thumbnails from Hugging Face (retries 3x). No model or replay. Workflow runs on push to `main`. Live at https://devxsubh.github.io/orbit-agent/ | **done, tested locally** (clean copy, all files load) |
| `src/eval/plot_results.py` | Draws `docs/results.png` (dark) from the final agents CSV (20 Oct: `..._20261020_0947.csv`, with agent v4 small head). Colour-blind-safe set checked with the dataviz validator; v4 = thick yellow, its wrong-label numbers written on the bars | **done, works** |
| `docs/` | README images (`results.png`, `demo_pass.png`, `demo_end.png`) + `experiment_log.md` (full history, moved out of README on 19 Oct) | done |
| `src/eval/confidence_test.py` | Runs the signals on 300 NEW AID photos (11-20/class), resumable; prints AUROC + wrong caught at 10/20/30% flagged; saves CSV + chart. `--evaluate-only` | **done, works** (quick 30-photo run done) |
| `src/model/head.py` | Agent v4 "decision head" (Jev-style): frozen image model (`--backbone siglip` 400M / `siglip-base` 90M / `dinov2`) makes embeddings (cached in `data/processed/`), logistic regression trained on AID photos 31+ (`--train-per-class`, 0 = all 9,100), C picked on tuning by log-loss. Saves `experiments/head_aid_<backbone>_{tuning,final}.csv`; `run_agent` adds agent v4 when one exists | **done, works** (quick Mac test) |
| `src/eval/sentinel_test.py` | Agent v4 on real Sentinel-2 (EuroSAT): test 1 AID head as is (30 AID classes matched to 10 EuroSAT via `MATCH`, fixed before testing), test 2 new head on EuroSAT (final 1,000 incl. Gemma's 100, tuning 500, train 100/class), test 3 replay (photos counted as 280 KB) | **done, works** |
| `src/model/cnn.py` | Tiny CNN baseline (MobileNetV3-Small, ImageNet start, all weights trained, flips/rot90, 20 epochs, best epoch + temperature on tuning). Same photo sets as the heads; CSV in head format -> `run_agent --head` / `sentinel_test --head-csv` | **done, works** (Kaggle T4, 7 min) |
| `src/eval/label_curve.py` + `src/eval/plot_label_curve.py` | Accuracy vs training photos per class (5/10/20/50/100, 3 seeds) for frozen head vs tiny CNN on AID and EuroSAT -> `experiments/label_curve_*.csv`, chart `docs/label_curve.png` (head yellow, CNN magenta, Gemma grey dashed) | **done, works** (Kaggle) |
| `src/eval/replay_bars.py` | Replay error bars: 7 days (19-25 Oct) x 3 stations x 3 seeds (42-44), AID (all agents + v4 head + v4 CNN) and EuroSAT (raw, v4 head, v4 CNN), 0.25-2 Mbps; mean ± std table, CSV, `--plot` -> `docs/replay_bars.png` | **done, works** (40 s) |
| `src/eval/overfit_check.py` | Overfitting checks for the small head: train/tuning/final/FRESH accuracy with bootstrap ranges, near-duplicate (cosine > 0.95) check, replay with fresh photos. FRESH = AID 131-160/class, EuroSAT next 100/class | **done, works** |
| `paper/outline.md` | Paper outline for IGARSS 2027 (4 pages, deadline ~10 Jan 2027): titles, sections with page budget, Tables 1-2 + Figures (results.png, label_curve.png) with numbers and source files, TODO list (Jetson first), timeline | draft |
| `src/cloud/kaggle_run.py` + `kaggle/` | Runs `head.py` on a free Kaggle T4 from the terminal: writes `kaggle/orbit-agent-head.ipynb` (clones the repo from GitHub, so push first), pushes, waits, copies CSVs to `experiments/` and embeddings to `data/processed/` | **done, works** (full run 20 Oct, 72 min on a T4) |
| `notes/research_review_2026-10-20.md` + `notes/pilots/` | Senior-researcher review: weak spots, 3 new directions, the pick (audited downlink), plan; two pilots (compress-to-fit, audit) | done |
| `src/data/s2_real.py` | **Real replay:** for every 30 s daylight slot of Sentinel-2A's real path, finds the image Sentinel-2A really took within 5 min (Element 84 Earth Search STAC, AWS COGs), cuts a 224 px chip (10 m), label = image's cloud map (SCL) if mostly cloud/snow else main ESA WorldCover 2021 class (8 classes). Resumable (`slots.csv`, `chips.csv`), 6-19 Oct 2026. Sim times match real images within 0-7 s; ~1 in 6 slots has an image (land + coast only) | **done, works** |
| `src/agent/audit.py` | **Audited downlink** policy: label + chances for every photo, then full photos picked at random with known chances (∝ expected error, floor x average chance), fixed count per pass (Pareto sampling). Also raw in order / raw random / labels only / triage (v4 rule) / stratified, all on arrays | **done, works** |
| `src/eval/estimate.py` | Ground side: count, raw, difference (Horvitz-Thompson), tuned (PPI++-style, cross-fitted), stratified (Olofsson); per-pass combine; error bars normal / t / safe / scaled / **scaled+1** (default). `python -m src.eval.estimate` = self-test on fakes | **done, works** |
| `src/model/faults.py` | Failure suite: new camera (mapped chances), silent swap (as sure as right answers), radiation bit flips in the 30 KB head (subtle: 5-30 pt drop), damaged camera (blur/noise/stripes, re-embedded); chances computed in 64-bit | **done, works** |
| `src/eval/audit_test.py` | Main audit experiment: `--data eurosat` (stand-in replay, India-like shares ASSUMED) or `--data real`; scenarios x policies x rates x 3 stations x 7 days x 3 seeds -> `experiments/audit_summary_*.csv` (+ every estimate in `data/processed/audit_runs_*.csv.gz`) | **done, works** |
| `src/eval/audit_floor.py`, `src/eval/compress_test.py`, `src/eval/audit_loop.py` | Floor sweep; R sweep with compress-to-fit (real JPEG ladder) + thumbnail audits; closed loop (new camera, zero labels, head retrained after every pass, used 1 pass later) | **done, works** |
| `src/eval/audit_tables.py`, `src/eval/plot_audit.py` | Markdown tables from the newest CSVs; charts `docs/audit_{honesty,floor,rsweep,loop,loop_real}.png` (palette validated: blue/yellow/magenta + grey) | **done, works** |
| `src/demo/export_audit.py` + `demo/audit.html` | Second demo page "the satellite reports its own error": per day, class shares truth vs audit (± bar) vs triage vs labels; onboard model switch; checked real photos; `?model=1&day=3`; tested at 1440 px and 390 px | **done, works** |
| `paper/outline_v2_audit.md` | New paper outline (audited downlink); old outline kept, its agent v4 = baseline | draft |

## How to run
```bash
cd ~/Downloads/SubhamPortfolio/orbit-agent
source .venv/bin/activate

python -m src.orbit.contact_windows                      # passes over Delhi, next 24 h, + map
python -m src.orbit.contact_windows --station svalbard   # other options: --hours, --rate, --satellite
python -m src.data.eurosat                               # download photos (once)
python -m src.eval.baseline_eurosat --per-class 3        # quick model test (30 photos)
python -m src.eval.baseline_eurosat                      # full baseline (100 photos)
python -m src.eval.baseline_eurosat --prompt v2          # prompt v0 (numbers), v1 (names, shuffled), v2 (names + descriptions)
python -m src.eval.baseline_eurosat --size 896           # image size (default 448)
python -m src.data.aid                                   # download 300 AID photos (once)
python -m src.eval.baseline_eurosat --dataset aid        # AID baseline (300 photos, ~40 min)
python -m src.orbit.replay --start 2026-10-19T00:00      # photo replay: queue + budget per pass (--rate, --every, --hours)
python -m src.data.aid --skip 10                         # download the 300 NEW test photos (once, done)
python -m src.eval.confidence_test                       # "not sure" signals on 300 new photos (~2 h, resumable)
python -m src.eval.run_agent                             # raw vs caption-all vs agent v1 vs v2 at 0.25-4 Mbps (seconds)
python -m src.eval.confidence_test --skip 20             # FINAL test photos 21-30 (~2 h)
python -m src.eval.run_agent --sizes jpeg               # use real JPEG sizes (default: equal sizes)
python -m src.demo.export && open demo/index.html       # demo page (dark theme, colour = meaning)
python -m src.eval.plot_results                          # redraw docs/results.png for the README
python -m src.demo.build_site && python -m http.server -d site   # website as GitHub Pages shows it
python -m src.model.vlm path/to/photo.jpg                # describe one photo
python -m src.model.head --backbone siglip-base --train-per-class 20   # agent v4 head, quick Mac test (~15 min first time)
python -m src.cloud.kaggle_run                           # agent v4 full run on a Kaggle GPU (push code first)
python check_setup.py                                    # health check

# Audited downlink (20 Oct)
python -m src.data.s2_real                               # real replay: what Sentinel-2A really photographed, 6-19 Oct (resumable)
python -m src.eval.estimate                              # self-test of the ground estimators (fakes, 10 s)
python -m src.eval.audit_test                            # audit vs policies, healthy/broken model, EuroSAT replay (~5 min)
python -m src.eval.audit_test --data real                # same on the real replay (head trained 6-12 Oct, tested 13-19)
python -m src.eval.audit_floor                           # floor sweep (~5 min)
python -m src.eval.compress_test                         # R sweep vs compress-to-fit (~3 min)
python -m src.eval.audit_loop --data real                # closed loop: new camera, zero labels
python -m src.eval.audit_tables                          # markdown tables from the newest CSVs
python -m src.eval.plot_audit                            # docs/audit_*.png
python -m src.demo.export_audit && open demo/audit.html  # second demo page (?model=1&day=3)
```

## Results so far
- **Orbit simulator (real run on my Mac, 18 Oct 2026):** Sentinel-2A passes over Delhi 2 times a day,
  about 10.4 min each (around 10:56 AM and 10:36 PM IST). That's about 7.7 GB per pass at 100 Mbps.
  The satellite can talk to Delhi only **1.4%** of the time. Svalbard (polar) gets about 10%.
- **EuroSAT baseline v0 (18 Oct 2026, gemma3:4b, 100 photos):** 26% accuracy (3 big groups: 66%),
  6.1 s per photo. Strong answer bias: "Forest" 40x, "SeaLake" 20x. 4 blue-water photos got
  "none of the options" → `unknown`. Full details in README. Mac has **8 GB RAM**, so gemma3:12b won't fit.
- **Prompt test (100 photos):** v0 26%, **v1 29% (best, keep)**, v2 22%. Prompts barely matter
  (within chance at n=100). "Forest" bias stays. v2's descriptions made water photos 0/10.
- **Image size test (v1, 100 photos):** 224 px 32%, 448 px 29%, 896 px 30%. Same answer on 87-88%
  of photos, so no real effect. Keep 448. Every tweak so far: 22-32%. Limit = 64x64 photos and/or 4B model.
- **AID quick test (30 photos, 1/class, v1):** **73%**, 8.1 s per photo. Sharp photos fix most of it;
  the 4B model is fine. Mistakes are close classes (medium vs sparse residential, etc.).
  `baseline_eurosat.py` now takes `--dataset eurosat|aid` (name kept so old commands still work).
- **AID full baseline (300 photos, v1):** **61%** (182/300), 7.5 s per photo. NAVI: 88.16% (check their
  model size/prompt/split in paper 04 before comparing). Quick-test 30 photos gave identical answers
  (repeatable). Worst: Playground→Stadium 9/10, SparseResidential→Forest 9/10. In 5/300 replies the
  model listed several answers = uncertainty signal, useful for the agent's abstain rule.
- **Photo replay (24 h from 2026-10-19 00:00 UTC, Delhi, 1 Mbps, every 30 s, daylight):** 3 passes,
  1,088 photos (308 MB) waiting; raw sending fits only 58%. Default rate is 1 Mbps because AID
  photos are ~0.3 MB JPEGs (at 100 Mbps everything fits). Photo content is a random AID stand-in.
- **Decision (Subham):** photos that don't fit in a pass are **dropped** (fresh data first), not carried over.
- **Agent v1 (same replay, 1,088 photos):** correct info, agent vs raw: 0.25 Mbps 67% vs 15%,
  1 Mbps 82% vs 60%, 2 Mbps 93% vs 89%, 4 Mbps both 100%. Caption-all: 61% but 429 wrong labels.
  "Correct per MB" alone is misleading (caption-all ~13,700/MB): always report correct + wrong + MB.
  Weak spot: wrong labels (193 at 1 Mbps). Only 5/300 answers "unsure", so abstain rule barely fires.
  Scoring assumes a full photo = correct (ground can see it). INTERESTING = human-activity AID classes.
- **Next: "not sure" step (plan in README).** Ollama 0.40.6 returns logprobs (`"logprobs": true,
  "top_logprobs": N` in /api/chat) → token-probability confidence is free. Test on 300 NEW AID photos
  (11–20 per class); first 300 only for learning rules. **Decision (Subham): an "unsure"-marked label
  scores as no info** (neither correct nor wrong).
- **"Not sure" step built (19 Oct 2026).** Logprobs are saturated (model ~100% sure even when wrong),
  so margin C uses the gap to the best DIFFERENT token. Quick test (30 new photos, 10 wrong, too few):
  AUROC A 0.47, B 0.74, C 0.46, D 0.69. v2 rule flags 53%, catches 7/10 wrong. Replay at 1 Mbps:
  v2 88 wrong labels vs v1 204, but correct info 75% vs 81% (trust vs coverage trade-off).
  Majority vote of 3 asks didn't help accuracy. **Don't retune the rule on test photos.**
- **"Not sure" full test (300 new photos, 19 Oct 2026):** accuracy 65% (194/300). AUROC: A 0.50,
  **B 0.79 (best, free)**, C 0.58, D 0.70, combined 0.69. D: all 3 agree -> 77% right, else ~32%.
  v2 rule (A or D or margin<8) flags 45%, catches 62% of wrong; it left out B and included weak C.
  Replay (new photos): 1 Mbps v1 85% correct / 158 wrong, v2 82% / 106 wrong (+90 unsure);
  2 Mbps v2 best on both (96% / 44 wrong); 0.25 Mbps v2 too careful (54% vs v1 72%, but 140 vs 310 wrong).
  Raw at 1 Mbps 59%. Test photos 11-20 are now "used": a v3 rule must be judged on photos 21-30.
- **Photo sets:** 1-10/class = first baseline (B learned here), 11-20 = tuning (v3 weights learned here),
  21-30 = FINAL test (`..._n10_skip20.csv`), never used for building.
- **Agent v3 (19 Oct):** p(right) = sigmoid(w0 + w1*logit(B) [+ w2*all-agree]), fit on tuning.
  Plain label if p>=0.5 else "unsure"; full photos by value-per-byte (1 - max(2p-1,0)). v3-lite = B only.
  **Found: with real JPEG sizes v3 partly wins by picking small (simple) photos** → `--sizes equal`
  is now the default (raw tiles are equal size). Tuning, equal sizes: 1 Mbps v1 86%/150 wrong,
  v3 93%/73 wrong; 0.25 Mbps v1 72%/304, v3 73%/171.
- **FINAL TEST (photos 21-30, 19 Oct): the paper's main numbers.** Accuracy 63%. AUROC B 0.77, combined 0.71,
  D 0.68, C 0.63, A 0.50. Agents, correct/wrong (equal sizes): 1 Mbps raw 60%/0, caption 63%/404,
  v1 84%/173, v2 89%/85, v3-lite 91%/99, **v3 92%/90**; 0.5 Mbps v1 74%/282, v3 79%/164;
  2 Mbps v1 95%/51, v3 99%/6; 0.25 Mbps v1 70%/331 vs v3 68%/192 (trade-off). v3-lite ≈ v3 with
  1 call instead of 3. Chances calibrated on final photos (~75% → 77% right). Demo exported from final photos.
- **Agent v4, decision head (20 Oct):** idea from TypeSafe's Jev (fixed answers, one pass, a chance per answer).
  Frozen SigLIP-base + logistic regression, 600 training photos (31+): **FINAL 95.7%** (Gemma 63%),
  calibration error 0.036, 26 ms/photo on the Mac. Replay: v4 100% correct / 0 wrong at 0.5-1 Mbps
  (v3 92% / 90 at 1 Mbps). Caveat for the paper: v4 is supervised on AID, Gemma is zero-shot.
  Details in `docs/experiment_log.md`.
- **v4 full run on Kaggle (20 Oct):** SigLIP so400m, 9,100 training photos, T4, 72 min (full precision;
  now fp16 on GPU). **FINAL 98.3%** (5 wrong), calibration error 0.021. But its 5 mistakes are all >96% sure,
  so on the replay it lets through 3-7 wrong labels vs 0-3 for the small head. More accurate != more
  trustworthy. Small head = better onboard deal. HF rate-limits (429) bulk photo downloads, so Kaggle
  uses the dataset `jiayuanchengala/aid-scene-classification-datasets` (same bytes). Next: non-AID photos.
- **v4 on real Sentinel-2 / EuroSAT (20 Oct):** AID head as is 33% (Gemma 29%; says "BareLand" for most blurry
  64 px photos) -> doesn't transfer. Same frozen SigLIP-base + new head on 1,000 EuroSAT photos: **92.8%**,
  calibration error 0.014 (Gemma's 100 photos: 93% vs 29%). Replay: v4 100%/0 wrong at 1 Mbps vs Gemma v1
  69%/333. Lesson: retrain the small head per camera (~1,000 labelled photos), not the big model.
- **Big model on Sentinel-2 (20 Oct, Kaggle fp16, 48 ms/photo on T4):** test 2 **94.8%** vs small 92.8%, same
  calibration (0.014); test 1 AID head 40.6% (small 33.2%). ~15-20x compute for +2 points -> small head is the
  main system, big = upper bound. Head = ~30 KB (768x10), could be uplinked per mission.
  Kaggle notebook installs `skyfield ollama`; keeps only files written after the clone; runner MERGES embeddings.
  The big AID embeddings (9,700 photos) were overwritten before that fix; remake with kaggle_run if needed.
- **Tiny CNN baseline (20 Oct):** MobileNetV3-Small 1.5M weights. AID 20/class **85.0%** (head 95.7%; replay 0.25 Mbps
  92%/91 wrong vs head 100%/3). Sentinel-2 100/class **94.6%** (head 92.8%, big 94.8%), ~1/300 compute of the head.
  Story = trade-off: head = label-efficient + 30 KB swap per mission; CNN = cheap compute. CNN still improving at
  epoch 20 on EuroSAT. Next: accuracy vs training photos per class (5-100) for both; Jetson timing/power.
- **Label curve (20 Oct):** 5/class: head AID 89.7% / S2 79.3% vs CNN 67.9% / 44.2% (CNN swings up to 19 pts by pick).
  S2 cross at ~50/class (91.4% both); 100/class CNN 95.4% vs head 92.8%. AID: head ahead up to 100 (94.7 vs 93.8).
  Paper message: frozen head for the start of a mission (few labels, 30 KB update), tiny CNN once labels are many.
- **Replay error bars (20 Oct):** Delhi 1 Mbps, 21 runs: raw 58.0±5.3%, v3 91.1±3.0% / 83±14 wrong, CNN 98.6±0.7% / 14±6,
  head 100.0±0.0% / 0. Policy order never changes. Svalbard (12 passes/day) raw 83% -> agent matters most at Delhi/Bengaluru.
- **Overfitting check (20 Oct):** FRESH never-used photos: AID 94.4% (final 94.3%), EuroSAT 92.5% (92.7%) -> no overfitting
  to final. AID head number moves 94.3-95.7% (C=1000 vs 3000 nearly tied on tuning) -> report ~95% ±2.5.
  EuroSAT: 723/1000 final photos have a near-twin in training; without them 88.4% -> possible spatial leakage;
  fix = spatial split (EuroSAT GeoTIFF has coords). NAVI 88.16% = AID cut to 18 classes; our Gemma re-scored = 76.5%.
  Closest work: Telepix embedding-uplink kNN triage (arXiv 2604.03301, ICLR 2026 ML4RS workshop; no budget, no calibration).
- **README, chart and demo show agent v4 (small head) since 20 Oct.** Screenshots made with headless Chrome:
  `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --window-size=1440,900
  --virtual-time-budget=5000 --screenshot=out.png "file://$PWD/demo/index.html?t=19900"` (t=86400 = end of day).
- **Research review + pivot (20 Oct evening, `notes/research_review_2026-10-20.md`):** the old score can't fail
  (a 50 B label = a 280 KB photo = "1 correct") and compress-to-fit (no AI) ties agent v4 in the old replay
  range. New direction = **audited downlink**: product = the day's class shares with a 90% error bar; policy =
  labels for all + random checks with known chances (floor 0.5) -> honest whatever the onboard model is like.
- **Real replay (20 Oct, `src/data/s2_real.py`):** 3,527 photos Sentinel-2A really took, 6-19 Oct (cloud 46%,
  water 17%, built-up 0.1%). Head trained 6-12 Oct = 82.3% right on 13-19 Oct. `audit_summary_real_20261020_2001.csv`,
  0.05 Mbps: healthy labels-only 1.9 pts / bars held 80% (rare 60%) -> even a healthy model misleads on real photos;
  EuroSAT head (no cloud class): labels 19.5 / 9%, triage 16.6 / 20%, audit 8.3 / 97%. Audit holds 85-99% everywhere
  (weakest: bit flips at 0.25 Mbps 85%; floor 1 = 98%). Price: with a badly broken model the audit is noisier than
  raw random (8.3 vs 5.9). Real closed loop (0.1 Mbps): 46% -> 83% right in 4 days from 215 checked photos,
  bars held 96%; launch head + triage 12%. Demo `demo/audit.html` uses this data at 0.1 Mbps.
- **Audit results, EuroSAT replay (20 Oct, `audit_summary_eurosat_20261020_2001.csv`, 0.25 Mbps, 63 days):**
  healthy head: triage 0.26 pts error (best, but no honest bar), audit 0.47 / bar held 98%. Broken models
  (new camera 28% right, silent swap, bit flips 84%, damaged camera 43-62%): labels-only / triage off by
  1.4-11.6 pts, bars held 0-80%; audit 1.2-2.7 pts, held 86-97%. Default bar = "scaled+1" (rare classes);
  still open: new camera, rare classes, 0.25-0.5 Mbps ~80-87% -> try betting intervals.
- **Floor sweep:** floor 0.5 = free insurance (healthy 0.47 pts like lower floors; bit flips held 95% vs 86-88%).
- **R sweep (compress_test, same bytes + same human checks):** R <= 14 compress-to-fit is exact (onboard AI adds
  nothing); R >= 140 thumbnail audits (160 px, 6.4 KB) give ±0.4-1.1 vs ±1.8-4.5 for compression; R* ~ 30.
- **Closed loop (EuroSAT, 0.25 Mbps):** head from zero labels reaches 91% by day 2, bars held 100% (EuroSAT photos
  repeat, so inflated); launch head + triage held 5%, 24 pts worst day. Loop CSVs now named `audit_loop_<data>_<rate>mbps_*`.
- **Kaggle:** username `subhammahapatra004`. Login works with the new token in `~/.kaggle/access_token` (20 Oct).
- **Work is on git branch `main`** (agent-v2 was merged via PRs #1 and #2). GitHub Pages deploys from main.

## Key decisions
- Satellites: Sentinel-2A (NORAD 40697), Sentinel-2B (42063). Default ground station: Delhi.
- Orbit data order: fresh saved copy, then CelesTrak, then TLE API mirror, then old saved copy,
  then built-in backup. TLE lines are checksum-checked.
- Downlink speed default: 100 Mbps (assumption for a small satellite; change with `--rate`).
- Times are shown in IST.
- Photos: EuroSAT RGB (Zenodo record 7711810). 64x64 pixels, so they're upscaled to 448x448 for the model.
- Classification prompt: numbered list of 10 classes, "reply with only the number", temperature 0.
  The parser looks for a number first, then for the label words.
- Compare our accuracy with NAVI's 88.16% on AID (AID photos are sharper than EuroSAT's).
- Big files stay out of git: `data/raw/*` and the PDFs are in `.gitignore`.

## Next steps (updated 20 Oct evening; steps 1-5 of the first plan are done)
1. Paper = the audited downlink (`paper/outline_v2_audit.md`, IGARSS 2027). Fill Table 2 from the real replay.
2. Error bars for rare classes under a badly wrong model (betting intervals, Waudby-Smith & Ramdas 2024).
3. Jetson Orin Nano: time + energy for encoder + head + sampler.
4. Demo video from `demo/audit.html` (switch to a broken model: triage's numbers go wrong, the audit's bars hold).
5. Optional: spatially separate EuroSAT split; more real days; a second station network.

## Papers (in `papers/`, reading order in `papers/README.md`)
01 Orbital Edge Computing (Denby & Lucia 2020) · 02 Kodan (2023, **missing**: my network blocks NSF,
download by hand from https://par.nsf.gov/servlets/purl/10424445) · 03 Φ-Sat-1 (**paywalled**, IEEE TGRS 2022)
· 04 NAVI-Orbital ⭐ · 05 FOOL · 06 On-Orbit Space AI survey ⭐ · 07 ML in embedded space systems review
· 08 Gemma 3 report · 09 AID dataset · 10 EuroSAT dataset.
Read first: 01, 04, 06.
