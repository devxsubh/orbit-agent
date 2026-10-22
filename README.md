# <img src="docs/logo.svg" width="36" alt=""> orbit-agent

**An AI on a satellite decides what to send down to Earth, and the ground can check how much to trust it.**

Frozen SigLIP + a 30 KB head onboard · random checks with known chances · tested on what Sentinel-2A really photographed · Python · laptop now, Jetson Orin later

![Healthy or broken onboard model on real Sentinel-2A photos: how far off each policy's class shares are, and how often its error bar held](docs/figures/audit_honesty_real.png)

## In one minute

- **The problem.** Satellites take far more photos than they can send. Sentinel-2A can talk to Delhi only
  about 1.5% of the day, so an onboard AI has to decide what goes down.
- **The catch.** Today's onboard designs send labels, plus full photos of what the model is *unsure* about.
  When the model is wrong in a *confident* way (a new camera, a radiation bit flip, a damaged lens), nobody on
  the ground can see it.
- **The idea.** Label every photo, then spend the rest of each pass on full photos picked **at random with
  known chances**. The ground corrects the labels with these checks and gets **honest error bars, whatever the
  onboard model is like**. A self-check switches to a safe mode when the checks show the model is broken.
- **The result**, on what Sentinel-2A really photographed (13-19 Oct 2026): with a broken model, labels alone
  are 10-20 points off and their 90% error bars hold only 9-27% of the time. The self-checking audit is 5-6
  points off and its bars hold 97-98%, with the same bytes.

<!-- EXP12-SUMMARY -->

## How to read this project

| Start here | What it is |
|---|---|
| [`notebooks/01_data.ipynb`](notebooks/01_data.ipynb) | **The data first:** what is in the photos, what is messy, can we trust the test? |
| [`notebooks/02_experiments.ipynb`](notebooks/02_experiments.ipynb) | **Every experiment in order** (1-12): question, what we did, result, lesson |
| `report/report.md` | The full report (build the PDF with `python report/build.py`) |
| [Live demo](https://devxsubh.github.io/orbit-agent/) | One day of the satellite, and a second page where you can break the model |

## The experiments, in order

| # | Date | Question | Answer |
|---|---|---|---|
| 0 | 22 Oct | What is in the data? | EuroSAT and AID are clean. 14% of real photos are messy; snow and cloud look identical in colour |
| 1 | 18 Oct | How often can the satellite talk to India? | 2-3 passes a day, about 1.5% of the time |
| 2 | 18 Oct | Can Gemma 3 (what NASA flew) label photos with no training? | 61-65% on AID, 22-32% on 64 px Sentinel-2 photos |
| 3 | 19 Oct | Can the agent tell when it is wrong? | A free "confusable class" signal works best (AUROC 0.77) |
| 4 | 20 Oct | Frozen image model + 30 KB head? | 95.7% vs Gemma's 63%, 300x faster; more accurate is not more trustworthy |
| 5 | 20 Oct | Real Sentinel-2 photos? | Retrain the 30 KB head per camera: 92.8% |
| 6 | 20 Oct | A tiny CNN instead? | Head wins with few labels; CNN catches up at ~50 per class |
| 7 | 20 Oct | Do results hold across days and stations? | Yes, but the old score sat at 100.0 ± 0.0%: a warning sign |
| 8 | 20 Oct | Is onboard AI needed at all? | Not below ~14x data taken / data sent: plain compression ties it. **The question changed** |
| 9 | 20 Oct | Can random checks keep the ground's numbers honest? | Yes, on real photos, whatever the model; plus a self-check |
| 10 | 22 Oct | Fair rivals, guaranteed error bars, hardware? | SigLIP zero-shot 66% > Gemma 63%; betting bars hold everywhere; 19 ms per photo |
| 11 | 22 Oct | Beyond accuracy? | Real photos: 82% accurate but 67% balanced; snow is never found |
| 12 | 22 Oct | Retrain using what the data showed? | <!-- EXP12-ROW --> |

<!-- EXP12-SECTION -->

## Honest limits

- **Labels of the real photos come from maps** (the image's cloud map and ESA WorldCover 2021), not people.
  An eye check of 150 photos found 97% right where the eye could tell (done by an AI; a person should re-check).
- **Laptop, not flight hardware.** All onboard steps take 19 ms per photo on a MacBook; the Jetson Orin script
  (`python -m src.eval.jetson_bench`) is ready but not run on a board yet.
- **One satellite, one product** (class shares). The EuroSAT replay uses assumed India-like class shares; the
  real replay does not.
- **The statistics are known** (survey sampling, prediction-powered inference); what is new is doing it onboard,
  per pass, under byte budgets, and testing it against silent failures.

## Run it yourself

Tested on macOS (Apple Silicon, 8 GB). Needs Python 3.10+; Gemma 3 experiments also need [Ollama](https://ollama.com).

```bash
./setup.sh && source .venv/bin/activate      # one-time setup
python check_setup.py                        # health check

# the data (downloads once)
python -m src.data.eurosat && python -m src.data.aid
python -m src.data.s2_real                   # what Sentinel-2A really photographed, 6-19 Oct (~1-2 h, resumable)
python -m src.data.s2_bands                  # infrared bands for every real photo (~1 h, resumable)
python -m src.eval.eda                       # the data analysis behind notebooks/01_data.ipynb

# the main experiments (each writes into results/<NN>_<name>/)
python -m src.eval.audit_test --data real    # experiment 9: audit vs today's policies, healthy or broken model
python -m src.model.retrain                  # experiment 12: retraining recipes
python -m src.model.retrain --audit          # experiment 12: the audit with the retrained head
```

Every experiment's own command is at the end of its section in `notebooks/02_experiments.ipynb`.

## Repo layout

| Path | What it is |
|---|---|
| `notebooks/` | `01_data.ipynb` (the data) and `02_experiments.ipynb` (experiments 1-12, in order) |
| `results/NN_name/` | Every result file, one folder per experiment (`src/paths.py` knows which goes where) |
| `src/orbit/` | Orbit simulator and photo replay |
| `src/data/` | Downloads: EuroSAT, AID, the real Sentinel-2A replay, its infrared bands |
| `src/model/` | Gemma 3, the decision head, tiny CNN, zero-shot SigLIP, fault suite, retraining |
| `src/agent/` | Agents v1-v4, the audited downlink, the self-check |
| `src/eval/` | One script per experiment, plus charts |
| `src/demo/`, `demo/` | The demo pages |
| `report/` | The report (markdown source + build script) |
| `paper/outline.md` | Paper outline (IGARSS 2027) |
| `docs/figures/` | Every chart |
| `archive/` | Superseded notes, logs, outlines and results (kept for the record) |

## Built on

- **NAVI-Orbital** (NASA JPL, arXiv 2606.18271): the first vision-language model run fully in orbit (Gemma 3 on a
  Jetson Orin, April 2026). This project asks the next question: *what should it send down, and can the ground trust it?*
- **Kodan** (ASPLOS 2023), **On-Orbit Space AI survey** (arXiv 2604.16518), **prediction-powered inference**
  (Angelopoulos et al. 2023), **Olofsson et al. 2014** (area estimation), **betting confidence intervals**
  (Waudby-Smith & Ramdas).
- Data: **EuroSAT** (Sentinel-2, 64 px), **AID** (aerial, 30 classes), **Sentinel-2 L2A** via Element 84 Earth
  Search (contains modified Copernicus Sentinel data 2026), **ESA WorldCover 2021**.
