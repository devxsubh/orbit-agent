# orbit-agent

**An AI agent that runs on a satellite and decides what is worth sending down to Earth.**

Gemma 3 (4B) vision-language model · real orbit data · Python · laptop now, Jetson Orin later

![Demo: the satellite passes over Delhi and sends down what the agent chose](docs/demo_pass.png)

## Main result

> With the same downlink budget (1 Mbps), the onboard agent gets **92%** of the information
> to the ground correctly. Sending raw photos gets **60%**.
> It also sends **about half the wrong labels** of a simple rule-based agent (90 vs 173).

![Correct info and wrong labels at each downlink speed](docs/results.png)

Tested on **photos never used while building the agent** (see [Fair testing](#fair-testing)).

## The problem
Satellites take far more photos than they can send to Earth. They can only talk to a ground
station for a few minutes per pass. Sentinel-2A can talk to Delhi only **1.4% of the time**
(2 passes a day, about 10 minutes each). So most photos are dropped or arrive late.

## The idea
Put a small AI model on the satellite. For each photo it gives a short label, like "airport" or
"farmland". Labels are tiny (about 50 bytes vs 280 KB for a photo), so all of them fit.
The agent then decides what else to send:

- **Sure about the label?** The label is enough.
- **Not sure?** Mark the label "unsure", so nobody on the ground trusts a guess.
- **Budget left?** Send full photos where they add the most, starting with the ones the model is least sure about.

## How it works

```
 Real orbit data ──> Photo queue ──> Gemma 3 (4B) ──> Chance it's right ──> Agent ──> Ground pass
   (TLE, Delhi)      (every 30 s)     labels it        (free signal +       decides     (budget =
                                                        3-ask agreement)    what goes    pass time
                                                                            down         x speed)
```

1. **Orbit** (`src/orbit/`): real orbit data for Sentinel-2A gives the exact times the satellite can talk to Delhi.
2. **Photo replay** (`src/orbit/replay.py`): a photo every 30 s in daylight gives 1,088 photos (304 MB) per day.
   At 1 Mbps, raw sending fits only 60% of them.
3. **Model** (`src/model/vlm.py`): Gemma 3 4B (Ollama, 4-bit, on a MacBook) labels each photo
   with one of 30 AID classes. About 7.5 s per photo, no training.
4. **"Not sure" check** (`src/model/confidence.py`): the agent estimates the chance each label is right.
   It uses a free signal (classes the model often mixes up) plus whether 3 shuffled asks agree.
5. **Agent v3** (`src/agent/decide.py`): plain label if the chance is at least 50%, else an "unsure" label.
   Leftover budget goes to full photos, best value per byte first.
6. **Score** (`src/eval/`): correct info, wrong labels and MB sent, always reported together.

## Final results

Same day of orbit (1,088 photos), same budget for every policy. Each cell: **correct info / wrong labels**.

| Downlink | Send all raw | Caption every photo | Agent v1 (rules) | Agent v2 | Agent v3-lite (1 call) | **Agent v3** (3 calls) |
|---|---|---|---|---|---|---|
| 0.25 Mbps | 15% / 0 | 63% / 404 | **70%** / 331 | 58% / 132 | 65% / 204 | 68% / 192 |
| 0.5 Mbps | 30% / 0 | 63% / 404 | 74% / 282 | 72% / 129 | 78% / 167 | **79%** / 164 |
| **1 Mbps** | **60% / 0** | 63% / 404 | 84% / 173 | 89% / 85 | 91% / 99 | **92% / 90** |
| 2 Mbps | 89% / 0 | 63% / 404 | 95% / 51 | 97% / 29 | 99% / 7 | **99% / 6** |
| 4 Mbps | 100% / 0 | 63% / 404 | 100% / 0 | 100% / 0 | 100% / 0 | 100% / 0 |

File: `experiments/agents_delhi_final_equal_20261019_1229.csv`

**How well can the agent tell when the model is wrong?** (300 final test photos; the model got 63% right)

| Signal | Score (AUROC: 0.5 = guessing, 1 = perfect) | Wrong answers caught when flagging 10% / 20% / 30% | Extra cost |
|---|---|---|---|
| **B. Classes the model often confuses** | **0.77** | 20% / 34% / 50% | **none** |
| D. Agreement over 3 shuffled asks | 0.68 | 20% / 34% / 48% | 2 more model calls |
| C. Answer margin (token probability) | 0.63 | 15% / 31% / 42% | none |
| A. Model lists several answers | 0.50 | 8% / 18% / 26% | none |

File: `experiments/confidence_aid_test_gemma3-4b_r3_n10_skip20.csv` (+ `.png` chart)

## What we learned
- **The agent helps most when the downlink is tight.** At 1 Mbps: 92% correct info vs 60% for raw.
  When everything fits (4 Mbps), all policies are equal.
- **Knowing when to say "not sure" is what makes v3 better.** It sends about half the wrong labels of
  v1 and still gets more correct info through.
- **The best "not sure" signal is free.** "Is this a class the model often gets wrong?" needs no extra
  model call. v3-lite (free signal only) is almost as good as v3 (91% vs 92% at 1 Mbps).
  That matters on a satellite with little power.
- **The model is overconfident.** Its token probability is about 100% even when it's wrong, so that
  signal is weak. Asking 3 times and checking agreement works better.
- **The chances are honest.** When v3 says about 75%, 77% of those labels are right; at about 94%, 91% are right.
- **At very tight budgets there is a trade-off.** At 0.25 Mbps, v1 gets a bit more correct info
  (70% vs 68%), but v3 sends 42% fewer wrong labels. Which is better depends on how costly a wrong label is.
- **"Correct per MB" alone is misleading.** Captioning every photo scores very high per MB because it
  sends almost nothing, but 404 of its labels are wrong. So we always report correct, wrong and MB together.

## Fair testing
The AID photos are split into three sets that never mix:

| Photos (per class) | Used for |
|---|---|
| 1–10 | First baseline. Signal B learns which classes the model confuses here |
| 11–20 | Tuning: agent v3 learns its weights here |
| **21–30** | **Final test.** Only used once, for the results above |

All photos get the same size (`--sizes equal`), because real satellite tiles are all the same size.
With real JPEG sizes, the agent could look better just by picking photos that compress well.

## Honest limits
- **Photos are stand-ins.** Time, position and budget are real, but each photo is a random AID aerial
  photo, not the real Sentinel-2 view at that spot.
- **Model accuracy is 61–65%** on AID's 30 classes, vs 88.16% for NASA JPL's NAVI-Orbital
  (Gemma 3 in orbit). Their model size, prompt and test split still need checking before comparing.
- **Laptop, not flight hardware.** Speed and power on a Jetson Orin are not measured yet.
- **Scoring assumes a full photo always counts as correct** (people on the ground can look at it).
- **One satellite, one ground station, one day.** Skipping cloudy photos and sending crops are not built yet.

## Demo
A web page that plays back one day of the satellite: the orbit, each photo and the agent's choice,
the queue waiting onboard, each downlink pass, and the running score (raw vs agent v3).
It uses the final test photos.

```bash
python -m src.demo.export      # makes demo/data.js and demo/thumbs/ (photos)
open demo/index.html           # opens in the browser, no server needed
```

Space = play/pause. "Next pass" jumps ahead. Passes slow down automatically.
`index.html?t=60000` opens at a set time (seconds after the start).

![End of the day: agent v3 92% vs raw 60%](docs/demo_end.png)

## Run it yourself
Tested on macOS (Apple Silicon, 8 GB). Needs Python 3.10+, and [Ollama](https://ollama.com) with a Gemma 3 model (`ollama pull gemma3:4b`).

```bash
./setup.sh                              # makes .venv, installs packages, downloads papers
source .venv/bin/activate
python check_setup.py                   # health check

python -m src.orbit.contact_windows     # passes over Delhi in the next 24 h, + map
python -m src.data.aid                  # download AID photos 1-10 per class (85 MB)
python -m src.data.aid --skip 20        # download the final test photos 21-30
python -m src.eval.confidence_test --skip 20   # model + "not sure" signals (~2 h, resumable)
python -m src.eval.run_agent            # all policies at 0.25-4 Mbps (seconds, no model calls)
python -m src.eval.plot_results         # redraws docs/results.png
```

More commands (EuroSAT, prompts, image sizes, replay options) are in [docs/experiment_log.md](docs/experiment_log.md).

## Project layout
| Path | What it is |
|---|---|
| `src/orbit/` | Orbit simulator (passes, budget per pass) and the photo replay |
| `src/data/` | Downloads AID (Hugging Face) and EuroSAT (Zenodo) |
| `src/model/` | Talks to Gemma 3 via Ollama, and the "not sure" signals |
| `src/agent/` | The agents (v1, v2, v3) and the baselines |
| `src/eval/` | Baseline tests, agent runs, scoring, charts |
| `src/demo/` + `demo/` | Demo export and the playback page |
| `experiments/` | Every run's results (CSV + charts) |
| `docs/` | README images and the full experiment log |
| `papers/`, `notes/` | Reading list and paper notes |

## Status
- [x] Orbit simulator with real orbit data
- [x] Model running locally: EuroSAT (about 30%, photos too small) and AID (61–65%)
- [x] Photo replay along the real orbit
- [x] Agent v1 (rules), v2 and v3 ("not sure"-aware)
- [x] Final test on unseen photos: **v3 92% vs raw 60% at 1 Mbps**
- [x] Demo page
- [ ] Check NAVI's setup (paper 04) to compare accuracy fairly
- [ ] Skip cloudy photos, send crops
- [ ] Run on a Jetson Orin Nano (speed and power)
- [ ] Demo video
- [ ] Paper draft

## Built on
- **NAVI-Orbital** (NASA JPL, arXiv 2606.18271): the first vision-language model run fully in orbit
  (Gemma 3 on a Jetson Orin, April 2026). This project asks the next question: *what should it send down?*
- **Kodan** (ASPLOS 2023): onboard processing cut downlink by about 89–97%.
- **On-Orbit Space AI survey** (arXiv 2604.16518): names trust in orbit and "what to send down" as open problems.
- Datasets: **AID** (30 classes, 600x600 px aerial photos) and **EuroSAT** (Sentinel-2, 64x64 px).

Full reading list: [papers/README.md](papers/README.md).
Full history of every experiment: [docs/experiment_log.md](docs/experiment_log.md).
