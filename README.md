# orbit-agent

**An AI agent that runs on a satellite and decides what is worth sending down to Earth.**

## The problem
Satellites take far more photos than they can send to Earth.
They only get a few minutes of connection per pass.
So most data is wasted or arrives late.

## The idea
Put a small AI model on the satellite. For each photo, an agent decides:

- **Skip** it (clouds, nothing useful)
- **Describe** it in a short text
- **Send** a crop or the full image
- **Not sure?** Send the full image instead of guessing

## Research question
> With a fixed amount of data we're allowed to send down, how much useful
> information can an onboard agent deliver, compared with sending everything raw?

## How we test it (a satellite on your desk)
- Your laptop (later a Jetson Orin Nano) acts as the satellite
- Free Sentinel-2 photos are replayed as if the satellite is flying over Earth
- Real orbit data decides when the satellite can "talk" to the ground
- We measure: useful info reached the ground ÷ data sent

## Folders
| Folder | What goes here |
|---|---|
| `papers/` | Starter papers. Start with `papers/README.md` |
| `notes/` | Your notes, one file per paper |
| `data/` | Satellite images (not saved in git) |
| `src/orbit/` | Where the satellite is and when it can talk to Earth |
| `src/model/` | Runs the AI model (Gemma 3 via Ollama) |
| `src/agent/` | The decision loop (LangGraph) |
| `src/eval/` | Scores the results |
| `experiments/` | Results of each run |

## Get started
```bash
./setup.sh          # makes the Python environment, installs packages, downloads papers
python check_setup.py   # checks everything works
```

## Plan
| Week | Goal |
|---|---|
| 1–2 | Read papers 01, 02, 04. Orbit simulator: contact windows from real satellite data |
| 3–4 | Run Gemma 3 on satellite photos. Baseline accuracy on EuroSAT / AID |
| 5–7 | Build the agent: skip / describe / send / not sure |
| 8–9 | Experiments vs baselines (send everything, cloud filter only, caption everything) |
| 10 | Write the paper + demo video |

Demo target: end of week 4.

## Results so far

### Orbit simulator (18 Oct 2026)
- Sentinel-2A passes over Delhi **2 times a day**, about **10.4 min** each.
- About **7.7 GB per pass** at 100 Mbps.
- The satellite can talk to Delhi only **1.4%** of the time (Svalbard: about 10%).

### EuroSAT baseline v0 (18 Oct 2026)
Setup: `gemma3:4b` (Ollama, MacBook 8 GB), no training, 100 photos (10 per class, seed 42),
photos upscaled 64→448 px, numbered-list prompt, temperature 0.
File: `experiments/baseline_eurosat_20261018_1505.csv`

| Measure | Result |
|---|---|
| Accuracy (10 classes) | **26%** (random = 10%) |
| Accuracy (3 groups: green / water / built-up) | **66%** |
| Speed | **6.1 s per photo** (first photo is slower, about 12 s, model loading) |
| Data size | photo 3.4 KB vs label 8 bytes (about 450x smaller) |

Per class (out of 10):

| Class | Correct | | Class | Correct |
|---|---|---|---|---|
| Industrial | 7 | | Residential | 2 |
| Forest | 4 | | River | 2 |
| Highway | 4 | | AnnualCrop | 1 |
| SeaLake | 4 | | Pasture | 0 |
| HerbaceousVegetation | 2 | | PermanentCrop | 0 |

Most common mix-ups: AnnualCrop → Forest (8), Forest → SeaLake (6), Pasture → Forest (6),
River → SeaLake (6), HerbaceousVegetation → Forest (5), PermanentCrop → Forest (5).

**What we learned**
- **The model is biased toward two answers.** It said "Forest" (option 2) 40 times and
  "SeaLake" (option 10) 20 times. Many mistakes come from this bias, not from what's in the photo.
- **4 photos got no answer.** On plain blue water photos the model said "none of the options fit"
  and called them space or atmosphere. Our parser marks these as `unknown`.
- **Fine green classes are hard** at 64x64 (crops, pasture, orchards, grassland). Big groups are
  much easier (66%). For the agent, "is this worth sending?" may matter more than the exact class.
- **Far below NAVI's 88.16%**, but that was on AID, which has much sharper photos. Not a fair comparison yet.
- **Speed is fine for the mission.** At about 6 s per photo, the laptop can check about 600 photos
  per hour between ground passes.
- An earlier quick run (30 photos) gave 20%: `experiments/baseline_eurosat_20261018_1449.csv`.

### Prompt test (18 Oct 2026)
Same 100 photos, `gemma3:4b`, 448 px. Only the prompt changes.

| Prompt | What it does | Accuracy | 3 groups | Speed | Most common answer |
|---|---|---|---|---|---|
| v0 | numbered list, "reply with the number" | 26% | 66% | 6.1 s | Forest (40x) |
| **v1** | class names, shuffled per photo | **29%** | **70%** | 6.1 s | Forest (45x) |
| v2 | v1 + what each class looks like | 22% | 62% | 7.0 s | AnnualCrop (39x) |

Files: `experiments/baseline_eurosat_gemma3-4b_v1_448px_20261018_1528.csv`,
`experiments/baseline_eurosat_gemma3-4b_v2_448px_20261018_1540.csv`

**What we learned**
- **Prompts barely matter.** With 100 photos, a gap of a few points is within chance
  (roughly ±9 points). v1 is the best and simplest, so we keep it as the default.
- **v1 fixed the number-related problems.** "SeaLake" answers dropped from 20 to 7, and
  there were no more "none of the options" replies.
- **The "Forest" bias is real.** It isn't caused by the numbers. The model calls most blurry
  green photos "forest".
- **Descriptions made it worse.** With v2, the model called 0 of 10 water photos "sea or lake"
  (it said farmland, grassland, etc.). The long text seems to pull attention away from the image.
- **So the limit is the image and the model, not the wording.** The next tests should change those.

### Image size test (18 Oct 2026)
Same 100 photos, `gemma3:4b`, prompt v1. Only the upscale size changes.

| Size | Accuracy | 3 groups | Speed |
|---|---|---|---|
| 224 px | 32% | 71% | 6.1 s |
| 448 px | 29% | 70% | 6.1 s |
| 896 px | 30% | 70% | 6.8 s |

**What we learned**
- **Size doesn't matter.** The three runs gave the same answer on 87–88 of 100 photos.
  The small gaps are chance. Gemma 3 resizes every photo to 896 px inside the model anyway.
- We keep 448 px as the default.
- **Every tweak so far lands between 22% and 32%.** The real limit is the tiny 64x64 photos
  and/or the 4B model. Fine green classes (grassland, pasture, orchards) are almost never right.

### AID quick test (18 Oct 2026), first look only
AID = the dataset NAVI used: 600x600 px Google Earth photos, 30 classes (random guessing = 3%).
`gemma3:4b`, prompt v1, **30 photos (1 per class)**.
File: `experiments/baseline_aid_gemma3-4b_v1_448px_20261018_1855.csv`

| Measure | Result |
|---|---|
| Accuracy (30 classes) | **73%** (22 of 30) |
| Speed | 8.1 s per photo |

- **Sharp photos change everything:** 73% on AID vs about 30% on EuroSAT, with the same model
  and prompt. So the small model is fine. The 64x64 photos were the main problem.
- Mistakes are between close classes: medium vs sparse residential, commercial vs city center,
  playground vs stadium, square vs park.
- 30 photos is too few to compare with NAVI's 88.16%. The full run (300 photos) comes next.

## Next step: better baseline (about 2–3 days)
Change one thing at a time, on the same 100 photos, so we know what helped.

1. ~~**Remove the answer bias:** class names, shuffled.~~ Done: v1 = 29%.
2. ~~**Better class descriptions.**~~ Done: v2 = 22%, worse.
3. ~~**Image size.**~~ Done: 224 / 448 / 896 px all about 30%, no real difference.
4. **Other model:** `gemma3n:e4b` (fits in 8 GB). `gemma3:12b` is too big for this Mac.
5. **AID dataset:** code done (`src/data/aid.py`), quick test 73%. Next: full run, 300 photos:
   `python -m src.eval.baseline_eurosat --dataset aid` (about 40 min).
6. **Better report:** a mix-up chart (confusion matrix) and group accuracy saved with each run.

## Status
- [x] Project set up
- [ ] Papers 01, 02, 04 read
- [x] Orbit simulator (`python -m src.orbit.contact_windows`)
- [x] Model running locally (`python -m src.eval.baseline_eurosat`): baseline v0 = 26%
- [ ] Better baseline (prompt, image size, other models, AID dataset)
- [ ] Agent v1
- [ ] Experiments
- [ ] Paper draft
