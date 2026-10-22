# Experiment log

Every experiment in the order it was done (18–19 Oct 2026), with the numbers and what we learned.
The short version is in the [README](../README.md).

Note: early tables use real JPEG sizes; from agent v3 on, every photo has the same size (see "Agent v3").

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

### AID full baseline (18 Oct 2026)
`gemma3:4b`, prompt v1, 448 px, **300 photos (10 per class, seed 42)**.
File: `experiments/baseline_aid_gemma3-4b_v1_448px_20261018_2302.csv`

| Measure | Result |
|---|---|
| Accuracy (30 classes) | **61%** (182 of 300; likely range about 55–67%) |
| Random guessing | 3% |
| NAVI-Orbital (in orbit, same dataset) | 88.16% |
| Speed | 7.5 s per photo |

Per class (out of 10):

| Easy (8–10) | Medium (5–7) | Hard (0–4) |
|---|---|---|
| Airport 10, Farmland 10, Parking 10, Port 10, Stadium 10, Forest 9, Mountain 9, BaseballField 8, Beach 8, Bridge 8, Viaduct 8 | DenseResidential 7, Industrial 7, Pond 7, BareLand 6, Desert 6, River 6, StorageTanks 6, Church 5, Resort 5, School 5 | Park 4, RailwayStation 4, Square 4, Meadow 3, MediumResidential 3, Center 2, Commercial 1, SparseResidential 1, Playground 0 |

Most common mix-ups: Playground → Stadium (9), SparseResidential → Forest (9),
Meadow → Farmland (6), Square → Park (6), BareLand ↔ Desert (4 each way),
MediumResidential → SparseResidential (4), RailwayStation → Industrial (4), StorageTanks → Industrial (4).

**What we learned**
- **61% vs NAVI's 88.16%.** The gap is real, but we don't know the cause yet. Before
  comparing, check in paper 04 which Gemma 3 size, prompt and test split NAVI used.
- **The quick test (73%) was lucky.** Those first 30 photos were easier. On them, the full run
  gave the same 30 answers, so results are repeatable.
- **Most mistakes are between close classes** (playground/stadium, bare land/desert,
  meadow/farmland). For "is this worth sending?", many of these mistakes may not matter.
- **SparseResidential → Forest (9 of 10):** houses hidden under trees. A real risk for the agent:
  it could skip photos that show buildings.
- **Sometimes the model lists several answers** (5 of 300, e.g. "parking lot / viaduct / forest").
  That's a sign it isn't sure. **Useful for the agent's "not sure → send full image" rule.**

### Photo replay (18 Oct 2026)
`python -m src.orbit.replay --start 2026-10-19T00:00`: Sentinel-2A, Delhi, 24 h,
a photo every 30 s in daylight, 1 Mbps downlink, stand-in photos from AID.

| Pass (IST) | Minutes | Photos waiting | Waiting | Can send | Fits |
|---|---|---|---|---|---|
| 19 Oct 10:56 | 10.4 | 329 | 91 MB | 78 MB | 85% |
| 19 Oct 22:07 | 10.1 | 658 | 190 MB | 76 MB | 40% |
| 19 Oct 23:49 | 4.2 | 101 | 26 MB | 31 MB | 100% |

- **Sending raw, only 58% of the data fits.** The agent has to choose the rest.
- **Why 1 Mbps:** AID photos are small JPEGs (about 0.3 MB). At 100 Mbps everything fits
  and there is nothing to decide. The downlink speed sets how tight the budget is,
  so experiments will try several speeds.
- **Limit:** time, position and budget are real, but each photo is a random AID photo,
  not the real view at that place.

### Agent v1 vs baselines (18 Oct 2026)
`python -m src.eval.run_agent`: same replay (1,088 photos, 754 show human activity), model
answers from the AID baseline run, unsent photos dropped.

**Agent v1 rules** (`src/agent/decide.py`): send a short label for every photo (about 50 bytes),
then spend what's left on full photos: model unsure first, then human activity, then the rest.
**Scoring:** full photo = correct; label = correct only if right; wrong label = "wrong info".

| Downlink | Policy | MB sent | Correct info | Wrong labels | Human-activity photos correct |
|---|---|---|---|---|---|
| 0.25 Mbps | send all raw | 46 | 15% | 0 | 16% |
| | caption all | 0.05 | 61% | 429 | 57% |
| | **agent v1** | 46 | **67%** | 360 | **66%** |
| 1 Mbps | send all raw | 179 | 60% | 0 | 59% |
| | caption all | 0.05 | 61% | 429 | 57% |
| | **agent v1** | 179 | **82%** | 193 | **84%** |
| 2 Mbps | send all raw | 269 | 89% | 0 | 88% |
| | **agent v1** | 269 | **93%** | 71 | **96%** |
| 4 Mbps | all policies (everything fits) | 308 | 100% | 0 | 100% |

**What we learned**
- **The agent wins when the budget is tight.** At 1 Mbps: 82% correct vs 60% for raw,
  with the same MB. At 0.25 Mbps: 67% vs 15%.
- **When everything fits (4 Mbps), the agent adds nothing.** The idea only helps when the downlink is the bottleneck.
- **"Correct per MB" alone is misleading.** Caption-all scores about 13,700 per MB because it sends
  almost nothing, yet it sends 429 wrong labels. We must report correct info, wrong info
  *and* MB together.
- **Wrong labels are the agent's weak spot** (193 at 1 Mbps). This is the trust problem from the
  survey. Next: smarter "not sure" detection, so fewer wrong labels go down.
- Only 5 of 300 model answers were "unsure" (several answers), so that rule barely mattered yet.

## Next step: better baseline (about 2–3 days)
Change one thing at a time, on the same 100 photos, so we know what helped.

1. ~~**Remove the answer bias:** class names, shuffled.~~ Done: v1 = 29%.
2. ~~**Better class descriptions.**~~ Done: v2 = 22%, worse.
3. ~~**Image size.**~~ Done: 224 / 448 / 896 px all about 30%, no real difference.
4. **Other model:** `gemma3n:e4b` (fits in 8 GB). `gemma3:12b` is too big for this Mac.
5. ~~**AID dataset.**~~ Done: 61% on 300 photos (NAVI: 88.16%).
6. **Better report:** a mix-up chart (confusion matrix) and group accuracy saved with each run.

## Agent v3: chance-based decisions (19 Oct 2026)
Instead of a yes/no "not sure" rule, v3 estimates **the chance each label is right** (p),
from the free signal B (+ agreement D in full v3). It learns this on the tuning photos (11–20).
- Label: plain if p ≥ 50%, else marked "unsure" (a label more likely wrong than right is worth less than nothing).
- Then full photos go to where they add the most value per byte.
- **v3-lite** = free signal only (1 model call per photo). **v3** = + agreement (3 calls).

**Photo sizes matter.** With real JPEG sizes, v3 partly won by choosing photos that compress well
(simple scenes). Real satellite tiles are all the same size. So experiments now give every photo
the same size by default (`--sizes equal`). Earlier tables above used real JPEG sizes.

Tuning photos 11–20, equal sizes (v3 learned on these, so this is NOT the final result):

| Downlink | Policy | Correct info | Wrong labels |
|---|---|---|---|
| 0.25 Mbps | agent v1 | 72% | 304 |
| | agent v3 | 73% | 171 |
| 1 Mbps | agent v1 | 86% | 150 |
| | agent v3 | 93% | 73 |

### Final test (19 Oct 2026): photos 21–30, never used while building
Files: `experiments/confidence_aid_test_gemma3-4b_r3_n10_skip20.csv` (+ `.png`),
`experiments/agents_delhi_final_equal_20261019_1229.csv`. Model accuracy on these photos: 63%.

**"Not sure" signals hold up on new photos:** B (free) 0.77, combined 0.71, D 0.68, C 0.63, A 0.50.

**Main result**: same replay (1,088 photos, equal sizes), correct info / wrong labels:

| Downlink | Send all raw | Caption all | Agent v1 | Agent v2 | Agent v3-lite (1 call) | **Agent v3** (3 calls) |
|---|---|---|---|---|---|---|
| 0.25 Mbps | 15% / 0 | 63% / 404 | **70%** / 331 | 58% / 132 | 65% / 204 | 68% / 192 |
| 0.5 Mbps | 30% / 0 | 63% / 404 | 74% / 282 | 72% / 129 | 78% / 167 | **79%** / 164 |
| 1 Mbps | 60% / 0 | 63% / 404 | 84% / 173 | 89% / 85 | 91% / 99 | **92%** / 90 |
| 2 Mbps | 89% / 0 | 63% / 404 | 95% / 51 | 97% / 29 | 99% / 7 | **99%** / 6 |
| 4 Mbps | 100% / 0 | 63% / 404 | 100% / 0 | 100% / 0 | 100% / 0 | 100% / 0 |

**What we learned**
- **At 1 Mbps, agent v3 delivers 92% correct info vs 60% for sending raw,** with the same MB, and
  about half the wrong labels of v1 (90 vs 173).
- **v3-lite is almost as good with no extra model calls** (91% vs 92% at 1 Mbps). The free signal
  does most of the work. That matters on a satellite with little power.
- **When the budget is very tight (0.25 Mbps), there's a trade-off:** v1 delivers a bit more correct info
  (70% vs 68%), v3 sends 42% fewer wrong labels (192 vs 331). Which is better depends on how costly a wrong label is.
- **The chances are honest on new photos:** when v3 says about 75%, 77% are right; about 94%, 91% are right.
- When everything fits (4 Mbps), all policies are equal. The agent only helps when the downlink is the bottleneck.

## Demo
A web page that plays back one day of the satellite: orbit, photos, the agent's choices and each
downlink pass, with "send all raw" vs the agent side by side.
```bash
python -m src.demo.export      # makes demo/data.js and demo/thumbs/ (photos)
open demo/index.html           # opens in your browser; no server needed
```
Tips for recording: press Space to play/pause; "Next pass" jumps ahead; passes slow down automatically.
`index.html?t=60000` opens at a set time (seconds after the start).

## Next step: better "not sure" detection (planned 18 Oct 2026)
Goal: fewer wrong labels sent down, without losing correct info.

1. **Fair test set:** 300 *new* AID photos (photos 11–20 per class). The first 300 are only
   used to learn rules, never to report results.
2. **Four "not sure" signals:**
   A. several answers (current) · B. often-confused classes (learned from the first 300) ·
   C. token probability of the answer (same call, free) · D. ask 3 times with shuffled options, check agreement.
3. **Judge them:** share of wrong labels caught when flagging 10 / 20 / 30% of photos, plus a chart.
4. **Agent v2:** least-trusted photos get full photos first. With no budget left, send the label
   marked "unsure", **scored as no info** (neither correct nor wrong).
5. Re-run raw vs caption-all vs agent v1 vs agent v2 at every downlink speed.

**Status (19 Oct 2026): built and tested. Waiting for the full run (300 new photos, about 2 hours):**
```bash
python -m src.eval.confidence_test      # safe to stop and restart: done photos are kept
python -m src.eval.run_agent            # then: all 4 policies, uses the new answers automatically
```

**Quick test (30 new photos, 10 wrong): too few to judge, early signs only**
File: `experiments/confidence_aid_test_gemma3-4b_r3_n1.csv` (+ `.png` chart)

| Signal | Ranking (0.5 = useless, 1 = best) | Wrong caught when flagging 20% |
|---|---|---|
| A. Several answers | 0.47 | 20% |
| B. Confused classes | 0.74 | 30% |
| C. Answer margin | 0.46 | 20% |
| D. Agreement (3 asks) | 0.69 | 30% |
| A + D + C combined | 0.58 | 20% |

- The model is very overconfident: most answers have about 100% token probability, even
  wrong ones. So the margin signal (C) looks weak so far.
- The agent v2 rule (fixed before testing) flags 53% of photos and catches 7 of 10 wrong answers.
  That's a lot of flags.
- In the replay at 1 Mbps, agent v2 sent **88 wrong labels vs 204 for agent v1**, but also had
  less correct info (75% vs 81%). **A trade-off: more trust, less coverage.**
- Asking 3 times and taking the majority vote did not improve accuracy (63% vs 67%).
- We do **not** change the rule based on these 30 photos. Wait for the full 300.

### "Not sure" full test (19 Oct 2026): 300 new photos
File: `experiments/confidence_aid_test_gemma3-4b_r3_n10.csv` (+ `.png` chart).
Accuracy on the new photos: **65%** (194/300), close to the first set's 61%.

| Signal | Ranking (0.5 = useless, 1 = best) | Wrong caught at 10% / 20% / 30% flagged | Extra cost |
|---|---|---|---|
| A. Several answers | 0.50 | 10% / 21% / 26% | none |
| **B. Confused classes** | **0.79** | **23% / 37% / 53%** | **none** |
| C. Answer margin | 0.58 | 11% / 25% / 37% | none |
| D. Agreement (3 asks) | 0.70 | 22% / 39% / 54% | 2 more model calls |
| A + D + C combined | 0.69 | 17% / 38% / 54% | 2 more model calls |

- **The best signal is free.** B ("is this a class the model often gets wrong?") needs no extra
  model call. It only uses what we learned from the first 300 photos. Good news for a satellite with little power.
- **Asking 3 times (D) works too:** when all 3 answers agree, 77% are right. When they don't, only about 32% are.
- **Token probability (C) is weak** (0.58): the model is about 100% sure even when it's wrong.
  "Several answers" (A) is useless (0.50); it happened only 7 times.
- **The agent v2 rule left out the best signal (B)** and included the weak one (C). It flags 45%
  of photos and catches 62% of wrong answers.

**Agents on the replay** (`experiments/agents_delhi_20261019_1031.csv`; 1,088 photos from the new set)

| Downlink | Policy | Correct info | Wrong labels | Unsure labels | Human-activity correct |
|---|---|---|---|---|---|
| 0.25 Mbps | send all raw | 14% | 0 | 0 | 15% |
| | caption all | 67% | 358 | 0 | 66% |
| | agent v1 | **72%** | 310 | 0 | 72% |
| | agent v2 | 54% | **140** | 360 | 56% |
| 1 Mbps | send all raw | 59% | 0 | 0 | 58% |
| | agent v1 | **85%** | 158 | 0 | 88% |
| | agent v2 | 82% | **106** | 90 | **90%** |
| 2 Mbps | send all raw | 87% | 0 | 0 | 86% |
| | agent v1 | 94% | 61 | 0 | 97% |
| | agent v2 | **96%** | **44** | 0 | **97%** |

- **At 1–2 Mbps, agent v2 is the best balance:** a third fewer wrong labels than v1, about the same
  correct info, and the most human-activity photos right.
- **At 0.25 Mbps, v2 is too careful:** it halves the wrong labels but loses a lot of correct info,
  because the budget is too small to send the unsure photos in full.
- **Both agents beat "send all raw" by a wide margin** whenever the downlink is tight.

## Agent v4: "decision head" (frozen image model + small classifier), 20 Oct 2026

Idea from "decision models" like TypeSafe's Jev: fix the answers in advance (30 classes) and get a
chance for every answer in one quick pass, with no text generated. For photos:
a **frozen** image model (SigLIP) makes an embedding, and a small logistic-regression head is trained on top.
Code: `src/model/head.py`. Agent v4 = agent v3's planner with the head's label and chance.

**Quick test on the Mac** (`siglip-base`, 90M parameters, 20 training photos per class = 600 photos
from AID photos 31+, which Gemma never saw). Setting C=1000 picked on tuning photos (11-20) by log-loss.

| Photos | Accuracy | Calibration error (0 = perfect) |
|---|---|---|
| tuning (11-20) | 95.7% | 0.026 |
| **FINAL (21-30)** | **95.7%** (287/300) | 0.036 |

Gemma 3 4B zero-shot on the same final photos: 63%. Speed: **26 ms per photo** (Mac M2 GPU) vs about 7,500 ms.
No photo is in two sets (checked). All 13 wrong answers are among the 58 least-sure photos.

**Agents on the replay** (`experiments/agents_delhi_final_equal_20261020_0947.csv`, equal sizes)

| Downlink | agent v3 (Gemma) correct / wrong | **agent v4 (head)** correct / wrong |
|---|---|---|
| 0.25 Mbps | 68% / 192 | **100% / 3** |
| 0.5 Mbps | 79% / 164 | **100% / 0** |
| 1 Mbps | 92% / 90 | **100% / 0** |
| 2 Mbps | 99% / 6 | 100% / 0 |

- **Not a fair fight, so say so:** the head is trained on 600 labelled AID photos (same dataset, same
  kind of photos). Gemma never saw a single labelled example. v4 is "supervised", v3 is "zero-shot".
- The replay reuses the 300 final photos about 3.6 times each, so 13 wrong photos become about 47
  wrong labels if sent; v4 sends almost all of them as full photos because its chances are honest.
- Next: full run with the 400M SigLIP and all 9,100 training photos on a Kaggle GPU
  (`python -m src.cloud.kaggle_run`); try `dinov2`; test on photos that are NOT from AID (real Sentinel-2).

### Full run on Kaggle (20 Oct 2026): SigLIP 400M, all 9,100 training photos

Run with `python -m src.cloud.kaggle_run` (Tesla T4, 72 min; AID read from the Kaggle copy, same bytes as
Hugging Face). Setting C=300 picked on tuning photos.

| Head | Training photos | Tuning | **FINAL (21-30)** | Calibration error (final) | Speed |
|---|---|---|---|---|---|
| SigLIP-base (90M) | 600 | 95.7% | 95.7% (13 wrong) | 0.036 | 26 ms/photo (Mac M2) |
| **SigLIP so400m (400M)** | 9,100 | 99.3% | **98.3%** (5 wrong) | 0.021 | 420 ms/photo (T4, full precision) |

Agents on the replay (`experiments/agents_delhi_final_equal_20261020_1111.csv`), correct / wrong labels:

| Downlink | send all raw | agent v3 (Gemma) | v4 small head | **v4 big head** |
|---|---|---|---|---|
| 0.25 Mbps | 15% / 0 | 68% / 192 | 100% / 3 | 99% / 7 |
| 0.5 Mbps | 30% / 0 | 79% / 164 | 100% / 0 | 100% / 5 |
| 1 Mbps | 60% / 0 | 92% / 90 | 100% / 0 | 100% / 3 |
| 2 Mbps | 89% / 0 | 99% / 6 | 100% / 0 | 100% / 0 |

- **More accurate is not the same as more trustworthy.** The big head's 5 mistakes all had chances
  above 96% (confidently wrong), so the agent trusted them. The small head's mistakes mostly had low
  chances, so the agent sent those photos in full. One photo (Park said Resort) fooled both heads.
- **For the satellite, the small head looks like the better deal:** about the same delivered info with
  1/4 of the size and much less compute (the two speeds above are on different hardware, so not
  directly comparable).
- The big run was slow because it used full precision on the T4. `head.py` now uses half precision (fp16)
  on Nvidia GPUs (checked on the Mac GPU: same embeddings, match 1.0). Not yet timed on a T4.

## Agent v4 on real Sentinel-2 photos (EuroSAT), 20 Oct 2026

`python -m src.eval.sentinel_test` (small head, SigLIP-base, on the Mac, 22 ms per photo).
EuroSAT photos are real Sentinel-2 (64x64 px, 10 m per pixel, 10 classes); AID photos are sharp
Google Earth photos. Sets: FINAL = 1,000 photos (100/class, including Gemma's 100 baseline photos),
tuning = 500, training = 1,000 (100/class). None overlap.

**Test 1: the AID head as it is (no retraining).** Its 30 classes were matched to EuroSAT's 10 before
testing (`MATCH` in the script; PermanentCrop and HerbaceousVegetation have no match).

| Photos | Right |
|---|---|
| All 1,000 | 33.2% (random guessing: 12%) |
| Only the 8 classes with an AID match | 41.5% |
| Gemma's 100 photos | 30% (Gemma 3 itself: 29%) |

- **It does not carry over.** It calls most blurry 64 px photos "BareLand" (crops, grass, orchards,
  even houses) and dark lakes "Forest". Only Forest (94%) and Residential (70%) work.
- So a head trained on one kind of photo is no better than Gemma on a different kind.

**Test 2: same frozen SigLIP, new head trained on 1,000 Sentinel-2 photos.** C=300 picked on tuning.

| Photos | Right | Calibration error |
|---|---|---|
| **FINAL 1,000** | **92.8%** | 0.014 |
| Gemma's 100 photos | 93% (Gemma 3: 29%) | |

**Test 3: satellite replay with Sentinel-2 photos** (each counted as 280 KB, like the main results).
Correct info / wrong labels (`experiments/sentinel_test3_replay_20261020_1126.csv`):

| Downlink | send all raw | caption all (Gemma) | agent v1 (Gemma) | **agent v4** |
|---|---|---|---|---|
| 0.25 Mbps | 15% / 0 | 29% / 776 | 37% / 687 | **97% / 29** |
| 0.5 Mbps | 30% / 0 | 29% / 776 | 49% / 557 | **100% / 5** |
| 1 Mbps | 60% / 0 | 29% / 776 | 69% / 333 | **100% / 0** |
| 2 Mbps | 89% / 0 | 29% / 776 | 92% / 86 | **100% / 0** |

Photos drawn from Gemma's 100 (Gemma answers exist only for those). With all 1,000 final photos,
agent v4 gets 98% / 17 wrong at 0.25 Mbps and 100% / 0-4 from 0.5 Mbps up.

- **The method works on real Sentinel-2 photos, but the head must be trained on the same kind of photos.**
  A satellite mission would train its head on labelled photos from its own camera (or a close one).
- The frozen image model is the same in both tests; only the tiny classifier changes. So moving to a new
  camera means retraining a small classifier from about 1,000 labelled photos, not the big model.

### Big model (SigLIP 400M) on Sentinel-2, 20 Oct 2026

Embeddings made on Kaggle (T4, half precision, **48 ms per photo**), then
`python -m src.eval.sentinel_test --backbone siglip` on the Mac (same orbit data as the small run).

| | Small head (SigLIP-base, 90M) | Big head (SigLIP so400m, 400M) |
|---|---|---|
| Test 1: AID head on Sentinel-2, all / 8 matched classes | 33.2% / 41.5% | 40.6% / 50.7% |
| **Test 2: head trained on 1,000 Sentinel-2 photos** | 92.8% | **94.8%** |
| Calibration error (test 2) | 0.014 | 0.014 |
| Replay, 1,000 photos, 0.25 Mbps (correct / wrong) | 98% / 17 | 99% / 10 |
| Replay, 1,000 photos, 1 Mbps | 100% / 0 | 100% / 1 |
| Compute per photo (estimate from model and input size) | 1x | about 15-20x |

- **The big model buys about 2 points on Sentinel-2 for about 15-20x the compute.** Both are equally honest.
  For a satellite (power-limited), the small head stays the main choice; the big one is the upper bound.
- Even the big AID head fails on Sentinel-2 (40.6%): the photo-type gap is the real problem, not model size.
- **The head is tiny:** 768 x 10 weights + 10 biases for the small model, about 30 KB. A new head for a new
  camera or mission could be sent UP to the satellite in seconds; the frozen image model never changes.
- Lost and remakeable: the 9,700-photo AID embeddings of the big model (`data/processed/embeddings_aid_siglip.npz`)
  were overwritten by this run's 900-photo file (runner bug, now fixed: it merges). The result CSVs are in git.

## Baseline: tiny CNN trained end to end (MobileNetV3-Small), 20 Oct 2026

`python -m src.cloud.kaggle_run --command "src.model.cnn"` (T4, 7 min). ImageNet start, every weight trained,
flips + quarter turns, 20 epochs, best epoch and a temperature (chance adjustment) picked on tuning photos.
Same training / tuning / final photos as the decision heads.

| | Tiny CNN (1.5M weights, ~3 MB) | Small head (SigLIP-base ~90M, frozen + 30 KB head) | Big head (400M) |
|---|---|---|---|
| AID, 20 training photos/class: right | 85.0% | **95.7%** | (98.3% with 9,100 photos) |
| AID calibration error | 0.049 | 0.036 | 0.021 |
| Sentinel-2, 100 training photos/class: right | 94.6% | 92.8% | **94.8%** |
| Sentinel-2 calibration error | 0.016 | 0.014 | 0.014 |
| Compute per photo (published figures, approx.) | ~0.06 GFLOPs | ~17.6 GFLOPs | roughly 15-20x the small head |

Replay, correct info / wrong labels:

| | 0.25 Mbps | 0.5 Mbps | 1 Mbps |
|---|---|---|---|
| AID, agent v4 with CNN | 92% / 91 | 96% / 40 | 99% / 12 |
| AID, agent v4 with small head | 100% / 3 | 100% / 0 | 100% / 0 |
| Sentinel-2 (1,000 photos), CNN | 98% / 19 | 99% / 9 | 100% / 3 |
| Sentinel-2 (1,000 photos), small head | 98% / 17 | 100% / 4 | 100% / 0 |

Files: `experiments/cnn_*_mobilenet_v3_small_*_final.csv`, `experiments/agents_delhi_final_equal_20261020_1207.csv`,
`experiments/sentinel_test3_replay_cnn_eurosat_mobilenet_v3_small_n100_20261020_1207.csv`

- **With few labelled photos, the frozen head wins clearly** (AID, 20/class: 95.7% vs 85.0%, and 3 vs 91 wrong
  labels at 0.25 Mbps).
- **With 100 photos per class, the CNN catches up** (Sentinel-2: 94.6% vs 92.8%) at about 1/300 of the compute.
  For a power-limited satellite that is a strong point for the CNN.
- So the honest story is a trade-off: the frozen head needs fewer labels and one shared model (swap a 30 KB head
  per mission); the CNN is far cheaper per photo but needs more labels and a full 3 MB retrain per mission.
- Caveat: on Sentinel-2 the CNN was still improving at epoch 20 (tuning log-loss 0.177 and falling), so it may
  do a bit better with more epochs. Speeds are not compared: the timings above include opening the photos and
  were on different machines. Real numbers need the Jetson.
- Next for the paper: accuracy vs number of training photos (5/10/20/50/100 per class) for both, to find
  where the CNN catches up.

## Label curve: how many labelled photos does each model need? 20 Oct 2026

`python -m src.cloud.kaggle_run --command "src.eval.label_curve"` (Kaggle T4), chart: `python -m src.eval.plot_label_curve`.
Frozen SigLIP-base + head vs tiny CNN (MobileNetV3-Small, up to 30 epochs, best on tuning), 5-100 training photos
per class, 3 random picks of photos each. Same tuning and final photos throughout.
File: `experiments/label_curve_20261020_0752.csv`, chart: `docs/label_curve.png`. Mean of 3 picks (lowest-highest):

| Photos/class | AID head | AID CNN | Sentinel-2 head | Sentinel-2 CNN |
|---|---|---|---|---|
| 5 | **89.7%** (88.7-90.7) | 67.9% (66.3-70.7) | **79.3%** (75.7-82.6) | 44.2% (36.2-55.4) |
| 10 | **91.6%** | 80.4% | **84.4%** | 61.7% |
| 20 | **94.2%** | 85.9% | **88.5%** | 81.4% |
| 50 | **94.8%** | 88.6% | 91.4% | 91.4% |
| 100 | 94.7% | 93.8% | 92.8% | **95.4%** |

- **The frozen head needs far fewer labels.** With 5 photos per class it gets 90% (AID) and 79% (Sentinel-2);
  the CNN gets 68% and 44%. It is also much steadier: the CNN's result swings by up to 19 points
  depending on which 5 photos were picked.
- **On Sentinel-2 the two cross at about 50 photos per class**; at 100 the CNN is ahead (95.4% vs 92.8%).
  On AID (30 classes, harder) the CNN is still behind at 100 (93.8% vs 94.7%).
- **Calibration:** with few labels the head's chances are more honest (Sentinel-2, 5/class: error 0.022 vs 0.093);
  at 100/class the CNN's are slightly better (0.012 vs 0.019).
- Note: at 100/class all 3 picks use the same 100 photos (that is the whole training pool), so the head's
  result there doesn't vary; the CNN's still does (random start).
- **For the paper:** a new satellite or camera starts with few labelled photos. The frozen head is the right
  tool then (and needs only a 30 KB update); once a few thousand labels exist, a tiny CNN is cheaper to run
  for the same accuracy. Both beat Gemma 3 with no training at every size.

## Replay error bars: 7 days x 3 stations x 3 photo orders, 20 Oct 2026

`python -m src.eval.replay_bars --plot` (saved answers, no model calls, 40 s). Days 19-25 Oct 2026,
stations Delhi / Bengaluru / Svalbard, photo orders (seeds) 42-44 = 21 runs per station, policy and rate.
File: `experiments/replay_bars_20261020_1413.csv`, chart: `docs/replay_bars.png`.
Check: Delhi, 19 Oct, seed 42 gives exactly the main-results numbers.

**AID final photos, 1 Mbps: correct info / wrong labels, mean ± spread (standard deviation)**

| Policy | Delhi | Bengaluru | Svalbard |
|---|---|---|---|
| Send all raw | 58.0 ± 5.3% / 0 | 55.5 ± 3.9% / 0 | 83.0 ± 2.2% / 0 |
| Caption all (Gemma) | 63.4 ± 0.7% / 372 ± 17 | 63.4 ± 0.7% / 378 ± 19 | 63.0 ± 0.7% / 462 ± 16 |
| Agent v1 | 84.3 ± 2.0% / 159 ± 16 | 83.4 ± 1.5% / 172 ± 15 | 94.2 ± 1.1% / 73 ± 15 |
| Agent v3 | 91.1 ± 3.0% / 83 ± 14 | 90.9 ± 1.4% / 93 ± 13 | 95.7 ± 1.8% / 45 ± 10 |
| Agent v4, tiny CNN | 98.6 ± 0.7% / 14 ± 6 | 98.6 ± 0.2% / 14 ± 2 | 99.2 ± 0.7% / 10 ± 10 |
| **Agent v4, frozen head** | **100.0 ± 0.0% / 0 ± 0** | **100.0 ± 0.0% / 0 ± 0** | **100.0 ± 0.0% / 0 ± 0** |

At 0.25 Mbps (Delhi): raw 18.4 ± 3.6%, v3 68.7 ± 2.1% / 167 ± 16, CNN 92.2 ± 0.9% / 76 ± 6,
head 99.3 ± 0.4% / 8 ± 4. Sentinel-2 photos, 1 Mbps (Delhi): raw 58.0 ± 5.3%, head 99.8 ± 0.3% / 2 ± 3,
CNN 99.6 ± 0.2% / 4 ± 2.

- **The one-day results are typical.** Raw 60% and v3 92% / 90 wrong (19 Oct) sit inside the spread.
- **The order of the policies never changes** across days, stations or photo orders: head > CNN > v3 > v1 > raw
  (AID); head ≈ CNN on Sentinel-2.
- **The agent matters most for low-latitude stations like Delhi and Bengaluru** (3 passes a day). Svalbard
  (polar, 12 passes a day) gets 4x the contact time, so raw sending already reaches 83% at 1 Mbps and the gain shrinks.
- Raw sending varies most (± 5% at Delhi): its result depends on pass lengths, which change day to day.
  The agents are steadier because labels always fit.

## Overfitting check and NAVI comparison, 20 Oct 2026

`python -m src.eval.overfit_check` (small head, SigLIP-base). FRESH = photos no model, setting or decision
ever touched: AID photos 131-160 per class (900), EuroSAT the next 100 per class (1,000).

| Set | AID (20 training photos/class) | Sentinel-2 / EuroSAT (100/class) |
|---|---|---|
| Training | 100.0% | 100.0% |
| Tuning | 96.0% (93.7-98.0) | 93.8% (91.6-95.8) |
| FINAL | 94.3% (91.7-96.7) | 92.7% (91.0-94.2) |
| **FRESH, never used** | **94.4% (92.9-95.8)** | **92.5% (90.9-94.1)** |
| Final photos with a near-twin (cosine > 0.95) in training | 25 of 300 | 723 of 1,000 |
| FINAL without those photos | 93.8% | 88.4% |
(95% bootstrap ranges in brackets. Gemma 3 on the AID final photos: 63.0%, range 57.7-68.3%.)

Replay with FRESH photos only (Delhi, 7 days x 3 photo orders), agent v4 vs raw:
AID 0.25 Mbps 98.7% / 13 wrong vs 18.4%; 1 Mbps 99.9% / 1 wrong vs 58.0%.
Sentinel-2 0.25 Mbps 97.8% / 22 wrong; 1 Mbps 99.8% / 2 wrong.

- **No overfitting to the final test photos:** fresh photos give the same accuracy (AID 94.4% vs 94.3%,
  Sentinel-2 92.5% vs 92.7%). Training accuracy is 100% (the head memorises its few training photos), but
  that doesn't carry over as a gap: tuning, final and fresh agree.
- **The AID number moves with tiny details:** with identical embeddings (cosine 1.000) the setting picker
  chooses C=1000 or C=3000 (tuning log-loss 0.140 vs 0.142), giving 94.3-95.7% on the final photos. Report
  "about 95% (95% range ±2.5)", not "95.7%".
- **EuroSAT may be inflated by neighbouring tiles** ("spatial leakage", a known EuroSAT problem; one preprint
  reports 3.5-21 points). 72% of final photos have a near-twin in training (similarity is high for all EuroSAT
  photos: 0.959 median, same as training-to-training 0.957). Without twins: 88.4%. The fresh set is drawn the
  same way, so it can't rule this out. Fix for the paper: a spatially separate split (needs the GeoTIFF
  version of EuroSAT, which has coordinates) or another Sentinel-2 dataset with locations.
- AID shows no such sign (twins 25/300, accuracy without them 93.8%).

**NAVI-Orbital's 88.16% is on AID cut to 18 classes** (arXiv 2606.18271, Sec. IV-B): 3 residential classes
merged; Meadow -> Agricultural; BareLand -> Desert; Park -> Forest; Viaduct -> Bridge (their table also has a
Highway Interchange class, so this is unclear); Stadium -> Venue; 7 classes removed (Baseball, Church, School,
Center, Resort, Square, Playground). They also prompt with the 18 names, use pan-and-scan high resolution,
temperature 0.2 and up to 10 retries. Our final photos re-scored with their 18 classes (our predictions mapped,
removed classes dropped, 230 photos): **Gemma 76.5%** (30 classes: 63.0%), small head 96.5%, tiny CNN 88.3%.
The rest of the gap to 88% fits their 18-option prompt and higher-resolution input.

## Audited downlink: can the ground trust the numbers? (20 Oct 2026, evening)

Why (full reasoning: `notes/research_review_2026-10-20.md`): agents v2-v4 send the photos the model is
**least sure** about. The photos it is sure about are never checked, so when the model is wrong in a confident
way, nothing on the ground can see it. And the old score can't fail: a 50-byte label and a 280 KB photo both
count as "1 correct", so a good labeller always scores ~100%.

New product the ground wants: **the share of the day's photos in each class, with a 90% error bar**
(crop area, forest cover, cloud...). New policy, the **audit** (`src/agent/audit.py`): a label + class chances
for every photo, then full photos picked **at random with known chances** (unsure photos more likely; every
photo keeps a floor of `floor` x the average chance), exactly as many as fit (Pareto sampling). The ground
(`src/eval/estimate.py`) adds the average mistake seen on the checked photos (each weighted by 1 / its chance)
to the labels' average: right on average whatever the onboard model is like.

Self-test on made-up data (`python -m src.eval.estimate`): the honest methods are right on average (worst
bias 0.1-0.3 points) vs 3-7 points for labels only and triage.

### Failure suite (`src/model/faults.py`), on 2,000 EuroSAT photos (FINAL + FRESH)
| Onboard model | Right |
|---|---|
| Healthy head (SigLIP-base + head, 100/class) | 92.6% |
| New camera: the AID head, classes matched to EuroSAT | 28.0% |
| Silent swap: 30% of AnnualCrop labelled PermanentCrop, as sure as right answers | 89.8% |
| Radiation bit flips: 8 random bits of the 30 KB head (first seed that hurt 5-30 points) | 84.2% |
| Damaged camera, moderate: blur 1 px / noise sigma 10 / stripes (column gain 8%) | 62.2% / 43.0% / 50.0% |
Finding on the side: the frozen SigLIP + head is **fragile to moderate camera damage** (43-62% right).

### Main result (`python -m src.eval.audit_test`, 7 days x 3 stations x 3 photo orders, India-like shares (assumed))
0.25 Mbps, 63 days per cell. Each cell: typical error (RMSE, percentage points) / 90% error bar held
(all classes, rare classes). Labels only and triage only have a "pretend every label is right" bar.
File: `experiments/audit_summary_eurosat_20261020_2023.csv` (every estimate: `data/processed/audit_runs_eurosat_*.csv.gz`).

| Onboard model | labels only | triage (v4) | raw random | stratified | **audit, floor 0.5** |
|---|---|---|---|---|---|
| healthy head | 0.69 / 96%, 95% | **0.26** / 100%, 100% | 2.18 / 91%, 96% | 0.79 / 99%, 99% | 0.47 / 98%, 99% |
| new camera (AID head) | 10.75 / **0%**, 0% | 7.99 / **8%**, 10% | 2.18 / 91%, 96% | 2.42 / 88%, 81% | 2.69 / 86%, 80% |
| silent swap | 4.39 / 79%, 95% | 3.05 / 80%, 100% | 2.18 / 91%, 96% | 1.05 / 97%, 98% | 1.22 / 97%, 100% |
| bit flips | 1.44 / 73%, 96% | 1.57 / 80%, 100% | 2.18 / 91%, 96% | 1.31 / 96%, 98% | 1.33 / 95%, 99% |
| damaged camera: blur | 5.09 / 26%, 17% | 2.90 / 57%, 84% | 2.18 / 91%, 96% | 1.68 / 93%, 96% | 1.22 / 94%, 98% |
| damaged camera: noise | 11.59 / 7%, 14% | 7.78 / 18%, 30% | 2.18 / 91%, 96% | 2.15 / 92%, 90% | 1.80 / 95%, 95% |
| damaged camera: stripes | 10.41 / 16%, 33% | 6.18 / 29%, 48% | 2.18 / 91%, 96% | 2.30 / 90%, 90% | 1.56 / 95%, 97% |

- **Healthy model: triage is the most accurate** (0.26 points), but nothing tells you that. The audit gets
  0.47 with an honest bar (±1.0).
- **Broken model: labels only and triage are off by 1.4-11.6 points with bars that hold 0-80%** of the time,
  and nothing on the ground shows it. The audit stays honest (86-97%) and is never much worse than raw random.
- Chart: `docs/audit_honesty.png` (`python -m src.eval.plot_audit`).

### Error bars that hold for rare classes (task: 90% bars should hold ~90%)
Plain normal / t bars held only 81-87% overall and down to 71% for rare classes (2-4% shares): at low budgets a
pass checks ~25 photos, often none of a rare class, so the pass reports ~0 spread. Fixes, compared pooled over
3 stations x 6 rates (`holds_*` columns in the summary CSV):
- **safe**: never narrower than what the model's chances predict. Works when the model is honest (97-99%),
  not when it is broken (new camera 83-85%).
- **scaled**: the model's prediction scaled up by (mistakes seen on the checked photos) / (mistakes the model
  expected), pooled over classes and passes. New camera 88-89%, silent swap / bit flips 93-97%.
- **scaled+1** (the default): also never narrower than "one checked photo we didn't see could have been this
  class" (plus-one rule, needs no model). New camera rare classes at 0.05-0.1 Mbps: 63-70% -> 87-96%.
  Cost: the healthy audit's average bar ±1.44 -> ±1.73 points (over all rates).
- **Still open:** with the new camera at 0.25-0.5 Mbps, rare classes hold ~80-87% (tuned: 82-98%). Next fix to
  try: finite-sample "betting" intervals (Waudby-Smith & Ramdas, JRSS-B 2024).
- A model-assisted regression estimator (GREG on all class chances) broke with ~160 checks (errors up to
  23 points); a per-class trust weight (PPI++ idea, "tuned", cross-fitted on two halves) never did worse than
  raw random with a broken model but costs accuracy with a healthy one (1.2-1.5 vs 0.5 points).

### How much should a satellite check itself? (`python -m src.eval.audit_floor`, 0.25 Mbps, 63 days per cell)
File: `experiments/audit_floor_20261020_1936.csv`, chart `docs/audit_floor.png`. Typical error / bar held:

| Floor | healthy head | new camera | silent swap | bit flips | blur | noise | stripes |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.50 / 97% | 2.74 / 88% | 1.51 / 95% | 2.31 / 86% | 1.24 / 94% | 1.73 / 93% | 1.61 / 92% |
| 0.1 | 0.50 / 98% | 2.73 / 89% | 1.65 / 95% | 2.27 / 88% | 1.08 / 95% | 1.78 / 95% | 1.69 / 94% |
| 0.35 | 0.49 / 98% | 2.62 / 89% | 1.43 / 97% | 1.57 / 95% | 1.20 / 93% | 1.86 / 94% | 1.65 / 93% |
| **0.5** | **0.47 / 98%** | 2.57 / 89% | 1.32 / 96% | **1.34 / 95%** | 1.26 / 96% | 1.96 / 92% | 1.76 / 93% |
| 1 | 0.71 / 97% | 2.29 / 88% | 1.12 / 98% | 1.23 / 97% | 1.50 / 95% | 1.83 / 92% | 1.72 / 95% |

- **A floor of 0.5 is nearly free insurance:** healthy error 0.47 (same as lower floors), while confident
  mistakes (bit flips) drop from 2.3 to 1.3 points error and the bars go from 86-88% to 95%.
- Floor 1 (plain random checks) costs 50% more error with a healthy model (0.71).
- No floor fixes a wrong camera (88-89% everywhere): that needs better error bars or a new head (closed loop).

### When does onboard AI beat plain compression? (`python -m src.eval.compress_test`, AID, Delhi, 21 days per row)
Compress to fit = every photo re-saved as JPEG small enough for the pass (real ladder: 600 px q50 = 44 KB,
93.7% still recognised; 224 px q90 = 19 KB, 93.3%; 160 px = 6.4 KB, 88.3%; 112 px = 3.6 KB, 54%).
Same bytes AND the same number of human checks for every method. ASSUMPTION: a person reads any photo >= 160 px.
"Audit, thumbnails" = the audit, with checked photos sent as 160 px thumbnails instead of full photos.
File: `experiments/compress_sweep_20261020_1938.csv`, chart `docs/audit_rsweep.png`.
Each cell: % of photos whose class reached the ground right / honest 90% bar on class shares.

| R (taken / sent) | raw random | compress to fit | triage (v4) | audit, full photos | audit, thumbnails |
|---|---|---|---|---|---|
| 1.4 (1 Mbps) | 58% / ±1.0 | 94% / exact | 100% / (±0.9 pretend) | 99% / ±0.1 | 100% / exact |
| 5.6 (0.25) | 18% / ±2.5 | 93% / exact | 99% | 96% / ±0.5 | 100% / exact |
| 28 (0.05) | 3.5% / ±6.6 | 80% / ±0.3 | 96% | 94% / ±1.4 | 100% / exact |
| 56 (0.025) | 1.7% / ±11 | 56% / ±0.9 | 95% | 94% / ±2.1 | 99.6% / ±0.1 |
| 140 (0.01) | 0.6% / ±40 | 28% / ±1.8 | 94% | 94% / ±30 | 98% / ±0.4 |
| 562 (0.0025) | 0% / ±50 | 7% / ±4.5 | 94% | 94% / ±50 | 95% / ±1.1 |

- **Honest negative result: in the paper's old range (R 1.4-5.6), plain compression is enough.** Every photo
  fits shrunk (people can see all of them, 93% still recognised by the head), and the class shares are exact.
  Under the old score ("full photo = correct") compress-to-fit ties agent v4 with zero onboard compute.
- **Onboard labels matter above R ~ 30:** compression shows only 56% of photos at R = 56 and 7% at R = 562,
  while labels keep ~94% of photos informative.
- **Send checks as thumbnails, not full photos:** a check only needs a photo a person can read. At R = 140 that
  is ~45x more checks per byte: honest bars ±0.4 vs ±30 with full photos and ±1.8 for compress-to-fit.

### Closed loop: a new camera with zero labels (`python -m src.eval.audit_loop`, Delhi, 0.25 Mbps, 3 photo orders)
The checked photos become training labels; the 30 KB head is retrained after every pass and used one pass
later (it only learned from earlier passes, so the error bars stay honest). Launch head = the AID head.
File: `experiments/audit_loop_eurosat_0.25mbps_20261020_1959.csv`, chart `docs/audit_loop.png`.

| Day | closed loop | launch head + audit | launch head + triage | head trained before launch | raw random | head right | labels |
|---|---|---|---|---|---|---|---|
| 1 | 1.93 / ±4.15 | 3.00 / ±4.07 | 9.42 (pretend ±1.1) | 0.49 / ±1.09 | 1.96 / ±3.69 | (no head yet) | 152 |
| 2 | 0.53 / ±1.32 | 2.09 / ±3.90 | 8.68 | 0.41 / ±0.98 | 2.60 / ±3.47 | 91% | 364 |
| 4 | 0.39 / ±1.22 | 3.42 / ±4.26 | 9.64 | 0.62 / ±1.35 | 3.08 / ±3.77 | 96% | 682 |
| 7 | 0.22 / ±1.19 | 1.74 / ±4.19 | 9.10 | 0.61 / ±1.40 | 2.14 / ±3.83 | 98% | 1,219 |
(typical error / honest bar, points.) Over the mission, 90% bars held: closed loop 100%, launch head + audit 84%,
launch head + triage **5%** (worst day 24 points off), head trained before launch 99%, raw random 88%.
- **Caveat:** the EuroSAT replay draws from 2,000 photos, so photos repeat across days and the closed-loop head
  partly memorises them (98% right, above the 92.6% of a head trained on 1,000 labels). The real replay has
  no repeats: see below.
- Note: a first run used the older "safe" bar; with the default "scaled+1" bar the numbers above hold.

### Real replay: what Sentinel-2A really photographed (13-19 Oct 2026)
`python -m src.data.s2_real` walks Sentinel-2A's real path (saved TLE) in 30 s steps, 6-19 Oct 2026, and for each
daylight slot asks Element 84's Earth Search catalogue for a Sentinel-2A image within 5 minutes at that spot.
- **The orbit sim matches reality:** where an image exists, its time is within 0-7 s of the simulated slot.
- 20,177 daylight slots -> **3,527 real photos** (17%: Sentinel-2 only images land and coasts), ~250 a day,
  224 px chips (10 m, 2.24 km), 200 MB. Label = the image's cloud map if >= 50% cloud (or snow), else the main
  ESA WorldCover 2021 class. **Real mix: cloud 46%, water 17%, shrubs/grass 12%, trees 11%, bare 8%,
  cropland 5%, snow 1%, built-up 0.1% (4 photos in 14 days).** Balanced datasets hide exactly this.
- Head (frozen SigLIP-base + logistic regression) trained on 6-11 Oct (1,525 photos), setting picked on 12 Oct,
  tested on 13-19 Oct (1,749 photos): **82.3% right** (cloud 93%, water 88%, bare 83%, trees 75%, cropland 69%,
  shrubs/grass 65%, snow 0%: it learned snow as cloud).
- Failure suite on real photos: EuroSAT head ("new place": Europe, no cloud/bare/snow class) 20.0%, silent swap
  (30% of cropland -> shrubs/grass) 81.2%, bit flips (seed 12) 57.5%, blur 77.4%, noise 77.0%
  (real 224 px photos survive blur/noise much better than EuroSAT's 64 px ones).
- Downlink speeds 0.01-0.25 Mbps (R ~ 1.7-42 for ~250 photos/day at 300 KB). Same 3 stations x 7 days x 3 seeds.

`python -m src.eval.audit_test --data real` -> `experiments/audit_summary_real_20261020_2023.csv`,
chart `docs/audit_honesty_real.png`. Typical error (points) / 90% bar held (all classes, rare classes):

| 0.05 Mbps (Delhi: ~30 checks a day) | labels only | triage (v4) | raw random | stratified | **audit, floor 0.5** |
|---|---|---|---|---|---|
| healthy head (82% right) | 1.89 / 80%, 60% | 1.54 / 84%, 67% | 5.89 / 96%, 100% | 4.08 / 99%, 100% | 3.53 / 99%, 100% |
| new place (EuroSAT head) | 19.54 / **9%**, 6% | 16.63 / **20%**, 27% | 5.89 / 96%, 100% | 7.22 / 95%, 100% | 8.30 / 97%, 99% |
| silent swap | 1.69 / 82%, 56% | 1.51 / 84%, 62% | 5.89 / 96%, 100% | 4.38 / 99%, 100% | 3.76 / 99%, 99% |
| bit flips | 10.63 / 27%, 32% | 10.78 / 27%, 43% | 5.89 / 96%, 100% | 6.22 / 95%, 99% | 8.12 / 91%, 98% |
| damaged camera: blur | 4.52 / 51%, 63% | 3.28 / 67%, 68% | 5.89 / 96%, 100% | 5.18 / 98%, 99% | 3.71 / 97%, 99% |
| damaged camera: noise | 3.96 / 51%, 59% | 2.67 / 71%, 68% | 5.89 / 96%, 100% | 5.12 / 100%, 100% | 4.21 / 97%, 99% |

| 0.25 Mbps | labels only | triage (v4) | raw random | stratified | **audit, floor 0.5** |
|---|---|---|---|---|---|
| healthy head | 1.89 / 80%, 60% | 0.67 / 96%, 89% | 2.22 / 92%, 97% | 1.48 / 96%, 97% | 1.01 / 95%, 95% |
| new place (EuroSAT head) | 19.54 / 9%, 6% | 9.08 / 58%, 79% | 2.22 / 92%, 97% | 2.06 / 90%, 93% | 3.49 / 92%, 96% |
| silent swap | 1.69 / 82%, 56% | 0.75 / 95%, 86% | 2.22 / 92%, 97% | 1.50 / 96%, 95% | 1.08 / 95%, 94% |
| bit flips | 10.63 / 27%, 32% | 6.66 / 55%, 79% | 2.22 / 92%, 97% | 2.17 / 88%, 94% | 2.78 / 85%, 88% |
| damaged camera: blur | 4.52 / 51%, 63% | 1.24 / 92%, 86% | 2.22 / 92%, 97% | 1.65 / 93%, 97% | 1.10 / 94%, 97% |
| damaged camera: noise | 3.96 / 51%, 59% | 1.12 / 93%, 87% | 2.22 / 92%, 97% | 1.84 / 93%, 92% | 1.31 / 94%, 94% |

- **On real photos even a healthy onboard model misleads:** at 82% right, labels only are 1.9 points off and their
  bars hold 80% (rare classes 60%). On EuroSAT (93% right) they held 96%. Real Earth is harder.
- **Triage is only honest when it can check a lot and the model is healthy** (0.25 Mbps, healthy: 96%); with tight
  budgets or a broken model its bars hold 20-84%. With the EuroSAT head, labels only are 19.5 points off: the head has
  no "cloud" class, and 46% of real photos are cloud.
- **The audit's bars hold 85-99% in every case.** Weakest: radiation bit flips at 0.25 Mbps (85%): the flipped head
  is confidently wrong on 42% of photos, and floor 0.5 still trusts its chances. Uniform checks (floor 1) hold 98%
  there (all variants: `audit, floor 1/0.1, tuned` rows in the CSV).
- **The honest price with a badly broken model:** the audit is noisier than random raw photos (new place: 8.3 vs 5.9
  points at 0.05 Mbps), because its check chances come from the broken model. With a healthy or mildly damaged model
  it beats raw random (3.5-4.2 vs 5.9).
- "Audit, thumbnails" is also in the CSV, but at these budgets every photo fits as a 160 px thumbnail, so it becomes
  a census (exact, but a person must read every photo): that is the compression baseline; `compress_test` compares
  them fairly with equal human checks.

### Real closed loop: a new camera learns from its own audits (`python -m src.eval.audit_loop --data real --rate 0.1`)
Launch head = the EuroSAT head (20% right). No photo repeats in the real replay, so this is the fair test.
Files: `experiments/audit_loop_real_0.1mbps_20261020_2000.csv` (chart `docs/audit_loop_real.png`) and `..._0.05mbps_...`.

| Day | closed loop | launch head + audit | launch head + triage | head trained before launch | raw random | head right | labels |
|---|---|---|---|---|---|---|---|
| 1 | 3.52 / ±4.25 | 4.54 / ±8.36 | 17.37 (pretend ±2.5) | 2.27 / ±3.27 | 4.24 / ±5.95 | 46% | 74 |
| 2 | 4.23 / ±7.75 | 4.83 / ±11.05 | 13.65 | 3.83 / ±5.38 | 5.27 / ±9.10 | 68% | 114 |
| 3 | 2.78 / ±4.09 | 8.37 / ±9.00 | 16.53 | 1.38 / ±3.69 | 5.16 / ±6.19 | 77% | 164 |
| 4 | 1.71 / ±3.73 | 5.58 / ±9.53 | 15.31 | 2.01 / ±3.82 | 3.35 / ±6.38 | 83% | 215 |
| 7 | 2.37 / ±3.74 | 5.30 / ±7.31 | 17.96 | 1.57 / ±3.50 | 3.96 / ±5.81 | 80% | 356 |
(typical error / honest bar, points; 0.1 Mbps, Delhi, 3 photo orders.) Over the mission, 90% bars held: closed loop
**96%**, launch head + audit 94%, launch head + triage **12%** (worst day 42 points off), trained before launch 96%,
raw random 95%.
- **From zero labels, the head reaches 77-83% right in 3-4 days on 164-215 checked photos** (the head trained on
  1,525 labels: 82%), and its error bars match the pre-trained head's from day 3 (±3.7-5.0 vs ±3.5-5.4), tighter
  than raw random (±5.8-9.1) and the launch head (±7-11).
- At 0.05 Mbps (~30 checks a day): head 43% -> 81% by day 4 (111 labels), bars held 98%; triage 11%.

### Self-checking audit: the satellite notices its own broken model (20 Oct, late)
Code: `src/agent/self_check.py`, method "audit, self-check" in `src/eval/audit_test.py` (memory across the 7 days
of a mission). After every pass the ground compares the mistakes it saw on the checked photos with the mistakes
the model's chances expected, over all checks so far. If **seen / expected > 1.5**, one bit goes up and the next
passes run in **safe mode**: (1) checks picked uniformly at random (floor 1), (2) the model's chances recalibrated
with a table learned from past checks (predicted class -> share of each true class among checked photos,
smoothed; fewer than 20 past checks: use the checks alone). Everything used comes from earlier passes, so the
error bars stay honest. This is the "conservative fallback mode" of the On-Orbit Space AI survey (Sec. 6B).

**Decided on the EuroSAT replay only, before any real-photo test** (scripts in the job's scratch folder, not kept):
- Signal: the healthy head's running ratio stayed 0.62-0.83 (per pass: median 0.82, 95th percentile 1.31, 2% of
  passes above 1.5); broken models 1.7-2.7 (71-99% of passes above 1.5). -> switch at 1.5.
- Safe mode, 3 options at 0.25 Mbps (error): checks alone 2.3-2.4 for every failure; model's own chances: new
  camera 2.50 (bars 87%); **recalibrated chances: new camera 2.21 (92%), swap 1.09, bit flips 1.24** -> chosen.

**Results** (`experiments/audit_summary_*_20261020_2023.csv`; the older methods' numbers are identical to the
earlier run). Typical error / 90% bar held / share of passes in safe mode:

| | plain audit (floor 0.5) | **self-checking audit** | raw random |
|---|---|---|---|
| EuroSAT, 0.25 Mbps, healthy head | 0.47 / 98% | 0.53 / 98% / 1% | 2.18 / 91% |
| EuroSAT, new camera | 2.69 / 86% (rare 80%) | **1.99 / 94% (rare 95%)** / 96% | 2.18 / 91% |
| EuroSAT, silent swap | 1.22 / 97% | **0.95 / 97%** / 94% | 2.18 / 91% |
| EuroSAT, bit flips | 1.33 / 95% | 1.30 / 96% / 95% | 2.18 / 91% |
| EuroSAT, blur | **1.22** / 94% | 1.47 / 95% / 96% | 2.18 / 91% |
| **Real**, 0.05 Mbps, healthy head | 3.53 / 99% | 3.56 / 98% / 3% | 5.89 / 96% |
| **Real**, new place (EuroSAT head) | 8.30 / 97% | **5.34 / 98%** / 97% | 5.89 / 96% |
| **Real**, bit flips | 8.12 / 91% | **5.50 / 97%** / 96% | 5.89 / 96% |
| **Real**, blur | 3.71 / 97% | 3.56 / 97% / 6% | 5.89 / 96% |
| **Real**, 0.25 Mbps, new place | 3.49 / 92% | **1.85 / 95%** / 97% | 2.22 / 92% |
| **Real**, 0.25 Mbps, bit flips | 2.78 / **85%** | **1.73 / 96%** / 97% | 2.22 / 92% |

- **The one real test confirms the EuroSAT choice.** With a badly broken model the audit used to be noisier
  than random raw photos; with the self-check it beats them (5.3-5.5 vs 5.9 points at 0.05 Mbps; 1.7-1.9 vs 2.2
  at 0.25) and its bars hold 95-98% (the 85% weak spot is now 96%). Healthy models: at most 3% of passes switched.
- It also mostly closes the rare-class gap with a wrong camera (EuroSAT 0.25 Mbps: rare classes 80% -> 95%).
- **Costs:** damaged cameras on EuroSAT (safe mode drops the model's per-photo confidence: blur 1.47 vs 1.22);
  at the tightest real budget (0.02 Mbps, ~6-12 checks a day) a few false switches (up to 12% of passes) made
  the silent-swap case worse (9.1 vs 5.7 points). Next idea: recalibrate with the full chance vector, not only
  the top label.
- Demo: `demo/audit.html` now uses the self-checking audit and shows, pass by pass, the mistake ratio and the mode.

## Fixing the "100%" flaws (22 Oct 2026)

The old per-photo score gave agent v4 100.0 ± 0.0%. That number is a ceiling, not strength: a 50-byte label
counts as much as a 280 KB photo, and plain compression ties it. The fixes below close the open weak spots.

**1. README and demo.** The audit is now the main result. The per-photo score is "Older result" with a
warning, and the table has a compress-to-fit row: at 0.25-1 Mbps (R = 1.1-7.1) compress-to-fit shows every
photo on all 21 days (old "lenient" score 100%, 0 wrong; `compress_sweep_20261020_1938.csv`). The demo page
(`demo/index.html`) has a "Careful" note under its score with a link to `audit.html`.

**2. A fair zero-shot rival** (`python -m src.model.zeroshot`). SigLIP-base with class names only
("a satellite photo of {words}.", same words as Gemma's prompt, fixed before testing), the same saved
embeddings as the head. One temperature fitted on the tuning photos (changes chances, not answers).

| Final photos | SigLIP zero-shot | calibration error (plain / temperature) | Gemma 3 4B | Small head (trained) |
|---|---|---|---|---|
| AID, 300 | **66.3%** | 0.067 / 0.048 | 63% | 95.7% |
| EuroSAT, 1,000 | **40.8%** | 0.189 / 0.102 | 29% (its 100) | 92.8% |

As agent v4's labeller on the old replay (Delhi, 1 Mbps): 94% / 67 wrong labels (Gemma agent v3 92% / 90).
File: `experiments/agents_delhi_final_equal_zeroshot_20261022_1310.csv`. So a 90M image model with no
training slightly beats the 4B VLM, 300x faster; the big gap to the trained head is the labels, not the model.

**3. Betting error bars** (`interval(e, "betting")` in `src/eval/estimate.py`). Hedged-capital betting
confidence interval (Waudby-Smith & Ramdas 2023) on each checked photo's weighted mistake
W = (y - p) / chance x checks / photos, whose limits are known from the chances sent down. No bell-curve
approximation. Assumption: checks treated as independent draws (ours are without replacement: less spread,
so the bar should err wide). Bet cap c = 0.75 chosen on made-up data (c = 0.5: ±20.4, 0.75: ±15.5 at 40
checks; both held 100%). Self-test (fakes): betting held 99-100% where normal bars held 86-89%.

Audit reruns (`audit_summary_{eurosat,real}_20261022_13*.csv`; every older column identical to 20 Oct).
Self-checking audit, mean over scenarios x stations: held / rare classes held / worst single class / ± points:

| | scaled+1 (default) | betting |
|---|---|---|
| EuroSAT 0.05 Mbps | 97% / 99% / 76% / ±8.7 | 100% / 100% / 95% / ±21.1 |
| EuroSAT 0.25 Mbps | 95% / 96% / 76% / ±2.5 | 100% / 100% / 90% / ±5.9 |
| EuroSAT 2 Mbps | 94% / 94% / 71% / ±0.4 | 100% / 100% / 100% / ±1.6 |
| Real 0.02 Mbps | 99% / 100% / 81% / ±24.1 | 100% / 100% / 100% / ±44.2 |
| Real 0.05 Mbps | 98% / 99% / 81% / ±11.7 | 100% / 100% / 100% / ±26.5 |
| Real 0.25 Mbps | 96% / 97% / 81% / ±7.0 | 100% / 100% / 95% / ±7.7 |

- The worst-class numbers of scaled+1 (71-81%) are partly chance: a cell has 21 runs per class, and 16/21 = 76%.
- Betting holds for every class in every cell (>= 90%), but costs width: ~2.4x on EuroSAT, 1.1x on real
  photos at 0.25 Mbps, ~2x at the tightest real budgets. Report both: scaled+1 as the working bar, betting as
  the guaranteed one.

**4. Real labels by eye** (`python -m src.eval.label_check`). 150 real photos from the test days (13-19 Oct;
~20 per class, all 1 BuiltUp, seed 0), 6 contact sheets. **Checked by Claude from the sheets, not by a person.**

| Label | Photos | Agree | Disagree | Can't tell |
|---|---|---|---|---|
| Cloud | 26 | 25 | 1 (thin haze over dry hills) | 0 |
| Water | 21 | 21 | 0 | 0 |
| Cropland | 20 | 19 | 1 (thick cloud) | 0 |
| Trees | 20 | 14 | 1 (all white: cloud) | 5 (half cloud) |
| ShrubGrass | 21 | 15 | 1 (sand dunes) | 5 |
| Bare | 21 | 16 | 0 | 5 (pale: bright desert or haze) |
| SnowIce | 20 | 8 | 0 | 12 (white: snow or cloud?) |
| **All** | 150 | **119** | **4** | 27 |

- Where the eye could tell: **119/123 = 97%** (also 97% weighted to the real class mix).
- Weak spots: RGB can't separate snow from cloud (the SWIR band can: snow is dark there, cloud bright);
  photos ~40-60% cloud get their land class. Next: a person re-checks the CSV; a "partly cloudy" rule.

**5. Sentinel-2 split by location** (`python -m src.data.eurosat_geo`, then `python -m src.eval.spatial_split`).
Coordinates of all 27,000 EuroSAT photos from the GeoTIFF version (one 2 GB stream from Zenodo, only the first
4 KB of each file kept; same-name GeoTIFF and JPEG checked to be the same photo: pixel correlation 0.94-0.99 vs
~0 for another photo). Photos grouped in 10 km blocks (50 km as a check, both fixed before looking); whole blocks
go to final / tuning / training (40/20/40% of blocks), same sizes as before (100 / 50 / 100 per class), 3 seeds.

| Split | Final accuracy (mean of 3 seeds) | 95% range (one seed) | Calibration error | Final photos with a near-twin in training |
|---|---|---|---|---|
| Random (20 Oct) | 92.7% | 91.0-94.2% | 0.014 | 723 / 1,000 |
| **10 km blocks** | **93.4% ± 0.6%** | 91.3-94.4% | 0.019 | 695 / 1,000 |
| 50 km blocks | 92.5% ± 0.3% | 90.7-94.0% | 0.016 | 672 / 1,000 |

- **No neighbouring-tile leak:** separating the places doesn't lower the accuracy.
- The near-twin test (cosine > 0.95) was misleading: two thirds of test photos still have one 50 km away.
  EuroSAT's 64 px photos of one class look alike to SigLIP. "88.4% without twins" (87-92% here) just drops the
  typical, easy photos.
- Files: `experiments/spatial_split_eurosat_20261022_1328.csv`, answers `head_eurosat_siglip-base_n100_spatial_final.csv`.

**6. Jetson script** (`python -m src.eval.jetson_bench`): times image model (batch 1 and 8) + head + audit sampler,
reads the Jetson's power meter (`tegrastats`, VDD_IN) for energy per photo. MacBook M2 (MPS, no power meter):
26.9 ms/photo at batch 1, 19.1 at batch 8; head 0.004 ms; sampler 0.001 ms per photo. `experiments/bench_Darwin-arm64_20261022_1320.csv`.
