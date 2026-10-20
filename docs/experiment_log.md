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
