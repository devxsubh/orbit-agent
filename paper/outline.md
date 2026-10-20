# Paper outline

**Target:** IGARSS 2027, 4-page full paper. Deadline about **10 January 2027** (check the official site;
listings say 10 or 11 Jan). Also post on arXiv. Backup: CVPR EarthVision 2027 (deadline usually early March).

Every number below comes from a file in `experiments/` and is explained in `docs/experiment_log.md`.
**TODO** = still to do before submitting.

---

## Title (pick one)
1. *What Should a Satellite Send Down? Calibrated Onboard Triage Under a Downlink Budget*
2. *Frozen Features, Honest Chances: Label-Efficient Onboard Triage for Downlink-Limited Earth Observation*
3. *Beyond Onboard Captioning: Calibrated Decision Heads for Satellite Downlink Triage*

## Abstract (about 150 words, write last)
- Problem: satellites take far more photos than they can send down; a ground pass lasts minutes.
- Gap: onboard vision-language models (NAVI-Orbital, 2026) caption photos, but don't decide what to send,
  and their confidence is not honest.
- What we do: an onboard agent that sends a label for every photo and spends the rest of the pass on the
  full photos it is least sure about, using the *chance a label is right*. Labeller is swappable:
  zero-shot Gemma 3, a frozen image model + tiny trained head, or a tiny CNN.
- Main results: at 1 Mbps, 100% of the information reaches the ground correctly with 0 wrong labels vs 60%
  for raw sending (Gemma agent: 92%, 90 wrong). On real Sentinel-2 photos the head reaches 92.8% from
  1,000 labels; with 5 labels per class it beats a tiny CNN by 35 points.
- Takeaway: honest chances matter more than raw accuracy; frozen head for new missions, tiny CNN once labels are many.

## 1. Introduction (about 0.5 page)
- Downlink bottleneck: Sentinel-2A sees Delhi 1.4% of the time (2 passes a day, ~10 min each). [our orbit sim]
- Onboard AI is now real: NAVI-Orbital ran Gemma 3 in orbit (Apr 2026); Phi-Sat-1/2; Kodan cut downlink 89-97%.
- Open problems named by the On-Orbit Space AI survey: what to send down, trust in orbit, power limits.
- **Contributions** (3 bullets):
  1. A downlink-aware agent: a label for every photo + full photos chosen by value per byte from the chance
     the label is right; a score that counts **wrong labels**, not only correct ones.
  2. A comparison of three onboard labellers under real orbit budgets: zero-shot VLM vs frozen encoder + head
     vs tiny CNN, including transfer to real Sentinel-2 photos.
  3. Two findings: (a) more accurate ≠ more trustworthy (a bigger head sends more wrong labels);
     (b) a label curve showing when each labeller should be used.

## 2. Related work (about 0.4 page)
- Onboard VLMs / foundation models: NAVI-Orbital (arXiv 2606.18271), DPhi Space, ImageSat Gemma-3n.
- Onboard prioritisation: 6U CubeSat CNN triage (arXiv 2408.14865), segmentation-based prioritisation
  (Remote Sensing 2024, 16(10):1729), HYPSO-1 onboard classification, onboard vessel detection (arXiv 2610.12182),
  Kodan (ASPLOS 2023), orbital edge computing (Denby & Lucia 2020).
- Compression baseline: FOOL. **TODO:** add FOOL as a baseline or explain why not.
- Calibration: temperature scaling (Guo et al. 2017); linear probes on frozen features (SigLIP, DINOv2).
- "Decision models" (TypeSafe Jev, 2026) as motivation only (blog, not peer reviewed): footnote at most.
- **Our difference:** none of these choose what to send with calibrated chances and a wrong-label-aware
  score, or compare against an onboard VLM.

## 3. Method (about 0.8 page)
### 3.1 Simulation ("a satellite on your desk")
- Real TLEs for Sentinel-2A, Delhi ground station, 24 h, photo every 30 s in daylight -> 1,088 photos.
- Budget per pass = pass time x downlink rate (0.25-4 Mbps). Unsent photos are dropped (fresh data first).
- All photos counted at the same size (280 KB, like raw tiles), so a policy can't win by picking small files.
### 3.2 Labellers
- **Gemma 3 4B**, zero-shot, 4-bit, 30-class prompt; chance from class trust + agreement of 3 shuffled asks (agent v3).
- **Decision head**: frozen SigLIP (base 90M / so400m 400M) + logistic regression; C picked on tuning by log-loss.
- **Tiny CNN**: MobileNetV3-Small (1.5M), trained end to end; best epoch + temperature on tuning.
### 3.3 Agent (one short equation)
- Label value = 2p - 1 if p >= 0.5 (plain label), else 0 ("unsure: X?" label). Full photo value = 1.
- Spend the leftover budget on the highest (1 - label value) / bytes first.
### 3.4 Score
- Per photo: full photo -> correct; right label -> correct; wrong label -> **wrong** (misleads the ground);
  "unsure" label -> no info; dropped -> nothing. Always report correct, wrong and MB together.
- **TODO:** write this as a formula.

## 4. Experiments (about 1.6 pages: 2 tables + 2 figures)
### Data and fair testing
- AID: photos 1-10/class first baseline, 11-20 tuning, **21-30 final test (used once)**, 31+ training (heads/CNN only).
- EuroSAT (real Sentinel-2): 1,000 final (incl. Gemma's 100), 500 tuning, up to 100/class training.

### Table 1: main replay result (AID, 1 Mbps; one row per policy)
| Policy | Correct info | Wrong labels |
|---|---|---|
| Send all raw | 60% | 0 |
| Caption every photo (Gemma, NAVI-style) | 63% | 404 |
| Agent v3 (Gemma + chances) | 92% | 90 |
| Agent v4, tiny CNN (600 labels) | 99% | 12 |
| **Agent v4, frozen head (600 labels)** | **100%** | **0** |
Files: `agents_delhi_final_equal_20261019_1229.csv`, `..._20261020_0947.csv`, `..._20261020_1207.csv`.
Option: Figure 1 = `docs/results.png` (all rates) instead of the table if space allows.

### Table 2: labellers (accuracy, calibration error, size, compute)
| Labeller | AID final | Sentinel-2 final | Calibration (AID) | Size |
|---|---|---|---|---|
| Gemma 3 4B, zero-shot | 63% | 29% (100 photos) | not measured (raw token chance ~100% even when wrong) | 4B |
| Frozen SigLIP-base + head | 95.7% | 92.8% | 0.036 | 90M + 30 KB |
| Frozen SigLIP so400m + head | 98.3% | 94.8% | 0.021 | 400M |
| Tiny CNN | 85.0% | 94.6% | 0.049 | 1.5M |
- Text: the 400M head's 5 mistakes are all > 96% sure -> it sends more wrong labels than the small head
  (3-7 vs 0-3). **More accurate ≠ more trustworthy.**
- Transfer: AID-trained heads on Sentinel-2 get 33-41% (no better than Gemma) -> retrain the head per camera.

### Figure 2: label curve (`docs/label_curve.png`)
- 5 labels/class: head 90% / 79%, CNN 68% / 44%. Sentinel-2 crossover at ~50/class; at 100 CNN 95% vs head 93%.

### Edge cost
- Compute per photo (published figures, approx.): CNN ~0.06 GFLOPs, SigLIP-base ~17.6, so400m ~15-20x more.
- **TODO (most important):** latency + power on a Jetson Orin Nano for all three; energy per photo.

## 5. Discussion and limits (about 0.3 page)
- Photos are stand-ins (not the true view at each spot); a full photo always counts as correct;
  photos repeat in the replay (300 / 1,000 final photos over 1,088 slots).
- Head and CNN are supervised, Gemma is zero-shot: different settings, say so.
- One satellite, one ground station, one day. **TODO:** more days, stations and seeds -> error bars on Table 1.
- No cloud handling or crops yet.

## 6. Conclusion (3-4 sentences)
- Calibrated chances, not raw accuracy, decide how much useful and trustworthy information reaches the ground.
- Start a mission with a frozen encoder + tiny head (few labels, 30 KB updates); switch to a tiny CNN once
  labels are plentiful.

---

## TODO list before submitting (in order)
1. **Jetson Orin Nano**: latency, power, energy per photo for Gemma, SigLIP heads and the CNN.
2. **Error bars for the replay**: several days, 2-3 ground stations (Delhi, Svalbard, Bengaluru), several seeds.
3. **Check NAVI-Orbital's setup** (paper 04) to compare Gemma's 63% with their 88.16% fairly.
4. Score as a formula; agent as pseudo-code (5 lines).
5. FOOL baseline or a clear reason to skip it.
6. Get the IEEE IGARSS LaTeX template; write sections 3-4 first, then 1, 2, 5, 6, abstract last.
7. Ask a teacher to read it before submitting.

## Rough timeline to 10 January 2027
| Weeks | Work |
|---|---|
| Late Oct - mid Nov | Jetson numbers (or a borrowed board), replay error bars |
| Mid Nov - early Dec | NAVI check, formula, first full draft |
| December | Feedback, rewrite, figures final |
| 1-10 January | Final check, submit, post on arXiv |
