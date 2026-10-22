# Paper outline v2: the audited downlink

**Target:** IGARSS 2027, 4 pages, deadline about 10 January 2027 (check the official site). Post on arXiv.
Backup: IEEE GRSL (letter), or an ML-for-remote-sensing workshop.
The first outline (`paper/outline.md`, calibrated triage) stays as is: its agent v4 is now the main **baseline**.

Every number below comes from a file in `results/` (one folder per experiment) and is explained in `archive/experiment_log.md`
(section "Audited downlink", 20 Oct 2026). **TODO** = still to do.

---

## Titles (pick one)
1. *How Much Should a Satellite Check Itself? Audited Downlink for Trustworthy Earth-Observation Statistics*
2. *Audit, Don't Trust: Honest Land-Cover Statistics from Onboard AI That May Be Wrong*
3. *Random Checks Make Onboard AI Honest: Downlink as a Probability Sample*

## Abstract (about 150 words, write last)
- Problem: satellites take more photos than passes can send, so onboard AI decides what goes down
  (NAVI-Orbital 2026; Kodan; SpaceVerse). Today's designs are **deterministic**: send the photos the
  model is least sure about. The photos it is sure about are never checked.
- Consequence (shown): when the onboard model is wrong in a confident way (new camera, radiation bit flip,
  damaged optics), the ground's numbers are off by up to 25 points and nothing on the ground shows it.
- Idea: make the downlink a **probability sample**. Label every photo; send a random sample of full photos
  with known chances (unsure photos more likely, every photo keeps a floor). The ground corrects the labels
  with the checked photos and gets an **honest error bar whatever the onboard model is like**.
- Results: honest bars in every failure we tried (89-99% of 90% bars hold), at almost no cost when the model
  is healthy (floor 0.5); checks sent as thumbnails give 4-5x narrower bars than compression at high data
  ratios; a new camera with zero labels learns from its own audits within days.

## 1. Introduction (0.5 page)
- Downlink bottleneck (Delhi 1.4% contact; our orbit sim). Onboard AI is real (NAVI-Orbital, Phi-Sat, Kodan).
- The survey's top blocker is **assurance**: "defensible evidence that unsafe behaviours will be detected and
  bounded" (On-Orbit Space AI survey, Sec. 6B).
- The gap: onboard triage decides deterministically, so its errors on unsent data are **unmeasurable**
  (missing-not-at-random data, Rubin 1976).
- **Contributions:**
  1. *Self-checking audited downlink*: onboard labels + a byte-budgeted random audit with known chances
     (floor); ground estimators with error bars that hold even for rare classes (the "scaled+1" bar); and a
     self-check that switches to a safe mode when the checked mistakes show the model is broken.
  2. A test bed that is real where it matters: real orbit (TLE) and passes, a **replay of what Sentinel-2A
     really photographed** on 13-19 Oct 2026 (clouds included), a failure suite (new camera, silent swap,
     bit flips, damaged camera), and a fair compression baseline.
  3. Findings: (a) triage and labels-only fail silently; audits stay honest; (b) a floor of 0.5 is nearly
     free insurance; (c) below R ~ 30 (data taken / data sent) plain compression is enough, above it onboard
     labels + thumbnail checks win 4-5x; (d) audits double as training labels (closed loop).

## 2. Related work (0.4 page)
- Onboard selection: NAVI-Orbital (arXiv 2606.18271: text summaries down, operators request photos by hand),
  SpaceVerse (ACM MM 2025, satellite-ground split with a confidence network), Kodan (ASPLOS 2023), Earth+
  (2025, change-only downlink), FOOL (2025, feature compression), CubeSat CNN triage (arXiv 2408.14865).
  None gives an error bar for the products made from the data.
- Statistics with ML predictions: PPI (Angelopoulos et al. 2023), PPI++ (2023), active statistical inference
  (Zrnic & Candès, ICML 2024); for remote sensing, Olofsson et al. 2014 (good-practice area estimation) and
  PPI on maps (Lu, Kluger, Bates & Wang, arXiv 2407.13659). All sample on the ground after the map exists;
  the cost there is field work, not downlink bytes, and none runs onboard per pass.
- Guarantees near the downlink: risk-controlled link scheduling (Bafarassat 2026, arXiv 2608.08635);
  onboard out-of-calibration detection (Bhattacharjee & Jung, arXiv 2405.02634). Neither covers the products.
- **Our difference:** the sample is chosen onboard, under hard per-pass byte budgets, with chances from the
  onboard model and a floor for confident mistakes; evaluated against silent failures.

## 3. Method (0.8 page)
### 3.1 Setting
- Passes p = 1..P, each with N_p photos and a byte budget B_p. A full photo costs c bytes, a label + chances
  K + 50 bytes. Product: class shares of the day's photos, with a 90% error bar.
### 3.2 Onboard policy (`src/agent/audit.py`)
- Label + chances p_i for every photo. Audit size n_p = (B_p - N_p (K+50)) / c.
- Chance of a check: pi_i = clip(s * sqrt(1 - sum_k p_ik^2), floor * n_p / N_p, 1), s set so sum pi_i = n_p
  (unsure photos more likely; the floor keeps confident ones checkable).
- Pick exactly n_p photos with chances pi_i (Pareto sampling, Rosén 1997); send them with pi_i.
### 3.3 Ground estimate (`src/eval/estimate.py`)
- Difference (Horvitz-Thompson) estimate per class: theta_k = mean_i p_ik + (1/N) sum_{checked} (y_ik - p_ik) / pi_i.
  Unbiased for any onboard model (design-based).
- Variance: Rosén's formula per pass, passes combined by size.
- **"scaled+1" error bar** (needed for rare classes): the variance is never smaller than (a) the model's own
  predicted variance, scaled by how much bigger the checked mistakes were than predicted (pooled over classes
  and passes), nor (b) the "plus-one" rule: one unseen checked photo might belong to the class.
### 3.4 Self-check and safe mode (`src/agent/self_check.py`)
- Ratio r = (sum over checks so far of |y - p|^2 / pi) / (sum of sum_k p(1-p) / pi). Honest model: r ~ 1.
- r > 1.5 (chosen on EuroSAT only): next passes use uniform checks (floor 1) and chances recalibrated from past
  checks (table: predicted class -> share of each true class). Uses only earlier passes -> still unbiased.
### 3.5 Closed loop (`src/eval/audit_loop.py`)
- Checked photos become training labels; the 30 KB head is retrained after every pass and used one pass
  later. Each pass's head is independent of that pass's sampling, so the error bars stay honest.

## 4. Experiments (1.6 pages)
### Data
- EuroSAT replay: real Sentinel-2 photos (FINAL + FRESH, 2,000) in the orbit replay, India-like class
  shares (assumed), 7 days x 3 stations x 3 photo orders, 0.05-2 Mbps.
- **Real replay:** Sentinel-2A photos at the places and times the satellite really took them, 6-19 Oct 2026
  (`src/data/s2_real.py`); truth = the image's cloud map, else ESA WorldCover 2021. Head trained on 6-12 Oct,
  tested on 13-19 Oct. 3,527 real photos (cloud 46%, water 17%, built-up 0.1%); sim and real image times
  match within 0-7 s.
- Failure suite (`src/model/faults.py`): new camera (AID head, 28% right), silent swap (30% of one class,
  as confident as right answers), radiation bit flips (8 bits, accuracy 93 -> 84%), damaged camera
  (blur / noise / stripes: 62 / 43 / 50% right).

### Table 1: can the ground trust the numbers? (EuroSAT replay, 0.25 Mbps, 63 days per cell)
Typical error (RMSE, percentage points) / how often the 90% error bar held. File: `audit_summary_eurosat_20261020_2023.csv`.

| Onboard model | Labels only | Triage (v4) | Raw random | Stratified | **Audit (floor 0.5)** |
|---|---|---|---|---|---|
| Healthy head (92.6% right) | 0.69 / 96% | **0.26** / 100% | 2.18 / 91% | 0.79 / 99% | 0.47 / 98% |
| New camera (28.0%) | 10.75 / 0% | 7.99 / 8% | 2.18 / 91% | 2.42 / 88% | 2.69 / 86% |
| Silent swap (89.8%) | 4.39 / 79% | 3.05 / 80% | 2.18 / 91% | 1.05 / 97% | 1.22 / 97% |
| Bit flips (84.2%) | 1.44 / 73% | 1.57 / 80% | 2.18 / 91% | 1.31 / 96% | 1.33 / 95% |
| Damaged camera: blur (62.2%) | 5.09 / 26% | 2.90 / 57% | 2.18 / 91% | 1.68 / 93% | 1.22 / 94% |
| Damaged camera: noise (43.0%) | 11.59 / 7% | 7.78 / 18% | 2.18 / 91% | 2.15 / 92% | 1.80 / 95% |
| Damaged camera: stripes (50.0%) | 10.41 / 16% | 6.18 / 29% | 2.18 / 91% | 2.30 / 90% | 1.56 / 95% |
(From `python -m src.eval.audit_tables`. Labels/triage bars are "pretend every label is right". Rare-class
coverage is in the experiment log.)

### Table 2: the same on what Sentinel-2A really photographed (13-19 Oct 2026, head trained 6-12 Oct, 0.05 Mbps)
File: `audit_summary_real_20261020_2023.csv`. Typical error (points) / 90% bar held.

| Onboard model | Labels only | Triage (v4) | Raw random | Stratified | Audit (floor 0.5) | **Self-checking audit** |
|---|---|---|---|---|---|---|
| Healthy head (82.3% right) | 1.89 / 80% | 1.54 / 84% | 5.89 / 96% | 4.08 / 99% | 3.53 / 99% | 3.56 / 98% |
| New place: EuroSAT head (20.0%) | 19.54 / 9% | 16.63 / 20% | 5.89 / 96% | 7.22 / 95% | 8.30 / 97% | **5.34 / 98%** |
| Silent swap (81.2%) | 1.69 / 82% | 1.51 / 84% | 5.89 / 96% | 4.38 / 99% | 3.76 / 99% | 3.64 / 98% |
| Bit flips (57.5%) | 10.63 / 27% | 10.78 / 27% | 5.89 / 96% | 6.22 / 95% | 8.12 / 91% | **5.50 / 97%** |
| Damaged camera: blur (77.4%) | 4.52 / 51% | 3.28 / 67% | 5.89 / 96% | 5.18 / 98% | 3.71 / 97% | 3.56 / 97% |
| Damaged camera: noise (77.0%) | 3.96 / 51% | 2.67 / 71% | 5.89 / 96% | 5.12 / 100% | 4.21 / 97% | 3.97 / 97% |
- Key real-data points: even the healthy model's labels mislead (bars 80%, rare 60%). The plain audit holds
  91-99% but is noisier than raw random with a badly broken model (8.3 vs 5.9); the self-check (rules fixed on
  EuroSAT, tested once here) fixes that (5.3-5.5) and holds 97-98%. Healthy: <= 3% of passes switched.
- 0.25 Mbps: plain audit 1.0-3.5 points, held 85-95% (bit flips 85%); self-check bit flips 1.73 / 96%,
  new place 1.85 / 95%.
- Self-check costs: EuroSAT damaged camera +0.1-0.3 points; 0.02 Mbps false switches (swap 9.1 vs 5.7).

### Figure 1: `docs/figures/audit_honesty_real.png` (Table 2 as a dot plot; `docs/figures/audit_honesty.png` for Table 1).
### Figure (closed loop): `docs/figures/audit_loop_real.png`: from zero labels, 46% -> 83% right in 4 days (215 checked
  photos), bars held 96%; launch head + triage held 12%.
### Figure 2: `docs/figures/audit_rsweep.png`: when onboard AI beats compression (same bytes, same human checks)
- Compress-to-fit is exact for statistics up to R ~ 14 and shows every photo up to R ~ 6: onboard AI adds
  nothing there (honest negative result). At R = 140-560 the thumbnail audit's bar is ±0.4-1.1 vs ±1.8-4.5.
### Figure 3 (if space): `docs/figures/audit_floor.png` (floor 0.5: healthy error 0.47 vs 0.41-0.50 at lower floors;
  bit flips: bars hold 95% vs 86-90% at floor <= 0.1) or `docs/figures/audit_loop.png`.

## 5. Discussion and limits (0.3 page)
- **Two kinds of error bar (22 Oct).** "scaled+1" (bell-curve approximation) holds 94-98% on average with the
  self-check, but a single class can drop to 71-81% in a 21-run cell. The **betting** bar (Waudby-Smith & Ramdas,
  hedged capital, c = 0.75 chosen on made-up data) needs no approximation: held 100% on average and >= 90% for
  every single class in every cell (EuroSAT and real). Price: 2.4x wider on EuroSAT (0.25 Mbps self-check: ±5.9
  vs ±2.5 points); on real photos at 0.25 Mbps only 1.1x (±7.7 vs ±7.0), at the tightest budgets ~2x.
  Assumption: checks treated as draws with replacement (ours are without, less spread -> errs wide).
  Files: `audit_summary_{eurosat,real}_20261022_13*.csv` (older columns identical to the 20 Oct run).
- Truth for real photos = WorldCover 2021 + cloud map (2021 vs 2026). **Eye check (22 Oct, 150 photos, by an
  AI from contact sheets; `results/label_check_real.csv`):** 119/123 right where the eye could tell (97%);
  can't tell: snow vs cloud in RGB (12 of 20 SnowIce), half-cloudy photos labelled as land (8). Wrong: 4.
  TODO: a person re-checks; SWIR band would separate snow from cloud.
- People reading checked photos (or thumbnails) is assumed correct; at scale a stronger ground model could do it.
- Per-pass budgets are modelled; uplink of the 30 KB head is assumed free; no Jetson numbers yet.
- EuroSAT replay photos repeat across days (2,000-photo pool): closed-loop accuracy is inflated there; the real
  replay has no repeats.

## 6. Conclusion (3-4 sentences)
- A satellite that only sends what it is unsure about can never show how wrong it is when it is sure.
- Making the downlink a random sample with known chances costs almost nothing and makes the ground's numbers
  honest, whatever happens to the onboard model.

---

## TODO before submitting (in order)
1. ~~Real replay numbers (Table 2) and the real closed loop~~ **done** (20 Oct).
2. ~~Fall back to uniform checks when the checks show the model is broken~~ **done**: the self-check (20 Oct).
   ~~Finite-sample (betting) error bars~~ **done** (22 Oct): valid for every class, 1.1-2.4x wider. Report both.
3. Jetson Orin Nano: time and energy of encoder + head + sampler. Script ready: `python -m src.eval.jetson_bench`
   (Mac M2: 19 ms/photo at batch 8, power n/a). Needs the board.
3a. ~~EuroSAT split by location~~ **done** (22 Oct): 10 km blocks 93.4% ± 0.6% vs random 92.7%: no leak.
3b. ~~Truth labels by eye~~ first pass done by an AI (97%); a person should re-check `results/label_check_real.csv`.
4. Write sections 3-4 first (IEEE template), then 1, 2, 5, 6, abstract last.
5. Ask a statistician (or teacher) to check Section 3.3.
