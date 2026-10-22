# Research review: weak spots, new directions, and a plan (20 Oct 2026)

Written by Claude as a "senior researcher" review. Short version is in the chat; this file has everything.

**Assumptions (no questions were asked):**
- Inputs: the repo, `paper/outline.md`, `docs/experiment_log.md`, `docs/project_review.pdf`, saved results.
- Target venue: **IGARSS 2027** (4 pages, deadline about 10 Jan 2027), as in the outline.
- Two small pilots were run on the saved models (no new training). They are **pilots, not paper results**.
  Code: `notes/pilots/pilot_compress.py` (Check A, ~2 min) and `notes/pilots/pilot_audit.py` (Check B, ~10 s).

---

## 1. Weak spots (blunt)

### Fatal unless fixed

**1. The headline number can't fail.**
The score counts a 50-byte label and a 280 KB photo as the same "1 correct" (`src/eval/run_agent.py`, `score()`).
A label is about 5,600x cheaper. So any calibrated labeller saturates: agent v4 gets 100.0 ± 0.0%.
A ± 0.0 is a ceiling, not strength. A metric where the best method scores 100% can't separate good from great.

**2. The obvious baseline is missing, and it ties you (Check A).**
"JPEG-compress every photo until the pass fits", no onboard AI. Your small head (frozen SigLIP-base +
logistic regression, trained exactly as in the paper) labels the shrunk AID FINAL photos "on the ground":

| Photo version | Mean size | Head right (ground) |
|---|---|---|
| Original AID JPEG | 280 KB | 94.3% |
| 600 px, JPEG quality 50 | 44 KB | 93.7% |
| 600 px, quality 20 | 24 KB | 93.3% |
| 224 px, quality 90 | 19 KB | 93.3% |
| 224 px, quality 40 | 6.7 KB | 87.3% |
| 160 px, quality 75 | 6.4 KB | 88.3% |
| 112 px, quality 75 | 3.6 KB | 54.0% |

At 0.25 Mbps (Delhi replay, 19 Oct), each pass picks the best setting that fits: **every photo goes down,
the head labels 93.4% right on the ground, and people can look at every photo.** Under your own rule
("full photo = correct") this scores **100% with 0 wrong labels: it ties or beats agent v4** (99.3 ± 0.4% / 8 ± 4
wrong at 0.25 Mbps) with zero onboard compute.
- "Send all raw" sends 280 KB high-quality JPEGs. No operator downlinks uncompressed like that.
- SigLIP-base looks at 224x224 anyway, so a 19 KB thumbnail gives the ground exactly what the onboard head saw.

**3. The orbit is decoration.**
Each photo slot gets a random dataset photo (`src/orbit/replay.py:86`). The orbit only sets how many photos
and how many bytes. There is no link between place and content: no sea, no night, no clouds (about 67% of
Earth is cloud-covered on average, King et al. 2013), no repeat visits, no rare events. `SKIP` is never used.
A reviewer will call "real TLE data" window dressing.

**4. The budget regime is hand-picked.**
The log says 1 Mbps was chosen because "at 100 Mbps everything fits". Sentinel-2's own X-band link is about
560 Mbps (2 x 280 Mbps in ESA design material; sources differ slightly). EuroSAT tiles (~3 KB JPEGs) are
"counted as 280 KB". The only number that matters is the ratio **R = data taken ÷ data you can send**.
Your results live at R ≈ 1.7 (1 Mbps) to 6.7 (0.25 Mbps), exactly where plain compression is enough (point 2).
Check A suggests compression holds up to roughly 40x (87-88% at 6-7 KB) and collapses by about 80x (54% at 3.6 KB).

### Serious

**5. Balanced datasets hide the real problem.** AID and EuroSAT have equal photos per class. Real scenes are
mostly a few classes, and the valuable things are rare. The "interesting" set is 21 of 30 classes = 69% of
photos (754 of 1,088). That's not a priority list; it's most of the data.

**6. The "agent" and "VLM" story no longer matches your own results.** Best system = logistic regression on frozen
SigLIP. Gemma loses by 30+ points. The planner is a greedy knapsack. Reviewers will ask where the agent is.
Supervised head vs zero-shot VLM is unfair (you say so); the fair zero-shot rival (SigLIP zero-shot with
class-name prompts) is missing.

**7. The idea space is crowded.** NAVI-Orbital already downlinks a text summary of every image and lets
operators request full images (their introduction and discussion). SpaceVerse (Fudan, ACM MM 2025) splits work between satellite
and ground with a confidence network. ImageSat proposes Gemma-3n prioritisation. "Label everything, send unsure
ones in full" = selective classification (Geifman & El-Yaniv 2017) + a knapsack. The novelty margin is thin.

**8. Small n behind some claims.** "More accurate ≠ more trustworthy" rests on 5 vs 13 errors among 300 photos.
Replay error bars come from reshuffling the same 300 photos (~3.6 uses each), so they understate uncertainty.
The EuroSAT near-twin leak is still open (88.4% without twins).

### The assumption nobody questioned

**9. The downlink choice is deterministic.** "Send the least-sure photos in full" means confident photos are
never checked, so you can never measure how wrong they are. Your own data shows both ways this bites:
a new camera (AID head on Sentinel-2: 33%) and confident mistakes (the 400M head's 5 errors were all > 96% sure).
For the ground this is "missing not at random" data (Rubin 1976): what you didn't see depends on why you didn't
send it, so no ground-side math can repair it afterwards.

---

## 2. Three directions a reviewer would not expect

### A. "Audit, don't trust": make the downlink a random sample with known chances
- **Borrowed from:** survey sampling (Horvitz-Thompson weights, 1952), election risk-limiting audits
  (Lindeman & Stark 2012), prediction-powered inference (PPI, Angelopoulos et al. 2023).
- **Idea:** send the onboard label (or class chances) for every tile. Spend the rest of the pass on full tiles
  picked **at random with known chances**: unsure tiles are more likely, but every tile keeps a minimum chance.
  The ground corrects the labels using the audited tiles. Result: numbers about Earth with honest error bars,
  whether the onboard model is right or wrong.
- **Contradicts the current approach:** deterministic "least sure first" (v2-v4) and the per-photo score.
  The goal becomes "true numbers with an honest error bar per MB", not "labels that look correct".

### B. "Pull, don't push": the ground decides, the satellite only describes (most direct contradiction)
- **Borrowed from:** video analytics. DDS (SIGCOMM 2020): the camera sends a cheap stream first; the server's
  model asks for the regions it needs in high quality.
- **Idea:** at the start of a pass the satellite sends a tiny "manifest": one embedding per tile (768 numbers,
  ~1.5 KB; much less after compression). The ground runs any model or new user query, compares with its archive,
  and requests the tiles it wants **within the same pass**. Nothing to retrain or certify in orbit. Bonus: tiles
  you never download stay searchable on the ground.
- **Risks:** NAVI already does this by hand with text; Sim (2026) sends embeddings the other way (ground to
  satellite). The same-pass round trip depends on the operator's ground system: **speculative** until checked.

### C. "Send surprise, not labels": value = what the ground could not predict
- **Borrowed from:** predictive coding in neuroscience (Rao & Ballard 1999) and coding with side information
  (Wyner & Ziv 1976).
- **Idea:** Sentinel-2 sees the same place every few days. A tile of unchanged forest is worth about zero even if
  labelled perfectly; a new flood is worth a lot. Rank tiles by how far their embedding moved from the last view
  of that place.
- **Risks:** Earth+ (UChicago + Microsoft, 2025) already sends only changed areas against a reference uploaded
  from the ground (3.3x less downlink). Your twist would be meaning-level change, or using the ground archive with
  no upload at all (learned Wyner-Ziv compressors exist: Özyılkan, Ballé & Erkip 2023): **research-grade,
  speculative** before January. Needs time series with change labels (OSCD has only 24 image pairs).

---

## 3. The pick: A, "audited downlink"

**Why (plain words):**
- It fixes the worst weak spot. The question becomes one people pay for: "how much farmland, forest, water and
  built-up did we see, and how sure are we?" (crop-area statistics, forest reporting, flood extent).
  Olofsson et al. 2014 is the standard for this kind of estimate, and IGARSS reviewers know it.
- It turns your failures into the motivation: the 33% camera-shift result and the confident mistakes are exactly
  what deterministic triage can't see.
- It reuses everything: heads, calibration, replay, label curve.
- It's cheap: no new training, simulations run in seconds.
- Pilot B already shows the problem is real and the fix works, with honest costs.

**Honest novelty check:** the statistics are known (survey sampling, PPI, active inference). What's new: choosing
the sample **onboard under per-pass byte budgets**, showing that today's onboard-AI designs **silently fail** for
statistics, the **"self-audit share"** as a design knob, and audits doubling as training labels. That's enough for
IGARSS or IEEE GRSL. For an ML venue it's incremental unless you add the two harder pieces: finite-sample error bars
for rare classes under per-pass designs, and a minimax rule for the floor.

### Pilot B: what happened

Setup: real Sentinel-2 tiles (EuroSAT FINAL 1,000), your onboard heads, 1,088 tiles a day with **assumed**
India-like class shares (crop 30%, water 18%, forest 12%, grass 12%, built-up 8%, rare classes 2-4%),
1,000 simulated days, one pooled budget per day. Task: estimate each class's share of the day.
Three onboard models: your EuroSAT head (92.7%), the AID head on Sentinel-2 (27.9%, real camera shift),
and your head with a **silent fault** (30% of AnnualCrop labelled PermanentCrop, exactly as confident as its
correct answers).

**0.25 Mbps (164 full tiles fit), typical error in percentage points (RMSE):**

| Scenario | Labels only | Triage (your v4) | Random audit (floor 0.5) | Raw random |
|---|---|---|---|---|
| Good head (92.7%) | 0.8 | **0.24** | 0.45 | 2.1 |
| Wrong camera (AID head on S2) | 10.6 (one class off by 25) | 8.8 (off by 21) | 2.3 | 2.1 |
| Silent fault | 4.4 (off by 10) | 3.7 (off by 8) | 1.3 | 2.1 |
| Does the 90% error bar hold? | 0% to 95% | 1% to 100% | 86-87% | 88-89% |

What it says:
- **Good model:** your triage is the most accurate, but it can't tell you that.
- **Wrong model:** triage and labels give confidently wrong numbers, and nothing on the ground shows it.
  Audits stay honest and are about as good as raw random.
- **Cost of the insurance:** 0.45 vs 0.24 points in the good case (both far below 1 point). At 1 Mbps: 0.07 vs 0.
- **Problems it exposed (real technical work, good for the paper):**
  - Error bars run a little narrow, worst for rare classes (79-88% for the audits, down to 70% for
    stratified, instead of 90%).
  - A fancier estimator (GREG regression on all class chances) broke with only 164 audits (errors up to 23 points).
  - A per-class trust weight (PPI++ idea) made audits never worse than raw random with a broken model, but cost
    some accuracy with a good one.

Full tables: Appendix.

---

## 4. Making it publishable

**Working title:** *How Much Should a Satellite Check Itself? Audited Downlink for Trustworthy Earth-Observation
Statistics.* (Drop "agent" from the title.)

**Core hypothesis.** If the satellite picks its full tiles as a random sample with known chances (more for unsure
tiles, a floor for all), the ground gets class-share estimates whose 90% error bars hold no matter how good or bad
the onboard model is. With a good model this costs only a fraction of a point compared with deterministic triage;
with a bad model it is no worse than random raw sampling. Deterministic triage and labels-only fail silently under
camera shift and confident faults.

**What's new compared with existing work:**

| Work | What it does | What's missing |
|---|---|---|
| NAVI-Orbital (2026), SpaceVerse (2025), CubeSat triage, your v1-v4 | onboard model picks what to send | deterministic: errors in unsent data can't be measured; no error bars for products |
| Kodan (2023), Earth+ (2025), FOOL (2025) | send fewer bytes (filter / change / compress) | no guarantee for numbers made from the data |
| Olofsson et al. 2014; Lu, Kluger, Bates & Wang (PPI for remote-sensing maps) | honest estimates from a map + a reference sample | sample collected on the ground after the map; cost = field work, not downlink bytes |
| Active statistical inference (Zrnic & Candès, ICML 2024) | uncertainty-weighted labelling with valid error bars | generic; expected budget, not a hard one; no onboard per-pass setting; no shift/fault tests |
| Bafarassat (2026) | risk-controlled downlink *scheduling* | guarantees the link, not the science products |
| Bhattacharjee & Jung (2024) | onboard out-of-calibration detection (conformal prediction) | flags drift onboard; no ground estimates under a byte budget |

**Experiments (in order):**
1. **Fix the testbed first.**
   a. Add compress-to-fit (Check A) and "raw random" baselines.
   b. Sweep R = data taken ÷ data sent from 2 to 1,000, not only Mbps. Prediction: onboard AI matters only above
      some R* (Check A hints R* ≈ 40-80 for these tiles; **to be measured**).
   c. Realistic class shares, not balanced ones.
   d. Spatially separate EuroSAT split. Better: real Sentinel-2 chips at the real places on the replayed track,
      with real clouds, labelled by ESA WorldCover (both on AWS open data; check your network can reach them).
      This kills "the orbit is decoration". About 1-2 weeks.
2. **Main table:** class-share estimates per day at 0.25-4 Mbps, Delhi / Bengaluru / Svalbard: coverage,
   error-bar width, RMSE, bytes needed for ±1 point.
3. **Silent-failure suite:** wrong camera (AID→S2; Europe→India if 1d is built), injected faults (confident class
   swaps; bit flips in the 30 KB head), image damage (blur, noise, stripes). Does each method's error bar hold?
4. **Self-audit trade-off:** floor from 0 to 1 → curve of good-case accuracy vs worst-case honesty. Choose the
   floor by minimax.
5. **Miss rate:** "how many rare-class tiles never came down?" Audits give an estimate with an error bar; triage
   can't answer at all.
6. **Closed loop:** new camera, zero labels. Audited tiles become labels; retrain the 30 KB head each pass using
   only past passes (so error bars stay valid); upload it. Uses the label curve (5 labels/class → 79-90%).
7. **Jetson:** time and energy for encoder + head + sampler (the sampler costs almost nothing).

**Baselines:** raw in time order (current), raw random, compress-to-fit + ground model, labels only (NAVI-style),
triage (your v4, the strongest current practice), Olofsson stratified, uniform PPI, active inference with an
expected budget (unadapted).

**Ablations:** floor share; how chances are set (uniform / 1-p / square root of Brier score); temperature
calibration on/off; estimator (difference / tuned / stratified / ridge-GREG); error-bar method (normal vs
finite-sample betting intervals, Waudby-Smith & Ramdas 2024); per-pass vs pooled budget; send top-1 label vs full
chances vs embedding; equal vs real tile sizes.

**Metrics:** (1) coverage of 90% error bars, overall and for the rarest class (target ≥ about 88%);
(2) error-bar width at fixed bytes; (3) bytes needed for ±1 point; (4) RMSE and worst bias;
(5) worst-case coverage over the failure suite; (6) share of budget spent on audits, and "interesting" tiles still
delivered (the cost of the insurance).

**What would prove the idea wrong:**
- **F1:** On real shifts (new region, new season) and realistic faults, triage's bias stays inside its own error
  bar. Then current practice is good enough and the motivation dies.
- **F2:** Honest error bars for rare classes are wider than the class itself (e.g. ±3 points on a 2% class) at
  realistic budgets. Then the guarantee is useless where it matters.
- **F3:** In the good case, the audit's error is clearly bigger than triage's at the precision users need
  (say ±1 point). Then operators won't pay for the insurance. (Pilot: 0.45 vs 0.24 points; passes here, must hold
  on real data.)
- **F4:** Compress-to-fit + ground model + ground-side audit (no onboard AI) matches the onboard design at the R of
  real missions. Then the onboard part adds nothing and this is a ground-statistics paper.

**Timeline to 10 Jan 2027 (about 11 weeks):**
- Weeks 1-2: testbed fixes (1a-1d), R sweep.
- Weeks 3-4: estimators, finite-sample error bars, failure suite, floor curve.
- Weeks 5-6: real Sentinel-2 chips + WorldCover; closed loop.
- Weeks 7-8: write. Then Jetson if a board appears, feedback, submit.
- Internship demo (end of Nov): "the satellite reports its own error": a live chart of estimate ± error bar,
  an injected fault, and the audit catching it. Ask interviewers what their R is.

**Keep from the current work:** triage v4 is the main baseline; calibration sets the audit chances; the label curve
powers the closed loop; the replay gives the budgets. "More accurate ≠ more trustworthy" becomes motivation, not a claim.

---

## Appendix: Pilot B full numbers

RMSE and worst bias in percentage points. "Holds" = how often the 90% interval contained the truth
(all classes / rare classes = Highway, Industrial, River). "vs raw" = how many times more bytes raw random would
need for the same squared error.

| Scenario | Rate | Method | RMSE | Worst bias | Holds (all / rare) | vs raw |
|---|---|---|---|---|---|---|
| Good head | 0.25 | raw random | 2.05 | 0.11 | 89% / 86% | 1.0x |
| | | labels only | 0.76 | 1.60 | 95% / 94% | 7.2x |
| | | triage (v4) | 0.24 | 0.41 | 100% / 100% | 71.7x |
| | | audit uniform | 0.67 | 0.03 | 86% / 85% | 9.4x |
| | | active, floor 0.1 | 0.40 | 0.02 | 87% / 88% | 25.9x |
| | | active, floor 0.5 | 0.45 | 0.02 | 86% / 87% | 20.9x |
| | | active tuned, floor 0.5 | 0.51 | 0.17 | 83% / 79% | 15.8x |
| | | stratified (Olofsson) | 0.75 | 0.07 | 74% / 72% | 7.5x |
| Good head | 1 | raw random | 0.74 | 0.03 | 90% / 90% | 1.0x |
| | | triage (v4) | 0.00 | 0.00 | 100% / 100% | exact |
| | | audit uniform | 0.23 | 0.02 | 90% / 89% | 10.0x |
| | | active, floor 0.5 | 0.07 | 0.00 | 87% / 87% | 107x |
| | | stratified | 0.27 | 0.03 | 89% / 88% | 7.7x |
| AID head on S2 | 0.25 | raw random | 2.06 | 0.16 | 88% / 85% | 1.0x |
| | | labels only | 10.60 | 25.19 | 0% / 0% | 0.0x |
| | | triage (v4) | 8.84 | 20.72 | 1% / 1% | 0.1x |
| | | audit uniform | 2.11 | 0.21 | 88% / 86% | 1.0x |
| | | active, floor 0.5 | 2.28 | 0.09 | 87% / 83% | 0.8x |
| | | active tuned, floor 0.5 | 2.04 | 0.09 | 86% / 82% | 1.0x |
| | | stratified | 1.94 | 0.08 | 87% / 84% | 1.1x |
| AID head on S2 | 1 | labels only | 10.58 | 25.14 | 0% / 0% | 0.0x |
| | | triage (v4) | 4.64 | 12.95 | 34% / 43% | 0.0x |
| | | active, floor 0.5 | 0.83 | 0.03 | 90% / 88% | 0.8x |
| | | stratified | 0.68 | 0.04 | 90% / 90% | 1.2x |
| Silent fault | 0.25 | labels only | 4.35 | 9.73 | 78% / 94% | 0.2x |
| | | triage (v4) | 3.71 | 8.28 | 80% / 100% | 0.3x |
| | | audit uniform | 1.08 | 0.17 | 86% / 85% | 3.7x |
| | | active, floor 0.5 | 1.27 | 0.05 | 87% / 88% | 2.7x |
| | | active tuned, floor 0.5 | 1.08 | 0.15 | 84% / 80% | 3.7x |
| | | stratified | 0.91 | 0.04 | 73% / 70% | 5.2x |
| Silent fault | 1 | triage (v4) | 2.18 | 4.83 | 80% / 100% | 0.1x |
| | | active, floor 0.5 | 0.47 | 0.05 | 86% / 88% | 2.4x |
| | | stratified | 0.31 | 0.03 | 88% / 88% | 5.6x |

Not in the table: GREG regression (all 10 class chances as inputs) was unstable at 164 audits (RMSE up to
23 points with the shifted head), so it was dropped from the pilot.

---

## Citations

**Checked online or read locally on 20 Oct 2026:**
- NAVI-Orbital, arXiv 2606.18271 (local PDF) · FOOL, Furutanpey et al. (local PDF) · On-Orbit Space AI survey,
  arXiv 2604.16518 (local PDF)
- [Earth+](https://knowledge.uchicago.edu/record/14843) (UChicago + Microsoft Research, 2025)
- [Lu, Kluger, Bates, Wang, arXiv 2407.13659](https://arxiv.org/abs/2407.13659) (PPI for remote-sensing maps;
  v1 title "Quantifying uncertainty in area and regression coefficient estimation from remote sensing maps")
- [Zrnic & Candès, Active Statistical Inference, ICML 2024](https://proceedings.mlr.press/v235/zrnic24a.html)
- [Angelopoulos et al., Prediction-Powered Inference, arXiv 2301.09633](https://arxiv.org/abs/2301.09633)
  (I believe it appeared in Science in 2023; my search didn't confirm the journal, so check)
- [PPI++, arXiv 2311.01453](https://arxiv.org/abs/2311.01453)
- [Olofsson et al. 2014, Remote Sensing of Environment 148:42-57](https://eprints.nottingham.ac.uk/44846/)
- [DDS, Du et al., SIGCOMM 2020](https://par.nsf.gov/biblio/10174381)
- [Sim, Embedding-Only Uplink, arXiv 2604.03301](https://arxiv.org/abs/2604.03301)
- [SpaceVerse, arXiv 2507.05731](https://arxiv.org/abs/2507.05731) (ACM MM 2025)
- [Bafarassat, Mean Gain Is Not a Guarantee, arXiv 2608.08635](https://arxiv.org/abs/2608.08635)
- [Bhattacharjee & Jung, arXiv 2405.02634](https://arxiv.org/abs/2405.02634) (venue not confirmed)
- [King et al. 2013, IEEE TGRS](https://modis-atmosphere.gsfc.nasa.gov/sites/default/files/ModAtmo/King_et_al.2013.pdf) (global cloud fraction ~67%)
- [Lindeman & Stark 2012, IEEE Security & Privacy](https://www.stat.berkeley.edu/~stark/Preprints/gentle12.pdf)
- [Waudby-Smith & Ramdas, JRSS-B 2024, arXiv 2010.09686](https://arxiv.org/abs/2010.09686)
- [Özyılkan, Ballé, Erkip, arXiv 2305.04380](https://arxiv.org/abs/2305.04380)
- [OSCD, Daudt et al., IGARSS 2018](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection)
- Sentinel-2 X-band 2 x 280 Mbps: [ESA Sentinel-2 mass memory presentation](https://indico.esa.int/event/67/contributions/3069/attachments/2476/2849/1055_-_Sentinel_2_Mass_Memory.pdf) (sources differ slightly)
- Kodan (ASPLOS 2023): title and venue confirmed via [NSF PAR](https://par.nsf.gov/biblio/10424445); authors not re-checked

**From memory, not re-checked today (well known, but verify before citing):**
Horvitz & Thompson 1952 (JASA) · Rubin 1976, "Inference and missing data" (Biometrika) · Geifman & El-Yaniv 2017,
selective classification (NeurIPS) · Rao & Ballard 1999, predictive coding (Nature Neuroscience) · Wyner & Ziv 1976
(IEEE Trans. Information Theory) · Särndal, Swensson & Wretman 1992, *Model Assisted Survey Sampling* (book) ·
Rosén 1997, Pareto πps sampling (J. Statistical Planning and Inference).
