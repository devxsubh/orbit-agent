# orbit-agent: an AI on a satellite whose numbers the ground can trust

Report, 22 October 2026 · Subham Mahapatra · [github.com/devxsubh/orbit-agent](https://github.com/devxsubh/orbit-agent) · [live demo](https://devxsubh.github.io/orbit-agent/)

> **In short.** Satellites take far more photos than they can send down, so an onboard AI must choose. Today's
> designs label every photo and send in full only what the model is unsure about, so when the model is wrong
> *and sure*, nobody on the ground can tell. This project labels every photo and spends the rest of each pass on
> **full photos picked at random with known chances**; the ground corrects the labels with them and gets honest
> error bars whatever the onboard model is like. On what Sentinel-2A really photographed (13-19 Oct 2026), with
> a broken model, labels alone were 10-20 points off and their error bars held 9-27% of the time; the
> self-checking audit was 5-6 points off and held 97-98%. Retraining with what the data analysis found (3 more weeks of real photos, infrared bands) lifted the onboard model's balanced accuracy from 67% to 73% and made the audit about 25% more precise at 0.25 Mbps; it stays honest either way.

## 1. The problem

A satellite can talk to a ground station only while it flies over it. Sentinel-2A, an Earth-observation
satellite, can talk to Delhi on 2-3 passes a day of up to about 10 minutes: **about 1.5% of the day**. Most of
what it photographs over India never comes down. NASA JPL's NAVI-Orbital (April 2026) showed that a small
vision-language model (Gemma 3) can run in orbit and describe photos. The next question is what to send
down, and whether the ground can trust what it makes from it.

## 2. The data first

Three data sets, each checked before trusting any result (`notebooks/01_data.ipynb`):

| Data set | What | Classes | Used for |
|---|---|---|---|
| EuroSAT | real Sentinel-2 photos, 64 x 64 px, Europe | 10, equal sizes | first tests; the head on a real satellite camera |
| AID | sharp aerial photos, 600 x 600 px | 30 | NAVI's data set; comparing labellers |
| **Real Sentinel-2A replay** | what Sentinel-2A *really* photographed every 30 s of its real orbit, 6-19 Oct 2026 | 8, very uneven | **the main results** |

The real replay was built from the satellite's real orbit: for each daylight slot, the image Sentinel-2A
really took there (within seconds) was found in the free Sentinel-2 archive and a 2.24 km chip cut out.
Labels come from the image's cloud map (mostly cloud, mostly snow) and else ESA WorldCover 2021.

**What the data analysis found:**

- **EuroSAT and AID are clean:** no duplicate photos, one size each, nothing broken. Keeping EuroSAT training
  and test photos 10-50 km apart gives the same accuracy (93.4% vs 92.7%): no neighbouring-tile leak.
- **The real replay is very uneven:** 46% cloud, 17% water, and only 43 snow and 4 built-up photos in two weeks.
- **14% of real photos are messy** (land label on a 20-50% cloudy photo, mixed land types, no-data strips,
  blown-out white). The model is wrong on 14% of clean photos but 42-85% of messy ones: messy photos carry 30%
  of the mistakes.
- **Snow and cloud are both blown-out white** in the colour photos (42 of 43 snow photos), so no colour model can
  tell them apart. In short-wave infrared they differ clearly.
- **Labels agree with similar photos** for Cloud (86%) and Water (84%), much less for Trees, ShrubGrass and
  Cropland (about 50%), which blend into each other at 10 m. An eye check of 150 photos found 97% of labels
  right where the eye could tell.
- **The numbers are stable:** other ways of splitting the days move accuracy by about 1 point; cleaning the
  test photos adds about 3.5; the main result holds on clean photos too.

[[FIGURE eda_quality.png | Figure 1. The onboard head's error rate on real test photos, by data problem. Messy photos are 14% of the test but carry 30% of the mistakes.]]

## [newpage] 3. The experiments, in order

Full detail, code and every table: `notebooks/02_experiments.ipynb`. Results: `results/NN_name/`.

| # | Date | Question | What we found |
|---|---|---|---|
| 1 | 18 Oct | How often can the satellite talk to India? | 2-3 passes a day, ~1.5% of the time (Svalbard ~8%) |
| 2 | 18 Oct | Can Gemma 3 label photos with no training? | AID 61-65%; EuroSAT 22-32% (64 px is too small); NAVI's 88% is on an easier 18-class AID, ours scored that way is 76.5% |
| 3 | 19 Oct | Can the agent tell when Gemma is wrong? | A free "confusable class" signal works best (AUROC 0.77); agents v1-v3 |
| 4 | 20 Oct | Frozen SigLIP + 30 KB head instead? | 95.7% vs 63%, 300x faster, honest chances. A 400M head is 98.3% right, but its 5 mistakes are all over 96% sure |
| 5 | 20 Oct | Real Sentinel-2 photos? | A head doesn't transfer between cameras (33%); retrain the 30 KB head per camera: 92.8% |
| 6 | 20 Oct | A tiny CNN instead? | Head wins with few labels (5 per class: 79-90% vs 44-68%); CNN catches up at ~50 per class |
| 7 | 20 Oct | Do results hold across days and stations? | The order never changes, no overfitting, but the head scored 100.0 ± 0.0%: a ceiling |
| 8 | 20 Oct | Is onboard AI needed at all? | Below ~14x (data taken / data sent), re-saving photos as smaller JPEGs sends everything: the old score can't fail. **New question: can the ground trust the numbers?** |
| 9 | 20 Oct | Do random checks keep the ground honest? | Yes, on real photos, healthy or broken model; a self-check adds a safe mode |
| 10 | 22 Oct | Fair rivals, guaranteed error bars, hardware? | SigLIP with class names only beats Gemma (66% vs 63%); betting error bars hold for every class; 19 ms per photo |
| 11 | 22 Oct | Beyond accuracy? | Real photos: 82% accurate, 67% balanced; snow never found |
| 12 | 22 Oct | Retrain using what the data showed? | More real training days + infrared: balanced accuracy 67% -> 73% (75% with a 4.5x bigger model); the audit's error -25% at 0.25 Mbps |

**Experiments 1-7: labelling photos.** The project started as "an agent that labels photos and sends the unsure
ones". Gemma 3, the model NASA flew, was 61-65% right on sharp aerial photos with no training and too slow
(7.5 s per photo). A frozen image model with a tiny trained classifier was far better (95.7%) and 300x faster.
On real Sentinel-2 photos, the classifier has to be retrained per camera (about 1,000 labelled photos).

**Experiment 8: the turning point.** The score used so far counted a 50-byte label the same as a 280 KB photo,
so any good labeller reached about 100%. Re-saving every photo as a smaller JPEG until the pass fits, with no AI
at all, also reached 100%. The question changed: **can the ground trust the numbers it makes from the data?**
The product became the share of the day's photos in each class (cloud, cropland, water...), with an honest
90% error bar.

[[FIGURE audit_rsweep.png | Figure 2. When onboard AI beats plain compression: only above about 30x data taken / data sent.]]

## 4. Main result: the audited downlink

**How it works.** (1) The satellite labels every photo and sends the labels with each class's chance (about 60
bytes each). (2) It gives every photo a chance of being checked: unsure photos are more likely, but every photo
keeps at least half the average chance, so confident mistakes can still be caught. (3) It sends exactly as many
full photos as fit, picked at random with those chances. (4) The ground reads the true class of the checked
photos and corrects the day's class shares (a Horvitz-Thompson difference estimate), with a 90% error bar.
(5) **Self-check:** if the checked mistakes are 1.5x bigger than the model expected, the next passes run in a
safe mode (fully random checks, chances recalibrated from past checks).

**Tested** on what Sentinel-2A really photographed, 13-19 Oct 2026 (head trained on 6-12 Oct), over Delhi,
Bengaluru and Svalbard, 7 days x 3 photo orders, against labels only (like NAVI), triage (send the least-sure
photos) and random raw photos, with a healthy model and five broken ones.

| Real photos, 0.05 Mbps | Labels only | Triage | Random raw photos | **Self-checking audit** |
|---|---|---|---|---|
| Healthy model (82% right) | 1.9 points off, bar held 80% | 1.5, 84% | 5.9, 96% | 3.6, **98%** |
| New place (head from Europe, no cloud class) | 19.5, **9%** | 16.6, **20%** | 5.9, 96% | **5.3, 98%** |
| Radiation bit flips in the head | 10.6, 27% | 10.8, 27% | 5.9, 96% | **5.5, 97%** |
| Damaged camera (blur) | 4.5, 51% | 3.3, 67% | 5.9, 96% | 3.6, 97% |

[[FIGURE audit_honesty_real.png | Figure 3. Healthy or broken onboard model on real Sentinel-2A photos: how far off each policy's class shares are (left), and how often its 90% error bar held (right).]]

- **Today's designs fail silently:** with a broken model, labels and triage are 10-20 points off and their bars
  rarely hold. Even a healthy model's labels mislead on real photos (bar held 80%).
- **The audit stays honest** in every case, and with the self-check it beats random raw photos when the model
  is broken.
- **Guarantees:** the default error bar held 94-98% on average; a betting error bar (no approximation) held for
  every class in every setting, at 1.1-2.4x the width.
- **A new camera learns from its own checks:** checked photos become training labels; from 46% to 83% right in
  4 days.

## 5. Retraining with what the data showed

The data analysis pointed at three fixable problems, so each fix was tried one at a time (experiment 12), with
the test days fixed and every setting picked on training data only. The recipe for the audit was chosen before
seeing any result.

| Real Sentinel-2A photos, test 13-19 Oct | Accuracy | Balanced accuracy | Macro-F1 | Snow found | Calibration error |
|---|---|---|---|---|---|
| Baseline (experiments 9-11) | 82.3% | 67.0% | 66.4% | 0% | 0.029 |
| + balanced classes | 63.9% | 69.1% | 62.4% | 50% | 0.165 |
| + chances corrected to the real class mix | 81.1% | 64.8% | 64.9% | 0% | 0.050 |
| + infrared bands | 82.2% | 67.0% | 68.1% | 7% | 0.052 |
| **+ 3 more weeks of training photos** | **84.7%** | **73.1%** | **75.9%** | **32%** | **0.012** |
| + bigger image model (SigLIP 400M, Kaggle) | 85.6% | 74.7% | 77.5% | 32% | 0.012 |

[[FIGURE retrain.png | Figure 4. Accuracy and balanced accuracy (with 95% range) of each recipe on real Sentinel-2A photos, 13-19 Oct.]]

- **More real data is the biggest win:** 3 more weeks (1,525 -> 7,158 training photos, snow 15 -> 120) lift balanced
  accuracy by 3.5 points on their own; with the infrared bands, by 6 (to 73.1%), and snow is found for the first time (32%).
- **Balancing alone is a trap:** it finds half the snow only by guessing rare classes far more often (accuracy 64%,
  dishonest chances); corrected back to the real class mix, the gain disappears.
- **Infrared helps a little, not a lot:** thick ice clouds also look snow-like in short-wave infrared.
- **A bigger image model adds little** (+1.6 points for 4.5x the weights): better data beats a bigger model.
- **Effect on the main result:** with the retrained head, the self-checking audit's error at 0.25 Mbps falls from
  1.10 to 0.83 points (bars still hold 96%), and triage's from 0.67 to 0.45. Labels alone still mislead (bar held
  87%), and with a broken model labels and triage still fail. Retraining improves the model; the audit keeps the
  numbers honest.

[[FIGURE confusion_retrained.png | Figure 5. Confusion matrices on real photos before and after retraining. Rows = true class, each row scaled to 100%.]]


## 6. Limits

- Truth labels for the real photos come from maps, not people (97% right in an eye check by an AI; a person
  should re-check).
- No flight hardware yet: all onboard steps take 19 ms per photo on a laptop; the Jetson Orin script is ready.
- One satellite, one product (class shares); the EuroSAT replay uses assumed India-like class shares.
- The statistics are known (survey sampling, prediction-powered inference, Olofsson et al. 2014); what is new is
  choosing the sample onboard, per pass, under byte budgets, and testing it against silent failures.

## 7. Next steps

1. A person re-checks the 150 eye-checked labels and the 395 photos whose neighbours disagree with their label.
2. Time and energy on a Jetson Orin Nano (`python -m src.eval.jetson_bench`).
3. A 60-90 s demo video of the audit page (break the model, watch the error bars hold).
4. Paper for IGARSS 2027 (4 pages, deadline about 10 January 2027; outline in `paper/outline.md`).

## How to reproduce

`notebooks/01_data.ipynb` (the data) and `notebooks/02_experiments.ipynb` (experiments 1-12) load every number
from `results/`. Each experiment's command is at the end of its section. Setup: `./setup.sh`.

## Sources

NAVI-Orbital (NASA JPL, arXiv 2606.18271) · On-Orbit Space AI survey (arXiv 2604.16518) · Kodan (ASPLOS 2023) ·
Horvitz & Thompson (1952) · Olofsson et al., Remote Sensing of Environment 148 (2014) · prediction-powered
inference (Angelopoulos et al., arXiv 2301.09633) · Waudby-Smith & Ramdas, betting confidence intervals
(arXiv 2010.09686) · EuroSAT (Helber et al.) · AID (Xia et al., 2017) · Sentinel-2 L2A via Element 84 Earth Search
(contains modified Copernicus Sentinel data 2026) · ESA WorldCover 2021 · SigLIP (Zhai et al., 2023).
