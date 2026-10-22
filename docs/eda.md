# Exploratory data analysis (22 Oct 2026)

**Question:** is part of the error a data problem, and how much do the numbers move if the data is cleaned
or split differently?

**Short answer:** EuroSAT and AID are clean (no duplicates, one photo size, no broken files). The real
Sentinel-2A replay has some messy photos (14% of them), and the model's mistakes pile up there. Cleaning
them adds about 3.5 points to the scores, and other day splits move them by only about 1 point. **The main
result, the audit, gives the same conclusions on clean data.**

Run it: `python -m src.eval.eda` (about 1 min, saved photos and embeddings only).
Files: `experiments/eda_summary_*.csv` (every number), `experiments/eda_real_chips.csv` (every real photo
with its flags), `experiments/eda_sensitivity_real_*.csv`, `experiments/eda_audit_clean_*.csv`.

## 1. The real Sentinel-2A replay (6-19 Oct 2026, 3,527 photos)

![Class mix, training vs test days](eda_classes.png)

- **Very uneven classes:** Cloud 46%, Water 17%, ShrubGrass 12%, Trees 11%, Bare 8%, Cropland 5%,
  SnowIce 1.2% (43 photos), BuiltUp 0.1% (4 photos).
- **Training days (6-12 Oct) and test days (13-19 Oct) have the same mix** (total difference 5%).
  SnowIce is the exception: 15 photos in training, 28 in test.
- **Every photo is from a different source image**, no names repeat, and the real image was taken within
  0-9 s of the simulated time.
- **Places don't repeat:** only 93 of 1,749 test photos are within 10 km of a training photo (median
  distance 198 km). Half the test photos still have a "near-twin" in training (cosine > 0.95), but that is
  because clouds and open water look alike everywhere, not a leak (same as found for EuroSAT).

![Where the real photos are](eda_map.png)

### Data problems (rules fixed before looking at any result)

| Problem | Photos | Mostly |
|---|---|---|
| Land label, but 20-50% of the photo is cloud | 230 (6.5%) | Water, Trees, ShrubGrass |
| Cloud label, but only 50-70% cloud | 128 (3.6%) | Cloud |
| Mixed: the label's land type covers under half the photo | 75 (2.1%) | ShrubGrass, Cropland, Trees |
| More than 5% "no data" (black strips at the image edge) | 57 (1.6%) | Cloud, Water |
| Land label, but more than half pure white | 18 (0.5%) | Bare |
| **Clean (none of the above)** | **3,043 (86%)** | |

- **Blown-out white:** 1,131 of 1,615 Cloud photos and 42 of 43 SnowIce photos are more than half pure white
  (the colour product saturates on bright surfaces). With no texture left, **snow and cloud can't be told
  apart in these photos**. That is a limit of the data, not only of the model. The infrared bands would fix it.
- **Label rule:** a photo up to 49% cloud gets its land label; a spot the 2021 land map doesn't cover counts
  as Water; land-map "herbaceous wetland" counts as Water.

### Do labels agree with similar photos?

For each photo, the share of its 10 most similar photos (in SigLIP space) that have the same label:

| Cloud | Water | Bare | Trees | ShrubGrass | Cropland | SnowIce | BuiltUp |
|---|---|---|---|---|---|---|---|
| 86% | 84% | 69% | 54% | 50% | 47% | 18% | 5% |

- The eye check confirms this measure works: photos the eye agreed with score 71%, "can't tell" 28%,
  "label wrong" 5%.
- Trees, ShrubGrass and Cropland agree only about half the time: at 10 m, these land types blend into each
  other, so some confusion is built into the classes themselves.
- 395 photos (11%) have a label that at most 2 of their 10 neighbours share: candidates for a person to check.

## 2. Where are the model's mistakes?

![Error rate by data problem](eda_quality.png)

Healthy head (trained 6-12 Oct), test days 13-19 Oct:

| Photos | Wrong |
|---|---|
| Clean | 14% |
| No-data strips | 17% |
| Cloud label, only 50-70% cloud | 23% |
| Land label, 20-50% cloud | 42% |
| Mixed land types | 74% |
| Land label, mostly pure white | 85% |

- **Flagged photos are 14% of the test days but carry 30% of the mistakes.** So part of the error is the
  data, but most of it (70%) is on clean photos: the model really mixes up Trees, ShrubGrass and Cropland.
- All 27 SnowIce mistakes (-> Cloud) are on photos that pass every flag: blown-out snow looks like cloud.

## 3. How much do the numbers move?

![How much the numbers move](eda_sensitivity.png)

Same head and steps each time; balanced accuracy and macro-F1 over the classes with at least 5 test photos
(BuiltUp has 0-1, so it is left out everywhere; otherwise one photo moves the score by 10 points).

| Variant | Accuracy (95% range) | Balanced accuracy | Macro-F1 |
|---|---|---|---|
| As used: train 6-11, tune 12, test 13-19 | 82.3% (80.5-84.0) | 67.0% | 66.4% |
| Same head, clean test photos only | 85.7% (83.9-87.4) | 70.8% | 69.2% |
| Clean training + clean test photos | 85.1% (83.3-86.8) | 69.2% | 68.6% |
| Reversed: train 13-18, test 6-12 | 81.8% (79.9-83.6) | 65.7% | 66.2% |
| Train even days, test odd days | 81.3% (79.4-83.0) | 65.5% | 65.2% |
| Train odd days, test even days | 82.1% (80.3-84.0) | 66.3% | 66.8% |

- **Other day splits move the scores by about 1 point.** The results don't depend on the lucky choice of days.
- **Cleaning the test photos adds about 3.5 points.** Cleaning the training photos adds nothing: messy
  photos make the test harder, they don't teach the head wrong things.
- The earlier "balanced accuracy 58.6%" counted BuiltUp (1 test photo, missed) as a full class; over the
  classes with enough photos it is 67%.

## 4. Does the audit still hold on clean photos?

The main experiment again (Delhi, Bengaluru, Svalbard x 7 days x 3 photo orders), all photos vs clean photos
only. Typical error of the day's class shares (points) / how often the 90% error bar held:

| Onboard model, speed | Labels only | Triage (agent v4) | Random raw photos | Self-checking audit |
|---|---|---|---|---|
| Healthy, 0.05 Mbps | 1.9/80% -> 1.8/84% | 1.5/84% -> 1.2/88% | 5.9/96% -> 5.8/96% | 3.6/98% -> 3.0/99% |
| Healthy, 0.25 Mbps | 1.9/80% -> 1.8/84% | 0.7/96% -> 0.6/95% | 2.2/92% -> 1.8/95% | 1.1/97% -> 1.1/95% |
| New place, 0.05 Mbps | 19.5/9% -> 20.4/8% | 16.6/20% -> 17.0/23% | 5.9/96% -> 5.8/96% | 5.3/98% -> 5.7/97% |
| New place, 0.25 Mbps | 19.5/9% -> 20.4/8% | 9.1/58% -> 9.3/66% | 2.2/92% -> 1.8/95% | 1.9/95% -> 1.8/94% |
| Bit flips*, 0.05 Mbps | 10.6/27% -> 5.8/58% | 10.8/27% -> 6.5/52% | 5.9/96% -> 5.8/96% | 5.5/97% -> 5.0/98% |

\* The bit flips are re-picked on the clean photos, so that broken model is a different (milder) one.

- **Same conclusions:** with a broken model, labels and triage still fail silently (bars hold 8-66%), and
  the self-checking audit still holds 94-99%.
- At the tightest budget the audit's error moves by about 0.5 points between runs just from which photos
  happen to be checked (3.0-3.6 for the healthy head): report these numbers with their spread.

## 5. EuroSAT (27,000 photos) and AID (2,400 downloaded)

| Check | EuroSAT | AID |
|---|---|---|
| Exact duplicate photos | 0 | 0 |
| Same photo under two classes | 0 | - |
| Photo sizes | all 64 x 64 | all 600 x 600 |
| Very dark or broken photos | 0 | - |
| Class sizes | 2,000-3,000 per class (Pasture smallest) | full set 220-420 per class; our sets are equal (10 or 20 per class) |
| Where | Europe only: latitude 28-65, 5,020 different 10 km blocks | Google Earth, many cities, mixed zoom levels |
| Other | | JPEG size by class 139 KB (Port) to 720 KB (SparseResidential): why the replay gives every photo the same size |

Both are clean. Their limits are known and already handled: EuroSAT is Europe-only and 64 px (so the head is
retrained per camera), AID is aerial, not satellite.

## What to do with this

1. **Report balanced accuracy and macro-F1 next to accuracy**, over classes with enough test photos.
2. **Report the audit's tight-budget numbers with their spread** (about ±0.5 points from the random checks).
3. **Say in the paper** that 14% of real photos are messy (partly cloudy, mixed, no-data, blown-out) and that
   the conclusions hold with and without them.
4. **Fixes worth trying:** add the infrared band (snow vs cloud), a "partly cloudy" class or rule, and more
   SnowIce / BuiltUp training photos (more days).
5. **A person should check** the 395 photos whose neighbours disagree with their label
   (`experiments/eda_real_chips.csv`, column `neighbour_agreement`).
