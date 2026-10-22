"""Pilot B (20 Oct 2026, exploratory, not part of the paper pipeline):
"audited downlink" vs current practice, for ESTIMATING what the satellite saw.

Run from the orbit-agent folder (about 10 s, uses saved embeddings only):
    python notes/pilots/pilot_audit.py

Question: after one day, what fraction of the tiles were cropland, forest, water...? (with an
honest error bar). Real Sentinel-2 tiles (EuroSAT FINAL 1,000), the paper's onboard heads.

Each simulated day: N = 1,088 tiles drawn with assumed India-like class shares.
Budget per day = pass bytes at that downlink speed; a full tile costs 280 KB, a label 50 B.

Methods (all spend the same bytes):
  raw random    : send random full tiles until full; estimate from those only (classic survey)
  labels only   : send every onboard label; count them (NAVI / caption-all style)
  triage (v4)   : labels for all + full tiles for the LEAST sure (current agent v4); count
  audit uniform : labels (probabilities) for all + RANDOM full tiles; difference estimator
  active f=...  : like audit uniform, but unsure tiles are more likely to be picked (known odds;
                  every tile keeps a minimum chance = f x the uniform chance);
                  inverse-probability-weighted correction
  ... tuned     : same sample, but the model's chances get a trust weight per class (PPI++ idea),
                  so a bad model is ignored instead of trusted
  stratified    : remote-sensing standard (Olofsson et al. 2014): random audit within each
                  predicted class, area estimate from the stratified sample
Only the audit / stratified / raw methods come with a valid 90% interval by design.
Columns: RMSE and worst bias in percentage points; "holds" = how often the 90% interval
contained the truth (all / common / rare classes); "vs raw" = how many times more bytes
"raw random" would need for the same squared error.
Simplifications: one pooled budget per day (not per pass); class shares are assumed;
the fault is injected; intervals use a normal approximation.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))  # append (not insert): stdlib always wins

import numpy as np

from src.data.eurosat import CLASSES, dataset_dir
from src.eval.sentinel_test import MATCH, photo_sets
from src.model.head import aid_head, embed, fit_head

Z90 = 1.645
N, TILE, LABEL = 1088, 280_000, 50
BUDGET_MB = {0.25: 46, 1: 179}  # per day, Delhi, from docs/experiment_log.md
SHARES = {"AnnualCrop": .30, "Forest": .12, "HerbaceousVegetation": .12, "Highway": .02,
          "Industrial": .02, "Pasture": .06, "PermanentCrop": .06, "Residential": .08,
          "River": .04, "SeaLake": .18}  # ASSUMPTION: India-like mix, not measured
DAYS = 1000
FLOORS = [0.1, 0.5]

# ---- onboard models (from saved embeddings, same steps as the paper) ----
final, tuning, train = photo_sets(100)
root = dataset_dir()
x_final, _ = embed(final, "siglip-base", root=root, dataset="eurosat")
x_tuning, _ = embed(tuning, "siglip-base", root=root, dataset="eurosat")
x_train, _ = embed(train, "siglip-base", root=root, dataset="eurosat")
classes = list(CLASSES)
K = len(classes)
y = np.array([classes.index(label) for label, _ in final])
Y = np.eye(K)[y]

own = fit_head(x_train, [l for l, _ in train], x_tuning, [l for l, _ in tuning], verbose=False)
p_own = own.predict_proba(x_final)[:, [list(own.classes_).index(c) for c in classes]]

aid = aid_head("siglip-base", 20)
raw = aid.predict_proba(x_final)
p_aid = np.zeros((len(final), K))
for j, c in enumerate(aid.classes_):
    if c in MATCH:
        p_aid[:, classes.index(MATCH[c])] += raw[:, j]
p_aid /= p_aid.sum(1, keepdims=True)

# "silent fault": the good head, but 30% of AnnualCrop tiles are labelled PermanentCrop.
rng0 = np.random.default_rng(7)
# The fault's confidence is drawn from the head's confidence on its CORRECT answers,
# so the wrong labels look exactly as sure as right ones (truly silent).
p_fault = p_own.copy()
hit = (y == classes.index("AnnualCrop")) & (rng0.random(len(y)) < 0.3)
right_conf = p_own.max(1)[p_own.argmax(1) == y]
conf = rng0.choice(right_conf, hit.sum())
p_fault[hit] = ((1 - conf) / (K - 1))[:, None]
p_fault[hit, classes.index("PermanentCrop")] = conf

models = {"own head (92.8%)": p_own, "AID head, shifted": p_aid, "own head + silent fault": p_fault}
for name, p in models.items():
    print(f"{name:<26} accuracy {np.mean(p.argmax(1) == y):.1%}")


# ---- one simulated day ----
def pareto_sample(pi, n, rng):
    """Fixed-size sample with (approximately) inclusion chances pi (Pareto pi-ps, Rosen 1997)."""
    u = rng.random(len(pi))
    lam = np.clip(pi, 1e-9, 1 - 1e-9)
    q = u * (1 - lam) / (lam * (1 - u))
    return np.argsort(q)[:n]


def ht_difference(f, d, pi, idx):
    """mean(f over all) + inverse-probability-weighted mean of the residual d on the audit."""
    n = len(idx)
    w = d[idx] / pi[idx, None]
    est = f.mean(0) + w.sum(0) / len(f)
    a = (1 - pi[idx])[:, None]
    centre = (a * w).sum(0) / a.sum()
    var = n / (n - 1) * (a * (w - centre) ** 2).sum(0) / len(f) ** 2
    return est, np.sqrt(var)


def tuned(Pd, Yd, pi, idx):
    """Power-tuned difference estimator (PPI++ idea, one trust weight per class):
    estimate = lam * mean(model chance) + weighted audit mean of (truth - lam * chance).
    lam ~ 1 when the model tracks the truth, ~ 0 when it doesn't (then it is a plain audit)."""
    w = 1 / pi[idx]
    X, Yi = Pd[idx], Yd[idx]
    xm = (w[:, None] * X).sum(0) / w.sum()
    ym = (w[:, None] * Yi).sum(0) / w.sum()
    cov = (w[:, None] * (X - xm) * (Yi - ym)).sum(0)
    var = (w[:, None] * (X - xm) ** 2).sum(0)
    lam = np.clip(np.divide(cov, var, out=np.zeros_like(cov), where=var > 1e-12), 0, 1)
    return ht_difference(Pd * lam, Yd - Pd * lam, pi, idx)


def one_day(p, rate, rng):
    shares = np.array([SHARES[c] for c in classes])
    tiles = np.concatenate([rng.choice(np.flatnonzero(y == k), size=m)
                            for k, m in enumerate(rng.multinomial(N, shares))])
    Yd, Pd = Y[tiles], p[tiles]
    hard = np.eye(K)[Pd.argmax(1)]
    sure = Pd.max(1)
    truth = Yd.mean(0)
    budget = BUDGET_MB[rate] * 1e6
    n_raw = int(budget // TILE)
    n_aud = int((budget - N * LABEL) // TILE)
    fpc = 1 - n_aud / N
    out = {}

    s = rng.choice(N, n_raw, replace=False)
    m = Yd[s].mean(0)
    out["raw random"] = (m, np.sqrt((1 - n_raw / N) * Yd[s].var(0, ddof=1) / n_raw))

    m = hard.mean(0)
    out["labels only"] = (m, np.sqrt(m * (1 - m) / N))  # what you'd report if labels were truth

    full = np.argsort(sure)[:n_aud]
    est = hard.copy()
    est[full] = Yd[full]
    m = est.mean(0)
    out["triage (v4)"] = (m, np.sqrt(m * (1 - m) / N))

    s = rng.choice(N, n_aud, replace=False)
    d = Yd - Pd
    out["audit uniform"] = (Pd.mean(0) + d[s].mean(0),
                            np.sqrt(fpc * d[s].var(0, ddof=1) / n_aud))
    out["uniform tuned"] = tuned(Pd, Yd, np.full(N, n_aud / N), s)

    need = np.sqrt(np.clip(1 - (Pd ** 2).sum(1), 1e-6, None))  # expected error size per tile
    for share in FLOORS:  # minimum chance = share x the uniform chance
        floor = share * n_aud / N
        lo, hi = 0.0, 1e6
        for _ in range(60):  # find the scale so the chances add up to the audit size
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if np.clip(mid * need, floor, 1).sum() < n_aud else (lo, mid)
        pi = np.clip(hi * need, floor, 1)
        idx = pareto_sample(pi, n_aud, rng)
        out[f"active f={share}"] = ht_difference(Pd, d, pi, idx)
        out[f"active tuned f={share}"] = tuned(Pd, Yd, pi, idx)

    strata = Pd.argmax(1)
    sizes = np.bincount(strata, minlength=K)
    used = int((sizes > 0).sum())  # at least 2 per stratum, total never above the audit size
    alloc = (2 + np.floor((n_aud - 2 * used) * sizes / N)).astype(int) * (sizes > 0)
    est, var = np.zeros(K), np.zeros(K)
    for h in np.flatnonzero(sizes):
        members = np.flatnonzero(strata == h)
        take = rng.choice(members, min(alloc[h], len(members)), replace=False)
        wh = len(members) / N
        est += wh * Yd[take].mean(0)
        if len(take) > 1:
            var += wh ** 2 * (1 - len(take) / len(members)) * Yd[take].var(0, ddof=1) / len(take)
    out["stratified"] = (est, np.sqrt(var))
    return truth, out


print(f"\nN = {N} tiles/day, full tile {TILE / 1e3:.0f} KB, label {LABEL} B, {DAYS} simulated days."
      " Error = how far the estimated class shares are from the truth (percentage points).")
for name, p in models.items():
    for rate in BUDGET_MB:
        rng = np.random.default_rng(42)
        err, cover, width = {}, {}, {}
        for _ in range(DAYS):
            truth, out = one_day(p, rate, rng)
            for method, (est, se) in out.items():
                err.setdefault(method, []).append(est - truth)
                cover.setdefault(method, []).append(np.abs(est - truth) <= Z90 * se)
                width.setdefault(method, []).append(Z90 * se)
        base = np.mean(np.array(err["raw random"]) ** 2)
        print(f"\n== {name}, {rate} Mbps ({int(BUDGET_MB[rate] * 1e6 // TILE)} full tiles fit) ==")
        print(f"{'method':<20} {'RMSE pts':>8} {'worst bias':>10} {'holds: all':>10} "
              f"{'common':>7} {'rare':>5} {'+/- pts':>7} {'vs raw':>8}")
        rare = np.array([SHARES[c] <= 0.04 for c in classes])
        for method in err:
            e = np.array(err[method])
            c = np.array(cover[method]).mean(0)
            mse = (e ** 2).mean()
            ratio = f"{base / mse:.1f}x" if mse > 1e-12 else "exact"
            print(f"{method:<20} {100 * np.sqrt(mse):>8.2f} "
                  f"{100 * np.abs(e.mean(0)).max():>10.2f} {c.mean():>10.0%} "
                  f"{c[~rare].mean():>7.0%} {c[rare].mean():>5.0%} "
                  f"{100 * np.mean(width[method]):>7.2f} {ratio:>8}")
