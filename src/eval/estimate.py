"""The ground's side of the audited downlink: turn what came down into numbers with error bars.

Question: what share of the photos in a pass (or a day) were cloud, cropland, water...?

Each method only uses what the satellite sent: labels and chances (if sent), the full photos
(their true class can be read on the ground), and each photo's chance of being sent in full.
It returns an estimate per class, its variance, and whether its error bar is honest by design.

- labels only / triage: count the labels (the truth for photos sent in full). No honest error bar
  exists: the one reported pretends every label is right.
- raw: average over the full photos. Honest only if they were picked at random.
- difference: average of the model's chances + the average mistake on the audited photos, each
  weighted by 1 / its chance (Horvitz-Thompson). Honest whatever the onboard model is like.
- tuned: like difference, but first learns per class how much to trust the model's chances
  (PPI++ idea). Learned on one half of the audited photos, used on the other half (cross-fitting).
- stratified: Olofsson et al. 2014: the average within each predicted class, weighted by its size.

Error bars (`interval`): normal / t / safe / scaled / scaled+1 use the bell-curve approximation.
"betting" (Waudby-Smith & Ramdas 2023) needs no approximation: valid for any number of checks,
also for rare classes, if the checks are draws with known chances (see `betting_interval`).

Self-test on made-up data (checks the honest methods are right on average and their error bars hold):
    python -m src.eval.estimate
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.agent.audit import Sent


@dataclass
class Estimate:
    value: np.ndarray       # estimated share of each class (0-1)
    var: np.ndarray         # variance of the estimate, from the photos that came down
    var_model: np.ndarray   # variance the model's own chances predict (used by the "safe" error bar)
    honest: bool            # is the error bar honest by design?
    checked: int            # photos sent in full (their truth was seen)
    seen: float = 0.0       # weighted sum of squared mistakes on the audited photos ...
    expected: float = 0.0   # ... and what the model's chances expected there (seen/expected ~ 1 if honest)
    photos: int = 0         # photos the estimate is about
    # for the betting error bar (None = this method can't give one): each checked photo's mistake
    # divided by its chance (one row per checked photo), the part that needs no check (mean of the
    # chances), and the smallest / largest value a row could have had (known before checking)
    terms: np.ndarray | None = None
    base: np.ndarray | None = None
    lowest: np.ndarray | None = None
    highest: np.ndarray | None = None


def one_hot(P: np.ndarray) -> np.ndarray:
    return np.eye(P.shape[1])[P.argmax(1)]


def _rosen_var(z: np.ndarray, chance: np.ndarray, N: int) -> np.ndarray:
    """Variance of sum(z / chance) / N for a fixed-size sample with unequal chances (Rosén 1997)."""
    n = len(z)
    if n < 2:
        return np.full(z.shape[1] if z.ndim > 1 else 1, np.inf)
    w = z / chance[:, None]
    a = (1 - chance)[:, None]
    if a.sum() <= 1e-12:  # every photo was sent: no sampling error left
        return np.zeros(z.shape[1])
    centre = (a * w).sum(0) / a.sum()
    return n / (n - 1) * (a * (w - centre) ** 2).sum(0) / N ** 2


def _model_var(P: np.ndarray, chance: np.ndarray, lam: np.ndarray | float = 1.0) -> np.ndarray:
    """Variance predicted from the chances alone: what we'd expect if the chances were honest.

    For each photo the expected squared mistake of y - lam*p is p(1-p) + (1-lam)^2 p^2.
    """
    N = len(P)
    keep = chance > 0
    c = chance[keep]
    expected = P[keep] * (1 - P[keep]) + ((1 - np.asarray(lam)) * P[keep]) ** 2
    return (((1 - c) / c)[:, None] * expected).sum(0) / N ** 2


def labels_count(P: np.ndarray, sent: Sent, Y_full: np.ndarray) -> Estimate:
    """Count the labels; photos sent in full count with their true class (labels only, triage)."""
    hard = one_hot(P)
    hard[sent.full] = Y_full
    m = hard.mean(0)
    pretend = m * (1 - m) / len(P)  # the error bar you'd report if every label were right
    return Estimate(m, pretend, pretend, False, len(sent.full), photos=len(P))


def raw_mean(P_shape: tuple[int, int], sent: Sent, Y_full: np.ndarray) -> Estimate:
    """Average over the photos sent in full (a random sample if they were picked at random)."""
    N, K = P_shape
    n = len(sent.full)
    if n == 0:
        return Estimate(np.full(K, 1 / K), np.full(K, np.inf), np.full(K, np.inf), True, 0, photos=N)
    m = Y_full.mean(0)
    var = (1 - n / N) * Y_full.var(0, ddof=1) / n if n > 1 else np.full(K, np.inf)
    # predicted variance if we knew nothing else: m(1-m), floored so a class never seen still counts
    floor = np.maximum(m * (1 - m), 1 / (n + 2) * (1 - 1 / (n + 2)))
    random = sent.policy == "raw random"
    e = Estimate(m, var, (1 - n / N) * floor / n, random, n, photos=N)
    if random:  # same as a difference estimate with all chances 0, every photo checked with chance n/N
        e.terms, e.base = Y_full * N / n, np.zeros(K)
        e.lowest, e.highest = np.zeros(K), np.full(K, N / n)
    return e


def difference(P: np.ndarray, sent: Sent, Y_full: np.ndarray) -> Estimate:
    """Model chances + weighted mistakes on the audited photos (Horvitz-Thompson difference)."""
    N, K = P.shape
    full, chance = sent.full, sent.chance
    if len(full) == 0:
        return Estimate(P.mean(0), np.full(K, np.inf), np.full(K, np.inf), True, 0, photos=N)
    z = Y_full - P[full]
    value = P.mean(0) + (z / chance[full, None]).sum(0) / N
    var = _rosen_var(z, chance[full], N)
    seen, expected = _seen_expected(z, P[full], chance[full])
    could = chance > 0
    return Estimate(value, var, _model_var(P, chance), True, len(full), seen, expected, N,
                    terms=z / chance[full, None], base=P.mean(0),
                    lowest=(-P[could] / chance[could, None]).min(0),
                    highest=((1 - P[could]) / chance[could, None]).max(0))


def _seen_expected(z: np.ndarray, P_s: np.ndarray, c: np.ndarray, lam=1.0) -> tuple[float, float]:
    """How big the audited mistakes were, and how big the model's chances said they would be
    (both weighted by 1 / chance, so they stand for all photos). Pooled over classes."""
    expected = (P_s * (1 - P_s) + ((1 - np.asarray(lam)) * P_s) ** 2).sum(1)
    return float(((z ** 2).sum(1) / c).sum()), float((expected / c).sum())


def _trust(P_s, Y_s, w) -> np.ndarray:
    """Per class: how much to trust the model's chances (0 = not at all, 1 = fully), by weighted
    least squares of truth on chance."""
    xm = (w[:, None] * P_s).sum(0) / w.sum()
    ym = (w[:, None] * Y_s).sum(0) / w.sum()
    cov = (w[:, None] * (P_s - xm) * (Y_s - ym)).sum(0)
    var = (w[:, None] * (P_s - xm) ** 2).sum(0)
    return np.clip(np.divide(cov, var, out=np.ones_like(cov), where=var > 1e-12), 0, 1)


def tuned(P: np.ndarray, sent: Sent, Y_full: np.ndarray, rng: np.random.Generator) -> Estimate:
    """Difference estimate with a learned trust per class, cross-fitted on two halves."""
    N, K = P.shape
    full, chance = sent.full, sent.chance
    n = len(full)
    if n < 8:  # too few audited photos to learn anything: plain difference
        return difference(P, sent, Y_full)
    half = np.zeros(n, dtype=bool)
    half[rng.choice(n, n // 2, replace=False)] = True
    parts = []
    for mine in (half, ~half):
        lam = _trust(P[full[~mine]], Y_full[~mine], 1 / chance[full[~mine]])  # learned on the other half
        c = chance[full[mine]] * mine.sum() / n                                 # chance of being in this half
        z = Y_full[mine] - lam * P[full[mine]]
        value = lam * P.mean(0) + (z / c[:, None]).sum(0) / N
        parts.append((value, _rosen_var(z, c, N), _model_var(P, chance * mine.sum() / n, lam),
                      _seen_expected(z, P[full[mine]], c, lam)))
    value = (parts[0][0] + parts[1][0]) / 2
    var = (parts[0][1] + parts[1][1]) / 4
    var_model = (parts[0][2] + parts[1][2]) / 4
    seen = (parts[0][3][0] + parts[1][3][0]) / 2
    expected = (parts[0][3][1] + parts[1][3][1]) / 2
    return Estimate(value, var, var_model, True, n, seen, expected, N)


def stratified_estimate(P: np.ndarray, sent: Sent, Y_full: np.ndarray) -> Estimate:
    """Olofsson et al. 2014: within each predicted class, the share of each true class among the
    audited photos; weighted by the predicted class's size."""
    N, K = P.shape
    strata = P.argmax(1)
    in_full = strata[sent.full]
    value, var, var_model = np.zeros(K), np.zeros(K), np.zeros(K)
    for h in np.unique(strata):
        members = strata == h
        Nh = members.sum()
        Wh = Nh / N
        Yh = Y_full[in_full == h]
        nh = len(Yh)
        expected = (P[members] * (1 - P[members])).mean(0) + P[members].var(0)  # predicted spread
        if nh == 0:  # nothing checked in this predicted class: use its labels, and since the true
            # shares inside it can be anything from 0 to 1, add the biggest spread that allows
            value += Wh * one_hot(P[members]).mean(0)
            var += Wh ** 2 * 0.25
            var_model += Wh ** 2 * 0.25
            continue
        value += Wh * Yh.mean(0)
        fpc = 1 - nh / Nh
        if nh == Nh:      # every photo of this predicted class was checked: no sampling error
            spread = np.zeros(K)
        elif nh > 1:
            spread = Yh.var(0, ddof=1)
        else:             # one photo can't show the spread: assume the biggest possible
            spread = np.full(K, 0.25)
        var += Wh ** 2 * fpc * spread / nh
        var_model += Wh ** 2 * fpc * expected / nh
    return Estimate(value, var, var_model, True, len(sent.full), photos=N)


def estimate(P: np.ndarray, sent: Sent, Y_full: np.ndarray, how: str,
             rng: np.random.Generator) -> Estimate:
    """Run one ground method. how = count / raw / difference / tuned / stratified."""
    if how == "count":
        return labels_count(P, sent, Y_full)
    if how == "raw":
        return raw_mean(P.shape, sent, Y_full)
    if how == "difference":
        return difference(P, sent, Y_full)
    if how == "tuned":
        return tuned(P, sent, Y_full, rng)
    if how == "stratified":
        return stratified_estimate(P, sent, Y_full)
    raise ValueError(how)


def combine(parts: list[Estimate], sizes: list[int]) -> Estimate:
    """Join per-pass estimates into one for the day (each pass weighted by its number of photos)."""
    w = np.array(sizes, dtype=float) / sum(sizes)
    value = sum(wi * e.value for wi, e in zip(w, parts))
    var = sum(wi ** 2 * e.var for wi, e in zip(w, parts))
    var_model = sum(wi ** 2 * e.var_model for wi, e in zip(w, parts))
    day = Estimate(value, var, var_model, all(e.honest for e in parts), sum(e.checked for e in parts),
                   sum(e.seen for e in parts), sum(e.expected for e in parts), sum(e.photos for e in parts))
    if all(e.terms is not None for e in parts):  # the day's value = its base + (sum of all terms) / photos
        day.terms = np.concatenate([e.terms for e in parts])
        day.base = sum(wi * e.base for wi, e in zip(w, parts))
        day.lowest = np.min([e.lowest for e in parts], axis=0)
        day.highest = np.max([e.highest for e in parts], axis=0)
    return day


def betting_interval(e: Estimate, level: float = 0.9, c: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Error bar from a betting game (Waudby-Smith & Ramdas 2023, "hedged capital", JRSSB).

    Idea: for each candidate share m, imagine betting against "the true share is m", one checked photo
    at a time. If m is wrong, the bets win and the money grows. Every m where the money would pass
    1 / (1 - level) is ruled out; what is left is the error bar. This holds for ANY number of checks
    (no bell-curve approximation), and a class the checks rarely see gets a wide bar automatically.

    Each checked photo gives W = mistake / chance x checks / photos: its average is exactly the
    correction the difference estimate adds. W must stay between known limits (`lowest`, `highest`,
    known before checking from the chances sent down). Assumption: the checks behave like independent
    draws with these chances. Our checks are drawn without putting photos back, which has less spread,
    so the bar should err on the wide side (tested in the self-test and src/eval/audit_test.py).
    """
    K = len(e.value)
    if e.terms is None or len(e.terms) == 0:
        return np.zeros(K), np.ones(K)
    n, N = len(e.terms), e.photos
    a, b = e.lowest * n / N, e.highest * n / N
    span = np.maximum(b - a, 1e-12)
    order = np.random.default_rng(0).permutation(n)  # the bets need the photos in a random order
    X = (e.terms[order] * n / N - a) / span          # each between 0 and 1
    # running mean and spread from the photos BEFORE each bet (so a bet never peeks at its photo)
    t = np.arange(1, n + 1)[:, None]
    mean = (0.5 + np.cumsum(X, 0)) / (t + 1)
    mean_before = np.vstack([np.full((1, K), 0.5), mean[:-1]])
    spread = (0.25 + np.cumsum((X - mean_before) ** 2, 0)) / (t + 1)
    spread_before = np.vstack([np.full((1, K), 0.25), spread[:-1]])
    lam = np.sqrt(2 * np.log(2 / (1 - level)) / (n * spread_before))
    limit = np.log(1 / (1 - level))

    def ruled_out(m: np.ndarray) -> np.ndarray:
        up = np.minimum(lam, c / m)          # bet on "the share is above m"
        down = np.minimum(lam, c / (1 - m))  # bet on "the share is below m"
        log_up = np.log1p(up * (X - m)).sum(0)
        log_down = np.log1p(-down * (X - m)).sum(0)
        return np.logaddexp(log_up, log_down) + np.log(0.5) >= limit

    # shares from 0 to 1, in the 0-1 betting scale
    edge_low = np.clip((0 - e.base - a) / span, 1e-9, 1 - 1e-9)
    edge_high = np.clip((1 - e.base - a) / span, 1e-9, 1 - 1e-9)
    centre = np.clip(X.mean(0), edge_low, edge_high)
    ends = []
    for edge in (edge_low, edge_high):  # walk from the centre to each edge (bisection, per class)
        inside, outside = centre.copy(), edge.copy()
        stop = ~ruled_out(edge)
        inside[stop] = edge[stop]       # nothing ruled out up to the edge
        for _ in range(30):
            mid = (inside + outside) / 2
            out = ruled_out(mid)
            outside = np.where(out & ~stop, mid, outside)
            inside = np.where(~out & ~stop, mid, inside)
        ends.append(inside)
    low, high = (e.base + a + ends[0] * span, e.base + a + ends[1] * span)
    return np.clip(low, 0, 1), np.clip(high, 0, 1)


def interval(e: Estimate, how: str = "safe", level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Error bar (low, high) per class, clipped to 0-1.

    normal: estimate ± z x standard error
    t:      Student t instead of z (wider when few photos were checked)
    safe:   t, and the standard error is never smaller than what the model's chances predict
            (stops a lucky audit with no mistakes in it from giving a too-narrow error bar)
    scaled: like safe, but the model's prediction is first scaled up by how much bigger the
            audited mistakes were than the model expected (pooled over all classes and passes,
            so it works even for a rare class the audit never saw). A healthy model -> ~1x.
    scaled+1: like scaled, and never narrower than "one checked photo we didn't see might have
            belonged to this class" (plus-one rule, no model needed: protects rare classes when
            the model itself is broken)
    betting: no bell-curve approximation (see betting_interval); falls back to scaled+1 for
            methods that can't give one (tuned, stratified)
    """
    if not e.honest:
        how = "normal"  # labels only / triage: only the "pretend every label is right" bar exists
    if how == "betting":
        if e.terms is not None:
            return betting_interval(e, level)
        how = "scaled+1"  # tuned / stratified: no betting bar, use the default instead
    if how in ("normal", "t"):
        var = e.var
    elif how == "safe":
        var = np.maximum(e.var, e.var_model)
    else:
        scale = max(1.0, e.seen / e.expected) if e.expected > 0 else 1.0
        var = np.maximum(e.var, e.var_model * scale)
        if how == "scaled+1":
            n, N = e.checked, max(e.photos, e.checked)
            q = 1 / (n + 2)
            var = np.maximum(var, (1 - n / N) * q * (1 - q) / n if n > 0 else np.inf)
    if how == "normal":
        q = stats.norm.ppf(0.5 + level / 2)
    else:
        q = stats.t.ppf(0.5 + level / 2, df=max(1, e.checked - 1))
    half = q * np.sqrt(np.nan_to_num(var, nan=np.inf))
    return np.clip(e.value - half, 0, 1), np.clip(e.value + half, 0, 1)


# ---- self-test on made-up data ----

def _fake_population(N: int, K: int, rng, accuracy: float, overconfident: bool):
    """Made-up photos: true classes with uneven shares, and a model's chances for each."""
    shares = rng.dirichlet(np.full(K, 0.7))
    y = rng.choice(K, N, p=shares)
    right = rng.random(N) < accuracy
    guess = np.where(right, y, rng.integers(0, K, N))
    P = np.full((N, K), 0.0)
    sure = rng.uniform(0.5, 0.99, N) if not overconfident else np.full(N, 0.99)
    P[np.arange(N), guess] = sure
    P += ((1 - sure) / (K - 1))[:, None] * (1 - np.eye(K)[guess])
    return np.eye(K)[y], P


def selftest(reps: int = 2000) -> None:
    from src.agent.audit import audit, labels_only, raw_random, stratified, triage

    print("Self-test on made-up data: 400 photos, 6 classes, budget = labels + 40 full photos.")
    print("An honest method should be right on average (bias ~0) and its 90% error bar should hold ~90%.\n")
    rng = np.random.default_rng(0)
    N, K, tile = 400, 6, 10_000
    budget = N * (50 + K) + 40 * tile
    methods = {"labels only": (labels_only, "count"), "triage (v4)": (triage, "count"),
               "raw random": (raw_random, "raw"), "stratified": (stratified, "stratified"),
               "audit 1 + difference": (lambda *a: audit(*a, floor=1.0), "difference"),
               "audit 0.5 + difference": (lambda *a: audit(*a, floor=0.5), "difference"),
               "audit 0.5 + tuned": (lambda *a: audit(*a, floor=0.5), "tuned")}
    for model, (acc, over) in {"honest model, 80% right": (0.8, False),
                               "overconfident model, 60% right": (0.6, True)}.items():
        print(f"== {model} ==")
        print(f"{'method':<24} {'bias (pts)':>10} {'RMSE (pts)':>10} {'holds: normal':>14} {'t':>5} {'safe':>5} {'scaled':>7} {'scaled+1':>9} {'betting':>8}")
        Y, P = _fake_population(N, K, rng, acc, over)
        truth = Y.mean(0)
        for name, (policy, how) in methods.items():
            errors, held = [], {"normal": [], "t": [], "safe": [], "scaled": [], "scaled+1": [], "betting": []}
            for _ in range(reps):
                sent = policy(P, budget, tile, rng)
                e = estimate(P, sent, Y[sent.full], how, rng)
                errors.append(e.value - truth)
                for kind in held:
                    lo, hi = interval(e, kind)
                    held[kind].append((lo - 1e-9 <= truth) & (truth <= hi + 1e-9))
            err = np.array(errors)
            print(f"{name:<24} {100 * np.abs(err.mean(0)).max():>10.2f} {100 * np.sqrt((err ** 2).mean()):>10.2f} "
                  + "".join(f"{np.mean(held[k]):>{w}.0%}" for k, w in (("normal", 14), ("t", 6), ("safe", 6), ("scaled", 8), ("scaled+1", 10), ("betting", 9))))
        print()


if __name__ == "__main__":
    selftest()
