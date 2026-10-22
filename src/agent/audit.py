"""Audited downlink: a label for every photo, then full photos picked AT RANDOM with known chances.

Why random? Agents v2-v4 send the photos the model is LEAST sure about. So the photos it is sure
about are never checked. If the model is wrong in a confident way (a new camera, a radiation bit
flip), nothing on the ground can see it. If instead the full photos are a random sample with known
chances, the ground can correct the labels and give an honest error bar (src/eval/estimate.py).

For each pass, the satellite:
1. sends a label and the model's chance for every class, for every photo (about 60 bytes each),
2. gives each photo an audit chance: unsure photos are more likely to be picked, but every photo
   keeps at least `floor` x the average chance (so confident mistakes can still be caught),
3. picks exactly as many full photos as fit, with those chances (Pareto sampling),
4. sends each picked photo with its chance (the ground needs it to weigh the photo correctly).

Also here, for comparison (same bytes): send raw photos in time order or at random, labels only,
"triage" (agent v4's rule: least sure first) and a stratified sample (the remote-sensing standard).
Everything works on arrays: P = the model's chances, one row per photo, one column per class.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TEXT_BYTES = 50  # time, position and label as text (same as agent v1-v4's descriptions)


def label_bytes(n_classes: int) -> int:
    """Bytes per photo for its label + one byte per class for the chances (0-255)."""
    return TEXT_BYTES + n_classes


@dataclass
class Sent:
    """What the satellite sent down in one pass."""

    policy: str
    labels: bool             # were labels (and chances) sent for every photo?
    full: np.ndarray         # positions (in this pass) of the photos sent in full
    chance: np.ndarray       # each photo's chance of being sent in full (known to the ground)
    bytes_used: int


def n_full(budget: int, n_photos: int, tile_bytes: int, labels_for: int) -> int:
    """How many full photos fit after sending labels for `labels_for` photos."""
    left = budget - labels_for
    return int(max(0, min(n_photos, left // tile_bytes)))


def expected_error(P: np.ndarray) -> np.ndarray:
    """How big the label's mistake is expected to be, per photo: sqrt(1 - sum of chances squared).

    0 when the model is 100% sure, larger when its chances are spread over several classes.
    (If the chances are honest, this is the typical size of the gap between label and truth.)
    """
    return np.sqrt(np.clip(1 - (P ** 2).sum(1), 1e-6, None))


def audit_chances(P: np.ndarray, n: int, floor: float) -> np.ndarray:
    """Chance of each photo being sent in full, adding up to n.

    Unsure photos get more (in proportion to expected_error), but no photo gets less than
    floor x n / N. floor = 1 means every photo has the same chance (a plain random sample).
    """
    N = len(P)
    if n >= N:
        return np.ones(N)
    if n <= 0:
        return np.zeros(N)
    need = np.maximum(expected_error(P), 1e-3)
    low = floor * n / N
    lo, hi = 0.0, 1.0 / need.min()  # at hi every chance is 1, so the total is N >= n
    for _ in range(80):  # find the scale that makes the chances add up to n
        mid = (lo + hi) / 2
        if np.clip(mid * need, low, 1).sum() < n:
            lo = mid
        else:
            hi = mid
    return np.clip(hi * need, low, 1)


def pareto_sample(chance: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick exactly n photos so that each photo's chance of being picked is about `chance`.

    Pareto sampling (Rosén 1997): give each photo a random key that is small when its chance is
    big, and take the n smallest keys.
    """
    u = rng.random(len(chance))
    lam = np.clip(chance, 1e-9, 1 - 1e-9)
    key = u * (1 - lam) / (lam * (1 - u))
    return np.sort(np.argsort(key)[:n])


# ---- the policies (all get the same budget) ----

def raw_in_order(P, budget, tile_bytes, rng) -> Sent:
    """No onboard AI: full photos in the order taken until the pass is full (today's default)."""
    N = len(P)
    n = n_full(budget, N, tile_bytes, 0)
    return Sent("raw in order", False, np.arange(n), np.full(N, n / N), n * tile_bytes)


def raw_random(P, budget, tile_bytes, rng) -> Sent:
    """No onboard AI: full photos picked at random until the pass is full."""
    N = len(P)
    n = n_full(budget, N, tile_bytes, 0)
    return Sent("raw random", False, np.sort(rng.choice(N, n, replace=False)), np.full(N, n / N),
                n * tile_bytes)


def labels_only(P, budget, tile_bytes, rng) -> Sent:
    """Only labels (NAVI / caption-all style). No photo is checked."""
    N = len(P)
    return Sent("labels only", True, np.array([], dtype=int), np.zeros(N), N * label_bytes(P.shape[1]))


def triage(P, budget, tile_bytes, rng) -> Sent:
    """Agent v4's rule: labels for all, then full photos for the LEAST sure first (not random)."""
    N, K = P.shape
    n = n_full(budget, N, tile_bytes, N * label_bytes(K))
    full = np.sort(np.argsort(P.max(1), kind="stable")[:n])
    chance = np.zeros(N)
    chance[full] = 1.0  # sent for sure; the others had no chance at all
    return Sent("triage (v4)", True, full, chance, N * label_bytes(K) + n * tile_bytes)


def audit(P, budget, tile_bytes, rng, floor: float = 0.5) -> Sent:
    """Audited downlink: labels for all, then a random sample with known chances (see top)."""
    N, K = P.shape
    n = n_full(budget, N, tile_bytes, N * label_bytes(K))
    chance = audit_chances(P, n, floor)
    full = pareto_sample(chance, n, rng)
    return Sent(f"audit (floor {floor:g})", True, full, chance, N * label_bytes(K) + n * tile_bytes)


def stratified(P, budget, tile_bytes, rng) -> Sent:
    """Remote-sensing standard (Olofsson et al. 2014): random full photos within each predicted
    class, in proportion to its size (at least 2 per class when possible)."""
    N, K = P.shape
    n = n_full(budget, N, tile_bytes, N * label_bytes(K))
    strata = P.argmax(1)
    sizes = np.bincount(strata, minlength=K)
    used = np.flatnonzero(sizes)
    base = 2 if n >= 2 * len(used) else 0
    take = np.zeros(K, dtype=int)
    take[used] = base + np.floor((n - base * len(used)) * sizes[used] / N).astype(int)
    take = np.minimum(take, sizes)
    full, chance = [], np.zeros(N)
    for h in used:
        members = np.flatnonzero(strata == h)
        if take[h]:
            full += list(rng.choice(members, take[h], replace=False))
        chance[members] = take[h] / len(members)
    full = np.sort(np.array(full, dtype=int))
    return Sent("stratified", True, full, chance, N * label_bytes(K) + len(full) * tile_bytes)


POLICIES = {"raw in order": raw_in_order, "raw random": raw_random, "labels only": labels_only,
            "triage (v4)": triage, "stratified": stratified,
            "audit (floor 1)": lambda P, b, t, r: audit(P, b, t, r, floor=1.0),
            "audit (floor 0.5)": lambda P, b, t, r: audit(P, b, t, r, floor=0.5),
            "audit (floor 0.1)": lambda P, b, t, r: audit(P, b, t, r, floor=0.1)}
