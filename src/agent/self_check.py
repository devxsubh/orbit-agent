"""Self-checking audit: the ground notices when the onboard model is broken and switches to safe mode.

Every checked photo shows the ground two things: how big the label's mistake was, and how big the
model's own chances said it would be. If the model is healthy, the two match. If the checked mistakes
are clearly bigger than the model expected (over all checks so far), the model is wrong in a way
it can't see itself: a new camera, a radiation bit flip, damaged optics.

Then the ground sends one bit up and the next passes run in **safe mode**:
1. checks are picked uniformly at random (floor 1): the broken model no longer decides what gets checked,
2. the model's chances are not trusted as they are: they are recalibrated with a table learned from the
   past checks ("when the model said water, the photo was really cloud 60% of the time").
Everything safe mode uses comes from EARLIER passes, so the error bars stay honest.

Decided on the EuroSAT replay only (src/eval/audit_test.py, healthy head never switched; broken models
switched on 74-96% of passes), before testing on real Sentinel-2A photos:
    switch when seen / expected mistakes > 1.5 over all checks so far (after at least one pass).
This is the "conservative fallback mode" the On-Orbit Space AI survey asks for.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.agent.audit import Sent, audit
from src.eval.estimate import Estimate, difference, raw_mean

SWITCH = 1.5      # safe mode when the checked mistakes are 50% bigger than the model expected
SMOOTH = 2.0      # pseudo-checks per predicted class in the recalibration table
MIN_CHECKS = 20   # fewer past checks than this: safe mode uses the checks alone


class SelfCheck:
    """What the ground remembers about the checks of one mission (one onboard model)."""

    def __init__(self, n_classes: int):
        self.K = n_classes
        self.seen = 0.0       # weighted sum of squared mistakes on all checked photos so far
        self.expected = 0.0   # ... and what the model's chances expected there
        self.pred: list[int] = []         # past checks: the model's label,
        self.truth: list[np.ndarray] = [] # the true class (one-hot),
        self.weight: list[float] = []     # and 1 / chance of being checked
        self.passes = self.safe_passes = 0

    @property
    def ratio(self) -> float:
        """Checked mistakes / expected mistakes over all checks so far (1 = the model is honest)."""
        return self.seen / self.expected if self.expected > 0 else 1.0

    @property
    def safe(self) -> bool:
        return self.expected > 0 and self.ratio > SWITCH

    def recalibrated(self, P: np.ndarray) -> np.ndarray:
        """Each photo's chances from the past checks: the share of each true class among checked photos
        that got the same label (smoothed toward the overall mix of the checked photos)."""
        w, Y, pred = np.array(self.weight), np.array(self.truth), np.array(self.pred)
        mix = (w[:, None] * Y).sum(0) / w.sum()
        table = np.tile(mix, (self.K, 1))
        for h in range(self.K):
            mine = pred == h
            if mine.any():
                table[h] = ((w[mine, None] * Y[mine]).sum(0) + SMOOTH * mix) / (w[mine].sum() + SMOOTH)
        return table[P.argmax(1)]

    def run_pass(self, P: np.ndarray, budget: int, tile_bytes: int, Y_of, rng) -> tuple[Sent, Estimate, bool]:
        """One pass: pick the checks (normal or safe mode), estimate, then learn from the checks.
        Y_of(full) gives the true classes of the checked photos (read on the ground)."""
        safe = self.safe
        sent = audit(P, budget, tile_bytes, rng, floor=1.0 if safe else 0.5)
        Y_full = Y_of(sent.full)
        check = difference(P, sent, Y_full)  # also measures this pass's seen / expected mistakes
        if not safe:
            estimate = check
        elif len(self.weight) < MIN_CHECKS:
            estimate = replace(raw_mean(P.shape, replace(sent, policy="raw random"), Y_full),
                               seen=0.0, expected=0.0)
        else:
            estimate = replace(difference(self.recalibrated(P), sent, Y_full), seen=0.0, expected=0.0)
        self.seen += check.seen
        self.expected += check.expected
        self.pred += [int(k) for k in P[sent.full].argmax(1)]
        self.truth += list(Y_full)
        self.weight += list(1 / sent.chance[sent.full])
        self.passes += 1
        self.safe_passes += safe
        return sent, estimate, safe
