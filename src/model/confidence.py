"""How sure is the model? Four signals for the agent's "not sure" rule.

A. several answers:  the reply names more than one class (or none)
B. confused classes: how often this predicted class was right on the FIRST 300 photos
C. answer margin:    how far ahead the chosen words were of the next different words
                     (token log-probabilities, free: same model call)
D. agreement:        ask again with the options shuffled; do the answers match?

The agent v2 rule (is_unsure) was fixed BEFORE testing on the new photos, so it isn't tuned
to the test set. The margin threshold (8) comes from looking at 16 photos of the first set.
"""
from __future__ import annotations

import csv
import math
import time
from collections import defaultdict
from pathlib import Path

from src.model.vlm import (MODEL, _parse_words, ask_detailed, build_prompt, labels_mentioned,
                           shuffled_names)

MARGIN_SURE = 8.0  # log-prob gap; below this counts as unsure (about 3,000x more likely)


def same_word(a: str, b: str) -> bool:
    """True if two tokens are really the same choice: "City" vs "city", "far" vs "farm"."""
    a, b = a.strip().lower(), b.strip().lower()
    return a == b or a.startswith(b) or b.startswith(a)


def answer_margin(tokens: list) -> float:
    """C: smallest gap between the chosen token and the best DIFFERENT token, over the first line.

    Big (e.g. 15) = the model never considered anything else. Small (e.g. 3) = close call.
    """
    gaps = []
    for text, logprob, others in tokens:
        if "\n" in text:
            break  # only the first answer counts
        if not text.strip():
            continue
        rivals = [lp for tok, lp in others if not same_word(tok, text)]
        if rivals:
            gaps.append(logprob - max(rivals))
        elif others:  # every alternative was the same word: at least this sure
            gaps.append(logprob - min(lp for _, lp in others))
    return round(min(gaps), 3) if gaps else 0.0


def check_photo(path: Path, labels: dict[str, str], intro: str, model: str = MODEL,
                seed: int = 42, repeats: int = 3) -> dict:
    """Ask about one photo `repeats` times (options shuffled differently each time).

    The first ask is exactly the baseline question (prompt v1, same order), so its answer
    matches the baseline run. Returns everything the four signals need.
    """
    guesses, answers, seconds, margin = [], [], 0.0, 0.0
    for i in range(repeats):
        names = shuffled_names(labels, path, seed + i)
        start = time.perf_counter()
        answer, tokens = ask_detailed(path, build_prompt("v1", names, labels, intro=intro), model)
        if i == 0:
            seconds = time.perf_counter() - start
            margin = answer_margin(tokens)
        guesses.append(_parse_words(answer, names, labels))
        answers.append(answer)

    majority = max(set(guesses), key=lambda g: (guesses.count(g), g == guesses[0]))
    return {
        "guess": guesses[0],
        "answer": answers[0],
        "seconds": round(seconds, 2),
        "several": len(labels_mentioned(answers[0], labels)) != 1,   # A
        "margin": margin,                                            # C
        "agree": guesses.count(guesses[0]),                          # D (out of `repeats`)
        "majority": majority,
        "other_guesses": "|".join(guesses[1:]),
    }


def class_trust(calibration_csv: Path) -> dict[str, float]:
    """B: for each class, how often the model was right when it said that class.

    Learned from the FIRST 300 photos only. Smoothed, so a class seen once isn't 0% or 100%.
    """
    said, right = defaultdict(int), defaultdict(int)
    with open(calibration_csv) as f:
        for row in csv.DictReader(f):
            said[row["guess"]] += 1
            right[row["guess"]] += row["correct"] == "True"
    return defaultdict(lambda: 0.5, {g: (right[g] + 1) / (said[g] + 2) for g in said})


def is_unsure(several: bool, agree: int, repeats: int, margin: float) -> bool:
    """Agent v2 rule (fixed before testing): unsure if ANY of A, C, D says so."""
    return several or agree < repeats or margin < MARGIN_SURE


# ---- Agent v3: the chance that a label is right ----
# Instead of a yes/no "unsure" rule, v3 estimates p = chance the model's label is right:
#     p = sigmoid(w0 + w1 * logit(B trust) [+ w2 * (all asks agree)])
# The weights are learned on the TUNING photos (11-20), never on the final test photos (21-30).


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def asks_made(row: dict) -> int:
    """How many times this photo was asked (1 + the other guesses saved in the CSV)."""
    return 1 + len(row["other_guesses"].split("|")) if row.get("other_guesses") else 1


def features(row: dict, trust: dict[str, float], use_agreement: bool) -> list[float]:
    x = [1.0, _logit(trust[row["guess"]])]
    if use_agreement:
        x.append(1.0 if int(row["agree"]) == asks_made(row) else 0.0)
    return x


def fit_right_chance(rows: list[dict], trust: dict[str, float], use_agreement: bool,
                     steps: int = 3000, learning_rate: float = 0.5) -> list[float]:
    """Learn the weights (plain logistic regression, gradient descent) from tuning photos."""
    xs = [features(r, trust, use_agreement) for r in rows]
    ys = [1.0 if r["guess"] == r["truth"] else 0.0 for r in rows]
    w = [0.0] * len(xs[0])
    for _ in range(steps):
        grad = [0.0] * len(w)
        for x, y in zip(xs, ys):
            error = _sigmoid(sum(wi * xi for wi, xi in zip(w, x))) - y
            for i, xi in enumerate(x):
                grad[i] += error * xi / len(xs)
        w = [wi - learning_rate * g for wi, g in zip(w, grad)]
    return w


def right_chance(row: dict, trust: dict[str, float], weights: list[float]) -> float:
    """p = chance the model's first label for this photo is right."""
    x = features(row, trust, use_agreement=len(weights) == 3)
    return _sigmoid(sum(wi * xi for wi, xi in zip(weights, x)))
