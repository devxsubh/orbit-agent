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
