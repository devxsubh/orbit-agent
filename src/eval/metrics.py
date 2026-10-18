"""How we score a run.

Main number: useful information that reached the ground per megabyte sent.
"""


def accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def useful_per_mb(correct: int, bytes_sent: int) -> float:
    """Correct answers delivered to the ground for every MB sent down."""
    mb = bytes_sent / 1_000_000
    return correct / mb if mb else 0.0
