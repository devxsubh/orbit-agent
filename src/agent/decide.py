"""The onboard agent: for each photo, decide what to send down.

Plain Python rules for now (LangGraph comes later, once the rules work).

How v1 works, for each pass:
1. DESCRIBE every photo: a short text label + time + position (about 50 bytes).
2. Spend the budget that's left on full photos (SEND_FULL), best first:
     a. the model is NOT SURE (it gave several answers, or none) -> send the photo, don't guess
     b. the model thinks it shows human activity (INTERESTING)
     c. everything else
3. Photos that don't fit are dropped (we chose "fresh data first").

v2 is the same, but:
- "not sure" uses more signals (several answers, answers change when asked again,
  small answer margin; see src/model/confidence.py)
- a not-sure photo that gets no full photo is sent as an "unsure: forest?" label,
  so the ground is warned instead of misled (scored as no info, not wrong).

Also here: two simple policies to compare against (send everything raw / caption everything).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.orbit.replay import Photo, Window


class Decision(Enum):
    SKIP = "skip"            # clouds / nothing useful (not used yet: AID has no cloudy photos)
    DESCRIBE = "describe"    # send a short text only
    DESCRIBE_UNSURE = "describe_unsure"  # short text marked "unsure" (v2)
    SEND_CROP = "send_crop"  # send the interesting part (not built yet)
    SEND_FULL = "send_full"  # send the whole photo (also used when not sure)
    DROPPED = "dropped"      # didn't fit in the pass


# Mission for v1: watch places with human activity. Natural scenes get a label only,
# unless there's spare budget. (AID classes; change this set for another mission.)
INTERESTING = {
    "Airport", "BaseballField", "Bridge", "Center", "Church", "Commercial", "DenseResidential",
    "Industrial", "MediumResidential", "Park", "Parking", "Playground", "Port", "RailwayStation",
    "Resort", "School", "SparseResidential", "Square", "Stadium", "StorageTanks", "Viaduct",
}


@dataclass
class ModelView:
    """What the onboard model said about one photo."""

    guess: str           # its best label ("unknown" if none)
    unsure: bool         # v1 signal: it gave several answers or none
    doubt: bool = False  # v2 signal: any of several answers / changing answers / small margin


@dataclass
class Choice:
    """What the satellite did with one photo, and what it cost."""

    photo: Photo
    decision: Decision
    label: str        # label sent down (for DESCRIBE)
    bytes_sent: int


def description(photo: Photo, label: str) -> str:
    """The short text sent instead of a photo."""
    return f"{photo.taken:%Y-%m-%dT%H:%M:%SZ},{photo.lat:.3f},{photo.lon:.3f},{label}"


def describe_bytes(photo: Photo, label: str) -> int:
    return len(description(photo, label).encode())


def _label_then_upgrade(window: Window, views: dict[str, ModelView], priority,
                        warn_unsure: bool) -> list[Choice]:
    """Shared by agent v1 and v2.

    1. A label for every photo, in time order. If warn_unsure, doubtful ones say "unsure: X?".
    2. Spend what's left on full photos, lowest `priority` number first.
    """
    choices = []
    budget = window.budget_bytes
    for photo in window.photos:
        view = views[photo.path.name]
        if warn_unsure and view.doubt:
            decision, label = Decision.DESCRIBE_UNSURE, f"unsure: {view.guess}?"
        else:
            decision, label = Decision.DESCRIBE, view.guess
        cost = describe_bytes(photo, label)
        if cost <= budget:
            choices.append(Choice(photo, decision, label, cost))
            budget -= cost
        else:
            choices.append(Choice(photo, Decision.DROPPED, "", 0))

    labelled = [c for c in choices if c.decision != Decision.DROPPED]
    for choice in sorted(labelled, key=priority):  # sort keeps time order within a priority
        extra = choice.photo.size_bytes - choice.bytes_sent  # full photo replaces the label
        if extra <= budget:
            budget -= extra
            choice.decision, choice.bytes_sent = Decision.SEND_FULL, choice.photo.size_bytes
    return choices


def plan_agent(window: Window, views: dict[str, ModelView]) -> list[Choice]:
    """Agent v1: describe everything, then upgrade the best photos to full images."""
    def priority(choice: Choice) -> int:
        view = views[choice.photo.path.name]
        if view.unsure:
            return 0                                   # not sure -> send, don't guess
        return 1 if view.guess in INTERESTING else 2   # then human activity, then the rest

    return _label_then_upgrade(window, views, priority, warn_unsure=False)


def plan_agent_v2(window: Window, views: dict[str, ModelView]) -> list[Choice]:
    """Agent v2: like v1, with the better "not sure" signal and "unsure" labels."""
    def priority(choice: Choice) -> tuple[int, int]:
        view = views[choice.photo.path.name]
        return (0 if view.doubt else 1, 0 if view.guess in INTERESTING else 1)

    return _label_then_upgrade(window, views, priority, warn_unsure=True)


def plan_raw(window: Window, views: dict[str, ModelView] | None = None) -> list[Choice]:
    """Baseline: no onboard AI. Send full photos in the order taken until the pass is full."""
    choices, budget = [], window.budget_bytes
    for photo in window.photos:
        if photo.size_bytes <= budget:
            choices.append(Choice(photo, Decision.SEND_FULL, "", photo.size_bytes))
            budget -= photo.size_bytes
        else:
            choices.append(Choice(photo, Decision.DROPPED, "", 0))
    return choices


def plan_caption(window: Window, views: dict[str, ModelView]) -> list[Choice]:
    """Baseline (NAVI-style): send only the model's label for every photo."""
    choices, budget = [], window.budget_bytes
    for photo in window.photos:
        label = views[photo.path.name].guess
        cost = describe_bytes(photo, label)
        if cost <= budget:
            choices.append(Choice(photo, Decision.DESCRIBE, label, cost))
            budget -= cost
        else:
            choices.append(Choice(photo, Decision.DROPPED, "", 0))
    return choices


POLICIES = {"send all raw": plan_raw, "caption all": plan_caption,
            "agent v1": plan_agent, "agent v2": plan_agent_v2}
