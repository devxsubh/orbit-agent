"""The onboard agent: for each photo, decide what to send down.

Week 5-7 goal: build this as a LangGraph loop.
"""
from dataclasses import dataclass
from enum import Enum


class Decision(Enum):
    SKIP = "skip"            # clouds / nothing useful
    DESCRIBE = "describe"    # send a short text only
    SEND_CROP = "send_crop"  # send the interesting part
    SEND_FULL = "send_full"  # send everything (also used when not sure)


@dataclass
class Scene:
    image_path: str
    lat: float
    lon: float
    bytes_full: int          # size if we send the whole image


def decide(scene: Scene, bytes_left: int) -> Decision:
    """Pick what to do with one photo, given how much data we can still send.

    TODO:
    1. Ask the model what's in the image and how confident it is
    2. Cloudy or empty?   -> SKIP
    3. Confident?         -> DESCRIBE (cheap)
    4. Interesting + room left in the budget -> SEND_CROP
    5. Not sure?          -> SEND_FULL (don't guess)
    """
    raise NotImplementedError
