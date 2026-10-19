"""Runs the vision-language model (Gemma 3) on satellite photos.

Needs: the Ollama app running, and the model downloaded (ollama pull gemma3:4b).
Try it on one photo:
    python -m src.model.vlm path/to/photo.jpg
"""
from __future__ import annotations

import io
import random
import re
import sys
import time
from pathlib import Path

import ollama
from PIL import Image

MODEL = "gemma3:4b"


def _model_names() -> list[str]:
    try:
        models = ollama.list()["models"]
    except Exception:
        raise SystemExit("Can't reach Ollama. Open the Ollama app and try again.")
    names = []
    for m in models:
        for key in ("model", "name"):
            try:
                if m[key]:
                    names.append(m[key])
                    break
            except (KeyError, TypeError):
                pass
    return names


def check_ready(model: str = MODEL) -> str:
    """Make sure Ollama is running and the model is downloaded. Returns the model to use."""
    names = _model_names()
    if model in names:
        return model
    gemma = [n for n in names if n.startswith("gemma3")]
    if gemma:
        print(f"Using {gemma[0]} (didn't find {model}).")
        return gemma[0]
    raise SystemExit(f"Model {model} not found. Run:  ollama pull {model}")


def _prepare(image_path, size: int = 448) -> bytes:
    """EuroSAT photos are tiny (64x64), so scale them up for the model to see clearly."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if max(img.size) < size:
            img = img.resize((size, size), Image.Resampling.BICUBIC)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()


def ask(image_path, prompt: str, model: str = MODEL, size: int = 448) -> str:
    """Show the model one photo, ask one question, return its answer."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [_prepare(image_path, size)]}],
        options={"temperature": 0},  # same photo -> same answer
    )
    return response["message"]["content"].strip()


def ask_detailed(image_path, prompt: str, model: str = MODEL, size: int = 448,
                 top_logprobs: int = 10) -> tuple[str, list]:
    """Like ask(), but also returns how sure the model was of each word (token).

    Returns (answer, tokens). Each token is (text, log-probability, [(other text, log-prob), ...]):
    the word it picked, and the other words it almost picked.
    """
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [_prepare(image_path, size)]}],
        options={"temperature": 0},
        logprobs=True,
        top_logprobs=top_logprobs,
    )
    tokens = [(t.token, t.logprob, [(a.token, a.logprob) for a in (t.top_logprobs or [])])
              for t in (response.logprobs or [])]
    return response.message.content.strip(), tokens


def describe_image(image_path, model: str = MODEL) -> str:
    return ask(image_path, "Describe this satellite photo in one sentence.", model)


# Prompt versions for the baseline. v0 is the first run (26%); keep it unchanged.
#   v0: numbered list, "reply with only the number"
#   v1: class names in a shuffled order, "reply with the words" (removes the number bias)
#   v2: v1 + what each class looks like + "the photo is blurry, always one of these"
PROMPTS = ["v0", "v1", "v2"]
EUROSAT_INTRO = "This is a satellite photo of Earth, about 640 metres across."


def build_prompt(version: str, names: list[str], labels: dict[str, str],
                 details: dict[str, str] | None = None,
                 intro: str = EUROSAT_INTRO) -> str:
    """Write the question for one photo. `names` is the order to list the options in.

    intro: first line, describing the photo (each dataset has its own).
    """
    if version == "v0":
        options = "\n".join(f"{i}. {labels[n]}" for i, n in enumerate(names, 1))
        return (f"{intro}\n"
                "Which ONE of these best describes it?\n"
                f"{options}\n"
                "Reply with only the number.")
    if version == "v1":
        options = "\n".join(f"- {labels[n]}" for n in names)
        return (f"{intro}\n"
                "Which ONE of these best describes it?\n"
                f"{options}\n"
                "Reply with only the words of your choice, exactly as written.")
    if version == "v2":
        if not details:
            raise ValueError("Prompt v2 needs class details")
        options = "\n".join(f"- {labels[n]}: {details[n]}" for n in names)
        return ("This is a low-resolution satellite photo (Sentinel-2, 64x64 pixels, "
                "about 640 metres across), so it looks blurry.\n"
                "It always shows exactly one of these land types:\n"
                f"{options}\n"
                "Pick the ONE closest match, even if you are not sure.\n"
                "Reply with only the name before the colon.")
    raise ValueError(f"Unknown prompt version: {version}. Use one of {PROMPTS}")


def shuffled_names(labels: dict[str, str], image_path, seed: int = 42) -> list[str]:
    """Label order for one photo (v1/v2). Same seed + same photo -> same order."""
    names = list(labels)
    random.Random(f"{seed}-{Path(image_path).name}").shuffle(names)
    return names


def classify_image(image_path, labels: dict[str, str], model: str = MODEL,
                   prompt: str = "v0", size: int = 448,
                   details: dict[str, str] | None = None,
                   seed: int = 42, intro: str = EUROSAT_INTRO) -> tuple[str, str, float]:
    """Ask the model to pick one label, with no training (zero-shot).

    prompt: which prompt version to use (see PROMPTS).
    size:   small photos are upscaled to this many pixels before the model sees them.
    seed:   v1/v2 shuffle the options per photo; same seed + photo -> same order.
    intro:  first line of the prompt, describing the photo.

    Returns (label it picked, its raw answer, seconds taken).
    """
    names = list(labels) if prompt == "v0" else shuffled_names(labels, image_path, seed)
    question = build_prompt(prompt, names, labels, details, intro)

    start = time.perf_counter()
    answer = ask(image_path, question, model, size)
    seconds = time.perf_counter() - start

    if prompt == "v0":
        return _parse(answer, names, labels), answer, seconds
    return _parse_words(answer, names, labels), answer, seconds


def _parse(answer: str, names: list[str], labels: dict[str, str]) -> str:
    """Turn the model's reply into a label. Looks for a number first, then for the words."""
    for number in re.findall(r"\b\d{1,2}\b", answer):  # first number that is a valid option
        if 1 <= int(number) <= len(names):
            return names[int(number) - 1]
    text = answer.lower()
    hits = [(text.find(labels[n].lower()), n) for n in names if labels[n].lower() in text]
    return min(hits)[1] if hits else "unknown"


SKIP_WORDS = {"or", "with", "for", "and", "the", "area"}


def _parse_words(answer: str, names: list[str], labels: dict[str, str]) -> str:
    """For word replies (v1, v2): find which label the model chose.

    1. If a full label appears (e.g. "dense residential area"), take the first one.
       If two start at the same place, the longer one wins.
    2. Otherwise, take the label whose key word appears first.
       Key words come from the label text, e.g. "orchards or vineyards" -> orchard, vineyard.
       So a short or singular reply like "orchard" or "lake" still counts.
    """
    text = answer.lower()
    full = [(m.start(), -len(labels[n]), n) for n in names
            if (m := re.search(rf"\b{re.escape(labels[n].lower())}\b", text))]
    if full:
        return min(full)[2]

    best_pos, best_name = len(text) + 1, "unknown"
    for name in names:
        for word in labels[name].lower().split():
            if word in SKIP_WORDS:
                continue
            stem = word[:-1] if word.endswith("s") else word  # crops -> crop
            match = re.search(rf"\b{re.escape(stem)}s?\b", text)  # "sea" won't match "seasonal"
            if match and match.start() < best_pos:
                best_pos, best_name = match.start(), name
    return best_name


def labels_mentioned(answer: str, labels: dict[str, str]) -> list[str]:
    """All different labels in a reply, e.g. "parking lot\\nviaduct" -> [Parking, Viaduct].

    More than one means the model isn't sure. Each line (or comma part) is read separately.
    """
    found = []
    for part in re.split(r"[\n,;/]", answer):
        name = _parse_words(part, list(labels), labels)
        if name != "unknown" and name not in found:
            found.append(name)
    return found


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.model.vlm path/to/photo.jpg")
        sys.exit(1)
    use = check_ready()
    print(describe_image(sys.argv[1], use))
