"""Gold and generated answer extraction, shared by every stage that scores an answer."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

ExtractionType = Literal["numeric", "boolean", "multiple_choice"]
ExtractionMode = Literal["flexible", "strict"]

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_FINAL_ANSWER_RE = re.compile(r"Final\s+Answer\s*:\s*", flags=re.IGNORECASE)
_BOOLEAN_RE = re.compile(r"\b(true|false|yes|no)\b", re.IGNORECASE)
_MC_RE = re.compile(r"\(?([A-Z])\)|\b([A-Z])\b")


def parse_number(text: object) -> float | None:
    """The last numeric token in ``text``, with thousands separators removed."""
    if text is None:
        return None
    matches = NUM_RE.findall(str(text))
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _format_number(value: float) -> str:
    """``70.0`` -> ``"70"``, ``70.50`` -> ``"70.5"``."""
    if value.is_integer():
        return str(int(value))
    text = f"{value:.15f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _parse_numeric(text: str) -> str | None:
    value = parse_number(text)
    return None if value is None else _format_number(value)


def _parse_boolean(text: str) -> str | None:
    """The last true/false/yes/no in ``text``, title-cased."""
    matches = _BOOLEAN_RE.findall(text)
    return matches[-1].capitalize() if matches else None


def _parse_multiple_choice(text: str) -> str | None:
    """The last capital letter in ``text``, bare or in parentheses."""
    matches = _MC_RE.findall(text)
    if not matches:
        return None
    last = matches[-1]
    return str(last[0] or last[1])


_PARSERS: dict[ExtractionType, Callable[[str], str | None]] = {
    "numeric": _parse_numeric,
    "boolean": _parse_boolean,
    "multiple_choice": _parse_multiple_choice,
}


def extract_answer(batch_text: list[str]) -> list[str | None]:
    """The gold answer after the last ``####`` line marker of each text, verbatim."""
    answers = []
    for text in batch_text:
        matches = re.findall(r"^\s*####\s+(.+?)\s*$", text, re.MULTILINE)
        answers.append(matches[-1].strip() if matches else None)
    return answers


def extract_final_answer(text: str, extraction_type: ExtractionType = "numeric") -> str | None:
    """The answer parsed from the text after the last ``Final Answer:`` label."""
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if not matches:
        return None
    tail = text[matches[-1].end() :].strip()
    if not tail:
        return None
    return _PARSERS[extraction_type](tail)


def extract_answer_from_generation(
    batch_text: list[str],
    mode: ExtractionMode = "flexible",
    extraction_type: ExtractionType = "numeric",
) -> dict[str, list[str | None]]:
    """Extract answers from generated texts.

    ``strict`` reads only the value after the last ``Final Answer:``. ``flexible``
    falls back to parsing the whole text when no answer can be parsed after the last
    ``Final Answer:`` (or the label is missing), taking the last number, boolean or
    capital letter anywhere in it.

    Returns ``{"answer_text": stripped texts, "final_answer": extracted answers}``.
    """
    answers: dict[str, list[str | None]] = {"answer_text": [], "final_answer": []}
    parser = _PARSERS[extraction_type]
    for raw in batch_text:
        text = raw.strip()
        result = extract_final_answer(text, extraction_type)
        if result is None and mode == "flexible":
            result = parser(text)
        answers["answer_text"].append(text)
        answers["final_answer"].append(result)
    return answers


def answers_match(generated: str | None, gold: str | None) -> bool:
    """Case- and whitespace-insensitive equality; a missing answer never matches."""
    if generated is None or gold is None:
        return False
    return generated.strip().lower() == gold.strip().lower()
