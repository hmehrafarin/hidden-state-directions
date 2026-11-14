"""Answer extraction, which produces every label and accuracy in the pipeline.

In ``flexible`` mode a missing ``Final Answer:`` falls back to the last number,
boolean or capital letter anywhere in the text. These tests pin that behaviour.
"""

from __future__ import annotations

import pytest

from ri.utils.extraction import (
    answers_match,
    extract_answer,
    extract_answer_from_generation,
    extract_final_answer,
    parse_number,
)


def test_gold_answer_comes_from_the_last_hash_marker() -> None:
    assert extract_answer(["reasoning\n#### 18"]) == ["18"]
    assert extract_answer(["#### 1\nmore\n#### 2"]) == ["2"]
    assert extract_answer(["no marker here"]) == [None]


def test_gold_answer_keeps_non_numeric_values_verbatim() -> None:
    assert extract_answer(["#### True", "#### (C)"]) == ["True", "(C)"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("18", 18.0),
        ("$18", 18.0),
        ("1,234", 1234.0),
        ("-5", -5.0),
        ("3.50", 3.5),
        ("no digits", None),
    ],
)
def test_parse_number_handles_currency_and_separators(text: str, expected: float | None) -> None:
    assert parse_number(text) == expected


def test_integral_floats_format_without_a_trailing_zero() -> None:
    result = extract_final_answer("Final Answer: 18.0", "numeric")
    assert result == "18"
    assert answers_match(result, "18")


def test_strict_mode_requires_the_scaffold() -> None:
    strict = extract_answer_from_generation(
        ["The result is 42."], mode="strict", extraction_type="numeric"
    )
    assert strict["final_answer"] == [None]


def test_flexible_mode_falls_back_to_the_last_number_anywhere() -> None:
    text = "First 9 * 5 = 45, then 46 - 22 = 24, so the total is 24. But note 7."
    flexible = extract_answer_from_generation([text], mode="flexible", extraction_type="numeric")
    assert flexible["final_answer"] == ["7"]


def test_scaffold_wins_over_the_flexible_fallback() -> None:
    text = "Working: 45, then 22. Final Answer: 24"
    for mode in ("strict", "flexible"):
        got = extract_answer_from_generation([text], mode=mode, extraction_type="numeric")
        assert got["final_answer"] == ["24"], mode


def test_last_scaffold_wins_when_repeated() -> None:
    text = "Final Answer: 1\nreconsidering\nFinal Answer: 2"
    assert extract_final_answer(text, "numeric") == "2"


def test_answer_text_is_the_stripped_generation() -> None:
    got = extract_answer_from_generation(["  Final Answer: 3 \n"], extraction_type="numeric")
    assert got["answer_text"] == ["Final Answer: 3"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Final Answer: True", "True"),
        ("Final Answer: yes", "Yes"),
        ("Final Answer: NO", "No"),
        ("Final Answer: false", "False"),
    ],
)
def test_boolean_answers_are_title_cased(text: str, expected: str) -> None:
    assert extract_final_answer(text, "boolean") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Final Answer: (C)", "C"),
        ("Final Answer: C", "C"),
    ],
)
def test_multiple_choice_accepts_bare_and_parenthesised_letters(text: str, expected: str) -> None:
    assert extract_final_answer(text, "multiple_choice") == expected


def test_multiple_choice_fallback_matches_a_capital_letter_in_prose() -> None:
    got = extract_answer_from_generation(
        ["The options are unclear so I"], mode="flexible", extraction_type="multiple_choice"
    )
    assert got["final_answer"] == ["I"]


@pytest.mark.parametrize(
    ("generated", "gold", "expected"),
    [
        ("18", "18", True),
        ("  18 ", "18", True),
        ("true", "True", True),
        ("c", "C", True),
        ("18", "19", False),
        (None, "18", False),
        ("18", None, False),
        (None, None, False),
    ],
)
def test_answers_match_is_case_and_whitespace_insensitive(
    generated: str | None, gold: str | None, expected: bool
) -> None:
    assert answers_match(generated, gold) is expected
