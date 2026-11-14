from __future__ import annotations

import json

import pytest

from ri.common.prompts import build_prompt_batch
from ri.prompts.prompter import TEMPLATES_DIR, Prompter

TEMPLATE_NAMES = sorted(path.stem for path in TEMPLATES_DIR.glob("*.json"))


def test_every_template_has_a_cot_and_a_direct_variant() -> None:
    stems = {name.rsplit("_", 1)[0] for name in TEMPLATE_NAMES}
    assert {f"{s}_{regime}" for s in stems for regime in ("cot", "direct")} == set(TEMPLATE_NAMES)


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
def test_prompt_batch_uses_the_template_system_and_query(name: str) -> None:
    prompter = Prompter(name)
    template = json.loads((TEMPLATES_DIR / f"{name}.json").read_text())
    rows = [{"question": "What is 2 + 2?", "answer": "#### 4"}, {"question": "Q2", "answer": ""}]

    convos = build_prompt_batch(prompter, rows)

    assert convos[0] == [
        {"role": "system", "content": template["system"]},
        {"role": "user", "content": "Question: What is 2 + 2?\nAnswer:"},
    ]
    assert convos[1][1]["content"] == "Question: Q2\nAnswer:"


def test_unknown_template_raises() -> None:
    with pytest.raises(FileNotFoundError, match="no prompt template"):
        Prompter("gsm8k_missing")
