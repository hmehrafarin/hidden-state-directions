from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ri.prompts.prompter import Prompter


def build_prompt_batch(
    prompter: Prompter,
    batch_row: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, str]]]:
    """System and user messages for each row, with the gold answer left out."""
    adjusted_rows = [{"question": row.get("question", ""), "answer": ""} for row in batch_row]
    system_text = prompter.template["system"]
    return [
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_prompt},
        ]
        for user_prompt in prompter.create_prompt(adjusted_rows)
    ]
