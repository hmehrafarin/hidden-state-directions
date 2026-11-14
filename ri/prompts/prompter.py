from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TEMPLATES_DIR = Path(__file__).parent / "templates"


class Prompter:
    """A prompt template loaded from ``ri/prompts/templates/<template_name>.json``."""

    __slots__ = ("template", "template_name")

    def __init__(self, template_name: str) -> None:
        path = TEMPLATES_DIR / f"{template_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"no prompt template {template_name!r} at {path}")
        self.template_name = template_name
        self.template: dict[str, Any] = json.loads(path.read_text())

    def create_prompt(self, batch_row: Sequence[Mapping[str, Any]]) -> list[str]:
        """The user prompt for each row, from the template's ``query`` field."""
        return [
            self.template["query"].format(question=row["question"], answer=row["answer"]).strip()
            for row in batch_row
        ]
