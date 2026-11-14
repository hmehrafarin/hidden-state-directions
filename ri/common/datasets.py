"""Dataset loading for GSM8K, StrategyQA and the BIG-Bench Hard subtasks."""

from __future__ import annotations

import re
from typing import Any

import torch
from datasets import load_dataset


def get_dataset(dataset_name: str) -> Any:
    """Load a task by slug (``gsm8k``, ``strategyqa`` or ``bbh_<subtask>``)."""
    if dataset_name == "gsm8k":
        return load_dataset("openai/gsm8k", "main")
    if dataset_name == "strategyqa":
        return load_dataset("hoomeh/strategyqa-formatted")
    if dataset_name.startswith("bbh_"):
        return _load_bbh(dataset_name[4:])
    raise ValueError(f"Dataset '{dataset_name}' not supported.")


_PAREN_LETTER_RE = re.compile(r"^\(([A-Z])\)$")


def _clean_bbh_target(target: str) -> str:
    """``(A)`` -> ``A``; every other target is only stripped."""
    m = _PAREN_LETTER_RE.match(target.strip())
    return m.group(1) if m else target.strip()


def _bbh_row(row: dict[str, Any]) -> dict[str, str]:
    return {"question": row["input"], "answer": f"#### {_clean_bbh_target(row['target'])}"}


def _load_bbh(subtask: str) -> Any:
    ds = load_dataset("lukaemon/bbh", subtask)
    return ds.map(_bbh_row, remove_columns=["input", "target"])


def choose_torch_dtype() -> torch.dtype | None:
    """bfloat16 on a GPU that supports it, otherwise ``None`` to let transformers decide."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None
