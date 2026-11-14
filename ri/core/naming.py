"""Short model identifiers used to namespace every output directory."""

from __future__ import annotations

import re


def model_shortname(model_name: str) -> str:
    """Compact identifier for a Hugging Face model id.

    ``meta-llama/Llama-3.1-8B-Instruct`` -> ``llama8B``,
    ``Qwen/Qwen2.5-7B-Instruct`` -> ``qwen7B``.
    """
    base = str(model_name).split("/")[-1]
    param = re.search(r"(\d+[Bb])", base)
    name = re.search(r"([A-Za-z]+)", base)
    if param and name:
        return f"{name.group(1).lower()}{param.group(1).upper()}"
    return base
