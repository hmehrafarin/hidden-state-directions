"""Layer selection derived from the model's depth.

Layers are 0-indexed: layer ``l`` is the output of transformer block ``l``, which is
``hidden_states[l + 1]`` in a Hugging Face forward pass (``hidden_states[0]`` is the
embedding output). The paper numbers layers from one.

The late band is the last ``band_size`` even-numbered layers below the final one:
``[26, 28, 30]`` for a 32-block model and ``[22, 24, 26]`` for a 28-block model. The
final layer is never in the band because ``hidden_states[-1]`` is returned after the
model's final norm, so a state cached there is not comparable to the raw block outputs
cached at every other layer, and a hook on that block would not edit the state it was
labelled with.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

DEFAULT_BAND_SIZE = 3

LayerSpec = str | Sequence[int]


@cache
def depth_of(model_name: str) -> int:
    """Number of transformer blocks of a model, from its ``config.json``.

    ``model_name`` is a local model directory or a Hugging Face model id. For an id
    only the config is fetched, from the local cache first, so no weights are needed.
    """
    local = Path(model_name)
    if local.is_dir():
        config_path = local / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"{model_name} is a directory without a config.json")
        return _read_depth(config_path)

    from huggingface_hub import hf_hub_download

    from ri.settings import HUGGINGFACE_CACHE_DIR

    errors = []
    for local_only in (True, False):
        try:
            path = hf_hub_download(
                model_name,
                "config.json",
                cache_dir=HUGGINGFACE_CACHE_DIR,
                local_files_only=local_only,
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        return _read_depth(Path(path))
    raise ValueError(
        f"cannot determine the depth of {model_name!r}, which is needed to resolve its "
        "layers: it is not a local model directory, and its config.json is neither in the "
        "Hugging Face cache nor downloadable. Set model.model_name to a local model "
        "directory or to a Hub id whose config.json is cached or reachable (RI_CACHE_DIR "
        f"selects a non-standard cache). Last error: {errors[-1]}"
    )


def _read_depth(config_path: Path) -> int:
    with config_path.open() as handle:
        return int(json.load(handle)["num_hidden_layers"])


def num_hidden_layers(model: Any) -> int:
    """Block count for a model, an HF config, a model id or directory, or an int."""
    if isinstance(model, bool):
        raise ValueError(f"cannot determine num_hidden_layers from {model!r}")
    if isinstance(model, int):
        return model
    if isinstance(model, str):
        return depth_of(model)
    config = getattr(model, "config", model)
    value = getattr(config, "num_hidden_layers", None)
    if value is None:
        raise ValueError(f"cannot determine num_hidden_layers from {type(model)!r}")
    return int(value)


def late_band(model: Any, size: int = DEFAULT_BAND_SIZE) -> list[int]:
    """The last ``size`` even-numbered layers, excluding the final layer."""
    if size < 1:
        raise ValueError(f"band size must be >= 1, got {size}")
    n = num_hidden_layers(model)
    final = n - 1
    candidates = [layer for layer in range(0, n, 2) if layer != final]
    if len(candidates) < size:
        raise ValueError(f"model has only {len(candidates)} candidate layers, need {size}")
    return candidates[-size:]


def primary_intervention_layer(model: Any, *, band_size: int = DEFAULT_BAND_SIZE) -> int:
    """The middle layer of the late band: 28 for 32 blocks, 24 for 28 blocks."""
    return late_band(model, band_size)[band_size // 2]


def resolve_layers(spec: LayerSpec, model: Any, *, band_size: int = DEFAULT_BAND_SIZE) -> list[int]:
    """Resolve ``"band"`` or an explicit layer list against a model.

    Negative entries count from the end, so ``-1`` is the final layer. A layer outside
    the model's depth raises instead of being dropped.
    """
    n = num_hidden_layers(model)
    if isinstance(spec, str):
        if spec == "band":
            return late_band(n, band_size)
        raise ValueError(f"unknown layer spec {spec!r}; use 'band' or a list of layers")

    resolved: list[int] = []
    for layer in spec:
        value = int(layer)
        if value < 0:
            value += n
        if not 0 <= value < n:
            raise ValueError(
                f"layer {layer} is invalid for a model with {n} transformer layers "
                f"(valid 0..{n - 1})"
            )
        resolved.append(value)
    return resolved
