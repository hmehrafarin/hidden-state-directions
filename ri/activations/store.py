"""HDF5 activation cache: one file per question.

Each file holds, per layer, the hidden state at every CoT position (float16), the
recovering/failed label of each position and the position index, plus the question's
metadata as file attributes. Layout:
``<cache_dir>/<model_short>/<dataset>/sample_NNNN.h5``. ``cache_dir`` may be an
``hf://<org>/<repo>`` dataset spec, which is downloaded on demand and read-only.

This module depends only on h5py and numpy so that readers do not import torch.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ri.core.naming import model_shortname
from ri.tasks import task_slug

HF_CACHE_SCHEME = "hf://"

_COMPRESS: dict[str, Any] = {"compression": "gzip", "compression_opts": 9, "shuffle": True}


@cache
def _hf_snapshot(repo_id: str, patterns: tuple[str, ...]) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=list(patterns))
    )


def resolve_cache_root(cache_dir: str | Path, subpath: str) -> tuple[Path, bool]:
    """Resolve a cache root, downloading the files under ``subpath`` for an ``hf://`` spec.

    Returns ``(root, writable)``; sample files live under ``root / subpath``. A local
    directory is returned as is and is writable; a Hub snapshot is read-only.
    """
    spec = str(cache_dir)
    if not spec.startswith(HF_CACHE_SCHEME):
        return Path(cache_dir), True
    repo_id = spec[len(HF_CACHE_SCHEME) :].removeprefix("datasets/").strip("/")
    return _hf_snapshot(repo_id, (f"{subpath}/*",)), False


def activation_sample_dir(cache_dir: str | Path, model_name: str, dataset: str) -> Path:
    """Directory holding one model's cached samples for one dataset.

    Nothing is created; :func:`write_activation_sample` creates the directory it
    writes into.
    """
    subpath = f"{model_shortname(model_name)}/{task_slug(dataset)}"
    root, writable = resolve_cache_root(cache_dir, subpath)
    path = root / subpath
    if not writable and not path.is_dir():
        raise FileNotFoundError(f"{cache_dir} has no files under {subpath}")
    return path


def activation_sample_path(
    cache_dir: str | Path,
    model_name: str,
    dataset: str,
    sample_idx: int,
) -> Path:
    """Path of one question's file, ``sample_NNNN.h5``."""
    return activation_sample_dir(cache_dir, model_name, dataset) / f"sample_{sample_idx:04d}.h5"


def sample_has_activation_layers(
    path: Path,
    layers: list[int],
    *,
    expected_metadata: dict[str, Any] | None = None,
) -> bool:
    """True when ``path`` holds every layer in ``layers`` and matches ``expected_metadata``.

    Metadata values are compared as strings; a None expectation is ignored.
    """
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            if "layers" not in handle:
                return False
            existing = {int(key) for key in handle["layers"]}
            if expected_metadata:
                for key, expected in expected_metadata.items():
                    if expected is None:
                        continue
                    if key not in handle.attrs or str(handle.attrs[key]) != str(expected):
                        return False
    except OSError:
        return False
    return all(layer in existing for layer in layers)


def write_activation_sample(
    path: Path,
    *,
    sample_idx: int,
    layer_hidden_states: dict[int, np.ndarray],
    layer_labels: dict[int, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    """Write one question's states and labels atomically.

    ``layer_hidden_states[l]`` is ``[n_positions, hidden_size]`` and is stored as float16;
    ``layer_labels[l]`` is ``[n_positions]`` booleans. None-valued metadata is skipped.
    """
    if set(layer_hidden_states) != set(layer_labels):
        raise ValueError("layer_hidden_states and layer_labels must contain the same layers")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as handle:
        handle.attrs["sample_idx"] = int(sample_idx)
        for key, value in metadata.items():
            if value is None:
                continue
            handle.attrs[key] = value

        layers_group = handle.create_group("layers")
        for layer, hidden_states in layer_hidden_states.items():
            labels = layer_labels[layer]
            if len(hidden_states) != len(labels):
                raise ValueError(
                    f"layer {layer} has {len(hidden_states)} hidden states but {len(labels)} labels"
                )
            group = layers_group.create_group(str(int(layer)))
            group.create_dataset(
                "hidden_states",
                data=np.asarray(hidden_states, dtype=np.float16),
                **_COMPRESS,
            )
            group.create_dataset("labels", data=np.asarray(labels, dtype=bool), **_COMPRESS)
            group.create_dataset(
                "source_positions",
                data=np.arange(len(labels), dtype=np.int32),
                **_COMPRESS,
            )

    tmp.replace(path)


def read_activation_labels(path: str | Path, layer: int) -> np.ndarray:
    """Recovering (True) / failed (False) label of every position at ``layer``."""
    with h5py.File(path, "r") as handle:
        return handle[f"layers/{int(layer)}/labels"][:].astype(bool)


def read_activation_layer_ids(path: str | Path) -> list[int]:
    """Sorted layers present in the file."""
    with h5py.File(path, "r") as handle:
        if "layers" not in handle:
            return []
        return sorted(int(key) for key in handle["layers"])


def read_activation_hidden_states(path: str | Path, layer: int) -> np.ndarray:
    """``[n_positions, hidden_size]`` states at ``layer``, as float32."""
    with h5py.File(path, "r") as handle:
        return handle[f"layers/{int(layer)}/hidden_states"][:].astype(np.float32)


def read_activation_metadata(path: str | Path) -> dict[str, Any]:
    """The file's attributes."""
    with h5py.File(path, "r") as handle:
        return dict(handle.attrs)
