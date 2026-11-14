from __future__ import annotations

from pathlib import Path

import pytest

from ri import paths

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
QWEN = "Qwen/Qwen2.5-7B-Instruct"


def test_results_root_defaults_to_the_working_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(paths.RESULTS_ROOT_ENV, raising=False)
    assert paths.results_root() == Path("results")


def test_results_root_honours_the_environment(results_root: Path) -> None:
    assert paths.results_root() == results_root


def test_model_is_the_outermost_axis(results_root: Path) -> None:
    assert paths.model_root(LLAMA) == results_root / "llama8B"
    assert paths.model_root(QWEN) == results_root / "qwen7B"


def test_stage_dir_appends_parts(results_root: Path) -> None:
    got = paths.stage_dir(QWEN, "probe", "layer24", "bbh_snarks")
    assert got == results_root / "qwen7B" / "probe" / "layer24" / "bbh_snarks"
    assert paths.stage_dir(LLAMA, "eval") == results_root / "llama8B" / "eval"
