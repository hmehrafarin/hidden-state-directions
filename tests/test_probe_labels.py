"""The control labels and the training pool of a fixed cache, pinned to the paper's draws."""

from __future__ import annotations

import numpy as np
import pytest

from ri.activations.store import activation_sample_path, write_activation_sample
from ri.probe.data import load_task_layer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TASK = "bbh_navigate"
LAYER = 28


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Ten questions of 3-6 states; state i of question q is recovering when 3 divides i + q."""
    root = tmp_path_factory.mktemp("hs_cache")
    for q in range(10):
        n = 3 + q % 4
        write_activation_sample(
            activation_sample_path(root, MODEL, TASK, q),
            sample_idx=q,
            layer_hidden_states={LAYER: np.full((n, 2), float(q))},
            layer_labels={LAYER: (np.arange(n) + q) % 3 == 0},
            metadata={"model_name": MODEL, "dataset": TASK, "num_tokens": n},
        )
    return str(root)


def _bits(values: list[bool] | list[int]) -> str:
    return "".join(str(int(v)) for v in values)


def test_the_labels_and_the_pool_are_the_paper_draws(cache: str) -> None:
    rows = load_task_layer(MODEL, cache, TASK, LAYER, seed=42).rows
    assert _bits(rows["fold"].to_list()) == "4441111000003333331114444222223333330002222"
    assert _bits(rows["label"].to_list()) == "1000010010011001000010100100100010010101001"
    assert _bits(rows["origin_label"].to_list()) == "0100100010010100110100100010010100110100100"
    assert (
        _bits(rows["within_position_label"].to_list())
        == "0000010010010001010011100100101010000101001"
    )
    assert _bits(rows["in_pool"].to_list()) == "1001010111011111010110111111110011010111011"


def test_the_draws_follow_the_seed(cache: str) -> None:
    pinned = load_task_layer(MODEL, cache, TASK, LAYER, seed=42).rows
    other = load_task_layer(MODEL, cache, TASK, LAYER, seed=43).rows
    assert pinned["label"].equals(other["label"])
    for column in ("origin_label", "within_position_label", "in_pool"):
        assert not pinned[column].equals(other[column])


def test_the_pool_balances_each_fold_by_downsampling_its_majority(cache: str) -> None:
    rows = load_task_layer(MODEL, cache, TASK, LAYER, seed=42).rows
    for fold in range(5):
        in_fold = rows.filter(rows["fold"] == fold)
        positives = int(in_fold["label"].sum())
        minority = min(positives, in_fold.height - positives)
        pool = in_fold.filter(in_fold["in_pool"])
        assert (int(pool["label"].sum()), pool.height) == (minority, 2 * minority)
