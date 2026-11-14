"""The patch lands where the sweep assumes, on Llama and Qwen2 blocks.

A patch that silently does nothing produces all-failed labels, which looks like a result
rather than a bug, so the layer index, the hook site and the position are pinned here.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from ri.core.hooks import remove_hooks, set_patch
from ri.patching.pipeline import hidden_state_index

NUM_LAYERS = 6
HIDDEN = 32
PATCH_LAYER = NUM_LAYERS - 2


@pytest.fixture(scope="module")
def prompt() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 128, (1, 12))


def _block_output(model: Any, prompt: torch.Tensor, layer: int) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["hs"] = output[0] if isinstance(output, tuple) else output

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            model(prompt)
    finally:
        handle.remove()
    return captured["hs"]


def _logits_with_patch(
    model: Any, prompt: torch.Tensor, layer: int, vector: torch.Tensor
) -> torch.Tensor:
    hooks = set_patch(model, [{"layer_to_patch": layer, "hs_position": [-1], "hs": [vector]}])
    try:
        with torch.inference_mode():
            return model(prompt).logits.clone()
    finally:
        remove_hooks(hooks)


def _resident(model: Any, prompt: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        hidden = model(prompt, output_hidden_states=True).hidden_states
    return hidden[hidden_state_index(PATCH_LAYER, NUM_LAYERS + 1)][0, -1, :].clone()


def test_layer_index_is_the_block_output(tiny_causal_lm: Any, prompt: torch.Tensor) -> None:
    """Layer ``l`` is ``hidden_states[l + 1]``, the output of block ``l``."""
    with torch.inference_mode():
        hidden = tiny_causal_lm(prompt, output_hidden_states=True).hidden_states
    assert len(hidden) == NUM_LAYERS + 1
    for layer in range(NUM_LAYERS - 1):
        index = hidden_state_index(layer, len(hidden))
        assert index == layer + 1
        torch.testing.assert_close(
            _block_output(tiny_causal_lm, prompt, layer), hidden[index], atol=1e-5, rtol=0
        )


def test_final_hidden_state_is_after_the_norm(tiny_causal_lm: Any, prompt: torch.Tensor) -> None:
    """``hidden_states[-1]`` is normed, which is why the late band excludes the final layer."""
    with torch.inference_mode():
        hidden = tiny_causal_lm(prompt, output_hidden_states=True).hidden_states
        last = _block_output(tiny_causal_lm, prompt, NUM_LAYERS - 1)
        assert not torch.allclose(last, hidden[-1], atol=1e-5)
        torch.testing.assert_close(tiny_causal_lm.model.norm(last), hidden[-1], atol=1e-5, rtol=0)


def test_negative_and_out_of_range_layers() -> None:
    assert hidden_state_index(-1, NUM_LAYERS + 1) == NUM_LAYERS
    for bad in (NUM_LAYERS, NUM_LAYERS + 2, -NUM_LAYERS - 1):
        with pytest.raises(ValueError, match="invalid"):
            hidden_state_index(bad, NUM_LAYERS + 1)


def test_identity_patch_is_a_noop(tiny_causal_lm: Any, prompt: torch.Tensor) -> None:
    with torch.inference_mode():
        baseline = tiny_causal_lm(prompt).logits.clone()
    patched = _logits_with_patch(
        tiny_causal_lm, prompt, PATCH_LAYER, _resident(tiny_causal_lm, prompt)
    )
    torch.testing.assert_close(baseline, patched, atol=1e-5, rtol=0)


def test_foreign_patch_changes_the_output(tiny_causal_lm: Any, prompt: torch.Tensor) -> None:
    with torch.inference_mode():
        baseline = tiny_causal_lm(prompt).logits.clone()
    resident = _resident(tiny_causal_lm, prompt)
    torch.manual_seed(7)
    foreign = torch.randn_like(resident) * resident.norm() / HIDDEN**0.5
    assert not torch.allclose(
        baseline, _logits_with_patch(tiny_causal_lm, prompt, PATCH_LAYER, foreign), atol=1e-4
    )
    assert not torch.allclose(
        baseline,
        _logits_with_patch(tiny_causal_lm, prompt, PATCH_LAYER, torch.zeros_like(resident)),
        atol=1e-4,
    )


def test_patch_only_touches_the_last_position(tiny_causal_lm: Any, prompt: torch.Tensor) -> None:
    index = hidden_state_index(PATCH_LAYER + 1, NUM_LAYERS + 1)
    with torch.inference_mode():
        before = tiny_causal_lm(prompt, output_hidden_states=True).hidden_states[index].clone()
    torch.manual_seed(11)
    hooks = set_patch(
        tiny_causal_lm,
        [{"layer_to_patch": PATCH_LAYER, "hs_position": [-1], "hs": [torch.randn(HIDDEN)]}],
    )
    try:
        with torch.inference_mode():
            after = tiny_causal_lm(prompt, output_hidden_states=True).hidden_states[index]
    finally:
        remove_hooks(hooks)
    torch.testing.assert_close(before[:, :-1], after[:, :-1], atol=1e-5, rtol=0)
    assert not torch.allclose(before[:, -1], after[:, -1], atol=1e-4)


def test_patch_is_skipped_on_decode_steps(tiny_causal_lm: Any, prompt: torch.Tensor) -> None:
    single = prompt[:, -1:]
    with torch.inference_mode():
        baseline = tiny_causal_lm(single).logits.clone()
    torch.manual_seed(13)
    patched = _logits_with_patch(tiny_causal_lm, single, PATCH_LAYER, torch.randn(HIDDEN))
    torch.testing.assert_close(baseline, patched, atol=1e-6, rtol=0)
