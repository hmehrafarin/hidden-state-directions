"""The patch hook and the state-intervention hook of ``ri.core.hooks``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from ri.core.hooks import (
    Operator,
    remove_hooks,
    rotate_towards,
    set_patch,
    set_state_intervention,
)

HID = 8


class _PassthroughLayer(nn.Module):
    """Stands in for a decoder block: returns its input as a ``(hidden_states,)`` tuple."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x,)


def _model() -> Any:
    return SimpleNamespace(model=SimpleNamespace(layers=[_PassthroughLayer()]))


def _run(model: Any, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        (out,) = model.model.layers[0](x)
    return out


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def _hooked(target: torch.Tensor | np.ndarray, **kwargs: Any) -> tuple[Any, list[Any]]:
    model = _model()
    hooks = set_state_intervention(model, layer=0, target=target, **kwargs)
    return model, hooks


def test_rotate_touches_only_the_last_prompt_position_and_no_decoding_step() -> None:
    torch.manual_seed(0)
    model, hooks = _hooked(torch.randn(HID), operator="rotate", dose=0.7)
    try:
        x = torch.randn(1, 5, HID)
        out = _run(model, x)
        assert torch.equal(out[:, :-1], x[:, :-1])
        assert not torch.allclose(out[:, -1], x[:, -1])
        step = torch.randn(1, 1, HID)
        assert torch.equal(_run(model, step), step)
    finally:
        remove_hooks(hooks)


def test_rotate_preserves_the_norm_and_follows_the_great_circle() -> None:
    torch.manual_seed(11)
    target = torch.randn(HID)
    x = torch.randn(1, 2, HID)
    h = x[0, -1]
    c_hat = target / target.norm()
    theta = torch.arccos(torch.clamp((h / h.norm()) @ c_hat, -1, 1))
    outs = {}
    for t in (0.0, 0.5, 1.0, 1.25):
        model, hooks = _hooked(target, operator="rotate", dose=t)
        try:
            outs[t] = _run(model, x)[0, -1]
        finally:
            remove_hooks(hooks)
        assert torch.allclose(outs[t].norm(), h.norm(), atol=1e-4)
    assert torch.allclose(outs[0.0], h, atol=1e-4)
    assert abs(_cos(outs[1.0], c_hat) - 1.0) < 1e-4
    for t in (0.5, 1.0, 1.25):
        assert abs(_cos(outs[t], h) - float(torch.cos(t * theta))) < 1e-3
    assert _cos(outs[1.25], c_hat) < 1.0 - 1e-6


def test_rotate_leaves_a_state_parallel_to_the_target_unchanged() -> None:
    target = torch.zeros(HID)
    target[0] = 1.0
    model, hooks = _hooked(target, operator="rotate", dose=0.5)
    try:
        x = torch.zeros(1, 2, HID)
        x[0, -1, 0] = 4.0
        assert torch.equal(_run(model, x), x)
    finally:
        remove_hooks(hooks)


@pytest.mark.parametrize("operator", ["rotate", "scale"])
def test_a_non_finite_target_is_rejected(operator: Operator) -> None:
    target = torch.ones(HID)
    target[2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        set_state_intervention(_model(), layer=0, target=target, operator=operator, dose=1.0)


def test_scale_inserts_dose_times_the_target() -> None:
    torch.manual_seed(40)
    target = torch.randn(HID) * 3.7
    for dose in (0.25, 1.0, 4.0):
        model, hooks = _hooked(target, operator="scale", dose=dose)
        try:
            x = torch.randn(1, 3, HID)
            out = _run(model, x)
            assert torch.equal(out[0, -1], dose * target)
            assert torch.equal(out[:, :-1], x[:, :-1])
            step = torch.randn(1, 1, HID)
            assert torch.equal(_run(model, step), step)
        finally:
            remove_hooks(hooks)


def test_every_token_rotates_the_last_prompt_position_and_every_decoding_step() -> None:
    torch.manual_seed(50)
    target = torch.randn(HID)
    model, hooks = _hooked(target, operator="rotate", dose=0.3, every_token=True)
    try:
        x = torch.randn(1, 5, HID)
        out = _run(model, x)
        assert torch.equal(out[:, :-1], x[:, :-1])
        assert not torch.allclose(out[:, -1], x[:, -1])
        step = torch.randn(1, 1, HID)
        rotated = _run(model, step)
        assert torch.allclose(rotated, rotate_towards(step, target.expand_as(step), 0.3))
        assert torch.allclose(rotated.norm(), step.norm(), atol=1e-5)
    finally:
        remove_hooks(hooks)


def test_every_token_at_dose_zero_is_a_no_op() -> None:
    torch.manual_seed(51)
    model, hooks = _hooked(torch.randn(HID), operator="rotate", dose=0.0, every_token=True)
    try:
        step = torch.randn(1, 1, HID)
        assert torch.allclose(_run(model, step), step, atol=1e-5)
    finally:
        remove_hooks(hooks)


def test_the_edit_is_computed_in_float32_and_cast_back() -> None:
    torch.manual_seed(5)
    target = torch.randn(HID)
    model, hooks = _hooked(target, operator="rotate", dose=1.0)
    try:
        x = torch.randn(1, 3, HID, dtype=torch.bfloat16)
        out = _run(model, x)
        assert out.dtype == torch.bfloat16
        expected = rotate_towards(x[:, -1:].float(), target.expand(1, 1, HID), 1.0)
        assert torch.equal(out[:, -1:], expected.to(torch.bfloat16))
    finally:
        remove_hooks(hooks)


def test_a_numpy_target_is_accepted() -> None:
    target = np.arange(HID, dtype=np.float32)
    model, hooks = _hooked(target, operator="scale", dose=2.0)
    try:
        out = _run(model, torch.zeros(1, 2, HID))
        assert torch.equal(out[0, -1], torch.from_numpy(2.0 * target))
    finally:
        remove_hooks(hooks)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"operator": "rotate", "dose": 2.5}, "dose"),
        ({"operator": "rotate", "dose": -0.1}, "dose"),
        ({"operator": "scale", "dose": 11.0}, "dose"),
        ({"operator": "scale", "dose": 1.0, "every_token": True}, "every_token"),
        ({"operator": "replace", "dose": 1.0}, "operator"),
    ],
)
def test_invalid_settings_are_rejected(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        set_state_intervention(_model(), layer=0, target=torch.ones(HID), **kwargs)


def test_set_patch_overwrites_the_given_positions_of_prompt_passes_only() -> None:
    torch.manual_seed(7)
    vectors = torch.randn(2, HID)
    model = _model()
    hooks = set_patch(model, [{"layer_to_patch": 0, "hs_position": [[1, -1]], "hs": [vectors]}])
    try:
        x = torch.randn(1, 4, HID)
        out = _run(model, x)
        assert torch.equal(out[0, 1], vectors[0])
        assert torch.equal(out[0, 3], vectors[1])
        assert torch.equal(out[0, [0, 2]], x[0, [0, 2]])
        step = torch.randn(1, 1, HID)
        assert torch.equal(_run(model, step), step)
    finally:
        remove_hooks(hooks)


@pytest.mark.parametrize(
    "hs",
    [[[torch.ones(HID)]], [torch.ones(HID)], [torch.ones(3, HID)], [torch.ones(1, 2, HID)]],
)
def test_set_patch_requires_a_vector_per_position(hs: list[Any]) -> None:
    with pytest.raises(ValueError, match="'hs'"):
        set_patch(_model(), [{"layer_to_patch": [0], "hs_position": [[0, 1]], "hs": hs}])


def test_hooks_on_a_real_model_change_generation_and_are_removed(tiny_causal_lm: Any) -> None:
    model = tiny_causal_lm
    torch.manual_seed(1)
    ids = torch.randint(3, 128, (1, 7))
    target = torch.randn(model.config.hidden_size) * 5

    def generate() -> torch.Tensor:
        with torch.no_grad():
            return model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=0,
            )

    seen: list[torch.Tensor] = []
    capture = model.model.layers[3].register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(
            (args[0] if args else kwargs["hidden_states"]).clone()
        ),
        with_kwargs=True,
    )
    try:
        baseline = generate()
        clean = list(seen)
        seen.clear()
        hooks = set_state_intervention(model, layer=2, target=target, operator="rotate", dose=1.0)
        try:
            generate()
        finally:
            remove_hooks(hooks)
        edited = list(seen)
    finally:
        capture.remove()

    assert torch.equal(edited[0][:, :-1], clean[0][:, :-1])
    assert _cos(edited[0][0, -1], target) > 1 - 1e-5
    assert torch.allclose(edited[0][0, -1].norm(), clean[0][0, -1].norm(), atol=1e-4)
    assert torch.equal(generate(), baseline)
