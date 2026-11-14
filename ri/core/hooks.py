"""Forward hooks that edit the hidden states a decoder block writes.

Two kinds of edit are supported. ``set_patch`` overwrites the states at given
positions with given vectors (the patch sweep). ``set_state_intervention``
moves the state at the last prompt position towards a per-question target,
either by rotating it (SLERP, norm preserved) or by replacing it with a scaled
copy of the target; the rotation can also be applied at every generated token.

Hooks are registered on ``model.model.layers[l]``, so a hook at layer ``l``
edits the state that ``output_hidden_states`` reports as ``hidden_states[l + 1]``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel

Operator = Literal["rotate", "scale"]

# Admissible doses: a rotation fraction may overshoot the target (t > 1), a
# scale is a multiple of the target's own norm.
DOSE_RANGE: dict[str, tuple[float, float]] = {"rotate": (0.0, 2.0), "scale": (0.0, 10.0)}

_EPS = 1e-8


def _unpack(output: Any) -> tuple[Any, Callable[[torch.Tensor], Any]]:
    """Split a block's forward output into its hidden-state tensor and a repacker."""
    if isinstance(output, tuple):
        lst = list(output)
        x = lst[0]

        def repack(nx: torch.Tensor) -> Any:
            lst[0] = nx
            return tuple(lst)

        return x, repack

    if hasattr(output, "last_hidden_state"):

        def repack_field(nx: torch.Tensor) -> Any:
            output.last_hidden_state = nx
            return output

        return output.last_hidden_state, repack_field

    def identity(nx: torch.Tensor) -> Any:
        return nx

    return output, identity


def _vector_count(vectors: Any) -> int:
    """How many vectors one batch row of ``hs`` holds."""
    if isinstance(vectors, torch.Tensor):
        if vectors.dim() == 1:
            return 1
        if vectors.dim() == 2:
            return int(vectors.shape[0])
        raise ValueError(f"'hs' tensors must be 1D or 2D, got shape {tuple(vectors.shape)}")
    if isinstance(vectors, (list, tuple)):
        return len(vectors)
    return 1


def set_patch(
    model: PreTrainedModel,
    patch_config: list[dict[str, Any]],
) -> list[RemovableHandle]:
    """Register hooks that overwrite hidden states at given positions.

    Each entry of ``patch_config`` holds ``layer_to_patch`` (int or list of ints),
    ``hs_position`` (per batch row, a position or list of positions; negative
    positions count from the end) and ``hs`` (per batch row, the vector or
    vectors to write, one per position). Only prompt passes (``T > 1``) are
    edited; decoding steps pass through.
    """

    def _make_hook(hs_position: list[list[int] | int], hs: list[Any]) -> Callable[..., Any]:
        for i, vectors in enumerate(hs):
            positions = hs_position[i]
            n_positions = 1 if isinstance(positions, int) else len(positions)
            n_vectors = _vector_count(vectors)
            # A tensor row carries exactly one vector per position; a list may carry more.
            too_few = n_vectors < n_positions
            mismatch = isinstance(vectors, torch.Tensor) and n_vectors != n_positions
            if too_few or mismatch:
                raise ValueError(
                    f"'hs' row {i} holds {n_vectors} vectors for {n_positions} positions"
                )

        def hook_fn(module: torch.nn.Module, inputs: Any, output: Any) -> Any:
            x, repack = _unpack(output)
            if not isinstance(x, torch.Tensor) or x.dim() != 3:
                return output

            B, T, _H = x.shape

            if T <= 1:
                return output

            y = x.clone()

            for i in range(B):
                pos_i = hs_position[i]
                pos_list = [pos_i] if isinstance(pos_i, int) else list(pos_i)
                pos_list = [p + T if p < 0 else p for p in pos_list]
                pos_list = [max(0, min(p, T - 1)) for p in pos_list]

                vec_i = hs[i]
                if isinstance(vec_i, torch.Tensor):
                    vec_tensor = vec_i.to(y.device, dtype=y.dtype)
                    if vec_tensor.dim() == 1:
                        vec_list = [vec_tensor]
                    else:
                        vec_list = [vec_tensor[m] for m in range(vec_tensor.shape[0])]
                elif isinstance(vec_i, (list, tuple)):
                    vec_list = []
                    for v in vec_i:
                        if isinstance(v, torch.Tensor):
                            vec_list.append(v.to(y.device, dtype=y.dtype))
                        else:
                            vec_list.append(torch.tensor(v, device=y.device, dtype=y.dtype))
                else:
                    vec_list = [torch.tensor(vec_i, device=y.device, dtype=y.dtype)]

                for j, pos in enumerate(pos_list):
                    y[i, pos, :] = vec_list[j]
            return repack(y)

        return hook_fn

    hooks: list[RemovableHandle] = []
    for cfg in patch_config:
        hook = _make_hook(hs_position=cfg["hs_position"], hs=cfg["hs"])
        layers = cfg["layer_to_patch"]
        if isinstance(layers, int):
            layers = [layers]
        hooks.extend(model.model.layers[layer].register_forward_hook(hook) for layer in layers)
    return hooks


def rotate_towards(states: torch.Tensor, target: torch.Tensor, t: float) -> torch.Tensor:
    """SLERP each state towards ``target`` by the fraction ``t`` of the angle between them.

    ``states`` is ``(B, S, H)`` float32 and ``target`` broadcasts to it. The
    result keeps each state's norm; ``t = 1`` lands on the target's direction and
    ``t > 1`` continues past it along the same great circle. ``cos theta`` is clamped
    away from +-1, so a state exactly anti-parallel to the target is mapped to zero
    at ``t = 0.5`` rather than left unchanged.
    """
    norm = states.norm(dim=-1, keepdim=True)
    h_hat = states / norm.clamp_min(_EPS)
    c_hat = target / target.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    cos_th = (h_hat * c_hat).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.arccos(cos_th)
    sin_th = torch.sin(theta)
    interp = (
        torch.sin((1.0 - t) * theta) * h_hat + torch.sin(t * theta) * c_hat
    ) / sin_th.clamp_min(_EPS)
    interp = interp / interp.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    safe = sin_th > 1e-4
    return torch.where(safe, norm * interp, states)


def set_state_intervention(
    model: PreTrainedModel,
    *,
    layer: int,
    target: torch.Tensor | np.ndarray,
    operator: Operator,
    dose: float,
    every_token: bool = False,
) -> list[RemovableHandle]:
    """Register a hook that moves the last prompt state at ``layer`` towards ``target``.

    ``rotate`` rotates the state by the fraction ``dose`` of its angle to the
    target, keeping its norm. ``scale`` replaces the state with ``dose * target``,
    so ``dose = 1`` inserts the target unchanged. Either edit is applied once, at
    the last position of the prompt pass. With ``every_token`` (``rotate`` only)
    the rotation is also applied to the state of every generated token.

    Generation must use a KV cache, as ``generate()`` does by default: without one
    every decoding step is a full pass, and the one-shot edit would be applied to
    each new token.

    The arithmetic runs in float32 and is cast back to the model's dtype.
    """
    if operator not in DOSE_RANGE:
        raise ValueError(f"unsupported operator: {operator!r}")
    if every_token and operator != "rotate":
        raise ValueError("every_token is only defined for operator='rotate'")
    lo, hi = DOSE_RANGE[operator]
    dose = float(dose)
    if not (lo <= dose <= hi):
        raise ValueError(f"dose must be in [{lo}, {hi}] for operator={operator}, got {dose}")
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(np.asarray(target)).float()
    if not torch.isfinite(target).all():
        raise ValueError("target must be finite")

    def _apply(seg: torch.Tensor) -> torch.Tensor:
        t = seg.detach().to(torch.float32)
        u = target.to(t.device, dtype=torch.float32).expand_as(t)
        if operator == "scale":
            return dose * u
        return rotate_towards(t, u, dose)

    def hook_fn(module: torch.nn.Module, inputs: Any, output: Any) -> Any:
        x, repack = _unpack(output)
        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            return output
        T = x.shape[1]

        # With a KV cache every decoding step is a single-position pass.
        if T <= 1:
            if every_token:
                return repack(_apply(x).to(x.dtype))
            return output

        y = x.clone()
        y[:, T - 1 : T, :] = _apply(x[:, T - 1 : T, :]).to(x.dtype)
        return repack(y)

    return [model.model.layers[layer].register_forward_hook(hook_fn)]


def remove_hooks(hooks: list[RemovableHandle]) -> None:
    for hook in hooks:
        hook.remove()
