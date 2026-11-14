"""Tokenizer helpers: chat rendering, input tensors, stop tokens and token counts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def strip_llama_default_metadata(tokenizer: Any) -> None:
    """Remove the knowledge-cutoff and current-date lines from a Llama chat template."""
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return

    markers = ("Cutting Knowledge Date:", "Today Date:")
    if not any(marker in template for marker in markers):
        return

    lines = template.splitlines(keepends=True)
    cleaned = [line for line in lines if not any(marker in line for marker in markers)]
    tokenizer.chat_template = "".join(cleaned)


def decode_tokens(tokenizer: Any, token_array: Any) -> Any:
    """Decode a 1D token array to a string, or a 2D array to a list of strings."""
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return tokenizer.decode(token_array)


def get_pad_id(tokenizer: Any) -> int:
    """The tokenizer's pad token id, or 0 when it has none."""
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_id, int) and pad_id >= 0:
        return pad_id
    return 0


def _flatten_special_values(values: Any, output: list[str]) -> None:
    if isinstance(values, str):
        output.append(values)
        return
    if isinstance(values, dict):
        for val in values.values():
            _flatten_special_values(val, output)
        return
    if isinstance(values, Iterable):
        for val in values:
            _flatten_special_values(val, output)


def _gather_special_tokens(tokenizer: Any) -> list[str]:
    """Every special token string the tokenizer declares, in first-seen order."""
    tokens: list[str] = []
    for attr in (
        "special_tokens_map",
        "special_tokens_map_extended",
        "all_special_tokens",
        "additional_special_tokens",
    ):
        values = getattr(tokenizer, attr, None)
        if values:
            _flatten_special_values(values, tokens)
    return list(dict.fromkeys(tok for tok in tokens if isinstance(tok, str) and tok))


def get_eot_token(tokenizer: Any) -> str | None:
    """The first declared special token whose string contains ``eot``, case-insensitively."""
    return next((tok for tok in _gather_special_tokens(tokenizer) if "eot" in tok.lower()), None)


# Turn terminators of tokenizers that declare no "eot"-named token (Qwen/ChatML).
# They are matched exactly against the declared special tokens, so Llama, whose
# tokens are <|eot_id|> and <|end_of_text|>, is unaffected.
_EXTRA_STOP_TOKENS: tuple[str, ...] = (
    "<|im_end|>",
    "<|endoftext|>",
)


def get_extra_stop_tokens(tokenizer: Any) -> list[str]:
    """The entries of ``_EXTRA_STOP_TOKENS`` that this tokenizer declares."""
    declared = set(_gather_special_tokens(tokenizer))
    return [tok for tok in _EXTRA_STOP_TOKENS if tok in declared]


def get_eos_token_ids(tokenizer: Any) -> int | list[int]:
    """Stop token id(s) for generation: EOS, end-of-turn and declared extra terminators.

    Returns a single int when there is one id, a list otherwise, and the pad id when
    nothing is found.
    """
    ids: list[int] = []

    def _append(val: Any) -> None:
        if isinstance(val, int) and val >= 0:
            ids.append(val)
        elif isinstance(val, (list, tuple)):
            for inner in val:
                _append(inner)

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token and callable(convert):
            eos_id = convert(eos_token)
    _append(eos_id)

    eot_token = get_eot_token(tokenizer)
    if eot_token and callable(convert):
        _append(convert(eot_token))

    if callable(convert):
        for tok in get_extra_stop_tokens(tokenizer):
            _append(convert(tok))

    unique = list(dict.fromkeys(ids))
    if not unique:
        return get_pad_id(tokenizer)
    if len(unique) == 1:
        return unique[0]
    return unique


def count_generated_tokens(tokenizer: Any, token_ids: Any) -> int:
    """Number of generated tokens, without the trailing run of stop and pad ids.

    ``token_ids`` is the generated continuation of one prompt (a 1D tensor or a
    sequence of ints). The stop ids are those of :func:`get_eos_token_ids`, and the
    pad id is that of :func:`get_pad_id`, which falls back to 0 when the tokenizer
    declares no pad token (Llama-3.1, where id 0 is ``!``). A generation that ends
    in such ids, including one cut at the token budget whose last tokens are id 0,
    therefore counts fewer tokens than ``len(token_ids)``.
    """
    eos_ids = get_eos_token_ids(tokenizer)
    stop_ids = {int(eos_ids)} if isinstance(eos_ids, int) else {int(i) for i in eos_ids}
    stop_ids.add(get_pad_id(tokenizer))

    row = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    length = len(row)
    while length > 0 and int(row[length - 1]) in stop_ids:
        length -= 1
    return length


def render_prompts(
    tokenizer: Any,
    prompts: Any,
    *,
    system_prompt: bool = False,
    add_generation_prompt: bool = False,
) -> list[str]:
    """Render a batch of conversations into the exact text passed to the tokenizer.

    With ``system_prompt`` the chat template is applied; otherwise only the user
    message of each conversation is kept.
    """
    if system_prompt:
        return [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            for convo in prompts
        ]

    rendered: list[str] = []
    for convo in prompts:
        if isinstance(convo, (list, tuple)) and len(convo) > 1 and isinstance(convo[1], dict):
            rendered.append(convo[1].get("content", ""))
        else:
            rendered.append(str(convo))
    return rendered


def make_inputs(
    tokenizer: Any,
    prompts: Any,
    device: str | None = None,
    *,
    system_prompt: bool = False,
    add_generation_prompt: bool = False,
    rendered_prompts: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Left-padded ``input_ids`` and ``attention_mask`` for a batch of conversations."""
    if rendered_prompts is None:
        rendered_prompts = render_prompts(
            tokenizer,
            prompts,
            system_prompt=system_prompt,
            add_generation_prompt=add_generation_prompt,
        )

    # A rendered chat template already contains the BOS token.
    add_special_tokens = not system_prompt
    token_lists = [
        tokenizer.encode(p, add_special_tokens=add_special_tokens) for p in rendered_prompts
    ]
    maxlen = max(len(toks) for toks in token_lists)
    pad_id = get_pad_id(tokenizer)
    input_ids = [[pad_id] * (maxlen - len(toks)) + toks for toks in token_lists]
    attention_mask = [[0] * (maxlen - len(toks)) + [1] * len(toks) for toks in token_lists]

    tensor_kwargs: dict[str, Any] = {}
    if device:
        tensor_kwargs["device"] = device

    return {
        "input_ids": torch.tensor(input_ids, **tensor_kwargs),
        "attention_mask": torch.tensor(attention_mask, **tensor_kwargs),
    }
