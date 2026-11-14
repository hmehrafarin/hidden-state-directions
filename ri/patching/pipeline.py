"""CoT generation with cached states, and patched generation on the direct prompt.

A CoT position ``i`` refers to the hidden state from which generated token ``i`` was
predicted: position 0 is the last prompt token, position ``i > 0`` the generated token
``i - 1``. Layer ``l`` is the output of decoder block ``l`` (``hidden_states[l + 1]``),
which is also where :func:`ri.core.hooks.set_patch` writes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
import transformers

from ri.common import build_prompt_batch
from ri.core.hooks import remove_hooks, set_patch
from ri.core.model import ModelAndTokenizer
from ri.prompts import Prompter
from ri.utils.extraction import ExtractionMode, ExtractionType, extract_answer_from_generation
from ri.utils.text import prompt_text_from_rendered
from ri.utils.tokenizer import decode_tokens, get_eos_token_ids, get_pad_id, make_inputs

log = logging.getLogger(__name__)

TRACE_COLUMNS = (
    "source_generation_id",
    "source_num_tokens",
    "source_token_ids_hash",
    "source_prompt_hash",
)


def hidden_state_index(layer: int, total_hidden_states: int) -> int:
    """Index into ``hidden_states`` (embeddings first) for decoder block ``layer``.

    Negative layers count from the end; the extra embedding entry absorbs the offset.
    """
    max_layers = total_hidden_states - 1
    if max_layers <= 0:
        raise ValueError("Model did not produce per-layer hidden states.")
    if layer >= 0:
        if layer >= max_layers:
            raise ValueError(
                f"layer {layer} is invalid for a model with {max_layers} transformer layers."
            )
        return layer + 1
    if layer < -max_layers:
        raise ValueError(
            f"layer {layer} is invalid for a model with {max_layers} transformer layers."
        )
    return total_hidden_states + layer


def token_text(tokenizer: Any, token_id: int) -> str:
    """The vocabulary entry of ``token_id``, falling back to its decoding."""
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(converter):
        tok = converter(int(token_id))
        if tok is not None:
            return str(tok)
    try:
        return str(tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False))
    except Exception:
        return str(int(token_id))


def _hash_jsonable(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _hash_ints(values: list[int]) -> str:
    return _hash_jsonable([int(v) for v in values])


def _real_input_ids(tokenized: dict[str, torch.Tensor], row: int) -> list[int]:
    ids = tokenized["input_ids"][row].detach().cpu().tolist()
    mask = tokenized["attention_mask"][row].detach().cpu().tolist()
    return [int(tok) for tok, keep in zip(ids, mask, strict=False) if int(keep) == 1]


def _model_name(mt: ModelAndTokenizer) -> str:
    return str(
        getattr(mt, "model_name", None)
        or getattr(mt.model, "name_or_path", None)
        or "unknown_model"
    )


def _tokenizer_name(tokenizer: Any) -> str:
    return str(getattr(tokenizer, "name_or_path", "") or type(tokenizer).__name__)


def source_trace(
    mt: ModelAndTokenizer,
    tokenized: dict[str, torch.Tensor],
    generated_token_ids: list[list[int]],
    *,
    max_gen_len: int,
    answer_type: str,
    eos_ids: int | list[int],
    pad_id: int,
) -> dict[str, list[Any]]:
    """Per-question identity of a CoT generation, one list entry per batch row.

    ``source_generation_id`` hashes everything that determines the token trace;
    ``source_token_ids_hash`` and ``source_prompt_hash`` hash the generated and prompt
    token ids. The activation collector checks these against the sweep's labels.
    """
    eos_list = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
    trace: dict[str, list[Any]] = {column: [] for column in TRACE_COLUMNS}
    for row in range(int(tokenized["input_ids"].shape[0])):
        prompt_hash = _hash_ints(_real_input_ids(tokenized, row))
        token_ids = generated_token_ids[row]
        token_ids_hash = _hash_ints(token_ids)
        payload = {
            "model_name": _model_name(mt),
            "tokenizer": _tokenizer_name(mt.tokenizer),
            "prompt_hash": prompt_hash,
            "token_ids_hash": token_ids_hash,
            "max_gen_len": max_gen_len,
            "answer_type": answer_type,
            "eos_token_id": eos_list,
            "pad_token_id": pad_id,
            "patch_from_generation": True,
            "transformers_version": transformers.__version__,
        }
        trace["source_generation_id"].append(_hash_jsonable(payload)[:24])
        trace["source_num_tokens"].append(len(token_ids))
        trace["source_token_ids_hash"].append(token_ids_hash[:24])
        trace["source_prompt_hash"].append(prompt_hash[:24])
    return trace


def build_cot_inputs(
    mt: ModelAndTokenizer,
    prompter: Prompter,
    batched_input: list[dict[str, str]],
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Tokenize the CoT prompts; returns ``(user_prompt_texts, tokenized)``."""
    convos = build_prompt_batch(prompter, batched_input)
    use_chat = mt.uses_chat_template
    tokenized = make_inputs(
        mt.tokenizer, convos, mt.device, system_prompt=use_chat, add_generation_prompt=use_chat
    )
    return [prompt_text_from_rendered(convo) for convo in convos], tokenized


def build_direct_inputs(
    mt: ModelAndTokenizer,
    prompter: Prompter,
    batched_input: list[dict[str, str]],
) -> dict[str, torch.Tensor]:
    """Tokenize the direct prompts."""
    convos = build_prompt_batch(prompter, batched_input)
    use_chat = mt.uses_chat_template
    return make_inputs(
        mt.tokenizer, convos, mt.device, system_prompt=use_chat, add_generation_prompt=use_chat
    )


def trim_generated_ids(
    sequences: torch.Tensor,
    input_len: int,
    batch_size: int,
    num_gen_steps: int,
    eos_ids: int | list[int],
) -> list[list[int]]:
    """Generated ids of each row up to and including the first stop token."""
    stop_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
    generated: list[list[int]] = []
    for row in range(batch_size):
        gen_seq = sequences[row, input_len:]
        valid_len = len(gen_seq)
        for eos_id in stop_ids:
            matches = (gen_seq == eos_id).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                valid_len = min(valid_len, int(matches[0].item()) + 1)
        valid_len = min(valid_len, num_gen_steps)
        generated.append([int(tok) for tok in gen_seq[:valid_len].tolist()])
    return generated


def generate_cot_states(
    mt: ModelAndTokenizer,
    tokenized: dict[str, torch.Tensor],
    layers: list[int],
    *,
    max_gen_len: int,
    eos_ids: int | list[int],
    pad_id: int,
) -> tuple[dict[int, torch.Tensor], list[list[int]]]:
    """Greedy CoT generation that keeps the state at every position for each layer.

    Returns ``({layer: [batch, n_positions, hidden]}, generated_token_ids)``, both cut to
    the longest generated row.
    """
    with torch.inference_mode():
        generation = mt.model.generate(
            **tokenized,
            max_new_tokens=max_gen_len,
            eos_token_id=eos_ids,
            pad_token_id=pad_id,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    hidden_steps = generation.hidden_states or ()
    if not hidden_steps:
        raise RuntimeError("generation did not return hidden states")

    per_layer: dict[int, list[torch.Tensor]] = {int(layer): [] for layer in layers}
    for step_hidden in hidden_steps:
        for layer, steps in per_layer.items():
            layer_hidden = step_hidden[hidden_state_index(layer, len(step_hidden))]
            # The first step covers the whole prompt; later steps hold one new token.
            if layer_hidden.dim() == 3 and layer_hidden.size(1) == 1:
                steps.append(layer_hidden[:, 0, :])
            else:
                steps.append(layer_hidden[:, -1, :])
    states = {layer: torch.stack(steps, dim=1) for layer, steps in per_layer.items()}

    input_len = int(tokenized["input_ids"].shape[1])
    generated_token_ids = trim_generated_ids(
        generation.sequences,
        input_len,
        int(tokenized["input_ids"].shape[0]),
        len(hidden_steps),
        eos_ids,
    )
    max_valid_len = max((len(ids) for ids in generated_token_ids), default=0)
    states = {layer: tensor[:, :max_valid_len, :] for layer, tensor in states.items()}
    return states, generated_token_ids


def _cache_path(
    cache_dir: str,
    prompt_texts: list[str],
    mt: ModelAndTokenizer,
    tokenized: dict[str, torch.Tensor],
    *,
    max_gen_len: int,
    answer_type: str,
    eos_ids: int | list[int],
    pad_id: int,
) -> str:
    """Cache file for one CoT generation; the key does not depend on the layer."""
    payload = {
        "source_prompt_texts": prompt_texts,
        "input_ids": [
            _real_input_ids(tokenized, row) for row in range(int(tokenized["input_ids"].shape[0]))
        ],
        "model_name": _model_name(mt),
        "tokenizer": _tokenizer_name(mt.tokenizer),
        "max_gen_len": max_gen_len,
        "answer_type": answer_type,
        "batch_size": 1,
        "eos_token_id": [eos_ids] if isinstance(eos_ids, int) else list(eos_ids),
        "pad_token_id": pad_id,
        "transformers_version": transformers.__version__,
    }
    return os.path.join(cache_dir, f"gen_cache_{_hash_jsonable(payload)[:32]}.pt")


@dataclass
class SweepContext:
    """Everything about one question that stays fixed while its CoT positions are swept."""

    states: dict[int, torch.Tensor]
    """``{layer: [1, n_positions, hidden]}``."""
    generated_token_ids: list[int]
    trace: dict[str, Any]
    direct_inputs: dict[str, torch.Tensor]
    patch_position: int
    """Absolute index of the last direct-prompt token."""

    @property
    def n_positions(self) -> int:
        return len(self.generated_token_ids)


def prepare_sweep_context(
    mt: ModelAndTokenizer,
    cot_prompter: Prompter,
    direct_prompter: Prompter,
    batched_input: list[dict[str, str]],
    layers: list[int],
    *,
    max_gen_len: int,
    answer_type: ExtractionType,
    gen_cache_dir: str | None = None,
) -> SweepContext:
    """Generate the CoT of one question and prepare its direct prompt.

    With ``gen_cache_dir`` the generated token ids are cached, and a later run whose
    generation differs from the cached one fails instead of mixing token traces.
    """
    prompt_texts, cot_inputs = build_cot_inputs(mt, cot_prompter, batched_input)
    pad_id = get_pad_id(mt.tokenizer)
    eos_ids = get_eos_token_ids(mt.tokenizer)

    cache_path = None
    cached: dict[str, Any] | None = None
    if gen_cache_dir:
        os.makedirs(gen_cache_dir, exist_ok=True)
        cache_path = _cache_path(
            gen_cache_dir,
            prompt_texts,
            mt,
            cot_inputs,
            max_gen_len=max_gen_len,
            answer_type=answer_type,
            eos_ids=eos_ids,
            pad_id=pad_id,
        )
        if os.path.exists(cache_path):
            try:
                cached = torch.load(cache_path, map_location=mt.device, weights_only=True)
            except Exception as exc:
                log.warning("failed to load generation cache %s: %s", cache_path, exc)

    states, generated_token_ids = generate_cot_states(
        mt, cot_inputs, layers, max_gen_len=max_gen_len, eos_ids=eos_ids, pad_id=pad_id
    )

    if cached is not None:
        if generated_token_ids != cached["generated_token_ids"]:
            raise RuntimeError(
                "CoT generation differs from the cached generation for the same key; "
                "refusing to mix token traces across layers."
            )
    elif cache_path is not None:
        output = [mt.tokenizer.decode(ids) for ids in generated_token_ids]
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        torch.save(
            {
                "generated_token_ids": generated_token_ids,
                "source_extracted_answers": extract_answer_from_generation(
                    output, extraction_type=answer_type
                ),
            },
            tmp_path,
        )
        os.replace(tmp_path, cache_path)

    trace = source_trace(
        mt,
        cot_inputs,
        generated_token_ids,
        max_gen_len=max_gen_len,
        answer_type=answer_type,
        eos_ids=eos_ids,
        pad_id=pad_id,
    )

    direct_inputs = build_direct_inputs(mt, direct_prompter, batched_input)
    mask = direct_inputs["attention_mask"][0]
    n_real = int(mask.sum().item())
    offset = len(mask) - n_real

    return SweepContext(
        states=states,
        generated_token_ids=generated_token_ids[0],
        trace={column: values[0] for column, values in trace.items()},
        direct_inputs=direct_inputs,
        patch_position=offset + n_real - 1,
    )


@dataclass(frozen=True)
class PatchOutcome:
    """The patched generation for one (layer, CoT position)."""

    source_token: str
    generated_answer: str | None
    generated_text: str


def patch_and_generate(
    mt: ModelAndTokenizer,
    ctx: SweepContext,
    layer: int,
    position: int,
    *,
    max_gen_len: int,
    extraction_mode: ExtractionMode,
    answer_type: ExtractionType,
) -> PatchOutcome:
    """Replace the last direct-prompt state at ``layer`` with CoT state ``position``
    (``0 <= position < ctx.n_positions``), then generate greedily."""
    state = ctx.states[layer][0, position, :].reshape(1, 1, -1)
    source_token = token_text(mt.tokenizer, ctx.generated_token_ids[position])

    pad_id = get_pad_id(mt.tokenizer)
    eos_ids = get_eos_token_ids(mt.tokenizer)
    hooks = set_patch(
        mt.model,
        [
            {
                "layer_to_patch": layer,
                "hs_position": [ctx.patch_position],
                "hs": state,
            }
        ],
    )
    try:
        with torch.inference_mode():
            output_toks = mt.model.generate(
                **ctx.direct_inputs,
                max_new_tokens=max_gen_len,
                eos_token_id=eos_ids,
                pad_token_id=pad_id,
                do_sample=False,
            )
    finally:
        remove_hooks(hooks)

    generated = output_toks[:, int(ctx.direct_inputs["input_ids"].shape[1]) :]
    extracted = extract_answer_from_generation(
        decode_tokens(mt.tokenizer, generated),
        extraction_mode,
        extraction_type=answer_type,
    )
    return PatchOutcome(
        source_token=source_token,
        generated_answer=extracted["final_answer"][0],
        generated_text=extracted["answer_text"][0] or "",
    )
