from __future__ import annotations

from typing import Any

from ri.common import build_prompt_batch
from ri.core.model import ModelAndTokenizer
from ri.prompts import Prompter
from ri.utils.extraction import extract_answer_from_generation
from ri.utils.tokenizer import (
    count_generated_tokens,
    decode_tokens,
    get_eos_token_ids,
    get_pad_id,
    make_inputs,
)

from .config import EvaluationConfig


def generate_batch_outputs(
    mt: ModelAndTokenizer,
    prompter: Prompter,
    batched_input: list[dict[str, str]],
    config: EvaluationConfig,
) -> dict[str, list[Any]]:
    """Greedy generation for a batch of questions, with the extracted answers.

    Returns per-row lists under ``generated_answer``, ``answer_cot`` (the stripped
    generation), ``n_generated_tokens`` and ``terminated``.
    """
    convos = build_prompt_batch(prompter, batched_input)
    use_chat = mt.uses_chat_template
    inputs = make_inputs(
        mt.tokenizer,
        convos,
        mt.device,
        system_prompt=use_chat,
        add_generation_prompt=use_chat,
    )
    pad_id = get_pad_id(mt.tokenizer)
    eos_ids = get_eos_token_ids(mt.tokenizer)

    output_toks = mt.model.generate(
        **inputs,
        max_new_tokens=config.max_gen_len,
        eos_token_id=eos_ids,
        pad_token_id=pad_id,
        do_sample=False,
    )
    generated_toks = output_toks[:, inputs["input_ids"].shape[1] :]
    output = decode_tokens(mt.tokenizer, generated_toks)

    # A row shorter than the budget after dropping trailing stop and pad ids counts as
    # stopped; with the pad fallback id 0 a row ending in token 0 counts as stopped too.
    n_generated = [count_generated_tokens(mt.tokenizer, row) for row in generated_toks]
    terminated = [n < generated_toks.shape[1] for n in n_generated]

    extracted = extract_answer_from_generation(
        output, mode=config.extraction_mode, extraction_type=config.answer_type
    )
    return {
        "generated_answer": extracted["final_answer"],
        "answer_cot": extracted["answer_text"],
        "n_generated_tokens": n_generated,
        "terminated": terminated,
    }
