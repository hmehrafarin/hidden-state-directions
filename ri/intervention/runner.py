"""Generate from the direct prompt with the last prompt state moved towards a target.

For every question with a direction (at least one recovering and one failed
state), the runner generates once without intervention (optional) and once per
dose with the hook of ``ri.core.hooks.set_state_intervention`` at one layer,
decodes greedily, and scores the extracted answer against the gold answer.
Questions without a direction are not run; the analysis counts them as incorrect.
Each question's rows are written to its own file once all of its doses are done
(see ``ri.intervention.results``), and resuming skips the stored doses.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm

from ri.common import build_prompt_batch, prepare_batch_data
from ri.core.hooks import remove_hooks, set_state_intervention
from ri.intervention.config import InterventionConfig
from ri.intervention.results import InterventionResult, RowOperator, done_cells, write_results
from ri.utils import (
    answers_match,
    count_generated_tokens,
    extract_answer,
    extract_answer_from_generation,
    get_eos_token_ids,
    get_pad_id,
    make_inputs,
)
from ri.utils.extraction import ExtractionType

if TYPE_CHECKING:
    from ri.core.model import ModelAndTokenizer
    from ri.directions.store import Means
    from ri.prompts import Prompter
    from ri.tracking import ExperimentTracker

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InterventionInputs:
    """Everything a run needs besides its configuration."""

    model_name: str
    dataset: str
    answer_type: ExtractionType
    layer: int
    mt: ModelAndTokenizer
    data: Any
    prompter: Prompter
    means: Means
    targets: np.ndarray
    """``(n, d)`` float32, row-aligned with ``means.sample_idx``."""


def eligible_samples(means: Means, start: int | None, end: int | None) -> list[int]:
    """The questions of ``means`` within ``[start, end)``, in index order.

    ``means`` holds only questions with a direction (``Means.with_direction``).
    """
    samples = [int(s) for s in means.sample_idx]
    if start is not None:
        samples = [s for s in samples if s >= start]
    if end is not None:
        samples = [s for s in samples if s < end]
    return samples


def _generate(
    inputs: InterventionInputs,
    config: InterventionConfig,
    tokenized: dict[str, torch.Tensor],
    target: np.ndarray,
    operator: RowOperator,
    dose: float | None,
) -> torch.Tensor:
    tokenizer = inputs.mt.tokenizer
    hooks = []
    if operator != "none" and dose is not None:
        hooks = set_state_intervention(
            inputs.mt.model,
            layer=inputs.layer,
            target=target,
            operator=operator,
            dose=dose,
            every_token=config.every_token,
        )
    try:
        with torch.no_grad():
            return inputs.mt.model.generate(
                **tokenized,
                max_new_tokens=config.max_gen_len,
                eos_token_id=get_eos_token_ids(tokenizer),
                pad_token_id=get_pad_id(tokenizer),
                do_sample=False,
                use_cache=True,
            )
    finally:
        remove_hooks(hooks)


def run_intervention(
    config: InterventionConfig,
    inputs: InterventionInputs,
    output_dir: Path,
    tracker: ExperimentTracker | None = None,
) -> list[InterventionResult]:
    """Run every (question, dose) not yet done under ``output_dir``; return the new rows."""
    tokenizer = inputs.mt.tokenizer
    means = inputs.means
    row_of = {int(s): i for i, s in enumerate(means.sample_idx)}

    plan: list[tuple[RowOperator, float | None]] = []
    if config.resolved_include_baseline:
        plan.append(("none", None))
    plan.extend((config.operator, dose) for dose in config.resolved_doses)

    samples = eligible_samples(means, config.sample_start, config.sample_end)
    log.info("%s: %d questions with a direction in range", inputs.dataset, len(samples))
    use_chat = inputs.mt.uses_chat_template

    new_rows: list[InterventionResult] = []
    for sidx in tqdm(samples, desc=f"{config.name} {inputs.dataset}"):
        done = done_cells(output_dir, sidx) if config.resume else set()
        todo = [(operator, dose) for operator, dose in plan if (operator, dose) not in done]
        if not todo:
            continue

        _questions, answers, batched = prepare_batch_data(inputs.data, sidx, 1)
        gold = extract_answer([answers[0]])[0]
        tokenized = make_inputs(
            tokenizer,
            build_prompt_batch(inputs.prompter, batched),
            device=inputs.mt.device,
            system_prompt=use_chat,
            add_generation_prompt=use_chat,
        )
        prompt_len = tokenized["input_ids"].size(1)
        row = row_of[sidx]

        sample_rows: list[InterventionResult] = []
        for operator, dose in todo:
            output_ids = _generate(inputs, config, tokenized, inputs.targets[row], operator, dose)
            generated = output_ids[0, prompt_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            answer = extract_answer_from_generation(
                [text], mode=config.extraction_mode, extraction_type=inputs.answer_type
            )["final_answer"][0]
            sample_rows.append(
                InterventionResult(
                    sample_idx=sidx,
                    dataset=inputs.dataset,
                    model_name=inputs.model_name,
                    layer=inputs.layer,
                    direction=config.label,
                    source_dataset=config.source_dataset,
                    operator=operator,
                    every_token=config.every_token and operator != "none",
                    dose=dose,
                    gold_answer=gold,
                    generated_answer=answer,
                    generated_text=text,
                    n_generated_tokens=count_generated_tokens(tokenizer, generated),
                    correct=answers_match(answer, gold),
                    n_recovering=int(means.n_plus[row]),
                    n_failed=int(means.n_minus[row]),
                )
            )

        write_results(sample_rows, output_dir)
        new_rows.extend(sample_rows)

    _log_summary(new_rows, tracker)
    return new_rows


def _log_summary(rows: list[InterventionResult], tracker: ExperimentTracker | None) -> None:
    by_cell: dict[str, list[InterventionResult]] = defaultdict(list)
    for result in rows:
        key = "baseline" if result.operator == "none" else f"{result.operator}_{result.dose:g}"
        by_cell[key].append(result)
    metrics: dict[str, float] = {}
    for key, cell in sorted(by_cell.items()):
        metrics[f"{key}/accuracy"] = sum(r.correct for r in cell) / len(cell)
        metrics[f"{key}/mean_generated_tokens"] = sum(r.n_generated_tokens for r in cell) / len(
            cell
        )
        log.info(
            "%s: %d new rows, accuracy %.3f, mean tokens %.1f",
            key,
            len(cell),
            metrics[f"{key}/accuracy"],
            metrics[f"{key}/mean_generated_tokens"],
        )
    if tracker is not None and metrics:
        tracker.log_metrics(metrics)
