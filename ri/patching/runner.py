"""The patch sweep: label every CoT state of every question as recovering or failed.

For each question the CoT is generated once (greedy, ``max_gen_len`` tokens) with its
states at the requested layers. Each state is then patched into the last position of
the direct prompt at the same layer and the direct prompt is decoded greedily; the state
is recovering when the extracted answer matches the gold answer.

Results go to ``combined_<dataset>.parquet``, one row per (question, layer, CoT
position). Each (question, layer) is first written to a working file and then merged
into the combined file under a file lock, so concurrent jobs on disjoint question ranges
can share one combined file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm

from ri.common import prepare_batch_data
from ri.core.model import ModelAndTokenizer
from ri.progress import ProgressTracker
from ri.prompts import Prompter
from ri.tracking import ExperimentTracker
from ri.utils.extraction import answers_match, extract_answer
from ri.utils.io import save_rows_parquet

from .config import PatchSweepConfig
from .pipeline import patch_and_generate, prepare_sweep_context


class PatchSweepRow(BaseModel):
    """One patched generation: CoT state ``source_position`` at ``layer`` of a question."""

    sample_idx: int
    layer: int
    source_position: int
    source_token: str = ""
    source_generation_id: str | None = None
    source_num_tokens: int | None = None
    source_token_ids_hash: str | None = None
    source_prompt_hash: str | None = None
    # Always true here; part of the combined-file format, and the activation
    # collector keeps only rows where both are set.
    patch_from_generation: bool = True
    sweep_positions: bool = True
    gold_final_answer: str | None = None
    generated_final_answer: str | None = None
    generated_text: str = ""
    correct: bool = False


class PatchSweepRunner:
    """Runs the sweep for one model and one task."""

    def __init__(
        self,
        mt: ModelAndTokenizer,
        data: Any,
        *,
        cot_prompter: Prompter,
        direct_prompter: Prompter,
        config: PatchSweepConfig,
        layers: list[int],
    ) -> None:
        self.mt = mt
        self.data = data
        self.cot_prompter = cot_prompter
        self.direct_prompter = direct_prompter
        self.config = config
        self.layers = layers

    def run(
        self,
        combined_path: Path,
        work_dir: Path,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        """Sweep every question in the configured range and merge it into ``combined_path``.

        With ``config.resume`` a (question, layer) already complete in the combined file
        is skipped; otherwise it is recomputed and replaced.
        """
        cfg = self.config
        progress = ProgressTracker(combined_path)

        for idx in tqdm(cfg.sample_indices(len(self.data))):
            layers = [
                layer
                for layer in self.layers
                if not (cfg.resume and progress.is_done(sample_idx=idx, layer=layer))
            ]
            if not layers:
                continue

            _questions, answers, batched_input = prepare_batch_data(self.data, idx, 1)
            gold = extract_answer(answers)[0]
            ctx = prepare_sweep_context(
                self.mt,
                self.cot_prompter,
                self.direct_prompter,
                batched_input,
                layers,
                max_gen_len=cfg.max_gen_len,
                answer_type=cfg.answer_type,
                gen_cache_dir=cfg.gen_cache_dir,
            )

            for layer in layers:
                rows: list[PatchSweepRow] = []
                for position in tqdm(range(ctx.n_positions), desc=f"q{idx} l{layer}", leave=False):
                    outcome = patch_and_generate(
                        self.mt,
                        ctx,
                        layer,
                        position,
                        max_gen_len=cfg.max_gen_len,
                        extraction_mode=cfg.extraction_mode,
                        answer_type=cfg.answer_type,
                    )
                    rows.append(
                        PatchSweepRow(
                            sample_idx=idx,
                            layer=layer,
                            source_position=position,
                            source_token=outcome.source_token,
                            **ctx.trace,
                            gold_final_answer=str(gold) if gold is not None else None,
                            generated_final_answer=outcome.generated_answer,
                            generated_text=outcome.generated_text,
                            correct=answers_match(outcome.generated_answer, gold),
                        )
                    )
                if not rows:
                    continue
                progress.absorb(save_rows_parquet(rows, work_dir, f"s{idx:04d}-l{layer}"))
                if tracker is not None:
                    tracker.log_metrics(
                        {
                            "sample_idx": idx,
                            "layer": layer,
                            "recovering_fraction": sum(r.correct for r in rows) / len(rows),
                        }
                    )
