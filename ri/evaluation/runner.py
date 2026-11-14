"""Baseline evaluation: answer every question of a task with the CoT or the direct prompt."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm

from ri.common import prepare_batch_data
from ri.core.model import ModelAndTokenizer
from ri.prompts import Prompter
from ri.utils.extraction import answers_match, extract_answer

from .config import EvaluationConfig
from .pipeline import generate_batch_outputs


class EvaluationSample(BaseModel):
    """One evaluated question. ``final_answer`` is the gold answer, extracted from ``answer``."""

    sample_idx: int
    question: str = ""
    answer: str = ""
    final_answer: str | None = None
    generated_final_answer: str | None = None
    generated_answer_cot: str = ""
    correct: bool = False
    n_generated_tokens: int
    terminated: bool


class EvaluationRunner:
    """Evaluates one model on one task with one prompt."""

    def __init__(
        self,
        mt: ModelAndTokenizer,
        data: Any,
        prompter: Prompter,
        config: EvaluationConfig,
    ) -> None:
        self.mt = mt
        self.data = data
        self.prompter = prompter
        self.config = config

    def run(self) -> list[EvaluationSample]:
        batch_size = self.config.batch_size
        samples: list[EvaluationSample] = []
        for batch_idx in tqdm(range(math.ceil(len(self.data) / batch_size)), desc="Evaluating"):
            questions, answers, batched_input = prepare_batch_data(self.data, batch_idx, batch_size)
            gold_answers = extract_answer(answers)
            outputs = generate_batch_outputs(self.mt, self.prompter, batched_input, self.config)
            for i, question in enumerate(questions):
                generated = outputs["generated_answer"][i]
                gold = gold_answers[i]
                samples.append(
                    EvaluationSample(
                        sample_idx=batch_idx * batch_size + i,
                        question=question,
                        answer=answers[i],
                        final_answer=str(gold) if gold is not None else None,
                        generated_final_answer=str(generated) if generated is not None else None,
                        generated_answer_cot=outputs["answer_cot"][i],
                        correct=answers_match(generated, gold),
                        n_generated_tokens=outputs["n_generated_tokens"][i],
                        terminated=outputs["terminated"][i],
                    )
                )
        return samples
