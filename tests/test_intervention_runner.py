"""The intervention runner on a tiny random model with a character-level tokenizer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

import ri.intervention.task as intervention_task
from ri.core.hooks import remove_hooks, set_patch
from ri.core.model import ModelAndTokenizer
from ri.directions import Means, means_path, targets, write_means
from ri.intervention.config import InterventionConfig
from ri.intervention.results import question_path, read_question
from ri.intervention.runner import InterventionInputs, eligible_samples, run_intervention
from ri.prompts import Prompter
from ri.tracking import ExperimentTracker

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CONF = Path(__file__).resolve().parents[1] / "ri" / "conf" / "task"
LAYER = 3
SPECIALS = ["<pad>", "<eos>", "<unk>", "<|sys|>", "<|usr|>", "<|asst|>"]
CHARS = list("0123456789 abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,:;?!#()-+*/=\n")


def _char_tokenizer() -> Any:
    vocab = {token: i for i, token in enumerate(SPECIALS + CHARS)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    backend.decoder = decoders.Fuse()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=["<|sys|>", "<|usr|>", "<|asst|>"],
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{% if m['role'] == 'system' %}<|sys|>"
        "{% elif m['role'] == 'user' %}<|usr|>{% else %}<|asst|>{% endif %}"
        "{{ m['content'] }}{% endfor %}{% if add_generation_prompt %}<|asst|>{% endif %}"
    )
    return tokenizer


@pytest.fixture(scope="module")
def mt(make_tiny_causal_lm: Callable[..., Any]) -> ModelAndTokenizer:
    model = make_tiny_causal_lm(
        "qwen2", max_position_embeddings=512, eos_token_id=1, pad_token_id=0
    )
    return ModelAndTokenizer(model=model, tokenizer=_char_tokenizer(), device="cpu")


@pytest.fixture
def means() -> Means:
    """Five questions; question 1 has no recovering state and question 4 no failed state."""
    rng = np.random.default_rng(0)
    c_plus = rng.standard_normal((5, 32))
    c_minus = rng.standard_normal((5, 32))
    c_plus[1] = np.nan
    c_minus[4] = np.nan
    return Means(
        sample_idx=np.arange(5),
        c_plus=c_plus,
        c_minus=c_minus,
        n_plus=np.array([3, 0, 2, 5, 4]),
        n_minus=np.array([1, 4, 2, 2, 0]),
    )


DATA = [
    {"question": f"What is {a} plus {b}?", "answer": f"#### {a + b}"}
    for a, b in [(1, 2), (3, 4), (2, 2), (5, 1), (4, 3)]
]


def _inputs(mt: ModelAndTokenizer, means: Means, config: InterventionConfig) -> InterventionInputs:
    selected = means.with_direction()
    return InterventionInputs(
        model_name=MODEL,
        dataset="gsm8k",
        answer_type="numeric",
        layer=LAYER,
        mt=mt,
        data=DATA,
        prompter=Prompter("gsm8k_direct"),
        means=selected,
        targets=targets(selected, config.direction, config.lambda_ if config.uses_lambda else None),
    )


def test_eligible_questions_are_those_with_both_means_within_the_range(means: Means) -> None:
    selected = means.with_direction()
    assert eligible_samples(selected, None, None) == [0, 2, 3]
    assert eligible_samples(selected, 2, 3) == [2]
    assert eligible_samples(selected, 1, None) == [2, 3]


def test_rows_cover_every_eligible_question_and_dose(
    mt: ModelAndTokenizer, means: Means, tmp_path: Path
) -> None:
    config = InterventionConfig(doses=[0.0, 1.0], max_gen_len=6)
    rows = run_intervention(config, _inputs(mt, means, config), tmp_path)
    assert [(r.sample_idx, r.operator, r.dose) for r in rows] == [
        (s, op, d)
        for s in (0, 2, 3)
        for op, d in [("none", None), ("rotate", 0.0), ("rotate", 1.0)]
    ]
    for result in rows:
        assert result.direction == "recovering_mean"
        assert result.layer == LAYER
        assert result.n_recovering == means.n_plus[result.sample_idx]
        assert result.n_failed == means.n_minus[result.sample_idx]
        assert 0 <= result.n_generated_tokens <= 6
        assert result.gold_answer is not None
        assert not result.every_token

    by_key = {(r.sample_idx, r.dose): r.generated_text for r in rows}
    for sample in (0, 2, 3):
        assert by_key[(sample, 0.0)] == by_key[(sample, None)]

    frame = pl.concat([pl.read_parquet(p) for p in tmp_path.glob("*.parquet")])
    assert frame.height == len(rows)
    assert frame.schema["source_dataset"] == pl.String
    assert frame.schema["dose"] == pl.Float64


def _stored(output_dir: Path) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in sorted(output_dir.glob("*.parquet"))])


def _assert_complete(output_dir: Path, plan: list[tuple[str, float | None]]) -> None:
    """One file per eligible question, holding exactly one row per planned cell."""
    assert sorted(p.name for p in output_dir.iterdir()) == [
        f"sample_{s:04d}.parquet" for s in (0, 2, 3)
    ]
    frame = _stored(output_dir)
    keys = frame.select("sample_idx", "operator", "dose")
    assert not keys.is_duplicated().any()
    assert sorted(keys.iter_rows(), key=str) == sorted(
        [(s, op, dose) for s in (0, 2, 3) for op, dose in plan], key=str
    )


def test_a_finished_run_resumes_to_nothing(
    mt: ModelAndTokenizer, means: Means, tmp_path: Path
) -> None:
    config = InterventionConfig(doses=[1.0], max_gen_len=4, sample_end=3)
    assert len(run_intervention(config, _inputs(mt, means, config), tmp_path)) == 4
    stamps = {p: p.stat().st_mtime_ns for p in tmp_path.glob("*.parquet")}
    assert run_intervention(config, _inputs(mt, means, config), tmp_path) == []
    assert {p: p.stat().st_mtime_ns for p in tmp_path.glob("*.parquet")} == stamps

    wider = InterventionConfig(doses=[1.0, 0.5], max_gen_len=4, sample_end=3)
    new = run_intervention(wider, _inputs(mt, means, wider), tmp_path)
    assert [(r.sample_idx, r.dose) for r in new] == [(0, 0.5), (2, 0.5)]
    assert read_question(tmp_path, 0).height == 3


@pytest.mark.parametrize("resume", [True, False])
def test_running_a_question_twice_leaves_one_copy_of_its_rows(
    mt: ModelAndTokenizer, means: Means, tmp_path: Path, resume: bool
) -> None:
    config = InterventionConfig(doses=[0.5, 1.0], max_gen_len=4, resume=resume)
    first = run_intervention(config, _inputs(mt, means, config), tmp_path)
    again = run_intervention(config, _inputs(mt, means, config), tmp_path)
    assert len(again) == (0 if resume else len(first))
    _assert_complete(tmp_path, [("none", None), ("rotate", 0.5), ("rotate", 1.0)])
    stored = _stored(tmp_path).sort("sample_idx", "operator", "dose")
    expected = pl.DataFrame([r.model_dump() for r in first]).sort("sample_idx", "operator", "dose")
    assert stored.select("generated_text").equals(expected.select("generated_text"))


def test_a_rerun_without_resume_replaces_only_the_cells_it_runs(
    mt: ModelAndTokenizer, means: Means, tmp_path: Path
) -> None:
    config = InterventionConfig(doses=[0.5, 1.0], max_gen_len=4)
    run_intervention(config, _inputs(mt, means, config), tmp_path)
    subset = InterventionConfig(doses=[1.0], include_baseline=False, max_gen_len=4, resume=False)
    assert len(run_intervention(subset, _inputs(mt, means, subset), tmp_path)) == 3
    _assert_complete(tmp_path, [("none", None), ("rotate", 0.5), ("rotate", 1.0)])


def test_a_question_written_but_cut_short_is_completed_on_resume(
    mt: ModelAndTokenizer, means: Means, tmp_path: Path
) -> None:
    config = InterventionConfig(doses=[0.5, 1.0], max_gen_len=4)
    run_intervention(config, _inputs(mt, means, config), tmp_path)
    path = question_path(tmp_path, 2)
    pl.read_parquet(path).filter(pl.col("dose").ne_missing(1.0)).write_parquet(path)
    new = run_intervention(config, _inputs(mt, means, config), tmp_path)
    assert [(r.sample_idx, r.operator, r.dose) for r in new] == [(2, "rotate", 1.0)]
    _assert_complete(tmp_path, [("none", None), ("rotate", 0.5), ("rotate", 1.0)])


def test_sharded_runs_then_a_resumed_run_are_complete_and_free_of_duplicates(
    mt: ModelAndTokenizer, means: Means, tmp_path: Path
) -> None:
    plan: list[tuple[str, float | None]] = [("none", None), ("rotate", 1.0)]
    for start, end in [(0, 2), (2, None), (1, 3)]:
        shard = InterventionConfig(doses=[1.0], max_gen_len=4, sample_start=start, sample_end=end)
        run_intervention(shard, _inputs(mt, means, shard), tmp_path)
    full = InterventionConfig(doses=[1.0], max_gen_len=4)
    assert run_intervention(full, _inputs(mt, means, full), tmp_path) == []
    _assert_complete(tmp_path, plan)


def test_scale_at_one_is_a_patch_of_the_target(
    mt: ModelAndTokenizer, means: Means, tmp_path: Path
) -> None:
    config = InterventionConfig(
        operator="scale", doses=[1.0], include_baseline=False, max_gen_len=6
    )
    inputs = _inputs(mt, means, config)
    rows = run_intervention(config, inputs, tmp_path)

    tokenizer = mt.tokenizer
    for result in rows:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": Prompter("gsm8k_direct").template["system"]},
                {
                    "role": "user",
                    "content": f"Question: {DATA[result.sample_idx]['question']}\nAnswer:",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)])
        row = int(np.flatnonzero(inputs.means.sample_idx == result.sample_idx)[0])
        vector = torch.from_numpy(inputs.targets[row])
        hooks = set_patch(
            mt.model, [{"layer_to_patch": LAYER, "hs_position": [-1], "hs": [vector]}]
        )
        try:
            with torch.no_grad():
                out = mt.model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=config.max_gen_len,
                    do_sample=False,
                    eos_token_id=1,
                    pad_token_id=0,
                )
        finally:
            remove_hooks(hooks)
        text = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
        assert text == result.generated_text


def test_every_token_rows_are_marked(mt: ModelAndTokenizer, means: Means, tmp_path: Path) -> None:
    config = InterventionConfig(name="steer", every_token=True, doses=[0.5], max_gen_len=6)
    rows = run_intervention(config, _inputs(mt, means, config), tmp_path)
    assert {(r.operator, r.every_token) for r in rows} == {("none", False), ("rotate", True)}


def _task_cfg(**task: object) -> DictConfig:
    cfg = OmegaConf.create(
        {
            "task": OmegaConf.load(CONF / "intervene.yaml"),
            "model": {"model_name": MODEL},
            "dataset": {
                "name": "gsm8k",
                "direct_prompt_template": "gsm8k_direct",
                "answer_type": "numeric",
            },
            "seed": 42,
        }
    )
    cfg.task.layer = LAYER
    cfg.task.max_gen_len = 4
    for key, value in task.items():
        cfg.task[key] = value
    return cfg


@pytest.mark.parametrize(
    ("direction", "label"),
    [
        ("recovering_mean", "recovering_mean"),
        ("failed_mean", "failed_mean"),
        ("contrastive", "contrastive_050"),
        ("task_level", "task_level_050"),
    ],
)
def test_the_task_runs_every_direction_on_the_questions_with_both_means(
    mt: ModelAndTokenizer,
    means: Means,
    results_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
    label: str,
) -> None:
    write_means(means_path(MODEL, "gsm8k", LAYER), means)
    monkeypatch.setattr(intervention_task, "ModelAndTokenizer", lambda **kwargs: mt)
    monkeypatch.setattr(intervention_task, "get_dataset", lambda name: {"test": DATA})
    intervention_task.run(
        _task_cfg(direction=direction, doses=[1.0]), ExperimentTracker(enabled=False)
    )

    out = results_root / "llama8B" / "intervention" / f"{label}_rotate_layer{LAYER}" / "gsm8k"
    frame = pl.concat([pl.read_parquet(p) for p in out.glob("*.parquet")])
    assert frame.height == 6
    assert sorted(set(frame["sample_idx"])) == [0, 2, 3]
    assert set(frame["direction"]) == {label}
    assert set(frame["model_name"]) == {MODEL}


def test_the_cross_task_run_applies_the_source_direction_under_its_own_directory(
    mt: ModelAndTokenizer, means: Means, results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_means(means_path(MODEL, "gsm8k", LAYER), means)
    write_means(means_path(MODEL, "bbh_snarks", LAYER), means)
    monkeypatch.setattr(intervention_task, "ModelAndTokenizer", lambda **kwargs: mt)
    monkeypatch.setattr(intervention_task, "get_dataset", lambda name: {"test": DATA})
    intervention_task.run(
        _task_cfg(direction="task_level", source_dataset="bbh_snarks"),
        ExperimentTracker(enabled=False),
    )

    out = (
        results_root
        / "llama8B"
        / "intervention"
        / f"task_transfer_layer{LAYER}"
        / "from_bbh_snarks"
        / "gsm8k"
    )
    frame = pl.concat([pl.read_parquet(p) for p in out.glob("*.parquet")])
    assert sorted(frame["sample_idx"]) == [0, 2, 3]
    assert set(frame["operator"]) == {"rotate"}
    assert set(frame["dose"]) == {1.0}
    assert set(frame["direction"]) == {"task_level_050"}
    assert set(frame["source_dataset"]) == {"bbh_snarks"}
