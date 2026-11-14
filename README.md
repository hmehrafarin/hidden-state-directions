# Pointed in the Right Direction: Hidden-State Directions Outperform Chain-of-Thought

Code for the paper *Pointed in the Right Direction: Hidden-State Directions Outperform
Chain-of-Thought*. The paper shows that changing one hidden state of a model prompted to answer
directly makes it answer with much shorter generations than chain-of-thought (CoT) prompting and,
on average, more accurately. Each hidden state of a question's CoT is labelled **recovering** if
patching it into the last position of the direct prompt yields the correct answer and **failed**
otherwise; a linear probe shows that the model encodes the two classes differently, and a direction
built from their means is used to rotate the direct prompt's last hidden state once, before
generation. Across 25 tasks these directions raise mean accuracy from 46% to 85% on LLaMA 3.1 8B
(CoT: 69%) and from 52% to 86% on Qwen 2.5 7B (CoT: 77%), with about a tenth of the tokens CoT
generates.

The experiments run in ten stages, each a task of one `ri` command:

1. **Evaluate** — the CoT and direct baselines of every task.
2. **Patch sweep** — patch every CoT state into the direct prompt and label it recovering or failed.
3. **Collect activations** — store the labelled CoT states of every question in an HDF5 cache.
4. **Probe** — one linear probe per task and layer, cross-fitted over five question folds.
5. **Probe transfer** — every task's probe scored on every other task (a 25 × 25 matrix).
6. **Probe report** — the probe tables of the paper as CSV.
7. **Directions** — the recovering and failed means of every question.
8. **Intervene** — rotate (or scale) the last state of the direct prompt towards a direction.
9. **Steer** — the same rotation applied at every generated token.
10. **Report** — the baseline, patch-label and intervention tables of the paper as CSV.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and package management.

```bash
uv sync
```

This creates a virtual environment in `.venv/`, installs the project in editable mode with the
development tools, and exposes the `ri` command. Run commands with `uv run` (e.g. `uv run ri ...`),
or activate the environment with `source .venv/bin/activate`. Python 3.11 or 3.12 is required.

On Linux, `pyproject.toml` takes PyTorch from the CUDA 12.1 wheel index on x86_64 and from the CUDA
12.4 index on aarch64; on other platforms it comes from PyPI. To submit jobs to a Slurm cluster
through Hydra (see [Running all 25 tasks](#running-all-25-tasks)), add the optional launcher:

```bash
uv sync --extra slurm
```

Every generation stage needs a GPU. Models are loaded with `device_map="auto"`, in bfloat16 when
the GPU supports it. The paper's runs used NVIDIA GH200 nodes and about 8,000 GPU hours in total.
The probes train on CUDA when it is available and on the CPU otherwise, in float64 on either.

## Models and tasks

| Config | Model |
|--------|-------|
| `model=llama_8b` | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (gated) |
| `model=qwen_7b` | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |

The 25 tasks are GSM8K (test split, 1,319 questions), StrategyQA (2,290 questions) and 23
BIG-Bench Hard subtasks (all questions). Each is a `dataset=<slug>` config in `ri/conf/dataset/`,
and the data is downloaded from the Hugging Face Hub (`openai/gsm8k`, `lukaemon/bbh` and
`hoomeh/strategyqa-formatted`). Reasoning types are those of the paper's Table 1 and answer
formats those of its Table 3.

| Slug | Reasoning type | Answer format |
|------|----------------|---------------|
| `gsm8k` | arithmetic | numeric |
| `bbh_multistep_arithmetic_two` | arithmetic | numeric |
| `bbh_object_counting` | arithmetic | numeric |
| `bbh_boolean_expressions` | logical | true/false |
| `bbh_logical_deduction_three_objects` | logical | multiple choice |
| `bbh_logical_deduction_five_objects` | logical | multiple choice |
| `bbh_logical_deduction_seven_objects` | logical | multiple choice |
| `bbh_penguins_in_a_table` | logical | multiple choice |
| `bbh_web_of_lies` | logical | yes/no |
| `bbh_date_understanding` | temporal and state tracking | multiple choice |
| `bbh_reasoning_about_colored_objects` | temporal and state tracking | multiple choice |
| `bbh_temporal_sequences` | temporal and state tracking | multiple choice |
| `bbh_tracking_shuffled_objects_three_objects` | temporal and state tracking | multiple choice |
| `bbh_tracking_shuffled_objects_five_objects` | temporal and state tracking | multiple choice |
| `bbh_tracking_shuffled_objects_seven_objects` | temporal and state tracking | multiple choice |
| `bbh_geometric_shapes` | spatial and geometric | multiple choice |
| `bbh_navigate` | spatial and geometric | yes/no |
| `strategyqa` | causal, commonsense and knowledge | true/false |
| `bbh_causal_judgement` | causal, commonsense and knowledge | yes/no |
| `bbh_movie_recommendation` | causal, commonsense and knowledge | multiple choice |
| `bbh_sports_understanding` | causal, commonsense and knowledge | yes/no |
| `bbh_disambiguation_qa` | linguistic and pragmatic | multiple choice |
| `bbh_hyperbaton` | linguistic and pragmatic | multiple choice |
| `bbh_ruin_names` | linguistic and pragmatic | multiple choice |
| `bbh_snarks` | linguistic and pragmatic | multiple choice |

The CoT and direct prompts of a task share the user turn and differ only in the system
instruction; the templates are in `ri/prompts/templates/` (paper, Appendix A). Every stage renders
both prompts with the tokenizer's chat template when the tokenizer has one, which both models do.

## Configuration

All experiments are driven through [Hydra](https://hydra.cc/) configs composed from `ri/conf/`:

```
ri/conf/
├── config.yaml          # root: picks one task / model / dataset / tracking; seed: 42
├── task/                # evaluate, patch_sweep, collect_activations, probe, probe_transfer,
│                        # probe_report, directions, intervene, steer, report
├── model/               # llama_8b, qwen_7b
├── dataset/             # the 25 tasks
└── tracking/            # disabled, wandb
```

The entry point is `ri/main.py`, exposed as the `ri` console script. A run selects one task, one
model and one dataset, and any value can be overridden from the command line with dotted paths:

```bash
uv run ri task=evaluate model=qwen_7b dataset=bbh_snarks task.prompt=direct
```

Unknown override keys are rejected before anything is loaded, and task parameters are validated
before model weights are read. `--cfg job` prints the composed config and exits without loading a
model, which checks that a set of overrides resolves. Every task YAML in
`ri/conf/task/` documents its parameters.

Weights & Biases logging is off by default. Enable it with `tracking=wandb`; the project name and
tags are in `ri/conf/tracking/wandb.yaml`, and every task writes its local outputs regardless.

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RI_RESULTS_ROOT` | Root of the results tree | `./results` |
| `HS_CACHE_DIR` | Root of the activation cache: a local directory, or `hf://` followed by the id of a Hugging Face dataset that holds a cache | `./hs_cache` |
| `RI_CACHE_DIR` | Directory passed as `cache_dir` to every model and config download | unset |
| `HF_TOKEN` | Hugging Face access token, needed for the gated Llama model | unset |

When `RI_CACHE_DIR` is unset, the Hugging Face libraries choose the cache themselves
(`$HF_HUB_CACHE`, otherwise `$HF_HOME/hub`, otherwise `~/.cache/huggingface/hub`), so a model
downloaded with `hf download` is found there. `model.model_name` may also be a local model
directory, e.g. `model.model_name=/models/Llama-3.1-8B-Instruct`; its results go under the short
name derived from the directory name.

An `hf://` activation cache is downloaded one task at a time and is only read: `collect_activations`
writes to a local directory and refuses an `hf://` cache before loading anything.

`uv run hf auth login` stores a token and replaces `HF_TOKEN`. Access to
`meta-llama/Llama-3.1-8B-Instruct` must first be requested on its model page. The analysis stages
of a Hub model read its `config.json` to derive the layers, from the local cache when it is there
and otherwise from the Hub, which for Llama needs the token too.

### Layers

Layers are 0-indexed in the code: layer `l` is the output of transformer block `l`, which is
`hidden_states[l + 1]` of a Hugging Face forward pass. The paper numbers layers from one. Every
stage derives its layers from the model's depth (`ri/core/layers.py`):

| | LLaMA 3.1 8B (32 blocks) | Qwen 2.5 7B (28 blocks) |
|---|---|---|
| Late band, in the code | 26, 28, 30 | 22, 24, 26 |
| Late band, in the paper | 27, 29, 31 | 23, 25, 27 |
| Primary layer (code / paper) | 28 / 29 | 24 / 25 |

`layers: band` (the default of every multi-layer stage) is the late band of the loaded model, and
`layer: null` in `intervene` and `steer` is its primary layer, the middle of the band. Explicit
layers are 0-indexed.

## Reproducing the paper

Each stage below gives the command for one model and one task, what the stage reads and writes,
and what in the paper it reproduces. Paths are relative to `$RI_RESULTS_ROOT` (default `results/`),
under the model's short name (`llama8B` or `qwen7B`); see [Results layout](#results-layout). The
paper's settings are the defaults: greedy decoding, batch size 1, seed 42, flexible answer
extraction. The release writes the numbers behind every table and figure as CSV; it does not draw
the figures.

### Running all 25 tasks

Hydra's `--multirun` (`-m`) runs one job per combination of comma-separated values, and
`dataset='glob(*)'` selects all 25 dataset configs:

```bash
uv run ri -m task=evaluate model=llama_8b task.prompt=cot,direct 'dataset=glob(*)'
```

By default the jobs run one after another in one process. With the `slurm` extra installed, the
submitit launcher submits each job to Slurm instead, as one job array:

```bash
uv run ri -m hydra/launcher=submitit_slurm \
    hydra.launcher.partition=<partition> hydra.launcher.gpus_per_node=1 \
    hydra.launcher.timeout_min=1440 \
    task=evaluate model=llama_8b task.prompt=cot,direct 'dataset=glob(*)'
```

`hydra/launcher=submitit_local` runs the same jobs as local subprocesses. Jobs do not change the
working directory; Hydra's `multirun/<date>/<time>/<job>` directories hold only the config snapshot
and the job log, and every result goes to the results tree.

### Sharding and resuming

`patch_sweep`, `collect_activations`, `intervene` and `steer` take `task.sample_start` and
`task.sample_end`, a half-open range `[sample_start, sample_end)` of question indices (`null` is
unbounded), so one task can be split across concurrent jobs:

```bash
uv run ri task=patch_sweep model=llama_8b dataset=gsm8k task.sample_start=0 task.sample_end=100
uv run ri task=patch_sweep model=llama_8b dataset=gsm8k task.sample_start=100 task.sample_end=200
```

What running a stage again does:

- **Evaluate.** No shards and no resume: each run writes a new timestamped parquet, and the
  reports read the newest one.
- **Patch sweep.** Each (question, layer) is written to a working file and then merged into the
  task's combined parquet under a file lock (`flock`), so concurrent jobs share one combined file
  on a filesystem where `flock` works. With `task.resume=false` (the default) every (question,
  layer) in the range is computed again and its rows in the combined parquet are replaced; with
  `task.resume=true` the pairs already complete in it are skipped.
- **Collect activations.** One HDF5 file per question, written atomically. With `task.resume=true`
  (the default) a question whose file already holds every layer with matching metadata and labels
  is skipped; with `task.resume=false` every file in the range is written again.
- **Probe and probe transfer.** Both stop before computing anything when a requested layer already
  has results (a probe's `metrics.json`, or the layer's `transfer.parquet`), and the error names
  `task.skip_existing=true`. With that option the finished layers are skipped and the others run.
  Existing results are never overwritten.
- **Directions.** An existing means file is skipped unless `task.overwrite=true`.
- **Intervene and steer.** A run directory holds one `sample_NNNN.parquet` per question, with one
  row per (operator, dose), written atomically. With `task.resume=true` (the default) the (operator,
  dose) cells a question already has are skipped and its missing cells are generated. With
  `task.resume=false` every cell of the run is generated again and replaces the stored row, and
  rows of doses the run does not include are kept. A question therefore never holds two rows for
  one cell, even after overlapping shards or an interrupted job.

### Step 1: Evaluate (CoT and direct)

```bash
uv run ri task=evaluate model=llama_8b dataset=gsm8k task.prompt=cot
uv run ri task=evaluate model=llama_8b dataset=gsm8k task.prompt=direct
```

Answers every question of the task with the CoT or the direct prompt, generating at most 1,024
tokens (`task.max_gen_len`). Writes one row per question (generation, extracted and gold answer,
correctness, number of generated tokens, whether generation stopped before the budget) to
`eval/<cot|direct>/<task>/*.parquet`. The token counts of this stage and of the interventions
leave out the trailing run of end-of-sequence, end-of-turn and pad ids. A tokenizer without a pad
token gets pad id 0, which for Llama 3.1 is the ordinary token `!`: trailing `!` tokens are not
counted either, and a generation cut at the budget after them is recorded as stopped before it.

Reproduces the baseline table (Section 3.1, Table 1) through the report of Step 10. `batch_size`
stays 1: batching pads prompts on the left and can change greedy generations.

### Step 2: Patch sweep (labels)

```bash
uv run ri task=patch_sweep model=llama_8b dataset=gsm8k
```

For every question, generates the CoT (at most 400 tokens) and caches its states at the late band,
then patches each state into the last position of the direct prompt at the same layer, generates
again (at most 400 tokens) and labels the state recovering if the extracted answer is correct and
failed otherwise. Writes `patch_sweep/combined_<task>.parquet`, one row per (question, layer, CoT
position), with the CoT token ids cached under `patch_sweep/generation_cache/` and per-question
working files under `patch_sweep/<task>/` while the job runs.

Reproduces the patching setup (Section 3.2) and the patch-label table (Appendix B, Table 4)
through the report of Step 10. This is the most expensive stage: one generation per CoT token,
layer and question. Shard it as shown above.

### Step 3: Collect activations

```bash
uv run ri task=collect_activations model=llama_8b dataset=gsm8k
```

Regenerates each question's CoT exactly as the patch sweep did and writes its states at the late
band, with the labels from `patch_sweep/combined_<task>.parquet`, to
`$HS_CACHE_DIR/<model_short>/<task>/sample_NNNN.h5` (float16, one file per question).
`task.strict_trace=true` (the default) checks the regenerated CoT against the trace hashes the
sweep recorded, and `task.max_gen_len` must equal the sweep's; a mismatch stops the job with an
error naming `task.max_gen_len`. `task.hs_cache_dir` overrides `$HS_CACHE_DIR` for this stage and
must be a local directory.

The cache is the input of the probes and the directions (Sections 4 and 5), which read it from
`$HS_CACHE_DIR`.

### Step 4: Linear probe

```bash
uv run ri task=probe model=llama_8b dataset=gsm8k
uv run ri task=probe model=llama_8b dataset=gsm8k task.layer=28
```

Trains the linear probe of one task at every layer of the late band (`task.layer=band`, the
default) or at one of them. The probe minimises an L2-penalised binary cross-entropy on standardised
states, with L-BFGS in float64. Questions are split into five folds seeded by the task name;
training rows are class-balanced within each fold; the penalty is chosen from 1e-4 to 1e3 by
nested cross-validation, and the pooled out-of-fold scores give the probe's AUROC. A full refit on
all questions is the source probe for transfer, and the origin and within-position control probes
give the selectivities. Reads the activation cache and writes
`probe/layer<L>/<task>/{metrics.json, directions.npz, question_folds.parquet,
out_of_fold_predictions.parquet}`.

Reproduces Section 4.1 and 4.2 and Appendix C.1 to C.4 (Figure 2 and Tables 5 and 6) through the
probe report of Step 6.

### Step 5: Probe transfer

```bash
uv run ri task=probe_transfer model=llama_8b
```

Scores the full-refit probe of every task on every state of every other task, together with the
within-position control probes, at every layer of the late band (`task.layers`). Needs a finished
probe of all 25 tasks at each layer, and checks for them before computing anything. Writes
`probe/layer<L>/transfer.parquet` and `probe/layer<L>/transfer_summary.json`.

Reproduces Section 4.3 and Appendix C.5 (Figures 3 and 6 and Table 7) through the probe report.

### Step 6: Probe report

```bash
uv run ri task=probe_report model=llama_8b
uv run ri task=probe_report 'task.models=[llama_8b,qwen_7b]' task.output_dir=paper_tables
```

Reads the evaluations, the probes and the transfer matrices and writes the probe tables as
`analysis/probe_*.csv` of each model, or, with `task.output_dir`, the tables of all models in one
directory, concatenated with a `model` column. Stops with an error naming the missing directory and
the command that produces it when an input is missing. `seed` (42) seeds the permutation tests.

| Table | Paper |
|-------|-------|
| `probe_per_task` | Figure 2, Table 5 |
| `probe_layers`, `probe_layer_comparisons` | Table 6, Appendix C.4 |
| `probe_accuracy_correlations` | Section 4.2, Appendix C.3 |
| `probe_transfer`, `probe_transfer_by_reasoning_type` | Figures 3 and 6 |
| `probe_transfer_summary` | Table 7, the permutation test of Appendix C.5 |
| `probe_direction_stability`, `probe_direction_stability_summary` | Appendix C.6 |

### Step 7: Directions

```bash
uv run ri task=directions model=llama_8b dataset=gsm8k
```

Averages each question's recovering states and its failed states at every layer of the late band
(`task.layers`), giving the recovering mean c+_q and the failed mean c-_q of question q. Reads the
activation cache and writes `directions/<task>/means_layer<L>.npz` (`c_plus`, `c_minus`, `n_plus`,
`n_minus`, `sample_idx`). A question has a direction only if it has at least one recovering and one
failed state. On a task without an activation cache the stage stops with an error naming the stage
that writes it.

The directions of Section 5.1 are built from these means when an intervention runs:

- the contrastive direction d_q(λ) = ‖c+_q‖ · unit(c+_q − λc-_q);
- the task-level direction d_T(λ) = mean_q d_q(λ), the average over the questions of the task that
  have a direction, rescaled to a question's ‖c+_q‖ when applied there.

The task-level and cross-task runs reported in the paper used, in place of d_T(λ), directions built
from each task's mean recovering state and mean failed state, which differ from it by a small angle.

### Step 8: Intervention

```bash
uv run ri task=intervene model=llama_8b dataset=gsm8k task.direction=contrastive task.lambda_=0.5
```

For every question with a direction, generates from the direct prompt with its last hidden state at
one layer (the primary layer unless `task.layer` is set) moved towards a target, once, before
generation, with at most 128 new tokens. Each question is also generated once without intervention
(`task.include_baseline`). Reads `directions/<task>/means_layer<L>.npz`, and stops with an error
naming `ri task=directions` when it is missing. Writes one row per (operator, dose) to
`intervention/<direction>_<operator>_layer<L>/<task>/sample_NNNN.parquet`, one file per question.

- `task.direction`: `recovering_mean`, `failed_mean`, `contrastive` (d_q(λ)) or `task_level`
  (d_T(λ)); `task.lambda_` is λ (default 0.5). The directory label is `recovering_mean`,
  `failed_mean`, `contrastive_025`/`050`/`075` or `task_level_050`.
- `task.operator=rotate` (default) rotates the state towards the target with SLERP, keeping its
  norm; the dose is the rotation fraction t, by default 0.25, 0.5, 0.75, 1 and 1.25. The main text
  uses t = 1.
- `task.operator=scale` replaces the state with the target scaled to s times its own norm; the dose
  is s, by default 0.25, 0.5, 1, 2 and 4.
- `task.doses` overrides the doses, e.g. `task.doses=[1.0]`.

The runs of the paper, for each model over all 25 tasks:

```bash
# Table 2 and Figure 4: recovering mean, failed mean and d_q(λ)
uv run ri -m task=intervene model=llama_8b 'dataset=glob(*)' \
    task.direction=recovering_mean,failed_mean
uv run ri -m task=intervene model=llama_8b 'dataset=glob(*)' \
    task.direction=contrastive task.lambda_=0.25,0.5,0.75

# Table 2 and Figure 5: task-level direction d_T(0.5) on its own task
uv run ri -m task=intervene model=llama_8b 'dataset=glob(*)' task.direction=task_level

# Table 8: scaled insertion of d_q(0.5) (the rotation-fraction block comes from the d_q(0.5) run)
uv run ri -m task=intervene model=llama_8b 'dataset=glob(*)' \
    task.direction=contrastive task.lambda_=0.5 task.operator=scale

# Table 9: recovering mean at the other layers of the late band
uv run ri -m task=intervene model=llama_8b 'dataset=glob(*)' task.layer=26,30
uv run ri -m task=intervene model=qwen_7b 'dataset=glob(*)' task.layer=22,26
```

Together these reproduce Sections 5.2 and 5.3, Tables 2, 8 (first two blocks), 9 and 10, and
Figures 4 and 5, through the report of Step 10; the task-level rows and Figure 5 only up to the
small angle described in Step 7.

**Cross-task transfer (Appendix D.4, Figure 7).** `task.source_dataset` applies the task-level
direction d_T(0.5) of another task to the questions of `dataset`, rotated in once at t = 1, without
a baseline generation. It requires `task.direction=task_level`, `task.lambda_=0.5` and the rotate
operator, and the means of the source task at the same layer. Results go to
`intervention/task_transfer_layer<L>/from_<source>/<task>/sample_NNNN.parquet`. The paper reports
Qwen, and its Figure 7 is reproduced up to the same small angle:

```bash
uv run ri task=intervene model=qwen_7b dataset=bbh_snarks \
    task.direction=task_level task.source_dataset=gsm8k

# every source on every other task
TASKS=$(uv run python -c "from ri.tasks import TASKS; print(' '.join(TASKS))")
for source in $TASKS; do
    targets=$(echo $TASKS | tr ' ' '\n' | grep -vx "$source" | paste -sd, -)
    uv run ri -m task=intervene model=qwen_7b "dataset=$targets" \
        task.direction=task_level task.source_dataset=$source
done
```

### Step 9: Every-token rotation

```bash
uv run ri task=steer model=llama_8b dataset=gsm8k
```

The rotation towards the recovering mean applied at the last prompt position and at every generated
token, with t in 0.25, 0.5 and 1 and at most 128 new tokens, plus a baseline generation. Reads the
means and writes `steering/recovering_mean_rotate_layer<L>/<task>/sample_NNNN.parquet`. It takes
the `task.layer`, `task.doses`, `task.include_baseline`, `task.sample_start`, `task.sample_end` and
`task.resume` options of `intervene`.

Reproduces the every-token block of Table 8 (Appendix D.1) through the report.

### Step 10: Report

```bash
uv run ri task=report model=llama_8b
uv run ri task=report 'task.models=[llama_8b,qwen_7b]' task.output_dir=paper_tables
```

Reads the results tree of each model and writes the tables as `analysis/*.csv`, or, with
`task.output_dir`, the tables of all models in one directory, concatenated with a `model` column.
The evaluations of both prompts are required; the patch-label, intervention, transfer and geometry
tables are written for the stages that have results, and the report logs the directories it skips.

Intervention accuracy is intention-to-treat: the denominator is every question of the task, and a
question without a direction or without a result row counts as incorrect. Averages are macro
averages over the tasks evaluated under both prompts, and the differences to the prompts, the
paired tests and the intervention and cross-task tables use the same tasks. When these are fewer
than 25, the report logs one warning listing the missing tasks; every summary row records the
number of tasks used in `n_tasks`.

An intervention or cross-task cell (run, dose, task) is complete when every question of the task
that has a direction, according to `directions/<task>/means_layer<L>.npz`, has a result row.
`interventions_per_task` and `transfer_matrix` give the number of these questions in `n_expected`
and the check in `complete`; both are null when the means file is missing, and the report logs
that completeness cannot be checked. `transfer_matrix` has a row for every source task with a
cross-task run, every other task the report covers and every dose run at that layer, besides the
diagonal from the task-level run. A cell the run never reached has `n_scored` 0 and `complete`
false, with or without a means file, unless its task has no question with a direction. Incomplete
cells are logged once per kind. Incomplete cross-task cells are left out of `transfer_summary` and
`transfer_by_target`: `n_cells` counts the off-diagonal cells kept, `n_cells_incomplete` those left
out, and `n_cells_unchecked` the kept cells whose completeness is unknown. The report reads every
parquet of a run directory and stops with an error naming the directory and the files when two
rows share a (task, question, operator, dose) key. `wilcoxon` records the method of each test
(`exact`, `permutation` or `asymptotic`) and the number of nonzero differences.

| Table | Paper |
|-------|-------|
| `baselines_per_task`, `baselines_summary` | Table 1 |
| `patch_labels` | Table 4, Appendix B |
| `interventions_summary` | Tables 2, 8 and 9 |
| `interventions_per_task` | Table 10, Figures 4 and 5 |
| `wilcoxon` | the Wilcoxon signed-rank tests of Appendix D.2 |
| `mcnemar_per_task` | a per-task exact McNemar test against each prompt, with Benjamini-Hochberg significance at 5% across tasks; not reported in the paper |
| `both_prompts_wrong` | the share of questions both prompts answer incorrectly that the direction answers correctly (Section 5.2) |
| `geometry_per_task`, `geometry_summary` | the cosines of Section 5.1 |
| `transfer_matrix`, `transfer_summary`, `transfer_by_target` | Figure 7, Appendix D.4 |

## Results layout

```
results/                                     # $RI_RESULTS_ROOT
└── <model_short>/                           # llama8B, qwen7B
    ├── eval/
    │   └── <cot|direct>/<task>/*.parquet
    ├── patch_sweep/
    │   ├── combined_<task>.parquet
    │   ├── generation_cache/
    │   └── <task>/                          # working files of running jobs
    ├── probe/
    │   └── layer<L>/
    │       ├── <task>/
    │       │   ├── metrics.json
    │       │   ├── directions.npz
    │       │   ├── question_folds.parquet
    │       │   └── out_of_fold_predictions.parquet
    │       ├── transfer.parquet
    │       └── transfer_summary.json
    ├── directions/
    │   └── <task>/means_layer<L>.npz
    ├── intervention/
    │   ├── <direction>_<operator>_layer<L>/<task>/sample_NNNN.parquet
    │   └── task_transfer_layer<L>/from_<source>/<task>/sample_NNNN.parquet
    ├── steering/
    │   └── recovering_mean_rotate_layer<L>/<task>/sample_NNNN.parquet
    └── analysis/
        ├── *.csv                            # ri task=report
        └── probe_*.csv                      # ri task=probe_report

hs_cache/                                    # $HS_CACHE_DIR
└── <model_short>/<task>/sample_NNNN.h5
```

`<L>` is a 0-indexed layer. The patch sweep's lock is `combined_<task>.lock`, next to the combined
parquet. Files written atomically carry a `.tmp` suffix until they are complete.

## Project structure

- `ri/main.py` — Hydra entry point and task registry
- `ri/conf/` — YAML config tree (task/model/dataset/tracking)
- `ri/tasks.py` — the 25 tasks, their reasoning types, answer formats and display names
- `ri/paths.py`, `ri/settings.py` — results tree and cache locations
- `ri/common/`, `ri/prompts/`, `ri/utils/` — datasets, seeding, prompt assembly, tokenizer helpers,
  answer extraction
- `ri/core/` — model loading, layer selection, forward hooks for patching and rotation
- `ri/evaluation/` — CoT and direct baselines
- `ri/patching/` — the patch sweep
- `ri/activations/` — the HDF5 activation cache
- `ri/probe/` — linear probes, transfer and the probe report
- `ri/directions/` — recovering and failed means, contrastive and task-level directions
- `ri/intervention/` — one-shot and every-token interventions
- `ri/analysis/` — intention-to-treat accuracy, paired tests, geometry and the report
- `tests/` — offline tests on CPU with tiny randomly initialised models and synthetic data

## Development

`uv sync` installs the development group (ruff, mypy, codespell, deptry, pre-commit, pytest). The
tests need no GPU, no model download and no network:

```bash
uv run pytest
```

Install the git hooks with `uv run pre-commit install`; they run ruff, ruff-format, codespell, mypy
and deptry inside the project environment. To run the checks by hand:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy ri
uv run codespell
uv run deptry .
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the same hooks and the tests on
every push to `main` and every pull request.

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Citation

Citation metadata for this software is in [CITATION.cff](CITATION.cff). The reference to the paper
will be added here on publication.
