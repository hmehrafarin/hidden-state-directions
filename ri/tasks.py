"""The 25 tasks, their reasoning and answer types, and display names."""

from __future__ import annotations

import re

TASKS: tuple[str, ...] = (
    "gsm8k",
    "strategyqa",
    "bbh_boolean_expressions",
    "bbh_causal_judgement",
    "bbh_date_understanding",
    "bbh_disambiguation_qa",
    "bbh_geometric_shapes",
    "bbh_hyperbaton",
    "bbh_logical_deduction_five_objects",
    "bbh_logical_deduction_seven_objects",
    "bbh_logical_deduction_three_objects",
    "bbh_movie_recommendation",
    "bbh_multistep_arithmetic_two",
    "bbh_navigate",
    "bbh_object_counting",
    "bbh_penguins_in_a_table",
    "bbh_reasoning_about_colored_objects",
    "bbh_ruin_names",
    "bbh_snarks",
    "bbh_sports_understanding",
    "bbh_temporal_sequences",
    "bbh_tracking_shuffled_objects_five_objects",
    "bbh_tracking_shuffled_objects_seven_objects",
    "bbh_tracking_shuffled_objects_three_objects",
    "bbh_web_of_lies",
)

# Reasoning types in the order of the paper's tables.
REASONING_TYPES: tuple[str, ...] = (
    "arithmetic",
    "logical",
    "temporal_state",
    "spatial_geometric",
    "causal_commonsense",
    "linguistic_pragmatic",
)

_REASONING_GROUPS: dict[str, tuple[str, ...]] = {
    "arithmetic": (
        "gsm8k",
        "bbh_multistep_arithmetic_two",
        "bbh_object_counting",
    ),
    "logical": (
        "bbh_boolean_expressions",
        "bbh_logical_deduction_three_objects",
        "bbh_logical_deduction_five_objects",
        "bbh_logical_deduction_seven_objects",
        "bbh_web_of_lies",
        "bbh_penguins_in_a_table",
    ),
    "temporal_state": (
        "bbh_temporal_sequences",
        "bbh_date_understanding",
        "bbh_tracking_shuffled_objects_three_objects",
        "bbh_tracking_shuffled_objects_five_objects",
        "bbh_tracking_shuffled_objects_seven_objects",
        "bbh_reasoning_about_colored_objects",
    ),
    "spatial_geometric": (
        "bbh_navigate",
        "bbh_geometric_shapes",
    ),
    "causal_commonsense": (
        "bbh_causal_judgement",
        "strategyqa",
        "bbh_sports_understanding",
        "bbh_movie_recommendation",
    ),
    "linguistic_pragmatic": (
        "bbh_hyperbaton",
        "bbh_disambiguation_qa",
        "bbh_snarks",
        "bbh_ruin_names",
    ),
}

REASONING_TYPE: dict[str, str] = {
    task: reasoning for reasoning, tasks in _REASONING_GROUPS.items() for task in tasks
}

_ANSWER_GROUPS: dict[str, tuple[str, ...]] = {
    "boolean": (
        "bbh_boolean_expressions",
        "bbh_causal_judgement",
        "bbh_navigate",
        "bbh_sports_understanding",
        "bbh_web_of_lies",
        "strategyqa",
    ),
    "multiple_choice": (
        "bbh_date_understanding",
        "bbh_disambiguation_qa",
        "bbh_geometric_shapes",
        "bbh_hyperbaton",
        "bbh_logical_deduction_three_objects",
        "bbh_logical_deduction_five_objects",
        "bbh_logical_deduction_seven_objects",
        "bbh_movie_recommendation",
        "bbh_penguins_in_a_table",
        "bbh_reasoning_about_colored_objects",
        "bbh_ruin_names",
        "bbh_snarks",
        "bbh_temporal_sequences",
        "bbh_tracking_shuffled_objects_three_objects",
        "bbh_tracking_shuffled_objects_five_objects",
        "bbh_tracking_shuffled_objects_seven_objects",
    ),
    "numeric": (
        "gsm8k",
        "bbh_multistep_arithmetic_two",
        "bbh_object_counting",
    ),
}

ANSWER_TYPE: dict[str, str] = {
    task: answer for answer, tasks in _ANSWER_GROUPS.items() for task in tasks
}

DISPLAY_NAME: dict[str, str] = {
    "gsm8k": "GSM8K",
    "strategyqa": "StrategyQA",
    "bbh_boolean_expressions": "Boolean Expr.",
    "bbh_causal_judgement": "Causal Judgement",
    "bbh_date_understanding": "Date Underst.",
    "bbh_disambiguation_qa": "Disambig. QA",
    "bbh_geometric_shapes": "Geometric Shapes",
    "bbh_hyperbaton": "Hyperbaton",
    "bbh_logical_deduction_five_objects": "Logical Ded. (5)",
    "bbh_logical_deduction_seven_objects": "Logical Ded. (7)",
    "bbh_logical_deduction_three_objects": "Logical Ded. (3)",
    "bbh_movie_recommendation": "Movie Rec.",
    "bbh_multistep_arithmetic_two": "Multistep Arith.",
    "bbh_navigate": "Navigate",
    "bbh_object_counting": "Object Counting",
    "bbh_penguins_in_a_table": "Penguins",
    "bbh_reasoning_about_colored_objects": "Colored Objects",
    "bbh_ruin_names": "Ruin Names",
    "bbh_snarks": "Snarks",
    "bbh_sports_understanding": "Sports Underst.",
    "bbh_temporal_sequences": "Temporal Seq.",
    "bbh_tracking_shuffled_objects_five_objects": "Shuffled Obj. (5)",
    "bbh_tracking_shuffled_objects_seven_objects": "Shuffled Obj. (7)",
    "bbh_tracking_shuffled_objects_three_objects": "Shuffled Obj. (3)",
    "bbh_web_of_lies": "Web of Lies",
}


def task_slug(name: str) -> str:
    """Filesystem slug of a task name; ``bbh/snarks`` becomes ``bbh_snarks``."""
    clean = str(name).strip().replace("\\", "/")
    if clean.startswith("bbh/"):
        clean = "bbh_" + clean.split("/", 1)[1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", clean).strip("_")
