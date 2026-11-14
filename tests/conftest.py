"""Shared fixtures: tiny randomly initialised models, a byte-level tokenizer, a few GSM8K-style
questions and an isolated results tree."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ri.paths import RESULTS_ROOT_ENV

# Small enough for millisecond forward passes on CPU, with grouped key/value heads.
TINY_HIDDEN_SIZE = 32
TINY_NUM_LAYERS = 6
TINY_VOCAB_SIZE = 128

CHAT_TEMPLATES = {
    "llama": (
        "{{ '<|begin_of_text|>' }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}"
        "<|end_header_id|>\n\n{{ m['content'] }}<|eot_id|>{% endfor %}"
        "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
    ),
    "chatml": (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
        "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    ),
}
SPECIAL_TOKENS: dict[str, dict[str, Any]] = {
    "llama": {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|eot_id|>",
        "additional_special_tokens": ["<|start_header_id|>", "<|end_header_id|>"],
    },
    "chatml": {
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": ["<|im_start|>"],
    },
}


@pytest.fixture(scope="session")
def make_tiny_causal_lm() -> Callable[..., Any]:
    """Factory ``(architecture, seed=0, **overrides)`` for a random causal LM in eval mode.

    ``architecture`` is ``"llama"`` or ``"qwen2"``; ``overrides`` replace config fields.
    """
    transformers = pytest.importorskip("transformers")
    import torch

    def _build(architecture: str, seed: int = 0, **overrides: Any) -> Any:
        torch.manual_seed(seed)
        kwargs: dict[str, Any] = {
            "vocab_size": TINY_VOCAB_SIZE,
            "hidden_size": TINY_HIDDEN_SIZE,
            "intermediate_size": 2 * TINY_HIDDEN_SIZE,
            "num_hidden_layers": TINY_NUM_LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            **overrides,
        }
        if architecture == "qwen2":
            model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**kwargs))
        elif architecture == "llama":
            model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**kwargs))
        else:
            raise ValueError(f"unknown architecture: {architecture!r}")
        return model.eval()

    return _build


@pytest.fixture(scope="module", params=["llama", "qwen2"])
def tiny_causal_lm(request: pytest.FixtureRequest, make_tiny_causal_lm: Callable[..., Any]) -> Any:
    """A tiny Llama and a tiny Qwen2 model, one per parameter."""
    return make_tiny_causal_lm(request.param)


@pytest.fixture(scope="session")
def make_byte_tokenizer() -> Callable[[str], Any]:
    """Factory ``(chat_format)`` for a byte-level tokenizer; no download needed.

    ``chat_format`` is ``"llama"`` (Llama 3 header tokens) or ``"chatml"`` (Qwen).
    """
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    def _build(chat_format: str) -> Any:
        alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
        backend = tokenizers.Tokenizer(
            tokenizers.models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[])
        )
        backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=False
        )
        backend.decoder = tokenizers.decoders.ByteLevel()
        tok = transformers.PreTrainedTokenizerFast(
            tokenizer_object=backend, **SPECIAL_TOKENS[chat_format]
        )
        tok.chat_template = CHAT_TEMPLATES[chat_format]
        return tok

    return _build


@pytest.fixture(scope="session")
def make_tiny_mt(
    make_tiny_causal_lm: Callable[..., Any], make_byte_tokenizer: Callable[[str], Any]
) -> Callable[..., Any]:
    """Factory ``(architecture, chat_format="llama", model_name=None)`` for a tiny instruct
    model with a byte-level tokenizer, wrapped in ``ModelAndTokenizer`` on CPU."""
    from ri.core.model import ModelAndTokenizer

    def _build(architecture: str, chat_format: str = "llama", model_name: str | None = None) -> Any:
        tok = make_byte_tokenizer(chat_format)
        model = make_tiny_causal_lm(architecture, vocab_size=len(tok), max_position_embeddings=1024)
        return ModelAndTokenizer(
            model_name=model_name or f"tiny-{architecture}-Instruct",
            model=model,
            tokenizer=tok,
            device="cpu",
        )

    return _build


@pytest.fixture
def gsm8k_rows() -> list[dict[str, str]]:
    """Three GSM8K-style questions with ``#### <answer>`` gold answers."""
    return [
        {"question": "Tom has 3 apples and buys 4 more. How many apples?", "answer": "#### 7"},
        {"question": "What is 6 times 7?", "answer": "#### 42"},
        {"question": "There are 9 birds and 2 fly away. How many are left?", "answer": "#### 7"},
    ]


@pytest.fixture
def results_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``RI_RESULTS_ROOT`` at an empty temporary directory and return it."""
    root = tmp_path / "results"
    root.mkdir()
    monkeypatch.setenv(RESULTS_ROOT_ENV, str(root))
    return root
