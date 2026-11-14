from __future__ import annotations

from typing import Any

import torch

from ri.core.model import ModelAndTokenizer, set_requires_grad


class _Tokenizer:
    chat_template = "x\nToday Date: 1 Jan\ny\n"


def test_model_load_kwargs_pass_dtype() -> None:
    kwargs = ModelAndTokenizer._model_load_kwargs(cache_dir="/cache", torch_dtype=torch.bfloat16)
    assert kwargs == {
        "cache_dir": "/cache",
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "dtype": torch.bfloat16,
    }


def test_model_load_kwargs_omit_none_dtype() -> None:
    kwargs = ModelAndTokenizer._model_load_kwargs(cache_dir=None, torch_dtype=None)
    assert "dtype" not in kwargs
    assert "torch_dtype" not in kwargs


def test_wraps_a_loaded_model(tiny_causal_lm: Any) -> None:
    mt = ModelAndTokenizer(model=tiny_causal_lm, tokenizer=_Tokenizer(), device="cpu")
    assert mt.model is tiny_causal_lm
    assert mt.device == "cpu"
    assert mt.tokenizer.chat_template == "x\ny\n"


def test_the_chat_template_is_used_whenever_the_tokenizer_has_one(tiny_causal_lm: Any) -> None:
    mt = ModelAndTokenizer("/models/llama", model=tiny_causal_lm, tokenizer=_Tokenizer())
    assert mt.uses_chat_template

    class _Plain:
        chat_template = None

    plain = ModelAndTokenizer("org/Tiny-Instruct", model=tiny_causal_lm, tokenizer=_Plain())
    assert not plain.uses_chat_template


def test_the_model_name_falls_back_to_the_loaded_model(tiny_causal_lm: Any) -> None:
    tiny_causal_lm.config._name_or_path = "org/Some-Model"
    try:
        derived = ModelAndTokenizer(model=tiny_causal_lm, tokenizer=_Tokenizer(), device="cpu")
        assert derived.model_name == "org/Some-Model"
    finally:
        tiny_causal_lm.config._name_or_path = ""


def test_set_requires_grad(make_tiny_causal_lm: Any) -> None:
    model = make_tiny_causal_lm("llama")
    set_requires_grad(False, model)
    assert not any(p.requires_grad for p in model.parameters())
    set_requires_grad(True, model)
    assert all(p.requires_grad for p in model.parameters())
