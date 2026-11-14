from __future__ import annotations

from typing import Any

import pytest
import torch

from ri.utils.tokenizer import (
    count_generated_tokens,
    decode_tokens,
    get_eos_token_ids,
    get_extra_stop_tokens,
    get_pad_id,
    make_inputs,
    render_prompts,
    strip_llama_default_metadata,
)

LLAMA_EOT_ID = 128009
QWEN_IM_END_ID = 151645
QWEN_ENDOFTEXT_ID = 151643


class FakeTokenizer:
    """Exposes only what the stop-token helpers read."""

    def __init__(
        self, eos_token: str, special: list[str], vocab: dict[str, int], pad_token_id: Any = 0
    ) -> None:
        self.eos_token = eos_token
        self.eos_token_id = vocab[eos_token]
        self.all_special_tokens = special
        self.pad_token_id = pad_token_id
        self._vocab = vocab

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab[token]


def _llama() -> FakeTokenizer:
    return FakeTokenizer(
        eos_token="<|eot_id|>",
        special=["<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>"],
        vocab={"<|begin_of_text|>": 128000, "<|end_of_text|>": 128001, "<|eot_id|>": LLAMA_EOT_ID},
    )


def _qwen() -> FakeTokenizer:
    return FakeTokenizer(
        eos_token="<|im_end|>",
        special=["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        vocab={
            "<|im_start|>": 151644,
            "<|im_end|>": QWEN_IM_END_ID,
            "<|endoftext|>": QWEN_ENDOFTEXT_ID,
        },
        pad_token_id=QWEN_ENDOFTEXT_ID,
    )


def test_llama_stop_set_is_exactly_eot_id() -> None:
    assert get_eos_token_ids(_llama()) == LLAMA_EOT_ID
    assert get_extra_stop_tokens(_llama()) == []


def test_qwen_stop_set_includes_endoftext() -> None:
    ids = get_eos_token_ids(_qwen())
    assert ids == [QWEN_IM_END_ID, QWEN_ENDOFTEXT_ID]
    assert get_extra_stop_tokens(_qwen()) == ["<|im_end|>", "<|endoftext|>"]


def test_extra_stop_tokens_ignored_when_not_declared() -> None:
    tok = FakeTokenizer(
        eos_token="</s>", special=["<s>", "</s>"], vocab={"<s>": 1, "</s>": 2, "<|endoftext|>": 999}
    )
    assert get_extra_stop_tokens(tok) == []
    assert get_eos_token_ids(tok) == 2


def test_pad_id_falls_back_to_zero() -> None:
    assert get_pad_id(_qwen()) == QWEN_ENDOFTEXT_ID
    for pad in (None, -1, "x"):
        assert get_pad_id(FakeTokenizer("</s>", ["</s>"], {"</s>": 2}, pad_token_id=pad)) == 0


def test_count_excludes_trailing_stop_and_pad_tokens() -> None:
    tok = FakeTokenizer("</s>", ["</s>"], {"</s>": 2}, pad_token_id=0)
    assert count_generated_tokens(tok, torch.tensor([7, 8, 9, 2, 0, 0])) == 3
    assert count_generated_tokens(tok, [7, 8, 9, 10]) == 4
    assert count_generated_tokens(tok, [2, 0, 0]) == 0
    assert count_generated_tokens(tok, [7, 2, 8]) == 3


def test_count_honours_every_qwen_stop_id() -> None:
    assert count_generated_tokens(_qwen(), [7, 8, QWEN_IM_END_ID]) == 2
    assert count_generated_tokens(_qwen(), [7, 8, QWEN_ENDOFTEXT_ID]) == 2


def test_strip_llama_default_metadata() -> None:
    class _Tok:
        chat_template = "a\nCutting Knowledge Date: Dec 2023\nToday Date: 1 Jan\nb\n"

    tok = _Tok()
    strip_llama_default_metadata(tok)
    assert tok.chat_template == "a\nb\n"


def _word_tokenizer() -> Any:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    words = ["<pad>", "<s>", "</s>", "<eot>", "hello", "world", "system", "user", "reply"]
    model = tokenizers.models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<pad>")
    backend = tokenizers.Tokenizer(model)
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 1)]
    )
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", bos_token="<s>", eos_token="</s>"
    )
    tok.chat_template = (
        "<s>{% for m in messages %} {{ m['role'] }} {{ m['content'] }} <eot>{% endfor %}"
        "{% if add_generation_prompt %} reply{% endif %}"
    )
    return tok


CONVOS = [
    [{"role": "system", "content": "world"}, {"role": "user", "content": "hello"}],
    [{"role": "system", "content": "world"}, {"role": "user", "content": "hello hello world"}],
]


def test_render_prompts_without_chat_template_keeps_the_user_message() -> None:
    assert render_prompts(None, CONVOS) == ["hello", "hello hello world"]


def test_make_inputs_pads_on_the_left() -> None:
    tok = _word_tokenizer()
    left = make_inputs(tok, CONVOS)
    assert left["input_ids"].tolist() == [[0, 0, 1, 4], [1, 4, 4, 5]]
    assert left["attention_mask"].tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]


def test_make_inputs_with_chat_template_adds_no_second_bos() -> None:
    tok = _word_tokenizer()
    rendered = render_prompts(tok, CONVOS[:1], system_prompt=True, add_generation_prompt=True)
    assert rendered == ["<s> system world <eot> user hello <eot> reply"]
    batch = make_inputs(tok, CONVOS[:1], system_prompt=True, add_generation_prompt=True)
    assert batch["input_ids"].tolist() == [[1, 6, 5, 3, 7, 4, 3, 8]]


def test_decode_tokens_handles_one_and_two_dimensions() -> None:
    tok = _word_tokenizer()
    assert decode_tokens(tok, torch.tensor([4, 5])) == "hello world"
    assert decode_tokens(tok, torch.tensor([[4], [5]])) == ["hello", "world"]
