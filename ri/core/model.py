from __future__ import annotations

from typing import Any

import torch
import transformers

from ri.settings import HUGGINGFACE_CACHE_DIR
from ri.utils.tokenizer import strip_llama_default_metadata


class ModelAndTokenizer:
    """A causal language model and its tokenizer, loaded in eval mode without gradients.

    Pass ``model`` and ``tokenizer`` to wrap objects that are already loaded.
    ``cache_dir`` defaults to ``$RI_CACHE_DIR``; when that is unset too, the
    Hugging Face libraries resolve their own cache. ``model_name`` may be a Hub id
    or a local model directory.
    """

    def __init__(
        self,
        model_name: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        *,
        torch_dtype: torch.dtype | None = None,
        device: str = "cuda",
        cache_dir: str | None = None,
    ) -> None:
        if cache_dir is None:
            cache_dir = HUGGINGFACE_CACHE_DIR

        if tokenizer is None:
            if model_name is None:
                raise ValueError("model_name is required to load a tokenizer")
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, use_fast=True, cache_dir=cache_dir
            )
        if model is None:
            if model_name is None:
                raise ValueError("model_name is required to load a model")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name, **self._model_load_kwargs(cache_dir=cache_dir, torch_dtype=torch_dtype)
            )
            set_requires_grad(False, model)
            model.eval()

        # Any causal LM and tokenizer with the Hugging Face interface; the stubs type the
        # Auto* factories too narrowly for generate() to type-check.
        self.tokenizer: Any = tokenizer
        strip_llama_default_metadata(self.tokenizer)
        self.model: Any = model
        self.device = device
        self.model_name = self._derive_model_name(model_name, model)

    @property
    def uses_chat_template(self) -> bool:
        """Whether prompts are rendered with the tokenizer's chat template.

        Every stage renders both prompts through the template whenever the
        tokenizer has one, as the instruct models do.
        """
        return bool(getattr(self.tokenizer, "chat_template", None))

    @staticmethod
    def _derive_model_name(model_name: str | None, model: Any) -> str | None:
        if model_name:
            return model_name
        direct = getattr(model, "name_or_path", None)
        if direct:
            return str(direct)
        config = getattr(model, "config", None)
        cfg_name = getattr(config, "_name_or_path", None)
        return str(cfg_name) if cfg_name else None

    @staticmethod
    def _model_load_kwargs(
        *, cache_dir: str | None, torch_dtype: torch.dtype | None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }
        if torch_dtype is not None:
            kwargs["dtype"] = torch_dtype
        return kwargs


def set_requires_grad(requires_grad: bool, *models: torch.nn.Module) -> None:
    """Set ``requires_grad`` on every parameter of the given modules."""
    for model in models:
        for param in model.parameters():
            param.requires_grad = requires_grad
