"""Cache locations, read from the environment once at import."""

from __future__ import annotations

import os

# The Hugging Face model cache, passed as ``cache_dir`` to every download. When
# RI_CACHE_DIR is unset this is None, and the libraries resolve the standard cache
# themselves (HF_HUB_CACHE, HF_HOME/hub, ~/.cache/huggingface/hub).
HUGGINGFACE_CACHE_DIR: str | None = os.getenv("RI_CACHE_DIR") or None

# A local directory, or an ``hf://<org>/<repo>`` dataset spec that is read-only.
HS_CACHE_DIR: str = os.getenv("HS_CACHE_DIR", "hs_cache")
