"""Model set and runtime settings. CHAT_MODEL is the only model the user may change."""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

# Fixed by the image on purpose: never read from the environment.
RERANKER_MODEL = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0"
OCR_MODEL = "ggml-org/GLM-OCR-GGUF:Q8_0"
EMBEDDING_MODEL = "google/embeddinggemma-2"
NER_MODEL_PATH = "/models/ner/en_deberta_v3_base_ner_historical_place/en_deberta_v3_base_ner_historical_place-0.0.0"
NER_WHEEL_URL = ("https://huggingface.co/NikosKprl/en_deberta_v3_base_ner_historical_place/resolve/main/"
                 "en_deberta_v3_base_ner_historical_place-1.0-py3-none-any.whl")
NER_WHEEL_SHA256 = "d1039567f7e48c128e9c6a072eb221858c1b0a460cc367d33f03dab8d93310e2"

# The LLM baked into the image, and the one actually requested.
BAKED_CHAT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"
CHAT_MODEL = (os.getenv("CHAT_MODEL") or BAKED_CHAT_MODEL).strip()
# repo/name[:quant] only: the value ends up on a llama-server command line.
if not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[\w.-]+(:[\w.-]+)?", CHAT_MODEL):
    raise SystemExit(f"Invalid CHAT_MODEL {CHAT_MODEL!r}. Expected 'owner/repo' or 'owner/repo:quant'.")

MODELS_DIR = Path("/models")                      # baked, read-only at runtime
DOWNLOAD_CACHE = Path(os.getenv("LLAMA_DOWNLOAD_CACHE", "/cache"))  # a non-baked CHAT_MODEL lands here


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    import torch
    return torch.cuda.is_available()


def device() -> str:
    return "cuda" if gpu_available() else "cpu"
