import importlib

import pytest

from app import config


@pytest.mark.parametrize("value", ["--hf-token=x", "owner/repo;rm -rf /", "-m/x", "noslash", "a/b:q q"])
def test_rejects_unsafe_chat_model(monkeypatch, value):
    monkeypatch.setenv("CHAT_MODEL", value)
    with pytest.raises(SystemExit):
        importlib.reload(config)


def test_accepts_repo_and_quant(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M")
    assert importlib.reload(config).CHAT_MODEL.endswith("Q4_K_M")
