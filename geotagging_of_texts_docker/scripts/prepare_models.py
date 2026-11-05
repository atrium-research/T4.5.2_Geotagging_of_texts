#!/usr/bin/env python3
"""Image-build step: make sure every model is in /models and actually loads.

Models already present in the copied ./models folder are not downloaded again.
Missing ones are fetched. Each model is then loaded once, so a broken file fails the build.
"""
from __future__ import annotations

import hashlib
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections import deque

from app.config import (BAKED_CHAT_MODEL, EMBEDDING_MODEL, NER_MODEL_PATH, NER_WHEEL_SHA256, NER_WHEEL_URL,
                        OCR_MODEL, RERANKER_MODEL)

SERVER = os.getenv("LLAMA_SERVER_BIN", "/app/llama-server")
TIMEOUT = int(os.getenv("MODEL_CACHE_TIMEOUT", "2400"))
PROGRESS_RE = re.compile(r"download|\bETA\b|\b\d+(?:\.\d+)?%", re.I)


def say(msg: str) -> None:
    print(f"[models] {msg}", flush=True)


def load_gguf(role: str, model: str, extra: list[str]) -> None:
    say(f"Initializing {role} model: {model}")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [SERVER, "-hf", model, "--host", "127.0.0.1", "--port", str(port), "--ctx-size", "512",
               "--no-webui", "--n-gpu-layers", "0", *extra]
    process = subprocess.Popen(command, env={**os.environ, "LLAMA_CACHE": "/models"},
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: deque[str] = deque(maxlen=40)

    def pump() -> None:
        for line in process.stdout:  # type: ignore[union-attr]
            tail.append(line.rstrip())
            if PROGRESS_RE.search(line):
                say(line.rstrip())

    threading.Thread(target=pump, daemon=True).start()
    started = time.monotonic()
    try:
        while time.monotonic() - started < TIMEOUT:
            if process.poll() is not None:
                raise RuntimeError(f"llama-server exited while loading {model} (code {process.returncode}):\n"
                                   + "\n".join(tail))
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.5) as response:
                    if response.status == 200:
                        say(f"OK {role}: {model} loaded.")
                        return
            except OSError:
                pass
            time.sleep(1)
        raise TimeoutError(f"Timed out loading {model}.\n" + "\n".join(tail))
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


def load_embedding() -> None:
    say(f"Initializing embedding model: {EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    vector = SentenceTransformer(EMBEDDING_MODEL, device="cpu").encode(
        "Athens: a major city in Greece.", prompt_name="SearchQuery", normalize_embeddings=True)
    say(f"OK embedding: {EMBEDDING_MODEL} loaded ({len(vector)} dimensions).")


def load_ner() -> None:
    say(f"Initializing NER model: {NER_MODEL_PATH}")
    if not os.path.isfile(os.path.join(NER_MODEL_PATH, "meta.json")):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = os.path.join(tmp, NER_WHEEL_URL.rsplit("/", 1)[1])
            urllib.request.urlretrieve(NER_WHEEL_URL, wheel)
            with open(wheel, "rb") as handle:
                if hashlib.sha256(handle.read()).hexdigest() != NER_WHEEL_SHA256:
                    raise RuntimeError("NER wheel checksum mismatch; refusing to install it.")
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", "/models/ner", wheel],
                           check=True)
    import spacy
    entities = spacy.load(NER_MODEL_PATH)("The army marched from Athens to Corinth.").ents
    say(f"OK NER: loaded, found {[e.text for e in entities]} in the test sentence.")


def main() -> None:
    load_gguf("chat", BAKED_CHAT_MODEL, ["--reasoning", "off"])
    load_gguf("reranker", RERANKER_MODEL, ["--embedding", "--pooling", "rank", "--reranking"])
    load_gguf("ocr", OCR_MODEL, [])
    load_embedding()
    load_ner()
    say("All models are in /models and load correctly.")


if __name__ == "__main__":
    main()
