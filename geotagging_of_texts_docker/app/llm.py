from __future__ import annotations

import base64
import os
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

import requests
from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI

from .config import BAKED_CHAT_MODEL, DOWNLOAD_CACHE, MODELS_DIR, gpu_available

PORTS = {"chat": 8088, "reranker": 8089, "ocr": 8090}  # container-local only, never published


@dataclass
class _Server:
    signature: tuple[str, str | None, bool]
    process: subprocess.Popen


class LlamaManager:
    """Keep one llama-server process resident for each GGUF role."""

    def __init__(self) -> None:
        self.bin = os.getenv("LLAMA_SERVER_BIN", "/app/llama-server")
        self.threads = int(os.getenv("LLAMA_THREADS", "0"))
        self.context = int(os.getenv("LLAMA_CONTEXT", "4096"))
        self._servers: dict[str, _Server] = {}
        self._lock = threading.RLock()

    def _client(self, role: str, timeout: float = 300.0) -> OpenAI:
        return OpenAI(base_url=f"http://127.0.0.1:{PORTS[role]}/v1", api_key="local-llama-server",
                      timeout=timeout, max_retries=0)

    @staticmethod
    def _healthy(port: int) -> bool:
        try:
            for path in ("/health", "/v1/models"):
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=1.2) as response:
                    if response.status != 200:
                        return False
            return True
        except Exception:
            return False

    @staticmethod
    def _stop_process(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def stop(self, role: str | None = None) -> None:
        """Stop one role, or every llama-server process during app shutdown."""
        with self._lock:
            for name in [role] if role is not None else list(self._servers):
                server = self._servers.pop(name, None)
                if server:
                    self._stop_process(server.process)

    def ensure(self, model: str, *, reasoning: str | None = None, reranking: bool = False,
               role: str = "chat", on_status=None, timeout: int = 900) -> OpenAI:
        model = model.strip()
        if not model:
            raise RuntimeError(f"No llama.cpp model is configured for role {role}.")
        if reasoning not in (None, "on", "off", "auto"):
            raise ValueError("reasoning must be one of: None, 'on', 'off', or 'auto'.")
        if reranking and reasoning is not None:
            raise ValueError("Reranking mode cannot be combined with reasoning mode.")

        port = PORTS[role]
        signature = (model, reasoning, reranking)
        with self._lock:
            existing = self._servers.get(role)
            if (existing and existing.signature == signature and existing.process.poll() is None
                    and self._healthy(port)):
                return self._client(role)
            self.stop(role)
            # Baked models load offline. A user-chosen LLM is downloaded once into the /cache volume.
            custom = role == "chat" and model != BAKED_CHAT_MODEL
            cache = DOWNLOAD_CACHE if custom else MODELS_DIR
            if on_status:
                on_status(f"Starting {role} model {model}{' (downloading, not in the image)' if custom else ''}.")

            command = [self.bin, "-hf", model, "--host", "127.0.0.1", "--port", str(port),
                       "--ctx-size", str(self.context), "--no-webui",
                       "--n-gpu-layers", "99" if gpu_available() else "0"]
            if not custom:
                command.append("--offline")
            if self.threads > 0:
                command.extend(["--threads", str(self.threads), "--threads-batch", str(self.threads)])
            if reasoning is not None:
                command.extend(["--reasoning", reasoning])
            if reranking:
                command.extend(["--embedding", "--pooling", "rank", "--reranking"])

            env = {**os.environ, "LLAMA_CACHE": str(cache)}
            if custom:  # the only place the app may reach the network
                env.pop("HF_HUB_OFFLINE", None)
                cache.mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(command, env=env)
            self._servers[role] = _Server(signature=signature, process=process)
            started = time.monotonic()
            while time.monotonic() - started < timeout:
                if process.poll() is not None:
                    self._servers.pop(role, None)
                    raise RuntimeError(
                        f"llama-server for role {role} exited while loading {model} "
                        f"(code {process.returncode}). Check the container logs and model compatibility."
                    )
                if self._healthy(port):
                    return self._client(role)
                elapsed = int(time.monotonic() - started)
                if on_status and elapsed in (15, 60, 180, 360):
                    on_status(f"Still loading {role} model {model}; elapsed {elapsed} seconds.")
                time.sleep(1.0)
            self.stop(role)
            raise TimeoutError(f"Timed out loading {role} model {model} after {timeout} seconds.")

    def start_all(self, chat_model: str, reranker_model: str, ocr_model: str, *, timeout: int = 2400) -> None:
        """Load every GGUF model once, warm it with a tiny request, and keep all three resident."""
        say = lambda msg: print(msg, flush=True)  # noqa: E731
        for role, model, options in (("chat", chat_model, {"reasoning": "off"}),
                                     ("reranker", reranker_model, {"reranking": True}),
                                     ("ocr", ocr_model, {})):
            self.ensure(model, role=role, on_status=say, timeout=timeout, **options)
            say(f"Loaded {role} model: {model}")
        self.chat(chat_model, [{"role": "user", "content": "Athens"}], reasoning="off", max_tokens=1)
        self.rerank(reranker_model, "Athens", ["Athens: a city in Greece."], top_n=1)
        say("All llama.cpp models are loaded and warmed up.")

    def status(self) -> dict[str, dict[str, int | bool]]:
        with self._lock:
            return {role: {"active": role in self._servers and self._servers[role].process.poll() is None,
                           "port": port} for role, port in PORTS.items()}

    def _with_connection_retry(self, model: str, request, *, role: str, reasoning: str | None = None,
                               on_status=None, timeout: int = 900):
        """Retry once after restarting only the disconnected role server."""
        for attempt in range(2):
            client = self.ensure(model, role=role, reasoning=reasoning, on_status=on_status, timeout=timeout)
            try:
                return request(client)
            except (APIConnectionError, APITimeoutError):
                self.stop(role)
                if attempt == 1:
                    raise
                if on_status:
                    on_status(f"The local {role} API disconnected; restarting that model once.")
                time.sleep(0.8)

    def chat(self, model: str, messages: list[dict[str, Any]], *, role: str = "chat",
             reasoning: str | None = None, on_status=None, max_tokens: int = 700,
             temperature: float = 0.1, timeout: float = 300.0) -> str:
        try:
            response = self._with_connection_retry(
                model,
                lambda client: client.chat.completions.create(
                    model=model, messages=messages, max_tokens=max_tokens, temperature=temperature),
                role=role, reasoning=reasoning, on_status=on_status, timeout=max(900, int(timeout)),
            )
        except BadRequestError as exc:
            if "context size" not in str(exc):
                raise
            raise RuntimeError(
                f"A paragraph is too long for the model's context window ({self.context} tokens). "
                "Open the .env file, raise LLAMA_CONTEXT (for example to 16384), then restart the app."
            ) from exc
        return (response.choices[0].message.content or "").strip()

    def ocr_page(self, model: str, png_bytes: bytes, *, page_number: int, on_status=None) -> str:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        messages = [
            {"role": "system", "content": "You are an OCR engine. Transcribe visible text accurately. Return only the text. Preserve paragraph breaks. Do not summarize or add commentary."},
            {"role": "user", "content": [
                {"type": "text", "text": f"Transcribe all text on PDF page {page_number}."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ]
        response = self._with_connection_retry(
            model,
            lambda client: client.chat.completions.create(
                model=model, messages=messages, max_tokens=2500, temperature=0.0),
            role="ocr", on_status=on_status, timeout=1200,
        )
        return (response.choices[0].message.content or "").strip()

    def rerank(self, model: str, query: str, documents: list[str], *, top_n: int,
               on_status=None, timeout: float = 300.0) -> dict[str, Any]:
        """Run llama.cpp's native reranking endpoint on the persistent reranker server."""
        for attempt in range(2):
            self.ensure(model, reranking=True, role="reranker", on_status=on_status,
                        timeout=max(900, int(timeout)))
            try:
                response = requests.post(
                    f"http://127.0.0.1:{PORTS['reranker']}/rerank",
                    json={"query": query, "documents": documents, "top_n": top_n},
                    timeout=(10, timeout),
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                    raise RuntimeError("llama.cpp returned an unexpected reranking response.")
                return payload
            except requests.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                self.stop("reranker")
                if attempt == 0 and (status is None or status >= 500):
                    if on_status:
                        on_status("The local reranking API disconnected; restarting that model once.")
                    time.sleep(0.8)
                    continue
                raise RuntimeError(f"llama.cpp reranking request failed: {exc}") from exc

    def close(self) -> None:
        self.stop()


llama = LlamaManager()
