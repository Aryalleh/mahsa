"""Wrapper around llama.cpp for the local Meta-Llama-3.1-8B-Instruct GGUF.

Inference is blocking/CPU-heavy, so callers should run :meth:`chat` in a
thread (e.g. ``asyncio.to_thread``) to avoid stalling the Telegram event loop.
A lock serialises access because a single ``Llama`` instance is not safe to
call concurrently.
"""
from __future__ import annotations

import threading
from pathlib import Path

from ..utils.logging import get_logger

log = get_logger("mahsa.llm")

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": ...}


class LlamaEngine:
    def __init__(
        self,
        model_path: str,
        context: int = 8192,
        gpu_layers: int = 0,
        temperature: float = 0.85,
        max_tokens: int = 400,
    ):
        self.model_path = model_path
        self.context = context
        self.gpu_layers = gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the model into memory. Raises a clear error if weights are missing."""
        if self._llm is not None:
            return
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model weights not found at '{self.model_path}'.\n"
                "Download the GGUF, e.g.:\n"
                "  huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF "
                "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models"
            )
        # Imported lazily so the rest of the app can be imported without the
        # heavy native dependency installed (e.g. for tests).
        from llama_cpp import Llama

        log.info("Loading model %s (ctx=%d, gpu_layers=%d) ...",
                 self.model_path, self.context, self.gpu_layers)
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.context,
            n_gpu_layers=self.gpu_layers,
            chat_format="llama-3",
            verbose=False,
        )
        log.info("Model loaded.")

    def chat(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run a chat completion and return the assistant's text."""
        if self._llm is None:
            self.load()
        with self._lock:
            result = self._llm.create_chat_completion(
                messages=messages,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.15,  # discourages the looping/rambling small quants do
                stop=["<|eot_id|>", "<|end_of_text|>"],
            )
        return result["choices"][0]["message"]["content"].strip()

    def chat_json(
        self,
        messages: list[Message],
        temperature: float = 0.1,
        max_tokens: int = 200,
    ) -> str:
        """Chat completion constrained to valid JSON (for intent detection)."""
        if self._llm is None:
            self.load()
        with self._lock:
            result = self._llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        return result["choices"][0]["message"]["content"].strip()
