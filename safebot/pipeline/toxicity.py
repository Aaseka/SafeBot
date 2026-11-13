"""Stage 2: local multi-label toxicity classifier (unitary/toxic-bert by default).

Runs off the event loop so the bot stays responsive while a message is scored.
"""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class ToxicityModel:
    def __init__(self, model_name: str, max_concurrency: int = 2):
        self.model_name = model_name
        self._pipe = None
        self._sem = asyncio.Semaphore(max_concurrency)

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline("text-classification", model=self.model_name, top_k=None)
        return self._pipe

    def _run(self, text: str) -> dict[str, float]:
        out = self._load()(text, truncation=True, max_length=512, function_to_apply="sigmoid")
        if out and isinstance(out[0], list):  # batched-shape output
            out = out[0]
        return {d["label"]: float(d["score"]) for d in out}

    async def warmup(self) -> None:
        await asyncio.to_thread(self._run, "hello")

    async def score(self, text: str) -> dict[str, float]:
        async with self._sem:
            return await asyncio.to_thread(self._run, text)
