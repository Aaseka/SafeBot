"""Minimal client for any OpenAI-compatible chat API.

Works with Groq, OpenRouter, OpenAI, or a local Ollama server
(LLM_BASE_URL=http://localhost:11434/v1). If LLM_BASE_URL is unset, the bot
runs without an LLM and falls back to the local models only.
"""
from __future__ import annotations

import httpx


class LLM:
    def __init__(self, base_url: str | None, api_key: str | None, model: str):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(timeout=20, headers=headers) if self.base_url else None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def chat(self, system: str, user: str, *, json_mode: bool = False, max_tokens: int = 400) -> str:
        if not self._client:
            raise RuntimeError("LLM is not configured")
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        resp = await self._client.post(f"{self.base_url}/chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
