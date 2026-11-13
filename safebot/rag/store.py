"""Per-server vector store for rules, FAQ entries and moderator feedback examples.

Embeddings live in SQLite; each server's matrix is cached in memory and searched
with a dot product (vectors are normalized, so that is cosine similarity). That
is plenty fast for the few thousand entries a server will ever have.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from ..db import DB


@dataclass
class Hit:
    id: int
    kind: str
    text: str
    meta: dict
    score: float


class KnowledgeStore:
    def __init__(self, db: DB, model_name: str):
        self.db = db
        self.model_name = model_name
        self._model = None
        self._cache: dict[int, tuple[list[dict], np.ndarray]] = {}

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed(self, texts: list[str]) -> np.ndarray:
        def run():
            vecs = self._load().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return vecs.astype(np.float32)

        return await asyncio.to_thread(run)

    async def warmup(self) -> None:
        await self.embed(["hello"])

    async def add_many(self, guild_id: int, kind: str, texts: list[str], meta: dict | None = None) -> int:
        texts = [t.strip() for t in texts if t.strip()]
        if not texts:
            return 0
        vecs = await self.embed(texts)
        for text, vec in zip(texts, vecs):
            await self.db.add_doc(guild_id, kind, text, meta or {}, vec.tobytes())
        self._cache.pop(guild_id, None)
        return len(texts)

    async def add(self, guild_id: int, kind: str, text: str, meta: dict | None = None) -> None:
        await self.add_many(guild_id, kind, [text], meta)

    async def delete(self, guild_id: int, doc_id: int) -> bool:
        ok = await self.db.delete_doc(guild_id, doc_id)
        self._cache.pop(guild_id, None)
        return ok

    async def _index(self, guild_id: int):
        if guild_id not in self._cache:
            rows = await self.db.docs(guild_id)
            if rows:
                mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            else:
                mat = np.zeros((0, 0), dtype=np.float32)
            self._cache[guild_id] = (rows, mat)
        return self._cache[guild_id]

    async def search(self, guild_id: int, query: str, kinds: set[str], k: int = 4,
                     min_score: float = 0.3) -> list[Hit]:
        rows, mat = await self._index(guild_id)
        if not any(r["kind"] in kinds for r in rows):
            return []
        q = (await self.embed([query]))[0]
        scores = mat @ q
        hits = [
            Hit(r["id"], r["kind"], r["text"], r["meta"], float(s))
            for r, s in zip(rows, scores)
            if r["kind"] in kinds and s >= min_score
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
