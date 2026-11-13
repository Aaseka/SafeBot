import json
from typing import ClassVar

import numpy as np
import pytest

from safebot.db import DB
from safebot.pipeline.heuristics import SpamTracker, check_links, check_mentions
from safebot.pipeline.judge import _parse
from safebot.rag.store import KnowledgeStore


# ---------------- heuristics ----------------
@pytest.mark.parametrize("text", [
    "free nitro here https://discord-gift.ru/claim",
    "https://dlscord.com/gift/abc",
    "https://steamcommunlty.com/tradeoffer",
    "https://xn--dscord-6ve.com",
])
def test_scam_links_flagged(text):
    assert max(s.score for s in check_links(text)) >= 0.8


@pytest.mark.parametrize("text", [
    "check https://discord.com/channels/1/2",
    "https://cdn.discordapp.com/attachments/x.png",
    "https://github.com/Aaseka/SafeBot",
    "no links at all",
])
def test_legit_links_pass(text):
    assert all(s.score < 0.8 for s in check_links(text))


def test_spam_duplicate_and_flood():
    t = SpamTracker()
    assert t.check(1, 1, "buy now", now=0) is None
    assert t.check(1, 1, "buy now", now=1) is None
    assert t.check(1, 1, "buy now", now=2).kind == "spam"
    t2 = SpamTracker()
    results = [t2.check(1, 2, f"msg {i}", now=i * 0.5) for i in range(6)]
    assert results[-1] is not None and results[0] is None


def test_mass_mentions():
    assert check_mentions(6, False) is not None
    assert check_mentions(2, False) is None


# ---------------- judge parsing ----------------
def test_parse_verdict():
    v = _parse('Sure! {"violation": true, "severity": 2, "rule": "No slurs", "explanation": "Slur."}')
    assert v.violation and v.rule == "No slurs" and v.severity == 2
    assert _parse("not json") is None


# ---------------- db + RAG store (fake embedder) ----------------
class FakeStore(KnowledgeStore):
    """Bag-of-words embedder so tests don't download a model."""
    VOCAB: ClassVar[list[str]] = ["spam", "slur", "nsfw", "link", "advert", "language", "channel", "art", "game"]

    async def embed(self, texts):
        vecs = []
        for t in texts:
            v = np.array([t.lower().count(w) for w in self.VOCAB], dtype=np.float32) + 1e-3
            vecs.append(v / np.linalg.norm(v))
        return np.stack(vecs)


@pytest.mark.asyncio
async def test_db_and_store(tmp_path):
    db = DB(str(tmp_path / "t.db"))
    await db.open()
    cfg = await db.config(42)
    assert cfg["max_warnings"] == 3
    await db.set_config(42, strictness="strict", bogus="x")
    assert (await db.config(42))["strictness"] == "strict"

    fid = await db.add_flag(42, 7, 1, "hi", "toxicity", "insult", {"insult": 0.9})
    await db.add_warning(42, 7, fid, "insult")
    assert await db.count_warnings(42, 7) == 1
    await db.remove_warning_for_flag(fid)
    assert await db.count_warnings(42, 7) == 0
    assert json.loads((await db.get_flag(fid))["scores"])["insult"] == 0.9

    store = FakeStore(db, "fake")
    await store.add_many(42, "rule", ["No spam or advert links", "No slur or hateful language", "Keep nsfw art out"])
    hits = await store.search(42, "he posted an advert link", {"rule"}, k=1, min_score=0.0)
    assert hits[0].text.startswith("No spam")
    assert await store.search(99, "anything", {"rule"}) == []  # other servers are isolated
    await db.close()
