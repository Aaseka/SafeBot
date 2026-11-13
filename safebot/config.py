"""Settings loaded from environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass

# (act-immediately threshold, send-to-judge threshold) per strictness level
THRESHOLDS = {
    "relaxed": (0.92, 0.75),
    "balanced": (0.85, 0.55),
    "strict": (0.70, 0.40),
}


# NSFW probability at which an image in a non-NSFW channel is removed
NSFW_THRESHOLDS = {"relaxed": 0.95, "balanced": 0.85, "strict": 0.70}


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    token: str
    db_path: str
    tox_model: str
    embed_model: str
    tox_concurrency: int
    image_moderation: bool
    nsfw_model: str
    ocr: bool
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str


def load() -> Settings:
    token = os.getenv("DISCORD_TOKEN", "")
    if not token:
        raise SystemExit("DISCORD_TOKEN is not set (see .env.example)")
    return Settings(
        token=token,
        db_path=os.getenv("SAFEBOT_DB", "safebot.db"),
        tox_model=os.getenv("SAFEBOT_TOX_MODEL", "unitary/toxic-bert"),
        embed_model=os.getenv("SAFEBOT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        tox_concurrency=int(os.getenv("SAFEBOT_TOX_CONCURRENCY", "2")),
        image_moderation=_flag("SAFEBOT_IMAGE_MODERATION"),
        nsfw_model=os.getenv("SAFEBOT_NSFW_MODEL", "Falconsai/nsfw_image_detection"),
        ocr=_flag("SAFEBOT_OCR"),
        llm_base_url=os.getenv("LLM_BASE_URL") or None,
        llm_api_key=os.getenv("LLM_API_KEY") or None,
        llm_model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
    )
