"""Stage 1: cheap, no-ML checks for scams, phishing links, spam and mass pings.

These run on every message in microseconds, so the ML model only sees what's left.
"""
from __future__ import annotations

import ipaddress
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from urllib.parse import urlparse

URL_RE = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)

SCAM_PHRASES = (
    "free nitro", "nitro for free", "discord nitro gift", "steam gift", "claim your gift",
    "airdrop", "i accidentally reported you", "gift for you", "crypto giveaway", "double your",
)
PROTECTED_DOMAINS = (
    "discord.com", "discord.gg", "discordapp.com", "discord.gift",
    "steamcommunity.com", "steampowered.com", "roblox.com", "epicgames.com", "paypal.com",
)
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "shorturl.at"}


@dataclass(frozen=True)
class Signal:
    kind: str     # scam | spam | mentions
    score: float  # 0..1 confidence
    reason: str


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _registered_domain(host: str) -> str:
    parts = host.lower().rstrip(".").split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def check_links(text: str) -> list[Signal]:
    signals: list[Signal] = []
    lowered = text.lower()
    urls = URL_RE.findall(text)
    if not urls:
        return signals

    if any(p in lowered for p in SCAM_PHRASES):
        signals.append(Signal("scam", 0.9, "link combined with a known scam phrase"))

    for url in urls:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            continue
        reg = _registered_domain(host)
        if any(label.startswith("xn--") for label in host.split(".")):
            signals.append(Signal("scam", 0.85, f"punycode lookalike domain `{host}`"))
        elif _is_ip(host):
            signals.append(Signal("scam", 0.6, f"link to a raw IP address `{host}`"))
        elif reg in SHORTENERS:
            signals.append(Signal("scam", 0.4, f"shortened link `{host}`"))

        if any(reg == real or host.endswith("." + real) for real in PROTECTED_DOMAINS):
            continue  # the genuine site
        for real in PROTECTED_DOMAINS:
            brand = real.split(".")[0]
            if _levenshtein(reg, real) <= 2 or (brand in reg and len(brand) >= 5):
                signals.append(Signal("scam", 0.95, f"`{host}` impersonates `{real}`"))
                break
    return signals


class SpamTracker:
    """Sliding-window flood and duplicate-message detection, per user per server."""

    def __init__(self, flood_count=6, flood_window=8.0, dup_count=3, dup_window=30.0):
        self.flood_count, self.flood_window = flood_count, flood_window
        self.dup_count, self.dup_window = dup_count, dup_window
        self._history: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=20))

    def check(self, guild_id: int, user_id: int, content: str, now: float | None = None) -> Signal | None:
        now = time.monotonic() if now is None else now
        hist = self._history[(guild_id, user_id)]
        norm = re.sub(r"\s+", " ", content.lower()).strip()
        hist.append((now, norm))

        recent = [t for t, _ in hist if now - t <= self.flood_window]
        if len(recent) >= self.flood_count:
            return Signal("spam", 0.85, f"{len(recent)} messages in {self.flood_window:.0f}s")
        if norm:
            dups = sum(1 for t, c in hist if c == norm and now - t <= self.dup_window)
            if dups >= self.dup_count:
                return Signal("spam", 0.85, f"same message repeated {dups} times")
        return None


SCAM_TEXT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"send\s+\d+(\.\d+)?\s*(btc|eth|sol|usdt|bnb)",
    r"(giveaway|airdrop|free).{0,40}\b(btc|eth|crypto|nitro|usdt)\b",
    r"double\s+your\s+(crypto|btc|eth|money|coins)",
    r"scan\s+(this|the)\s+qr",
    r"verify\s+(your\s+)?(account|discord|identity)\b.{0,60}(scan|qr|login)",
    r"(withdraw|claim)\s+your\s+(reward|bonus|winnings)",
)]


def check_scam_text(text: str) -> Signal | None:
    """Scam wording without needing a link, e.g. text read from a screenshot."""
    lowered = text.lower()
    if any(p in lowered for p in SCAM_PHRASES) or any(p.search(text) for p in SCAM_TEXT_PATTERNS):
        return Signal("scam", 0.85, "image contains scam wording")
    return None


def check_qr(payloads: list[str], surrounding_text: str = "") -> list[Signal]:
    signals: list[Signal] = []
    for payload in payloads:
        low = payload.lower()
        # Discord's own QR login: whoever scans it hands their account to the poster.
        if "discord.com/ra/" in low or "discordapp.com/ra/" in low:
            signals.append(Signal("scam", 0.97, "Discord login QR code (account takeover scam)"))
            continue
        link_signals = check_links(payload if "://" in payload else "")
        signals.extend(link_signals)
        if "://" in payload and not link_signals and check_scam_text(surrounding_text):
            signals.append(Signal("scam", 0.9, "QR code link with scam wording"))
    return signals


def check_mentions(mention_count: int, mentions_everyone: bool) -> Signal | None:
    if mention_count >= 5:
        return Signal("mentions", 0.9, f"mass mention ({mention_count} users)")
    return None
