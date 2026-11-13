# SafeBot

A Discord moderation bot that understands **your server's rules**, not just bad words.
It combines fast heuristics, a local toxicity model, and retrieval-augmented LLM judgment,
and it learns from your moderators' decisions.

## How it works

```
message ─▶ 1. Heuristics ──────────▶ scam/phishing links, lookalike domains, spam floods, mass pings
              (microseconds)          └─ confident? remove + warn
         ─▶ 2. Toxicity model ─────▶ unitary/toxic-bert, 6 labels, runs locally off the event loop
              (local, ~20 ms CPU)     ├─ high score → remove, cite the closest server rule (RAG)
                                      ├─ mods marked something near-identical as fine → allow
                                      └─ borderline ↓
         ─▶ Images ─────────────────▶ QR codes (Discord login-QR takeover scams, phishing links),
              (attachments)           OCR → scam wording + toxicity on text inside images,
                                      NSFW model (skipped in channels marked NSFW),
                                      perceptual-hash memory of images mods removed or allowed
         ─▶ 3. RAG + LLM judge ────▶ retrieves this server's rules + past mod decisions on similar
              (borderline only)       messages + recent conversation; LLM decides and explains
```

- **Context-aware.** Trash talk in a gaming server and the same words in a support server get different verdicts. The judge sees the rules and the conversation.
- **Explains itself.** Users see *which rule* they broke, not just "flagged".
- **Learns from mods.** Each mod-log entry has *False positive / Timeout / Ban* buttons. Every click is stored as a labelled example, embedded, and retrieved for future decisions. No retraining needed.
- **Cheap.** About 95% of messages stop at stage 1 or 2. The LLM is only called on borderline cases, and it's optional.
- **Image moderation.** Catches the QR "verify your account" scam that steals Discord accounts, crypto-giveaway screenshots, hateful text in memes, and NSFW images outside NSFW channels. When a mod bans or dismisses an image, its perceptual hash is remembered, so reposts (resized or recompressed) are handled automatically.
- **Server FAQ bot.** `/ask` answers questions from the server's own rules and FAQ, with citations.

## Commands

| Command | Who | What |
|---|---|---|
| `/setup` | Manage Server | Mod-log channel, max warnings, auto-timeout length, strictness (relaxed/balanced/strict), image scanning on/off |
| `/rules add` · `import` · `list` · `remove` | Manage Server | Server rules (import a .txt, one rule per line) |
| `/faqs add` · `import` · `list` · `remove` | Manage Server | FAQ entries (import a .txt, entries separated by blank lines) |
| `/ask` | Everyone | Answer from the rules and FAQ |
| `/check` | Moderators | Dry-run: how would SafeBot judge this text or image? |
| `/warnings` · `/clearwarnings` | Moderators | View or reset a member's warnings (they decay after 30 days) |

Moderators (anyone with *Manage Messages*) are exempt from filtering.

## Run it

1. Create a bot at <https://discord.com/developers/applications>. Under **Bot**, enable the **Message Content** and **Server Members** intents. Invite it with the `bot` and `applications.commands` scopes and these permissions: Manage Messages, Moderate Members, Ban Members, Send Messages, Embed Links, Read Message History.
2. `cp .env.example .env` and fill in `DISCORD_TOKEN`. Optionally add an LLM (a Groq free tier key, or a local Ollama server).
3. Start it:

```bash
# Docker (recommended)
docker compose up -d --build

# or locally
sudo apt install tesseract-ocr        # OCR for images
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m safebot
```

4. In your server, run `/setup log_channel:#mod-log`, then `/rules import` with your rules file.

**Hosting:** the bot needs about 2 GB of RAM with image moderation on (three small models on CPU), or about 1.5 GB with `SAFEBOT_IMAGE_MODERATION=0`. That fits Oracle Cloud's free Ampere VM, a 2 GB VPS, Railway or Fly.io (2 GB machine), or a Raspberry Pi 5. Data lives in a single SQLite file in the `/data` volume.

## Development

```bash
pip install -r requirements-dev.txt
ruff check safebot tests && pytest -q
```

```
safebot/
  bot.py              Discord client, moderation flow, mod-log buttons, slash commands
  pipeline/
    heuristics.py     scam links, lookalike domains, spam and mention detection
    toxicity.py       local transformer classifier
    images.py         NSFW model, OCR, QR decoding, perceptual hashing
    judge.py          RAG + LLM context-aware verdicts
  rag/store.py        per-server embedding store (SQLite + numpy cosine search)
  llm.py              OpenAI-compatible client (Groq, OpenRouter, Ollama, ...)
  db.py               config, flags, warnings, knowledge base
```

## Roadmap

- Fine-tune a small classifier per server from the collected mod decisions
- Multilingual model option (e.g. `unitary/multilingual-toxic-xlm-roberta`)
- Raid detection on member joins (account age, join-rate spikes)
- Web dashboard for flags and stats
