# SafeBot

- **Context-aware.** Trash talk in a gaming server and the same words in a support server get different verdicts. The judge sees the rules and the conversation.
- **Explains itself.** Users see *which rule* they broke, not just "flagged".
- **Learns from mods.** Each mod-log entry has *False positive / Timeout / Ban* buttons. Every click is stored as a labelled example, embedded, and retrieved for future decisions. No retraining needed.
- **Cheap.** About 95% of messages stop at stage 1 or 2. The LLM is only called on borderline cases, and it's optional.
- **Image moderation.** Catches the QR "verify your account" scam that steals Discord accounts, crypto-giveaway screenshots, hateful text in memes, and NSFW images outside NSFW channels. When a mod bans or dismisses an image, its perceptual hash is remembered, so reposts (resized or recompressed) are handled automatically.
- **Server FAQ bot.** `/ask` answers questions from the server's own rules and FAQ, with citations.

