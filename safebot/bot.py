"""SafeBot: Discord client, moderation flow, mod-log buttons and slash commands."""
from __future__ import annotations

import json
import logging
from datetime import timedelta

import discord
from discord import app_commands
from discord.ext import commands

from .config import NSFW_THRESHOLDS, THRESHOLDS, Settings
from .db import DB
from .llm import LLM
from .pipeline.heuristics import SpamTracker, check_links, check_mentions, check_qr, check_scam_text
from .pipeline.images import MAX_BYTES, ImageModerator, ImageReport, hamming
from .pipeline.judge import Judge
from .pipeline.toxicity import ToxicityModel
from .rag.store import KnowledgeStore

log = logging.getLogger("safebot")

LEARNED_ALLOW_SIMILARITY = 0.88  # a mod already marked a near-identical message as fine
MAX_IMPORT_ENTRIES = 200
HASH_MATCH_BITS = 6  # dHash distance treated as "the same image"
MAX_IMAGES_PER_MESSAGE = 4


# --------------------------------------------------------------------------- #
# Persistent mod-log buttons (survive restarts via custom_id templates)
# --------------------------------------------------------------------------- #
BUTTONS = {
    "fp": ("False positive", discord.ButtonStyle.secondary),
    "to": ("Timeout 1h", discord.ButtonStyle.primary),
    "ban": ("Ban", discord.ButtonStyle.danger),
}


class ModButton(discord.ui.DynamicItem[discord.ui.Button], template=r"sb:(?P<action>fp|to|ban):(?P<flag>\d+)"):
    def __init__(self, action: str, flag_id: int):
        label, style = BUTTONS[action]
        super().__init__(discord.ui.Button(label=label, style=style, custom_id=f"sb:{action}:{flag_id}"))
        self.action = action
        self.flag_id = flag_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["action"], int(match["flag"]))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        perms = interaction.permissions
        needed = perms.ban_members if self.action == "ban" else perms.moderate_members
        if not needed:
            await interaction.response.send_message("You don't have permission for that.", ephemeral=True)
        return bool(needed)

    async def callback(self, interaction: discord.Interaction):
        await interaction.client.handle_mod_action(interaction, self.action, self.flag_id)


def mod_view(flag_id: int) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    for action in BUTTONS:
        view.add_item(ModButton(action, flag_id))
    return view


# --------------------------------------------------------------------------- #
# Bot
# --------------------------------------------------------------------------- #
class SafeBot(commands.Bot):
    def __init__(self, settings: Settings):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(command_prefix=commands.when_mentioned, intents=intents)
        self.settings = settings
        self.db = DB(settings.db_path)
        self.store = KnowledgeStore(self.db, settings.embed_model)
        self.tox = ToxicityModel(settings.tox_model, settings.tox_concurrency)
        self.llm = LLM(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
        self.judge = Judge(self.llm, self.store)
        self.spam = SpamTracker()
        self.images = ImageModerator(settings.nsfw_model, settings.ocr) if settings.image_moderation else None

    async def setup_hook(self) -> None:
        await self.db.open()
        log.info("Loading models...")
        await self.tox.warmup()
        await self.store.warmup()
        if self.images:
            await self.images.warmup()
        self.add_dynamic_items(ModButton)
        register_commands(self)
        await self.tree.sync()
        log.info("Ready. LLM judge %s", "enabled" if self.llm.enabled else "disabled")

    async def close(self) -> None:
        await self.llm.close()
        await self.db.close()
        await super().close()

    # ---------------------------- moderation flow ---------------------------- #
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.guild is None or not isinstance(message.author, discord.Member):
            return
        if message.author.guild_permissions.manage_messages:  # mods/admins are exempt
            return
        try:
            await self.moderate(message)
        except Exception:
            log.exception("Moderation failed for message %s", message.id)

    async def on_message_edit(self, before: discord.Message, after: discord.Message) -> None:
        if before.content != after.content:
            await self.on_message(after)

    async def moderate(self, message: discord.Message) -> None:
        guild_id = message.guild.id
        content = message.content.strip()
        cfg = await self.db.config(guild_id)

        # Stage 1: heuristics (no ML)
        signals = check_links(content)
        for extra in (
            self.spam.check(guild_id, message.author.id, content),
            check_mentions(len(message.raw_mentions), message.mention_everyone),
        ):
            if extra:
                signals.append(extra)
        if signals:
            top = max(signals, key=lambda s: s.score)
            if top.score >= 0.8:
                await self.enforce(message, cfg, top.kind, top.reason, {}, rule=None)
                return

        # Images: NSFW model, OCR, QR codes, remembered mod decisions
        if (message.attachments and self.images and cfg["images_enabled"]
                and await self.moderate_images(message, cfg)):
            return

        # Stage 2: local toxicity model
        if len(content) < 2 or not any(c.isalpha() for c in content):
            return
        scores = await self.tox.score(content)
        label, peak = max(scores.items(), key=lambda kv: kv[1])
        act_at, review_at = THRESHOLDS.get(cfg["strictness"], THRESHOLDS["balanced"])
        if peak < review_at:
            return

        # Learned exceptions: mods already said a near-identical message was fine
        similar = await self.store.search(guild_id, content, {"example"}, k=1, min_score=LEARNED_ALLOW_SIMILARITY)
        if similar and similar[0].meta.get("label") == "allowed":
            return

        if peak >= act_at:
            rule = await self.top_rule(guild_id, content)
            await self.enforce(message, cfg, "toxicity", f"{label.replace('_', ' ')} ({peak:.0%})", scores, rule)
            return

        # Stage 3: borderline -> RAG + LLM judge with conversation context
        if not self.llm.enabled:
            return
        context = [
            f"{m.author.display_name}: {m.content[:300]}"
            async for m in message.channel.history(limit=6, before=message)
        ][::-1]
        verdict = await self.judge.judge(
            guild_id, message.author.display_name, content, context, scores, cfg["strictness"]
        )
        if verdict and verdict.violation:
            reason = verdict.explanation or f"{label.replace('_', ' ')} ({peak:.0%})"
            await self.enforce(message, cfg, "context", reason, scores, verdict.rule)

    async def moderate_images(self, message: discord.Message, cfg: dict) -> bool:
        attachments = [
            a for a in message.attachments
            if (a.content_type or "").startswith("image/") and a.size <= MAX_BYTES
        ][:MAX_IMAGES_PER_MESSAGE]
        if not attachments:
            return False
        guild_id = message.guild.id
        known = await self.db.image_hashes(guild_id)
        nsfw_channel = bool(getattr(message.channel, "is_nsfw", lambda: False)())

        for att in attachments:
            try:
                data = await att.read()
            except discord.HTTPException:
                continue
            report = await self.images.analyze(data, want_nsfw=not nsfw_channel)
            if report is None:
                continue
            match = next((k for k in known if hamming(k["hash"], report.dhash) <= HASH_MATCH_BITS), None)
            if match and match["label"] == "allowed":
                continue
            if match and match["label"] == "blocked":
                await self.enforce(message, cfg, "image", "this image was removed by moderators before", {},
                                   rule=None, image_hashes=[report.dhash])
                return True
            result = await self.assess_image(guild_id, report, cfg, nsfw_channel)
            if result:
                category, reason, scores, rule, details = result
                await self.enforce(message, cfg, category, reason, scores, rule,
                                   image_hashes=[report.dhash], details=details)
                return True
        return False

    async def assess_image(self, guild_id: int, report: ImageReport, cfg: dict, nsfw_channel: bool):
        """Returns (category, reason, scores, rule, details) for a violation, else None."""
        strictness = cfg["strictness"] if cfg["strictness"] in THRESHOLDS else "balanced"
        ocr = report.ocr_text
        details = f"Text in image: {ocr[:300]}" if ocr else None

        signals = check_qr(report.qr_payloads, ocr)
        if ocr:
            signals += check_links(ocr)
            scam = check_scam_text(ocr)
            if scam:
                signals.append(scam)
        if signals:
            top = max(signals, key=lambda s: s.score)
            if top.score >= 0.8:
                return "scam", top.reason, {}, None, details

        if report.nsfw is not None and not nsfw_channel and report.nsfw >= NSFW_THRESHOLDS[strictness]:
            rule = await self.top_rule(guild_id, "no NSFW, explicit or sexual images")
            return "nsfw", f"explicit image ({report.nsfw:.0%}) outside an NSFW channel", \
                {"nsfw": report.nsfw}, rule, None

        if len(ocr) >= 8:
            scores = await self.tox.score(ocr)
            label, peak = max(scores.items(), key=lambda kv: kv[1])
            if peak >= THRESHOLDS[strictness][0]:
                rule = await self.top_rule(guild_id, ocr)
                return "toxicity", f"text in image: {label.replace('_', ' ')} ({peak:.0%})", scores, rule, details
        return None

    async def top_rule(self, guild_id: int, content: str) -> str | None:
        hits = await self.store.search(guild_id, content, {"rule"}, k=1, min_score=0.2)
        return hits[0].text if hits else None

    async def enforce(self, message: discord.Message, cfg: dict, category: str, reason: str,
                      scores: dict, rule: str | None, image_hashes: list[str] | None = None,
                      details: str | None = None) -> None:
        guild, user = message.guild, message.author
        flag_id = await self.db.add_flag(guild.id, user.id, message.channel.id, message.content, category, reason,
                                         scores, image_hashes)
        await self.db.add_warning(guild.id, user.id, flag_id, reason)
        count = await self.db.count_warnings(guild.id, user.id)
        limit = cfg["max_warnings"]

        try:
            await message.delete()
        except (discord.Forbidden, discord.NotFound):
            pass

        notice = f"{user.mention}, your message was removed: {reason}."
        if rule:
            notice += f"\n> Rule: {rule[:300]}"
        notice += f"\nWarning {count}/{limit}"
        try:
            await message.channel.send(notice, delete_after=30,
                                       allowed_mentions=discord.AllowedMentions(users=[user]))
        except discord.HTTPException:
            pass

        escalated = False
        if count >= limit:
            try:
                await user.timeout(timedelta(minutes=cfg["timeout_minutes"]),
                                   reason=f"SafeBot: reached {limit} warnings")
                escalated = True
            except discord.HTTPException:
                log.warning("Could not time out %s in %s", user.id, guild.id)

        await self.send_mod_log(guild, cfg, user, message, category, reason, rule, scores, count, flag_id,
                                escalated, details)

    async def send_mod_log(self, guild, cfg, user, message, category, reason, rule, scores, count, flag_id,
                           escalated, details=None):
        channel = guild.get_channel(cfg["log_channel_id"]) if cfg["log_channel_id"] else None
        if not isinstance(channel, discord.TextChannel):
            return
        embed = discord.Embed(
            title=f"Flagged: {category}",
            description=f"**User:** {user.mention} (`{user.id}`)\n**Channel:** {message.channel.mention}\n"
                        f"**Reason:** {reason}",
            color=discord.Color.red() if escalated else discord.Color.orange(),
        )
        embed.add_field(name="Message", value=(message.content or "*(no text)*")[:1000], inline=False)
        images = sum(1 for a in message.attachments if (a.content_type or "").startswith("image/"))
        if images:
            embed.add_field(name="Images", value=f"{images} attached (not reposted here)", inline=False)
        if details:
            embed.add_field(name="Details", value=details[:1000], inline=False)
        if rule:
            embed.add_field(name="Matched rule", value=rule[:1000], inline=False)
        if scores:
            top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
            embed.add_field(name="Scores", value=", ".join(f"{k} {v:.2f}" for k, v in top), inline=False)
        footer = f"Warning {count}/{cfg['max_warnings']} - flag #{flag_id}"
        if escalated:
            footer += f" - auto timeout {cfg['timeout_minutes']}m"
        embed.set_footer(text=footer)
        try:
            await channel.send(embed=embed, view=mod_view(flag_id))
        except discord.HTTPException:
            log.warning("Could not post to mod log in %s", guild.id)

    async def handle_mod_action(self, interaction: discord.Interaction, action: str, flag_id: int) -> None:
        flag = await self.db.get_flag(flag_id)
        if not flag or flag["guild_id"] != interaction.guild_id:
            await interaction.response.send_message("That flag no longer exists.", ephemeral=True)
            return
        if flag["status"] != "open":
            await interaction.response.send_message(f"Already handled ({flag['status']}).", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        guild = interaction.guild
        mod = interaction.user
        hashes = json.loads(flag.get("image_hashes") or "[]")
        has_text = bool(flag["content"].strip())
        try:
            if action == "fp":
                await self.db.remove_warning_for_flag(flag_id)
                if has_text:
                    await self.store.add(guild.id, "example", flag["content"], {"label": "allowed", "by": mod.id})
                status, msg = "dismissed", "Marked as a false positive. Warning removed and SafeBot will learn from it."
            elif action == "to":
                member = guild.get_member(flag["user_id"]) or await guild.fetch_member(flag["user_id"])
                await member.timeout(timedelta(hours=1), reason=f"SafeBot flag #{flag_id} by {mod}")
                if has_text:
                    await self.store.add(guild.id, "example", flag["content"], {"label": "violation", "by": mod.id})
                status, msg = "timed_out", f"<@{flag['user_id']}> timed out for 1 hour."
            else:
                await guild.ban(discord.Object(id=flag["user_id"]), reason=f"SafeBot flag #{flag_id} by {mod}",
                                delete_message_seconds=0)
                if has_text:
                    await self.store.add(guild.id, "example", flag["content"], {"label": "violation", "by": mod.id})
                status, msg = "banned", f"<@{flag['user_id']}> was banned."
        except discord.NotFound:
            await interaction.followup.send("That user is no longer in the server.", ephemeral=True)
            return
        except discord.Forbidden:
            await interaction.followup.send("I'm missing permissions for that (check my role position).", ephemeral=True)
            return

        if hashes:
            await self.db.label_image_hashes(guild.id, hashes, "allowed" if action == "fp" else "blocked")
        await self.db.set_flag_status(flag_id, status)
        if interaction.message and interaction.message.embeds:
            embed = interaction.message.embeds[0]
            embed.add_field(name="Resolved", value=f"{status} by {mod.mention}", inline=False)
            embed.color = discord.Color.green() if status == "dismissed" else discord.Color.dark_red()
            await interaction.message.edit(embed=embed, view=None)
        await interaction.followup.send(msg, ephemeral=True)


# --------------------------------------------------------------------------- #
# Slash commands
# --------------------------------------------------------------------------- #
async def _read_attachment(file: discord.Attachment) -> str:
    if file.size > 200_000:
        raise ValueError("File is too large (max 200 KB).")
    return (await file.read()).decode("utf-8", errors="replace")


def _split_entries(text: str, by_blank_line: bool) -> list[str]:
    if by_blank_line:
        parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    else:
        parts = [line.strip(" -*\t") for line in text.splitlines()]
    return [p[:1000] for p in parts if p][:MAX_IMPORT_ENTRIES]


def register_commands(bot: SafeBot) -> None:
    tree = bot.tree
    admin = app_commands.default_permissions(manage_guild=True)
    mod = app_commands.default_permissions(moderate_members=True)

    @tree.command(name="setup", description="Configure SafeBot for this server")
    @admin
    @app_commands.guild_only()
    @app_commands.choices(strictness=[app_commands.Choice(name=s, value=s) for s in THRESHOLDS])
    async def setup(interaction: discord.Interaction, log_channel: discord.TextChannel | None = None,
                    max_warnings: app_commands.Range[int, 1, 20] | None = None,
                    timeout_minutes: app_commands.Range[int, 1, 40320] | None = None,
                    strictness: app_commands.Choice[str] | None = None,
                    image_scanning: bool | None = None):
        await bot.db.set_config(
            interaction.guild_id,
            log_channel_id=log_channel.id if log_channel else None,
            max_warnings=max_warnings,
            timeout_minutes=timeout_minutes,
            strictness=strictness.value if strictness else None,
            images_enabled=None if image_scanning is None else int(image_scanning),
        )
        cfg = await bot.db.config(interaction.guild_id)
        log_ch = f"<#{cfg['log_channel_id']}>" if cfg["log_channel_id"] else "not set"
        await interaction.response.send_message(
            f"**SafeBot settings**\nMod log: {log_ch}\nMax warnings (30 days): {cfg['max_warnings']}\n"
            f"Auto timeout: {cfg['timeout_minutes']} min\nStrictness: {cfg['strictness']}\n"
            f"Image scanning: {'on' if cfg['images_enabled'] and bot.images else 'off'}\n"
            f"LLM judge: {'on' if bot.llm.enabled else 'off'}",
            ephemeral=True,
        )

    # ---- knowledge base: rules & faq ----
    for kind, by_blank in (("rule", False), ("faq", True)):
        group = app_commands.Group(name=f"{kind}s", description=f"Manage this server's {kind}s",
                                   guild_only=True, default_permissions=discord.Permissions(manage_guild=True))

        def make(kind=kind, by_blank=by_blank, group=group):
            @group.command(name="add", description=f"Add one {kind}")
            async def add(interaction: discord.Interaction, text: app_commands.Range[str, 3, 1000]):
                await interaction.response.defer(ephemeral=True)
                await bot.store.add(interaction.guild_id, kind, text)
                await interaction.followup.send(f"Added {kind}.", ephemeral=True)

            @group.command(name="import", description=f"Import {kind}s from a .txt or .md file")
            async def import_(interaction: discord.Interaction, file: discord.Attachment):
                await interaction.response.defer(ephemeral=True)
                try:
                    entries = _split_entries(await _read_attachment(file), by_blank)
                except ValueError as e:
                    await interaction.followup.send(str(e), ephemeral=True)
                    return
                n = await bot.store.add_many(interaction.guild_id, kind, entries)
                hint = "one per line" if not by_blank else "separated by blank lines"
                await interaction.followup.send(f"Imported {n} {kind}s ({hint}).", ephemeral=True)

            @group.command(name="list", description=f"List {kind}s")
            async def list_(interaction: discord.Interaction):
                docs = await bot.db.docs(interaction.guild_id, {kind})
                if not docs:
                    await interaction.response.send_message(f"No {kind}s yet.", ephemeral=True)
                    return
                lines = [f"`{d['id']}` {d['text'][:150]}" for d in docs]
                text = "\n".join(lines)
                await interaction.response.send_message(text[:1900] + ("\n..." if len(text) > 1900 else ""),
                                                        ephemeral=True)

            @group.command(name="remove", description=f"Remove a {kind} by id")
            async def remove(interaction: discord.Interaction, id: int):
                ok = await bot.store.delete(interaction.guild_id, id)
                await interaction.response.send_message("Removed." if ok else "Not found.", ephemeral=True)

        make()
        tree.add_command(group)

    @tree.command(name="ask", description="Ask a question about this server (answers from its rules and FAQ)")
    @app_commands.guild_only()
    async def ask(interaction: discord.Interaction, question: app_commands.Range[str, 3, 300]):
        await interaction.response.defer(ephemeral=True)
        hits = await bot.store.search(interaction.guild_id, question, {"rule", "faq"}, k=4, min_score=0.3)
        if not hits:
            await interaction.followup.send("I couldn't find anything about that in this server's rules or FAQ.",
                                            ephemeral=True)
            return
        if not bot.llm.enabled:
            await interaction.followup.send("Most relevant entries:\n" + "\n".join(f"- {h.text[:400]}" for h in hits),
                                            ephemeral=True)
            return
        sources = "\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, 1))
        try:
            answer = await bot.llm.chat(
                "You answer questions about a Discord server using ONLY the numbered sources. "
                "Cite sources like [1]. If the sources don't answer it, say you don't know. Be brief.",
                f"Sources:\n{sources}\n\nQuestion: {question}",
                max_tokens=300,
            )
        except Exception:
            log.exception("ask failed")
            answer = "Most relevant entries:\n" + "\n".join(f"- {h.text[:400]}" for h in hits)
        await interaction.followup.send(answer[:1900], ephemeral=True)

    @tree.command(name="warnings", description="Show a member's recent warnings")
    @mod
    @app_commands.guild_only()
    async def warnings(interaction: discord.Interaction, member: discord.Member):
        rows = await bot.db.list_warnings(interaction.guild_id, member.id)
        count = await bot.db.count_warnings(interaction.guild_id, member.id)
        if not rows:
            await interaction.response.send_message(f"{member.mention} has no warnings.", ephemeral=True)
            return
        lines = [f"<t:{int(r['created_at'])}:R> {r['reason'][:120]}" for r in rows]
        await interaction.response.send_message(
            f"{member.mention}: {count} active (last 30 days)\n" + "\n".join(lines), ephemeral=True
        )

    @tree.command(name="clearwarnings", description="Clear all of a member's warnings")
    @mod
    @app_commands.guild_only()
    async def clearwarnings(interaction: discord.Interaction, member: discord.Member):
        await bot.db.clear_warnings(interaction.guild_id, member.id)
        await interaction.response.send_message(f"Cleared warnings for {member.mention}.", ephemeral=True)

    @tree.command(name="check", description="Test how SafeBot would judge text or an image (nothing is enforced)")
    @mod
    @app_commands.guild_only()
    async def check(interaction: discord.Interaction, text: app_commands.Range[str, 1, 1000] | None = None,
                    image: discord.Attachment | None = None):
        await interaction.response.defer(ephemeral=True)
        cfg = await bot.db.config(interaction.guild_id)
        if image is not None:
            if not bot.images:
                await interaction.followup.send("Image scanning is disabled on this bot.", ephemeral=True)
                return
            if not (image.content_type or "").startswith("image/") or image.size > MAX_BYTES:
                await interaction.followup.send("Attach an image under 8 MB.", ephemeral=True)
                return
            report = await bot.images.analyze(await image.read())
            if report is None:
                await interaction.followup.send("Couldn't read that image.", ephemeral=True)
                return
            result = await bot.assess_image(interaction.guild_id, report, cfg, nsfw_channel=False)
            lines = [f"**Decision:** {'remove: ' + result[1] if result else 'allow'}"]
            if report.nsfw is not None:
                lines.append(f"**NSFW score:** {report.nsfw:.2f}")
            if report.qr_payloads:
                lines.append("**QR codes:** " + ", ".join(f"`{p[:100]}`" for p in report.qr_payloads))
            if report.ocr_text:
                lines.append(f"**Text found:** {report.ocr_text[:400]}")
            await interaction.followup.send("\n".join(lines)[:1900], ephemeral=True)
            return
        if not text:
            await interaction.followup.send("Give me some text or an image to check.", ephemeral=True)
            return
        act_at, review_at = THRESHOLDS[cfg["strictness"]]
        sigs = check_links(text)
        scores = await bot.tox.score(text)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:4]
        peak = top[0][1]
        decision = "remove" if peak >= act_at else "LLM review" if peak >= review_at else "allow"
        rule = await bot.top_rule(interaction.guild_id, text)
        out = [f"**Decision:** {decision} (strictness {cfg['strictness']})",
               "**Scores:** " + ", ".join(f"{k} {v:.2f}" for k, v in top)]
        if sigs:
            out.append("**Link signals:** " + "; ".join(s.reason for s in sigs))
        if rule:
            out.append(f"**Closest rule:** {rule[:300]}")
        await interaction.followup.send("\n".join(out), ephemeral=True)
