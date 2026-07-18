"""Incoming-message handling: chat replies + admin commands.

Because Mahsa runs on a *real user account*, we are deliberately conservative:

* Only respond to private (one-on-one) messages, never groups/channels, so she
  never spams. (Adjust ``_should_reply`` if you want group behaviour.)
* Never reply to her own outgoing messages.
* Admin commands are only accepted from configured admin user ids.
"""
from __future__ import annotations

import asyncio

from telethon import TelegramClient, events

from ..brain import Brain
from ..utils.logging import get_logger

log = get_logger("mahsa.telegram")

HELP_TEXT = (
    "Admin commands:\n"
    "/teach <fact>   — teach Mahsa something about herself (persists)\n"
    "/facts          — list learned facts with ids\n"
    "/forget <id>    — delete a learned fact\n"
    "/learnstyle @ch [n] — learn a public channel's texting vibe (default 200 msgs)\n"
    "/stylecount     — how many style snippets she's learned\n"
    "/forgetstyle    — wipe all learned style snippets\n"
    "/pending        — people waiting for your approval\n"
    "/approve <uid>  — let Mahsa chat with this person (a friend)\n"
    "/block <uid>    — block this person\n"
    "/friends        — list approved friends\n"
    "/mood           — show her current mood\n"
    "/post           — write & publish today's diary post now\n"
    "/reset <uid>    — clear a user's conversation history\n"
    "/help           — this message"
)


def register_handlers(
    client: TelegramClient,
    brain: Brain,
    admin_ids: list[int],
    channel: str,
    post_daily,  # coroutine fn: async def(force: bool) -> str | None
    whitelist_enabled: bool = True,
) -> None:

    def is_admin(uid: int) -> bool:
        return uid in admin_ids

    @client.on(events.NewMessage(incoming=True))
    async def on_message(event: events.NewMessage.Event):
        # Only private chats, and never messages we sent ourselves.
        if not event.is_private or event.out:
            return

        sender = await event.get_sender()
        uid = event.sender_id
        text = (event.raw_text or "").strip()
        if not text:
            return

        # Never engage other bots — avoids pointless bot-to-bot loops.
        if getattr(sender, "bot", False):
            return

        # ---- admin commands ---------------------------------------------
        if text.startswith("/"):
            # Style + contact commands need the client (channels / messaging users).
            if is_admin(uid) and await _handle_style_command(client, event, text, brain):
                return
            if is_admin(uid) and await _handle_contact_command(client, event, text, brain):
                return
            if await _handle_command(event, text, uid, is_admin(uid), brain, post_daily):
                return
            # non-admin or unknown command falls through to normal chat
            if is_admin(uid):
                return

        display = getattr(sender, "first_name", None)

        # ---- whitelist gating -------------------------------------------
        # Admins are always allowed. Everyone else must be an approved contact.
        if whitelist_enabled and not is_admin(uid):
            status = brain.memory.get_contact_status(uid)
            if status == "blocked":
                return  # silently ignore
            if status != "approved":
                if status is None:
                    # brand-new person: register + ask the admins
                    brain.memory.upsert_contact(uid, display, "pending")
                    await _notify_admins_new_contact(client, admin_ids, uid, display, text)
                    await event.reply(
                        "سلام 🌙 من مهسام. الان یه‌کم سرم شلوغه، بذار ببینم و بهت جواب می‌دم."
                    )
                # pending (already asked) → stay quiet until an admin decides
                return

        log.info("Message from %s (%s): %s", display, uid, text[:80])

        try:
            async with client.action(event.chat_id, "typing"):
                reply = await brain.reply(uid, text, display=display)
        except FileNotFoundError as e:
            log.error("Model missing: %s", e)
            await event.reply("(Mahsa is offline — model weights not loaded.)")
            return
        except Exception:  # noqa: BLE001
            log.exception("Failed to generate reply")
            await event.reply("…sorry, my mind went blank for a second. say that again?")
            return

        await event.reply(reply)

        # Occasionally refresh the remembered note about this person.
        asyncio.create_task(_maybe_summarise(brain, uid))

    log.info("Handlers registered (admins=%s).", admin_ids)


async def _handle_style_command(client, event, text: str, brain: Brain) -> bool:
    """Handle the channel style-learning commands. Returns True if handled."""
    parts = text.split()
    cmd = parts[0].lower().lstrip("/")
    if cmd not in {"learnstyle", "stylecount", "forgetstyle"}:
        return False

    if cmd == "stylecount":
        await event.reply(f"I've picked up {brain.memory.count_style_samples()} style snippets.")
        return True

    if cmd == "forgetstyle":
        n = brain.memory.clear_style_samples()
        await event.reply(f"Cleared {n} style snippets.")
        return True

    # learnstyle @channel [limit]
    if len(parts) < 2:
        await event.reply("Usage: /learnstyle @channel [how_many]\n"
                          "Reads a PUBLIC channel and learns its casual texting vibe.")
        return True
    channel = parts[1]
    limit = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 200

    await event.reply(f"Reading {channel} … 🌙")
    try:
        from .harvest import harvest_channel
        stored = await harvest_channel(client, channel, brain.memory, limit=limit)
        total = brain.memory.count_style_samples()
        await event.reply(
            f"Learned {stored} new snippets from {channel}. "
            f"I now have {total} in my style memory. ✨"
        )
    except Exception:  # noqa: BLE001
        log.exception("learnstyle failed")
        await event.reply(
            "Couldn't read that channel — is it public and spelled right? "
            "(Private channels I'm not a member of won't work.)"
        )
    return True


async def _notify_admins_new_contact(client, admin_ids, uid, display, first_msg) -> None:
    """Ping every admin that a new person wants to talk, with approve/block hints."""
    preview = (first_msg or "")[:120]
    text = (
        "👋 یه نفر جدید به مهسا پیام داد:\n"
        f"• نام: {display or '؟'}\n"
        f"• آیدی: {uid}\n"
        f"• پیام اول: «{preview}»\n\n"
        f"دوستته؟ برای تأیید: /approve {uid}\n"
        f"برای بلاک: /block {uid}"
    )
    warmed = False
    for admin in admin_ids:
        try:
            await client.send_message(admin, text)
        except ValueError:
            # Telethon hasn't cached this admin's entity yet. Warm the dialog
            # cache once and retry; if that still fails, the admin has never
            # DMed this account, so we can't initiate a chat with them.
            try:
                if not warmed:
                    await client.get_dialogs()
                    warmed = True
                await client.send_message(admin, text)
            except Exception:  # noqa: BLE001
                log.warning(
                    "Could not reach admin %s. They must send Mahsa one direct "
                    "message so she can DM them back.", admin
                )
        except Exception:  # noqa: BLE001
            log.exception("Could not notify admin %s", admin)


async def _handle_contact_command(client, event, text: str, brain: Brain) -> bool:
    """Whitelist admin commands: /approve /block /pending /friends. Returns True if handled."""
    parts = text.split()
    cmd = parts[0].lower().lstrip("/")
    if cmd not in {"approve", "block", "pending", "friends"}:
        return False

    if cmd == "pending":
        rows = brain.memory.list_contacts("pending")
        if not rows:
            await event.reply("کسی توی صف تأیید نیست.")
        else:
            await event.reply("در انتظار تأیید:\n" +
                              "\n".join(f"• {d} — {u}  (/approve {u})" for u, d, _ in rows))
        return True

    if cmd == "friends":
        rows = brain.memory.list_contacts("approved")
        if not rows:
            await event.reply("هنوز دوستی تأیید نشده.")
        else:
            await event.reply("دوستای تأییدشده:\n" +
                              "\n".join(f"• {d} — {u}" for u, d, _ in rows))
        return True

    # approve / block need a user id
    if len(parts) < 2 or not parts[1].lstrip("-").isdigit():
        await event.reply(f"Usage: /{cmd} <user id>")
        return True
    target = int(parts[1])

    if cmd == "approve":
        brain.memory.upsert_contact(target, None, "approved")
        await event.reply(f"✅ {target} تأیید شد. حالا مهسا باهاش راحت چت می‌کنه.")
        try:
            await client.send_message(target, "سلام دوباره 🌸 ببخشید معطل شدی، بگو چه خبر؟")
        except Exception:  # noqa: BLE001
            log.exception("Could not greet approved user %s", target)
    else:  # block
        brain.memory.upsert_contact(target, None, "blocked")
        await event.reply(f"🚫 {target} بلاک شد.")
    return True


async def _maybe_summarise(brain: Brain, uid: int) -> None:
    try:
        # Refresh roughly every time history crosses a small threshold.
        turns = brain.memory.recent_turns(uid, brain.max_turns)
        if len(turns) % 6 == 0:
            await brain.summarise_user(uid)
    except Exception:  # noqa: BLE001
        log.exception("Failed to summarise user %s", uid)


async def _handle_command(
    event, text: str, uid: int, admin: bool, brain: Brain, post_daily
) -> bool:
    """Returns True if the message was a recognised command (handled)."""
    parts = text.split(maxsplit=1)
    cmd = parts[0].lower().lstrip("/")
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd not in {"teach", "facts", "forget", "mood", "post", "reset", "help", "start"}:
        return False

    if not admin:
        # Silently ignore commands from non-admins (they get normal chat instead).
        return False

    if cmd in ("help", "start"):
        await event.reply(HELP_TEXT)
    elif cmd == "teach":
        if not arg:
            await event.reply("Usage: /teach <something true about Mahsa>")
        else:
            ok = brain.memory.add_fact(arg, source=str(uid))
            await event.reply("Got it, I'll remember that. 🌙" if ok else "I already knew that.")
    elif cmd == "facts":
        facts = brain.memory.list_facts_with_ids()
        if not facts:
            await event.reply("I haven't learned any facts yet.")
        else:
            await event.reply("\n".join(f"[{i}] {c}" for i, c in facts))
    elif cmd == "forget":
        if not arg.isdigit():
            await event.reply("Usage: /forget <fact id>")
        else:
            ok = brain.memory.delete_fact(int(arg))
            await event.reply("Forgotten." if ok else "No fact with that id.")
    elif cmd == "mood":
        mood = brain.memory.latest_mood()
        await event.reply(f"Right now: {mood}" if mood else "I haven't journaled a mood yet.")
    elif cmd == "post":
        await event.reply("Writing today's post…")
        result = await post_daily(True)
        await event.reply("Posted to the channel. ✨" if result else "Couldn't post — check logs.")
    elif cmd == "reset":
        if not arg.isdigit():
            await event.reply("Usage: /reset <user id>")
        else:
            brain.memory.clear_user(int(arg))
            await event.reply("Cleared that conversation.")
    return True
