"""Incoming-message handling: chat replies + admin commands.

Because Mahsa runs on a *real user account*, we are deliberately conservative:

* Only respond to private (one-on-one) messages, never groups/channels, so she
  never spams. (Adjust ``_should_reply`` if you want group behaviour.)
* Never reply to her own outgoing messages.
* Admin commands are only accepted from configured admin user ids.
"""
from __future__ import annotations

import asyncio
import random

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
    "/name <uid> <name> — give a friend a name Mahsa uses\n"
    "/tell <name|@user|id> <msg> — have Mahsa pass a message to a friend\n"
    "/lover <uid|group_id> — intimate mode for a consenting adult, or a private group\n"
    "/unlover <uid>  — remove lover status\n"
    "/lovers         — list lovers\n"
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
    watch=None,             # config.WatchConfig | None
    own_channel: str = "",  # Mahsa's own channel, skipped when following others
    group_chat_enabled: bool = True,
) -> None:

    def is_admin(uid: int) -> bool:
        return uid in admin_ids

    _me_cache: dict[str, int] = {}

    async def my_id() -> int:
        if "id" not in _me_cache:
            _me_cache["id"] = (await client.get_me()).id
        return _me_cache["id"]

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

        # Ignore *unapproved* bots (avoids bot-to-bot spam/loops), but allow bots
        # the admin has explicitly approved (e.g. via /approve <bot_id>).
        if getattr(sender, "bot", False):
            if not is_admin(uid) and brain.memory.get_contact_status(uid) != "approved":
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
                        f"سلام 🌙 من {brain.persona.name_fa}‌ام. الان یه‌کم سرم شلوغه، "
                        "بذار ببینم و بهت جواب می‌دم."
                    )
                # pending (already asked) → stay quiet until an admin decides
                return

        log.info("Message from %s (%s): %s", display, uid, text[:80])

        # Approved service/menu bots: if the message has inline ("glass") buttons,
        # let Mahsa pick and press one instead of sending a persona chat reply.
        if getattr(sender, "bot", False):
            if await _handle_bot_buttons(event, brain):
                return

        if is_admin(uid):
            rel = "admin"
        elif brain.memory.is_lover(uid):
            rel = "lover"       # intimate chat, but no admin powers
        else:
            rel = "friend"
        try:
            async with client.action(event.chat_id, "typing"):
                result = await brain.plan(uid, text, sender_name=display, relationship=rel)
        except FileNotFoundError as e:
            log.error("Model missing: %s", e)
            await event.reply("(Mahsa is offline — model weights not loaded.)")
            return
        except Exception:  # noqa: BLE001
            log.exception("Failed to generate reply")
            await event.reply("…sorry, my mind went blank for a second. say that again?")
            return

        if result["kind"] == "relay":
            to_name = result["to_name"]
            try:
                await client.send_message(result["to_id"], result["text"])
                brain.memory.add_message(result["to_id"], "assistant", result["text"])
                brain.memory.add_message(uid, "user", result["user_text"])
                ack = f"چشم، به {to_name} رسوندم 🌸"
                brain.memory.add_message(uid, "assistant", ack)
                log.info("Relayed a message to %s.", to_name)
                await event.reply(ack)
            except Exception:  # noqa: BLE001
                log.warning("Relay to %s failed", to_name)
                await event.reply(
                    f"خواستم به {to_name} برسونم ولی نشد — تا وقتی اون یه بار به من "
                    "پیام نده، نمی‌تونم بهش پیام بدم."
                )
        else:
            await event.reply(result["text"])

        # Occasionally refresh the remembered note about this person.
        asyncio.create_task(_maybe_summarise(brain, uid))

    # ---- follow joined channels: vibe-comment + react -------------------
    if watch and getattr(watch, "enabled", False):

        @client.on(events.NewMessage(func=lambda e: e.is_channel and not e.is_group))
        async def on_channel_post(event: events.NewMessage.Event):
            if event.out:
                return  # her own post
            uname = getattr(event.chat, "username", None)
            if uname and own_channel and ("@" + uname).lower() == own_channel.lower():
                return  # skip her own channel
            text = (event.raw_text or "").strip()

            # Wait a random moment so it isn't an instant, robotic reaction.
            await asyncio.sleep(random.uniform(2, max(2, watch.max_delay)))

            # Pick a mood-matched reaction (and a possible comment). For posts
            # with little/no text we can't judge a mood, so use a neutral default.
            comment, reaction = "", "❤️"
            if len(text) >= 10:
                try:
                    comment, reaction = await brain.channel_vibe(text)
                except FileNotFoundError:
                    return  # model not loaded
                except Exception:  # noqa: BLE001
                    log.exception("Failed to read channel vibe")

            # React to EVERY post, and make sure it actually lands: if the
            # chosen emoji isn't allowed by the channel, fall back to common ones.
            if watch.react:
                await _react_with_fallback(client, event.chat_id, event.message.id, reaction)

            # Commenting is spammier/riskier, so keep it probabilistic.
            if watch.comment and comment and random.random() <= watch.chance:
                try:
                    await client.send_message(
                        event.chat_id, comment, comment_to=event.message.id
                    )
                    log.info("Vibe-commented on %s.", uname or event.chat_id)
                except Exception:  # noqa: BLE001
                    log.debug("Comments not open on %s; skipped comment.", uname or event.chat_id)

        log.info("Channel-watching on (react=%s, comment=%s, chance=%.2f).",
                 watch.react, watch.comment, watch.chance)

    # ---- group chat: reply when replied-to, mentioned, or named ---------
    if group_chat_enabled or (watch and getattr(watch, "enabled", False)):

        persona_name = brain.persona.name
        persona_name_fa = brain.persona.name_fa

        @client.on(events.NewMessage(func=lambda e: e.is_group))
        async def on_group_message(event: events.NewMessage.Event):
            if event.out:
                return
            sender = await event.get_sender()
            if getattr(sender, "bot", False):
                return
            text = (event.raw_text or "").strip()
            if not text:
                return

            # A group marked as a lover-group is a private intimate space: she
            # replies to everything there. Elsewhere, only when clearly addressed.
            lover_group = brain.memory.is_lover(event.chat_id)
            is_reply_to_me = False
            if event.is_reply:
                replied = await event.get_reply_message()
                is_reply_to_me = bool(replied and replied.sender_id == await my_id())
            mentioned = bool(getattr(event.message, "mentioned", False))
            named = persona_name.lower() in text.lower() or persona_name_fa in text
            if not (lover_group or is_reply_to_me or mentioned or named):
                return

            status = brain.memory.get_contact_status(event.sender_id)
            trusted = (is_admin(event.sender_id)
                       or brain.memory.is_lover(event.sender_id)
                       or status == "approved")
            if lover_group:
                relationship = "lover" if trusted else "public"
            else:
                relationship = "friend" if trusted else "public"

            display = getattr(sender, "first_name", None)
            log.info("Group message addressing Mahsa from %s (%s): %s",
                     display, event.sender_id, text[:80])
            await asyncio.sleep(random.uniform(2, 12))  # human-like pause
            try:
                async with client.action(event.chat_id, "typing"):
                    reply = await brain.reply(
                        event.sender_id, text, display=display, relationship=relationship
                    )
            except FileNotFoundError:
                return
            except Exception:  # noqa: BLE001
                log.exception("Failed to reply in group")
                return
            await event.reply(reply)

        log.info("Group chat on (name=%s).", persona_name)

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


async def _handle_bot_buttons(event, brain: Brain) -> bool:
    """For an approved menu bot: let Mahsa choose and click a button.

    Returns True if the message had buttons (so we should NOT also send a chat
    reply), False if there were none (fall through to a normal text reply).
    """
    rows = getattr(event.message, "buttons", None) or []
    flat = []  # (row, col, label)
    for i, row in enumerate(rows):
        for j, btn in enumerate(row):
            flat.append((i, j, getattr(btn, "text", "") or ""))
    if not flat:
        return False

    labels = [lbl for _, _, lbl in flat]
    try:
        idx = await brain.choose_button(event.raw_text or "", labels)
    except Exception:  # noqa: BLE001
        log.exception("Failed to choose a button")
        return True  # had buttons; just don't chat at it

    if idx is None:
        log.info("Mahsa chose to press no button (buttons: %s).", labels)
        return True

    i, j, label = flat[idx]
    try:
        await event.message.click(i, j)
        log.info("Mahsa pressed button '%s'.", label)
    except Exception:  # noqa: BLE001
        log.exception("Failed to press button '%s'", label)
    return True


async def _react_with_fallback(client, entity, msg_id, reaction: str) -> bool:
    """React with `reaction`; if the channel doesn't allow it, try common ones.

    Returns True if any reaction landed. Ensures a post gets a reaction even
    when the model's mood-picked emoji isn't in that channel's allowed set.
    """
    tried = []
    for emo in [reaction, "❤️", "👍", "🔥", "🙏"]:
        if not emo or emo in tried:
            continue
        tried.append(emo)
        try:
            await client.send_reaction(entity, msg_id, emo)
            log.info("Reacted %s to a post in %s.", emo, entity)
            return True
        except Exception:  # noqa: BLE001
            continue
    log.debug("Could not react to %s (channel may disable reactions).", entity)
    return False


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
    if cmd not in {"approve", "block", "pending", "friends", "name", "tell",
                   "lover", "unlover", "lovers"}:
        return False

    if cmd == "lovers":
        rows = brain.memory.list_lovers()
        if not rows:
            await event.reply("هنوز کسی به‌عنوان lover ثبت نشده.")
        else:
            await event.reply("Lovers (حالت صمیمی):\n" +
                              "\n".join(f"• {d} — {u}" for u, d in rows))
        return True

    if cmd in ("lover", "unlover"):
        if len(parts) < 2 or not parts[1].lstrip("-").isdigit():
            await event.reply(f"Usage: /{cmd} <user id>")
            return True
        target = int(parts[1])
        if cmd == "lover":
            brain.memory.add_lover(target)
            if target > 0:
                # A real user → also approve them so Mahsa chats with them.
                brain.memory.upsert_contact(target, None, "approved")
                await event.reply(
                    f"❤️ {target} به‌عنوان lover ثبت شد. مهسا باهاش صمیمیه — ولی دستور "
                    "ادمین یا خوندن چت بقیه رو نداره."
                )
            else:
                # Negative id → a group/channel becomes an intimate space.
                await event.reply(
                    f"❤️ گروهِ {target} به‌عنوان فضای صمیمی ثبت شد. مهسا اونجا حالتِ "
                    "lover داره و به همه‌ی پیام‌ها جواب می‌ده."
                )
        else:
            ok = brain.memory.remove_lover(target)
            await event.reply("برداشته شد." if ok else "این آیدی lover نبود.")
        return True

    if cmd == "tell":
        # /tell <name | @username | id> <message>  → Mahsa DMs that friend
        bits = text.split(maxsplit=2)
        if len(bits) < 3:
            await event.reply("Usage: /tell <name | @username | id> <message>")
            return True
        target_raw, message = bits[1].strip(), bits[2].strip()

        # Resolve the recipient.
        recipient = None
        if target_raw.lstrip("-").isdigit():
            recipient = int(target_raw)
        elif target_raw.startswith("@"):
            recipient = target_raw
        else:
            matches = brain.memory.find_contacts_by_name(target_raw)
            if len(matches) == 1:
                recipient = matches[0][0]
            elif len(matches) > 1:
                await event.reply("چند نفر با این اسم دارم:\n" +
                                  "\n".join(f"• {d} — {u}" for u, d in matches) +
                                  "\nبا آیدی بگو: /tell <id> <پیام>")
                return True
            else:
                await event.reply(f"«{target_raw}» رو توی دوستام پیدا نکردم. /friends رو ببین.")
                return True

        to_name = target_raw.lstrip("@") if not target_raw.lstrip("-").isdigit() else "دوستت"
        relay = await brain.compose_relay(to_name, message)
        try:
            await client.send_message(recipient, relay)
            if isinstance(recipient, int):
                brain.memory.add_message(recipient, "assistant", relay)
            await event.reply("رسوندم بهش ✅")
        except Exception:  # noqa: BLE001
            log.warning("Relay to %s failed", recipient)
            await event.reply("نشد بهش پیام بدم — تا وقتی اون یه بار به من پیام نده، نمی‌تونم بهش پیام بدم.")
        return True

    if cmd == "name":
        # /name <uid> <name...>  — give a friend a name Mahsa will use
        bits = text.split(maxsplit=2)
        if len(bits) < 3 or not bits[1].lstrip("-").isdigit():
            await event.reply("Usage: /name <user id> <name>")
        else:
            # Upsert as an approved contact so naming also *registers* the friend
            # (so relay/recall can find them), not just renames an existing row.
            brain.memory.upsert_contact(int(bits[1]), bits[2].strip(), "approved")
            await event.reply(
                f"باشه، «{bits[2].strip()}» رو به‌عنوان دوست ثبت کردم و از این به بعد "
                "همین صداش می‌زنم."
            )
        return True

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
        await event.reply(f"✅ {target} تأیید شد. حالا {brain.persona.name_fa} باهاش راحت چت می‌کنه.")
        # Only greet real users. Negative ids are channels/groups (can't be DMed).
        if target > 0:
            try:
                await client.send_message(target, "سلام دوباره 🌸 ببخشید معطل شدی، بگو چه خبر؟")
            except Exception:  # noqa: BLE001
                log.warning("Approved %s but couldn't send greeting.", target)
                await event.reply(
                    "تأیید شد، ولی نتونستم بهش سلام بدم (شاید هنوز به من پیام نداده)."
                )
    else:  # block
        brain.memory.upsert_contact(target, None, "blocked")
        await event.reply(f"🚫 {target} بلاک شد.")
    return True


async def _maybe_summarise(brain: Brain, uid: int) -> None:
    try:
        # Refresh the per-user note only occasionally: it's a whole extra model
        # generation that competes for the engine, so keep it rare to stay fast.
        turns = brain.memory.recent_turns(uid, brain.max_turns)
        if len(turns) >= 6 and len(turns) % 20 == 0:
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
