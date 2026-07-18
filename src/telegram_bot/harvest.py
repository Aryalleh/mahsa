"""Harvest short text samples from public channels to learn a texting *style*.

This stores snippets only to imitate tone/rhythm/word-choice — not to copy
content or impersonate any specific real person. Use public channels.
"""
from __future__ import annotations

import re

from telethon import TelegramClient

from ..memory import MemoryStore
from ..utils.logging import get_logger

log = get_logger("mahsa.harvest")

_URL = re.compile(r"https?://|t\.me/|www\.", re.IGNORECASE)


def _usable(text: str) -> bool:
    """Keep short, chatty, self-contained lines; drop links/ads/long posts."""
    if not text:
        return False
    text = text.strip()
    if not (3 <= len(text) <= 240):
        return False
    if _URL.search(text):
        return False
    if text.count("\n") > 2:
        return False
    # Skip obvious promo / forwarded-ad noise.
    if any(tok in text.lower() for tok in ("تبلیغ", "ادمین", "join", "عضو شوید", "لینک")):
        return False
    return True


async def harvest_channel(
    client: TelegramClient,
    channel: str,
    memory: MemoryStore,
    limit: int = 200,
) -> int:
    """Read up to `limit` recent messages from `channel`; store usable snippets.

    Returns the number of new samples stored.
    """
    stored = 0
    try:
        async for msg in client.iter_messages(channel, limit=limit):
            text = (getattr(msg, "message", None) or "").strip()
            if _usable(text) and memory.add_style_sample(text, source=str(channel)):
                stored += 1
    except Exception:  # noqa: BLE001
        log.exception("Failed harvesting %s", channel)
        raise
    log.info("Harvested %d new style samples from %s.", stored, channel)
    return stored
