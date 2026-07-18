"""Telethon client construction for the real user account."""
from __future__ import annotations

from pathlib import Path

from telethon import TelegramClient


def build_client(session: str, api_id: int, api_hash: str) -> TelegramClient:
    # Ensure the sessions directory exists so the .session file can be written.
    Path(session).parent.mkdir(parents=True, exist_ok=True)
    return TelegramClient(session, api_id, api_hash)
