"""Mahsa — entry point.

Boots the local model, the persona, memory, the Telegram user client, message
handlers and the daily-post scheduler, then runs until interrupted.

First run performs the interactive Telegram login (it will ask for the code
Telegram sends you, and your 2FA password if enabled). After that the
.session file keeps you logged in.
"""
from __future__ import annotations

import asyncio
import logging

from config import load_config
from src.brain import Brain
from src.llm import LlamaEngine
from src.memory import MemoryStore
from src.persona import Persona
from src.scheduler import DailyPoster
from src.telegram_bot import build_client, register_handlers
from src.utils.logging import get_logger, setup_logging

log = get_logger("mahsa")


async def run() -> None:
    setup_logging(logging.INFO)
    cfg = load_config()

    # ---- persona + memory ------------------------------------------------
    persona = Persona.load(cfg.persona_file)
    memory = MemoryStore(cfg.database_path)

    # Seed an initial mood on very first run so replies/posts have context.
    if memory.latest_mood() is None and persona.initial_mood():
        memory.add_journal(entry=persona.initial_mood(), mood=persona.initial_mood())
        log.info("Seeded initial mood.")

    # ---- local model -----------------------------------------------------
    engine = LlamaEngine(
        model_path=cfg.llm.model_path,
        context=cfg.llm.context,
        gpu_layers=cfg.llm.gpu_layers,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
    )
    try:
        await asyncio.to_thread(engine.load)
    except FileNotFoundError as e:
        log.warning("%s", e)
        log.warning("Starting anyway; Mahsa will apologise until the model is present.")

    brain = Brain(engine, persona, memory, max_turns=cfg.memory_max_turns)

    # ---- telegram --------------------------------------------------------
    client = build_client(cfg.telegram.session, cfg.telegram.api_id, cfg.telegram.api_hash)

    poster = DailyPoster(
        client=client,
        brain=brain,
        channel=cfg.telegram.channel,
        hour=cfg.schedule.hour,
        minute=cfg.schedule.minute,
    )

    register_handlers(
        client=client,
        brain=brain,
        admin_ids=cfg.telegram.admin_user_ids,
        channel=cfg.telegram.channel,
        post_daily=poster.post_now,
        whitelist_enabled=cfg.whitelist_enabled,
    )

    log.info("Connecting to Telegram as %s ...", cfg.telegram.phone)
    await client.start(phone=cfg.telegram.phone)
    me = await client.get_me()
    log.info("Logged in as %s (id=%s).", getattr(me, "first_name", "?"), me.id)

    if cfg.schedule.enabled:
        poster.start()
    else:
        log.info("Daily posting disabled by config.")

    log.info("Mahsa is live. Press Ctrl+C to stop.")
    try:
        await client.run_until_disconnected()
    finally:
        poster.shutdown()
        memory.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Shutting down.")


if __name__ == "__main__":
    main()
