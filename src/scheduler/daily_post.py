"""Publishes Mahsa's daily emotional diary post to her channel."""
from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telethon import TelegramClient

from ..brain import Brain
from ..utils.logging import get_logger

log = get_logger("mahsa.scheduler")


class DailyPoster:
    def __init__(
        self,
        client: TelegramClient,
        brain: Brain,
        channel: str,
        hour: int = 21,
        minute: int = 0,
    ):
        self.client = client
        self.brain = brain
        self.channel = channel
        self.hour = hour
        self.minute = minute
        self._scheduler = AsyncIOScheduler()

    async def post_now(self, force: bool = False) -> str | None:
        """Write and publish today's entry. Skips if already posted today (unless force)."""
        if not force and self.brain.memory.has_journal_today():
            log.info("Already journaled today; skipping.")
            return None
        try:
            journal_id, entry = await self.brain.make_journal_and_store()
        except FileNotFoundError as e:
            log.error("Cannot write daily post — model missing: %s", e)
            return None
        except Exception:  # noqa: BLE001
            log.exception("Failed to generate daily entry")
            return None

        try:
            await self.client.send_message(self.channel, entry)
            self.brain.memory.mark_journal_posted(journal_id)
            log.info("Daily post published to %s.", self.channel)
            return entry
        except Exception:  # noqa: BLE001
            log.exception("Failed to send daily post to channel %s", self.channel)
            return None

    def start(self) -> None:
        self._scheduler.add_job(
            self.post_now,
            "cron",
            hour=self.hour,
            minute=self.minute,
            id="daily_post",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        self._scheduler.start()
        log.info("Daily poster scheduled for %02d:%02d local time.", self.hour, self.minute)

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
