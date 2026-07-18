"""Central configuration, loaded from environment / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _get(name: str, default: str | None = None) -> str:
    val = os.getenv(name, default)
    if val is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _get_id_list(name: str) -> list[int]:
    raw = os.getenv(name, "") or ""
    return [int(x) for x in raw.replace(" ", "").split(",") if x]


@dataclass
class TelegramConfig:
    api_id: int
    api_hash: str
    phone: str
    session: str
    channel: str
    admin_user_ids: list[int]


@dataclass
class LLMConfig:
    model_path: str
    context: int
    gpu_layers: int
    temperature: float
    max_tokens: int


@dataclass
class ScheduleConfig:
    hour: int
    minute: int
    enabled: bool


@dataclass
class WatchConfig:
    enabled: bool          # follow joined channels at all
    react: bool            # add an emoji reaction to new posts
    comment: bool          # comment her "vibe" when the post has comments open
    chance: float          # probability of acting on any given post (0..1)
    min_len: int           # skip posts shorter than this many chars
    max_delay: int         # random 0..max_delay second wait before acting (human-like)


@dataclass
class Config:
    telegram: TelegramConfig
    llm: LLMConfig
    schedule: ScheduleConfig
    persona_file: str
    database_path: str
    memory_max_turns: int
    schedule_watch: "WatchConfig | None" = None
    whitelist_enabled: bool = True
    admin_user_ids: list[int] = field(default_factory=list)


def load_config() -> Config:
    admins = _get_id_list("ADMIN_USER_IDS")
    return Config(
        telegram=TelegramConfig(
            api_id=_get_int("TELEGRAM_API_ID", 0),
            api_hash=_get("TELEGRAM_API_HASH", ""),
            phone=_get("TELEGRAM_PHONE", ""),
            session=_get("TELEGRAM_SESSION", "sessions/mahsa"),
            channel=_get("TELEGRAM_CHANNEL", ""),
            admin_user_ids=admins,
        ),
        llm=LLMConfig(
            model_path=_get("LLM_MODEL_PATH", "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
            context=_get_int("LLM_CONTEXT", 8192),
            gpu_layers=_get_int("LLM_GPU_LAYERS", 0),
            temperature=_get_float("LLM_TEMPERATURE", 0.75),
            max_tokens=_get_int("LLM_MAX_TOKENS", 160),
        ),
        schedule=ScheduleConfig(
            hour=_get_int("DAILY_POST_HOUR", 21),
            minute=_get_int("DAILY_POST_MINUTE", 0),
            enabled=_get_bool("DAILY_POST_ENABLED", True),
        ),
        persona_file=_get("PERSONA_FILE", "src/persona/mahsa.yaml"),
        database_path=_get("DATABASE_PATH", "data/mahsa.db"),
        memory_max_turns=_get_int("MEMORY_MAX_TURNS", 20),
        schedule_watch=WatchConfig(
            enabled=_get_bool("CHANNEL_WATCH_ENABLED", False),
            react=_get_bool("CHANNEL_REACT", True),
            comment=_get_bool("CHANNEL_COMMENT", True),
            chance=_get_float("CHANNEL_ACTION_CHANCE", 0.6),
            min_len=_get_int("CHANNEL_MIN_LEN", 15),
            max_delay=_get_int("CHANNEL_MAX_DELAY", 45),
        ),
        whitelist_enabled=_get_bool("WHITELIST_ENABLED", True),
        admin_user_ids=admins,
    )
