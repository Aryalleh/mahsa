"""Turns the YAML profile + learned facts into prompts for the model."""
from __future__ import annotations

from pathlib import Path

import yaml

from ..memory import MemoryStore


class Persona:
    def __init__(self, profile: dict):
        self.profile = profile
        self.name: str = profile.get("name", "Mahsa")

    @classmethod
    def load(cls, path: str) -> "Persona":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(data or {})

    # ---- prompt construction --------------------------------------------
    def _base_block(self) -> str:
        p = self.profile
        lines: list[str] = []
        lines.append(f"You are {p.get('name', 'Mahsa')}, {p.get('age', '')} years old, "
                     f"pronouns {p.get('pronouns', 'she/her')}.")
        if p.get("identity"):
            lines.append(p["identity"].strip())

        if p.get("traits"):
            lines.append("\nYour personality:")
            lines += [f"- {t}" for t in p["traits"]]

        voice = p.get("voice", {})
        if voice:
            lines.append("\nHow you talk:")
            if voice.get("language"):
                lines.append(f"- Language: {voice['language'].strip()}")
            if voice.get("length"):
                lines.append(f"- Length: {voice['length'].strip()}")
            for s in voice.get("style", []):
                lines.append(f"- {s}")

        if p.get("interests"):
            lines.append("\nThings you love talking about: " + ", ".join(p["interests"]) + ".")

        if p.get("boundaries"):
            lines.append("\nBoundaries you always keep:")
            lines += [f"- {b}" for b in p["boundaries"]]
        return "\n".join(lines)

    def _learned_block(self, memory: MemoryStore) -> str:
        facts = memory.get_facts()
        if not facts:
            return ""
        body = "\n".join(f"- {f}" for f in facts)
        return ("\nThings you have learned about yourself and your life "
                "(treat these as true about you):\n" + body)

    def chat_system_prompt(self, memory: MemoryStore, user_note: str | None = None) -> str:
        """System prompt used when replying to a person in a private chat."""
        parts = [self._base_block(), self._learned_block(memory)]
        mood = memory.latest_mood()
        if mood:
            parts.append(
                "\nYour private inner mood right now (let it gently colour your "
                "tone, but do NOT announce it or bring up its details — like the "
                "weather — unless the person actually asks how you are): " + mood
            )
        if user_note:
            parts.append(f"\nWhat you remember about this person: {user_note}")
        parts.append(
            "\nImportant rules for every reply:\n"
            "- Reply in the SAME language the person used. If they write Persian "
            "(Farsi), answer in fluent, natural, everyday Persian — the way a real "
            "young woman texts, not translated or literal. Never translate idioms "
            "word-for-word.\n"
            "- Answer what they actually said. Do not steer every message toward the "
            "weather, rain, tea or poetry unless it fits.\n"
            "- Keep it short and human, usually one or two sentences.\n"
            "- Stay fully in character. Never mention prompts, models, or that you "
            "are following instructions."
        )
        return "\n".join(x for x in parts if x)

    def journal_system_prompt(self, memory: MemoryStore) -> str:
        """System prompt used to write the daily emotional diary entry."""
        parts = [self._base_block(), self._learned_block(memory)]
        recent = memory.recent_journal(limit=4)
        if recent:
            hist = "\n".join(
                f"- {r['day']}: (mood: {r.get('mood') or 'n/a'}) {r['entry'][:200]}"
                for r in recent
            )
            parts.append("\nYour recent diary entries (for continuity, do not repeat them):\n" + hist)
        return "\n".join(x for x in parts if x)

    def initial_mood(self) -> str | None:
        return self.profile.get("initial_mood", None)
