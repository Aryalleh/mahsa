"""The reasoning core: builds context and asks the local model to speak/write as Mahsa."""
from __future__ import annotations

import asyncio
import json
import re
from datetime import date

from .llm import LlamaEngine
from .memory import MemoryStore
from .persona import Persona
from .utils.logging import get_logger

log = get_logger("mahsa.brain")


class Brain:
    def __init__(
        self,
        engine: LlamaEngine,
        persona: Persona,
        memory: MemoryStore,
        max_turns: int = 20,
    ):
        self.engine = engine
        self.persona = persona
        self.memory = memory
        self.max_turns = max_turns

    # ---- chat reply ------------------------------------------------------
    async def reply(
        self,
        user_id: int,
        text: str,
        display: str | None = None,
        relationship: str = "friend",
    ) -> str:
        """Generate Mahsa's reply to a user's message and persist the exchange."""
        self.memory.add_message(user_id, "user", text)
        note = self.memory.get_user_note(user_id)
        known = self._friends_roster(exclude=user_id)
        system = self.persona.chat_system_prompt(
            self.memory, user_note=note, relationship=relationship, known_people=known
        )

        history = self.memory.recent_turns(user_id, self.max_turns)
        messages = [{"role": "system", "content": system}]
        for turn in history:
            messages.append({"role": turn.role, "content": turn.content})

        answer = await asyncio.to_thread(self.engine.chat, messages)
        self.memory.add_message(user_id, "assistant", answer)
        return answer

    def _friends_roster(self, exclude: int, limit: int = 12) -> list[tuple[str, str]]:
        """Her approved friends as (name, short note), so she can mention them."""
        roster: list[tuple[str, str]] = []
        for fid, display, _ in self.memory.list_contacts("approved"):
            if fid == exclude:
                continue
            note = self.memory.get_user_note(fid) or ""
            # keep it to the first sentence so prompts stay small
            short = note.split(".")[0].strip()
            if len(short) > 120:
                short = short[:120].rstrip() + "…"
            roster.append((display or "?", short))
            if len(roster) >= limit:
                break
        return roster

    # Words that hint the user wants a message passed along.
    _RELAY_HINTS = ("بگو", "بهش", "بگه", "بفرست", "برسون", "پیام بده", "سلام برسون",
                    "tell", "say to", "pass ")

    async def plan(
        self,
        user_id: int,
        text: str,
        sender_name: str | None = None,
        relationship: str = "friend",
    ) -> dict:
        """Decide what to do with an incoming message.

        Returns either {"kind": "reply", "text": ...} or a relay action
        {"kind": "relay", "to_id", "to_name", "text", "ack"}.
        Relay is only ever considered for approved friends, never public strangers.
        """
        if relationship == "friend":
            det = await self._detect_relay(text)
            if det:
                who, message = det
                matches = [m for m in self.memory.find_contacts_by_name(who) if m[0] != user_id]
                if matches:
                    to_id, to_name = matches[0]
                    relay = self.compose_relay(sender_name or "دوستت", to_name, message)
                    return {"kind": "relay", "to_id": to_id, "to_name": to_name,
                            "text": relay, "user_text": text}
                # Detected a relay request but couldn't match the friend — say so
                # honestly instead of pretending it was delivered.
                self.memory.add_message(user_id, "user", text)
                ack = f"«{who}» رو توی دوستام پیدا نکردم. مطمئنی اسمش درسته؟"
                self.memory.add_message(user_id, "assistant", ack)
                return {"kind": "reply", "text": ack}
        reply = await self.reply(user_id, text, display=sender_name, relationship=relationship)
        return {"kind": "reply", "text": reply}

    async def _detect_relay(self, text: str) -> tuple[str, str] | None:
        """Detect 'tell <friend> <message>' intent. Returns (who, message) or None."""
        friends = [d for _, d, _ in self.memory.list_contacts("approved") if d]
        if not friends:
            return None
        low = text.lower()
        # Cheap pre-filter: skip the extra LLM call unless a friend name or a
        # relay hint word is present.
        if not (any(f.lower() in low for f in friends)
                or any(h in low for h in self._RELAY_HINTS)):
            return None
        names = ", ".join(friends)
        system = (
            "You decide whether the user is asking Mahsa to pass a message to one of "
            f"her friends. Her friends are: {names}. Respond with JSON only: "
            '{"relay": true or false, "who": "<the friend\'s name or empty>", '
            '"message": "<what to tell that friend, or empty>"}. Set relay=true ONLY '
            "if the user clearly asks to tell/say/pass something to a specific named "
            "friend from that list; otherwise false."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
        try:
            raw = await asyncio.to_thread(self.engine.chat_json, messages)
            data = json.loads(raw)
        except Exception:  # noqa: BLE001
            return None
        if not data.get("relay"):
            return None
        who = str(data.get("who", "")).strip()
        message = str(data.get("message", "")).strip()
        if not who or not message:
            return None
        return who, message

    @staticmethod
    def compose_relay(from_name: str, to_name: str, content: str) -> str:
        """A fixed relay template.

        Deliberately NOT model-generated: letting the LLM rewrite the message
        triggered refusals and leaked its internal reasoning. A template always
        delivers exactly what the sender asked, in Mahsa's warm style.
        """
        return f"سلام {to_name} جان 🌸 {from_name} گفت بهت بگم: «{content}»"

    # ---- daily emotional diary post -------------------------------------
    async def write_daily_entry(self) -> tuple[str, str]:
        """Compose today's diary entry. Returns (mood, entry_text)."""
        system = self.persona.journal_system_prompt(self.memory)
        prompt = (
            f"Today is {date.today().strftime('%A, %B %d')}. Write a short, "
            "first-person diary post for your public channel about how your day "
            "felt and what you are feeling. Two to five sentences, warm and "
            "honest, in your own voice. Start with one line that names your mood "
            "in a few words, then the entry. Do not use headings or bullet points."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        text = await asyncio.to_thread(self.engine.chat, messages, 0.9, 300)
        mood, entry = self._split_mood(text)
        return mood, entry

    async def make_journal_and_store(self) -> tuple[int, str]:
        """Write today's entry, store it, return (journal_id, post_text)."""
        mood, entry = await self.write_daily_entry()
        journal_id = self.memory.add_journal(entry=entry, mood=mood)
        return journal_id, entry

    # ---- reacting to channel posts --------------------------------------
    # A small, common reaction set most channels allow.
    _REACTIONS = ["❤️", "🔥", "😍", "👍", "😂", "😮", "😢", "🤔", "🙏", "👏"]

    async def channel_vibe(self, post_text: str) -> tuple[str, str]:
        """Read a channel post and return (comment_text, reaction_emoji) in her voice."""
        system = self.persona.chat_system_prompt(self.memory)
        allowed = " ".join(self._REACTIONS)
        prompt = (
            "You just saw this post in a channel you follow:\n\n"
            f"\"{post_text[:800]}\"\n\n"
            "React the way you naturally would. Reply in EXACTLY this format:\n"
            f"first line: one emoji from this set only -> {allowed}\n"
            "second line: one short, casual sentence in your own voice saying the "
            "vibe you got from it. Nothing else, no quotes."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        text = await asyncio.to_thread(self.engine.chat, messages, 0.8, 120)
        return self._parse_vibe(text)

    def _parse_vibe(self, text: str) -> tuple[str, str]:
        reaction = "❤️"
        for ch in text:
            if ch in self._REACTIONS:
                reaction = ch
                break
        # also match multi-codepoint emoji like ❤️ that the loop above may miss
        for emo in self._REACTIONS:
            if emo in text:
                reaction = emo
                break
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        comment = ""
        for l in lines:
            stripped = l
            for emo in self._REACTIONS:
                stripped = stripped.replace(emo, "")
            stripped = stripped.strip(" -–—:•\"'")
            if len(stripped) >= 3:
                comment = stripped
                break
        return comment, reaction

    # ---- service-bot menu navigation ------------------------------------
    async def choose_button(self, bot_text: str, labels: list[str]) -> int | None:
        """Pick which inline ("glass") button to press on a menu bot, or None.

        Given the bot's message and its button labels, the model returns the
        index of the button that best continues the interaction.
        """
        if not labels:
            return None
        numbered = "\n".join(f"{i}: {l}" for i, l in enumerate(labels))
        messages = [
            {
                "role": "system",
                "content": (
                    "You are navigating a Telegram service/menu bot. You are shown "
                    "the bot's latest message and a numbered list of its buttons. "
                    "Pick the ONE button that best continues the interaction in a "
                    "natural, sensible way. Answer with ONLY that number. If none "
                    "make sense, answer 'none'."
                ),
            },
            {
                "role": "user",
                "content": f"Bot message:\n{bot_text}\n\nButtons:\n{numbered}\n\nAnswer:",
            },
        ]
        ans = (await asyncio.to_thread(self.engine.chat, messages, 0.2, 8)).strip().lower()
        if "none" in ans:
            return None
        m = re.search(r"\d+", ans)
        if not m:
            return None
        idx = int(m.group())
        return idx if 0 <= idx < len(labels) else None

    @staticmethod
    def _split_mood(text: str) -> tuple[str, str]:
        """Best-effort split of a leading mood line from the entry body."""
        lines = [l for l in text.strip().splitlines() if l.strip()]
        if not lines:
            return "", text.strip()
        first = lines[0].strip()
        lowered = first.lower()
        if lowered.startswith("mood") or len(first) <= 60:
            mood = first.split(":", 1)[-1].strip(" .:-—") if ":" in first else first
            body = "\n".join(lines[1:]).strip() or first
            return mood, body
        return "", text.strip()

    async def summarise_user(self, user_id: int) -> None:
        """Update the short remembered note about a user from recent history."""
        history = self.memory.recent_turns(user_id, self.max_turns)
        if len(history) < 4:
            return
        transcript = "\n".join(f"{t.role}: {t.content}" for t in history)
        messages = [
            {
                "role": "system",
                "content": (
                    "You maintain a private one-paragraph memory note about a person "
                    f"{self.persona.name} chats with. Summarise who they seem to be, "
                    "what they talked about, and their tone. Third person, concise."
                ),
            },
            {"role": "user", "content": f"Conversation:\n{transcript}\n\nWrite the note:"},
        ]
        note = await asyncio.to_thread(self.engine.chat, messages, 0.4, 200)
        self.memory.set_user_note(user_id, note)
