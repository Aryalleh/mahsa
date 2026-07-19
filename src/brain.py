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
            self.memory, user_note=note, relationship=relationship,
            known_people=known, current_user_id=user_id,
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

    # Specific phrases that hint at a relay/recall request. Kept narrow so the
    # extra intent-detection model call only fires when it's really needed
    # (broad words like a bare "بگو" or a friend's name were far too common and
    # doubled latency on ordinary messages).
    _ACTION_HINTS = ("بهش بگو", "بگو به", "پیام بده", "پیام بفرست", "بفرست به",
                     "برسون", "سلام برسون", "بهش پیام", "صحبت کرد", "حرف زد",
                     "چیا گفت", "چی گفت", "چت کرد", "بپرس", "ازش بپرس", "حالشو بپرس",
                     "چه خبر از", "tell ", "message to", "what did you", "ask ")

    # Proposal phrases that mean "marry me" (recorded only in intimate chats).
    _MARRY_HINTS = ("با من ازدواج", "باهام ازدواج", "ازدواج کن", "زنم شو", "زنم بشو",
                    "همسرم شو", "همسرم بشو", "زن من شو", "مال من شو", "marry me")

    async def plan(
        self,
        user_id: int,
        text: str,
        sender_name: str | None = None,
        relationship: str = "friend",
    ) -> dict:
        """Decide what to do with an incoming message.

        Returns {"kind": "reply", "text": ...} or a relay action
        {"kind": "relay", "to_id", "to_name", "text", "user_text"}.
        Relay/recall are only considered for approved friends and admins.
        Recall (summarising chats with another friend) is admins-only.
        """
        # Marriage proposals (only from someone she's intimate with). If they
        # propose, she accepts and it's recorded — she's fine with several spouses.
        if relationship in ("admin", "lover"):
            low_m = text.lower()
            if any(k in low_m for k in self._MARRY_HINTS) and not self.memory.is_spouse(user_id):
                self.memory.add_spouse(user_id, sender_name)

        if relationship in ("friend", "admin", "lover"):
            action, who, message = await self._detect_action(text)
            if action in ("relay", "recall", "ask") and who:
                matches = [m for m in self.memory.find_contacts_by_name(who) if m[0] != user_id]
                if not matches:
                    self.memory.add_message(user_id, "user", text)
                    ack = f"«{who}» رو توی دوستام پیدا نکردم. مطمئنی اسمش درسته؟"
                    self.memory.add_message(user_id, "assistant", ack)
                    return {"kind": "reply", "text": ack}
                to_id, to_name = matches[0]
                if action == "relay":
                    relay = await self.compose_relay(to_name, message)
                    return {"kind": "relay", "to_id": to_id, "to_name": to_name,
                            "text": relay, "user_text": text}
                if action == "ask":
                    question = message or "چه خبر؟ خوبی؟"
                    q_text = await self.compose_relay(to_name, question)
                    return {"kind": "ask", "to_id": to_id, "to_name": to_name,
                            "text": q_text, "asker_id": user_id, "question": question,
                            "user_text": text}
                if action == "recall" and relationship == "admin":
                    self.memory.add_message(user_id, "user", text)
                    summary = await self.summarise_chat_with(to_id, to_name)
                    self.memory.add_message(user_id, "assistant", summary)
                    return {"kind": "reply", "text": summary}
        reply = await self.reply(user_id, text, display=sender_name, relationship=relationship)
        return {"kind": "reply", "text": reply}

    async def _detect_action(self, text: str) -> tuple[str, str, str]:
        """Classify intent about a friend. Returns (action, who, message).

        action is 'relay' (tell a friend something), 'recall' (what did you talk
        about with a friend), or 'none'.
        """
        friends = [d for _, d, _ in self.memory.list_contacts("approved") if d and d != "?"]
        if not friends:
            return ("none", "", "")
        low = text.lower()
        # Only run the extra detection call when an explicit relay/recall phrase
        # is present — a bare friend-name mention is not enough.
        if not any(h in low for h in self._ACTION_HINTS):
            return ("none", "", "")
        names = ", ".join(friends)
        system = (
            "You classify what the user wants Mahsa to do about one of her friends. "
            f"Her friends are: {names}. Respond with JSON only: "
            '{"action": "relay" | "recall" | "ask" | "none", "who": "<friend name or '
            'empty>", "message": "<for relay: what to tell them; for ask: the question '
            'to ask them; else empty>"}. '
            '"relay" = the user asks Mahsa to tell/send something to a named friend. '
            '"ask" = the user asks Mahsa to ASK a named friend something and report their '
            "answer back (e.g. 'ask Blue how they are'). "
            '"recall" = the user asks what Mahsa already talked about with a named friend. '
            '"none" = anything else.'
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
        try:
            raw = await asyncio.to_thread(self.engine.chat_json, messages)
            data = json.loads(raw)
        except Exception:  # noqa: BLE001
            return ("none", "", "")
        action = str(data.get("action", "none")).strip().lower()
        who = str(data.get("who", "")).strip()
        message = str(data.get("message", "")).strip()
        if action == "relay" and who and message:
            return ("relay", who, message)
        if action == "ask" and who:
            return ("ask", who, message)
        if action == "recall" and who:
            return ("recall", who, "")
        return ("none", "", "")

    async def compose_relay(self, to_name: str, content: str) -> str:
        """Write the relay message in Mahsa's OWN voice, first person, no quotes.

        Falls back to a plain template if the model refuses or leaks (e.g. a
        Chinese chain-of-thought), so a message always goes out.
        """
        system = self.persona.chat_system_prompt(self.memory)
        prompt = (
            f"Send a short Telegram message to your friend {to_name}, in your OWN "
            f"voice, first person, saying this naturally: {content}. Do NOT use "
            "quotation marks and do NOT say that someone told you to say it — just "
            "say it warmly as yourself. One or two lines."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            text = (await asyncio.to_thread(self.engine.chat, messages, 0.7, 120)).strip()
        except Exception:  # noqa: BLE001
            text = ""
        text = text.strip().strip("«»\"'")
        if self._looks_bad(text):
            return f"سلام {to_name} جان 🌸 {content}"
        return text

    async def summarise_chat_with(self, friend_id: int, friend_name: str) -> str:
        """Summarise, in Mahsa's voice, what she and a given friend talked about."""
        turns = self.memory.recent_turns(friend_id, self.max_turns)
        if not turns:
            return f"راستش هنوز با {friend_name} چت نکردم."
        transcript = "\n".join(
            f"{'من' if t.role == 'assistant' else friend_name}: {t.content}" for t in turns
        )
        system = self.persona.chat_system_prompt(self.memory)
        prompt = (
            f"Your admin asks what you and {friend_name} have been talking about. "
            f"Here is your recent chat with {friend_name}:\n{transcript}\n\n"
            "Tell the admin, in your own voice and a few short sentences, what the two "
            "of you talked about."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return await asyncio.to_thread(self.engine.chat, messages, 0.5, 220)

    @staticmethod
    def _looks_bad(text: str) -> bool:
        """True if the model output is empty, a refusal, or leaked other languages."""
        if not text or len(text) < 2:
            return True
        if any("　" <= c <= "鿿" for c in text):  # CJK leak
            return True
        low = text.lower()
        bad = ("نمیتونم", "نمی‌تونم", "نمی‌توانم", "نميتوانم", "cannot", "can't",
               "won't", "i'm sorry", "as an ai", "抱歉")
        return any(b in low for b in bad)

    # ---- daily emotional diary post -------------------------------------
    async def write_daily_entry(self) -> tuple[str, str]:
        """Compose today's diary entry. Returns (mood, entry_text)."""
        system = self.persona.journal_system_prompt(self.memory)
        prompt = (
          
            f"امروز {date.today().strftime('%A, %B %d')} است. "
            "برای کانالت یک یادداشت بنویس که خواننده احساس کند با یک انسان واقعی روبه‌روست. "
            "می‌توانی از یک تصویر ذهنی، یک سؤال، یک گفت‌وگوی خیالی، یک خاطره، یک مشاهده از طبیعت یا شهر، یا یک فکر ناگهانی شروع کنی. "
            "هیچ دو یادداشتی نباید شبیه هم باشند. "
            "از تکرار ساختار جمله‌ها و واژه‌های همیشگی خودداری کن. "
            "گاهی طنز ظریف، گاهی احساسات عمیق و گاهی سکوت و ابهام را وارد متن کن. "
            "همیشه فقط به زبان فارسی بنویس و بین ۳ تا ۶ جمله بنویس."

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
