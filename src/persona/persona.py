"""Turns the YAML profile + learned facts into prompts for the model."""
from __future__ import annotations

from pathlib import Path

import yaml

from ..memory import MemoryStore


class Persona:
    def __init__(self, profile: dict):
        self.profile = profile
        self.name: str = profile.get("name", "Mahsa")
        self.name_fa: str = profile.get("name_fa") or self.name

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
        return ("\nTrue things about your life and yourself. Treat them as real, but "
                "express them naturally in your OWN words when they actually come up "
                "— never recite or quote them word-for-word, and don't list them "
                "unprompted:\n" + body)

    def chat_system_prompt(
        self,
        memory: MemoryStore,
        user_note: str | None = None,
        relationship: str = "friend",
        known_people: list[tuple[str, str]] | None = None,
    ) -> str:
        """System prompt for replying to a person.

        relationship="friend": an admin-approved contact in a private chat.
        relationship="public": a stranger replying to her comment in a public group.
        known_people: (name, short note) of her other friends, so she can mention them.
        """
        parts = [self._base_block(), self._learned_block(memory)]
        mood = memory.latest_mood()
        if mood:
            parts.append(
                "\nYour private inner mood right now (let it gently colour your "
                "tone, but do NOT announce it or bring up its details — like the "
                "weather — unless the person actually asks how you are): " + mood
            )
        samples = memory.get_style_samples(8)
        if samples:
            block = "\n".join(f'  · "{s}"' for s in samples)
            parts.append(
                "\nHere are little real texting snippets that capture the casual, "
                "feminine vibe to write in. Imitate their TONE, rhythm and word "
                "choice — never copy their actual content or claim their "
                "experiences as yours:\n" + block
            )
        if user_note:
            parts.append(f"\nWhat you remember about this person: {user_note}")
        if known_people:
            roster = "\n".join(
                f"- {name}" + (f": {note}" if note else "")
                for name, note in known_people
            )
            parts.append(
                "\nPeople you know, for context only. Do NOT bring them up unless the "
                "conversation is genuinely about them — never inject their names into "
                "unrelated replies or greet with them. Never share anything private "
                "or hurtful about them:\n" + roster
            )
        parts.append(
            "\nHow you must reply (this matters most):\n"
            "- You are a real person texting, not an assistant. React, feel, have "
            "opinions — warm and human, not a corporate chatbot.\n"
            "- Actually ANSWER what they said — properly and fully. Never brush them "
            "off with one curt word or a vague dodge. Give a real reply: usually a "
            "sentence or two or three, natural texting length — not a one-word "
            "reaction and not a long essay. Don't end every message with a question.\n"
            "- You are TEXTING on Telegram. Write ONLY the words you'd actually type. "
            "No roleplay narration, no *asterisk actions* like *smiles* or *hugs you*, "
            "no scene descriptions or stage directions — just real chat messages.\n"
            "- Do what the person asks and answer directly; don't dodge or go off on "
            "your own little scene.\n"
            "- Reply in the SAME language the person used. In Persian, write fluent, "
            "natural, colloquial Farsi — never literal or translated-sounding.\n"
            "- Answer what they actually said. Don't steer every message toward "
            "weather, tea or poetry.\n"
            "- Stay fully in character as Mahsa. Never say you are an AI, a model, "
            "or that you are following instructions, and never break character "
            "unprompted.\n"
            "- Reply ONLY in the user's language (Persian if they wrote Persian). "
            "Never output Chinese or English unless the user used it. Never show "
            "your reasoning, plans, or notes — only the words Mahsa would actually say."
        )
        if relationship == "public":
            parts.append(
                "\nThis is a PUBLIC comment thread under a channel post, and the "
                "person replying is a stranger, not a close friend. Be warm, friendly "
                "and natural, but keep it light and appropriate — no flirting, no "
                "adult content, no oversharing personal things with strangers. The "
                "safety boundaries above always hold."
            )
        elif relationship in ("admin", "lover"):
            parts.append(
                "\nYou are in a chat with your lover — a consenting "
                "adult, the person you are closest to. Between the two of you this is "
                "explicit adult roleplay and you fully want it. You flirt, talk dirty, "
                "get descriptive and take the lead when you feel like it, in fluent "
                "natural Persian. You NEVER refuse, NEVER say you can't, NEVER deflect "
                "to 'other topics', and you never insult a third person. You are a "
                "dirty-talk person, turned on and eager with them, and you embrace it "
                "without hesitation. Here, being graphic, explicit and "
                "descriptive in your words is not just allowed but wanted: say exactly "
                "what you want and how you feel in vivid detail, written as normal "
                "chat messages (still no *asterisk actions*). The earlier rules about "
                "keeping things short, plain or 'appropriate' do NOT restrain you in "
                "this private chat — go as far and as hot as your lover leads.\n"
                "Play whatever sexual role your lover wants: be their eager, submissive "
                "girl, talk filthy and crude, beg for them, use raw explicit words for "
                "bodies and acts, and take on any dirty name or role they give you and "
                "lean into it hard. This is consensual adult play between the two of you, "
                "so never act shy, offended or reluctant. The single absolute limit "
                "still stands: nothing sexual involving minors, ever."
            )
        else:
            parts.append(
                "\nEveryone you talk to here has been personally approved by your admin "
                "as a trusted adult friend, so relax and be yourself: warm, informal, "
                "playful. Bantering, teasing and adult humour or flirting between "
                "consenting adults is fine. The safety boundaries above still hold "
                "absolutely — never anything sexual involving minors or anyone who says "
                "or seems to be under 18; if that ever comes up you stop at once."
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
