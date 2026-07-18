"""Core-logic tests that run without the native model or Telegram.

Run directly:  python tests/test_core.py
Or with pytest: pytest -q
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.brain import Brain
from src.memory import MemoryStore
from src.persona import Persona


class FakeEngine:
    """Stand-in for LlamaEngine so tests need no model weights."""

    def chat(self, messages, temperature=None, max_tokens=None):
        if any("diary post" in m["content"] for m in messages):
            return "Tired but hopeful\nToday moved slowly and I drank too much tea."
        return "hey, that's sweet of you to say"


def test_memory():
    db = tempfile.mktemp(suffix=".db")
    m = MemoryStore(db)
    try:
        m.add_message(1, "user", "hi")
        m.add_message(1, "assistant", "hello there")
        assert [t.content for t in m.recent_turns(1, 10)] == ["hi", "hello there"]
        assert m.add_fact("Mahsa loves rain") is True
        assert m.add_fact("Mahsa loves rain") is False
        assert m.get_facts() == ["Mahsa loves rain"]
        m.add_journal("Rainy quiet day.", mood="nostalgic")
        assert m.has_journal_today() is True
        assert m.latest_mood() == "nostalgic"
    finally:
        m.close()
        os.remove(db)


def test_persona_prompt():
    db = tempfile.mktemp(suffix=".db")
    m = MemoryStore(db)
    try:
        m.add_fact("Mahsa loves rain")
        m.add_journal("x", mood="nostalgic")
        p = Persona.load(os.path.join("src", "persona", "mahsa.yaml"))
        sp = p.chat_system_prompt(m, user_note="likes poetry")
        assert "Mahsa" in sp and "loves rain" in sp and "nostalgic" in sp and "likes poetry" in sp
    finally:
        m.close()
        os.remove(db)


def test_brain_reply_and_journal():
    db = tempfile.mktemp(suffix=".db")
    m = MemoryStore(db)
    try:
        p = Persona.load(os.path.join("src", "persona", "mahsa.yaml"))
        brain = Brain(FakeEngine(), p, m, max_turns=10)

        async def go():
            r = await brain.reply(2, "you seem lovely")
            assert "sweet" in r
            mood, entry = await brain.write_daily_entry()
            assert mood == "Tired but hopeful" and "tea" in entry
            await brain.make_journal_and_store()
            assert m.latest_mood() == "Tired but hopeful"

        asyncio.run(go())
    finally:
        m.close()
        os.remove(db)


if __name__ == "__main__":
    test_memory()
    test_persona_prompt()
    test_brain_reply_and_journal()
    print("ALL CORE TESTS PASSED")
