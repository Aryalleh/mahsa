# Mahsa 🌙

A platform that runs a **local** LLM (`Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`)
as an AI persona named **Mahsa**, who:

- **chats with people on Telegram** through a real user account (Telethon),
- **posts a daily diary entry** about her mood and her day to her own channel,
- **learns her personality** over time from a profile file + facts you teach her.

Everything runs on your own machine. The model never leaves your computer; only
the Telegram messages Mahsa chooses to send go out.

---

## ⚠️ Read this first — responsible use

Mahsa runs on a **real Telegram user account**, not the official Bot API.

- **Automating a user account can get the phone number limited or banned.** Use a
  dedicated number you don't mind losing, keep message volume human-like, and don't
  cold-message strangers. This is inherently against the spirit of Telegram's ToS —
  you accept that risk.
- **Be honest.** Mahsa is a *fictional AI persona*. Don't present her as a specific
  real person in order to deceive the people she talks to. Her profile already makes
  her admit she's an AI if asked directly — keep it that way.
- **Consent & safety.** Don't use this to impersonate a real individual, to
  manipulate people romantically/financially, or to target minors. The persona file
  contains safety boundaries; don't strip them out.

If you're building an AI-companion character for a consenting audience, you're in
the right place.

---

## Architecture

```
main.py                  boots everything, handles Telegram login, runs the loop
config.py                loads settings from .env
src/
  llm/engine.py          llama.cpp wrapper around the GGUF model
  persona/persona.py     builds the system prompt from the profile + learned facts
  persona/mahsa.yaml     Mahsa's editable personality profile
  memory/store.py        SQLite: chat history, learned facts, mood journal, notes
  brain.py               ties model + persona + memory together (reply / journal)
  telegram_bot/client.py Telethon client
  telegram_bot/handlers.py   incoming messages + admin commands
  scheduler/daily_post.py    APScheduler job that publishes the daily post
tests/test_core.py       core-logic tests (no model / no network needed)
```

Data flow for a reply:

```
user DM ─▶ handlers ─▶ brain.reply()
                         ├─ load recent history + learned facts + mood  (memory)
                         ├─ build system prompt                          (persona)
                         └─ run local model                              (llm)
                       ◀─ reply text ─▶ sent back over Telegram
```

---

## Setup

### 1. Python deps
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
> `llama-cpp-python` compiles a native library. For GPU/Metal acceleration, install
> it with the appropriate `CMAKE_ARGS` (see the llama-cpp-python docs).

### 2. Download the model
```bash
pip install huggingface_hub
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models
```

### 3. Telegram credentials
1. Go to <https://my.telegram.org> → **API development tools** → create an app.
2. Copy the `api_id` and `api_hash`.
3. Create the channel Mahsa will post to and **make her account an admin** of it.

### 4. Configure
```bash
cp .env.example .env
# then edit .env — api id/hash, phone, channel, admin ids, model path
```
Find your own numeric Telegram user id (to be an admin) by messaging
`@userinfobot`, and put it in `ADMIN_USER_IDS`.

### 5. Run
```bash
python main.py
```
On the **first run** Telegram sends a login code to that account — enter it (and
your 2FA password if you have one). A `sessions/mahsa.session` file is written so
you stay logged in afterwards.

---

## Teaching Mahsa her personality

Two layers:

1. **Base profile** — edit `src/persona/mahsa.yaml` (traits, voice, interests,
   boundaries, starting mood). Takes effect on restart.
2. **Learned facts** — from an admin account, DM Mahsa:
   - `/teach She grew up in Shiraz and misses the orange blossoms.`
   - `/facts` — list what she's learned (with ids)
   - `/forget 3` — remove fact #3

Learned facts are stored in SQLite and injected into every prompt, so she stays
consistent across restarts. She also keeps a short private note about each person
she talks to, so she "remembers" them.

### Admin commands (DM from an admin id)
| Command | Effect |
|---|---|
| `/teach <fact>` | Add a permanent personality/life fact |
| `/facts` | List learned facts with ids |
| `/forget <id>` | Delete a fact |
| `/mood` | Show her current mood |
| `/post` | Write & publish today's diary post right now |
| `/reset <uid>` | Clear one user's conversation history |
| `/help` | Show commands |

---

## Daily emotional posts

Every day at `DAILY_POST_HOUR:DAILY_POST_MINUTE` (local time) Mahsa writes a short,
first-person diary entry — a mood line plus a few sentences about how her day felt —
and posts it to `TELEGRAM_CHANNEL`. Each entry is stored in the `journal` table and
recent entries are fed back in for continuity, so her moods evolve over time instead
of resetting. Trigger one on demand any time with `/post`.

---

## Testing
```bash
python tests/test_core.py     # or: pytest -q
```
These exercise memory, prompt-building, and the reasoning core using a fake engine,
so they need neither the model weights nor a network connection.

---

## Notes & limits
- Inference runs in a worker thread so one slow generation doesn't freeze the
  Telegram loop, but the 8B model on CPU is not instant — expect a few seconds per
  reply. Use `LLM_GPU_LAYERS` to offload to a GPU.
- Mahsa only replies in **private chats** and never to her own messages, to avoid
  spamming groups. Adjust `src/telegram_bot/handlers.py` if you want different scope.
- Secrets (`.env`, `*.session`) and the model/`data/` folders are gitignored — keep
  them off version control.
