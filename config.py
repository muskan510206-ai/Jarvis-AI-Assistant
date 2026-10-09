"""
config.py
---------
Loads environment variables from .env and exposes them as plain constants.

This is the FIRST file the workshop builds (Section 3, "Test Groq"). Every
other module imports its settings from here instead of calling os.getenv()
directly - one place to look when something is misconfigured.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---- The one key the whole project needs (text AND voice mode) ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---- Model selection (safe defaults, overridable via .env) ----
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Voice (--mode voice only). Both STT and TTS run on Groq - same key, zero
# extra signups. GROQ_TTS_MODEL needs a one-time terms acceptance in the
# Groq console: see voice.py's docstring.
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
GROQ_TTS_MODEL = os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
GROQ_TTS_VOICE = os.getenv("GROQ_TTS_VOICE", "tara")

# ---- Agent loop safety valve ----
MAX_ITERATIONS = int(os.getenv("JARVIS_MAX_ITERATIONS", "6"))


def require_groq_key() -> None:
    """Fail fast, with a clear message, if the one truly required key is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "Missing GROQ_API_KEY.\n"
            "  1. Copy jarvis/.env.example to jarvis/.env\n"
            "  2. Get a free key at https://console.groq.com\n"
            "  3. Paste it into .env as GROQ_API_KEY=...\n"
        )


if __name__ == "__main__":
    # python config.py -> quick sanity check, no output means success.
    require_groq_key()
    print("config.py OK - GROQ_API_KEY is set.")