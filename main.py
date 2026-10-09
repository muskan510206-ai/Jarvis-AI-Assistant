"""
main.py
-------
JARVIS's entry point.

    python main.py                 -> interactive text mode (default)
    python main.py --mode text     -> same, explicit
    python main.py --mode voice    -> microphone in, speaker out

Text mode and voice mode call the exact same LangGraph `app` from graph.py -
voice just wraps it with a mic on one end and a speaker on the other. Both
modes need only GROQ_API_KEY - voice runs its STT/TTS on Groq too.
"""

import argparse
import sys

from config import require_groq_key
from graph import app, new_turn_state, resolve_pending_confirmation

# LLMs sometimes emit Unicode punctuation (em-dashes, narrow spaces, curly
# quotes) that the default Windows console codepage (cp1252) can't encode,
# which crashes print() outright. Force UTF-8 with a safe fallback instead.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def next_state(state: dict | None, user_text: str) -> dict:
    """Route to the human-in-the-loop resolver if JARVIS is waiting on a
    yes/no from last turn; otherwise run a normal graph turn."""
    if state and state.get("pending_confirmation"):
        return resolve_pending_confirmation(state, user_text)
    return app.invoke(new_turn_state(state, user_text))


def run_text_mode() -> None:
    require_groq_key()
    print("🤖 JARVIS is online. Type 'exit' to quit.\n")
    state = None
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit"):
            break

        state = next_state(state, user_text)
        print(f"JARVIS: {state['final_response']}\n")


def run_voice_mode() -> None:
    require_groq_key()  # the only key voice mode needs too - same as text mode
    from voice import play_audio, record_audio, speech_to_text, text_to_speech

    print("🎙️  JARVIS voice mode. Ctrl+C to quit.\n")
    state = None
    while True:
        try:
            audio_path = record_audio()
            user_text = speech_to_text(audio_path)
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"[voice error - STT] {e}\n")
            continue

        if not user_text.strip():
            print("(didn't catch that - try again)\n")
            continue
        print(f"You said: {user_text}")

        state = next_state(state, user_text)
        reply = state["final_response"]
        print(f"JARVIS: {reply}\n")

        try:
            reply_path = text_to_speech(reply)
            play_audio(reply_path)
        except Exception as e:
            print(f"[voice error - TTS/playback] {e} (reply was printed above)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="JARVIS - your personal AI operating assistant.")
    parser.add_argument("--mode", choices=["text", "voice"], default="text")
    args = parser.parse_args()

    if args.mode == "voice":
        run_voice_mode()
    else:
        run_text_mode()


if __name__ == "__main__":
    main()