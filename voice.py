"""
voice.py
--------
The ONLY file that knows which provider powers JARVIS's speech. graph.py,
agent.py and tools.py never import this module - main.py is the sole
caller, and only when running with --mode voice. Swapping providers later
means editing only the functions below.

Both STT and TTS run on Groq - the same account/key already used for the
LLM, so voice mode adds zero new signups. (An earlier version of this file
used OpenRouter, whose free tier covers TTS but gates ALL audio input
behind a $0.50 minimum account balance - confirmed live, no way around it
at $0. Groq's free tier covers real STT usage with no such wall.)

Endpoints, live-verified against a real Groq account:
  - STT: POST https://api.groq.com/openai/v1/audio/transcriptions
         multipart/form-data: file=<audio bytes>, model=...
         -> JSON response with a "text" field.
  - TTS: POST https://api.groq.com/openai/v1/audio/speech
         JSON body: {"model": ..., "input": <text>, "voice": ..., "response_format": "mp3"}
         -> raw audio bytes in the response body.

The TTS model (GROQ_TTS_MODEL, default canopylabs/orpheus-v1-english)
requires a one-time terms acceptance in the Groq console before your key
can use it: visit
https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english
and accept - no payment involved, just a consent click.

Model IDs drift over time - check https://console.groq.com/docs/speech-to-text
and https://console.groq.com/docs/text-to-speech before a live session.
"""

from pathlib import Path

import requests

from config import GROQ_API_KEY, GROQ_STT_MODEL, GROQ_TTS_MODEL, GROQ_TTS_VOICE, require_groq_key

_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
_TTS_URL = "https://api.groq.com/openai/v1/audio/speech"


def speech_to_text(audio_path: str) -> str:
    """Transcribe a local audio file (wav/mp3/m4a/...) to text via Groq's
    Whisper endpoint."""
    require_groq_key()
    with open(audio_path, "rb") as f:
        resp = requests.post(
            _STT_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (Path(audio_path).name, f, "application/octet-stream")},
            data={"model": GROQ_STT_MODEL},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()["text"]


def text_to_speech(text: str, out_path: str = "reply.mp3") -> str:
    """Convert text to speech via Groq TTS, save the audio to out_path."""
    require_groq_key()
    resp = requests.post(
        _TTS_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_TTS_MODEL,
            "input": text,
            "voice": GROQ_TTS_VOICE,
            "response_format": "mp3",
        },
        timeout=30,
    )
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path


def record_audio(path: str = "input.wav", seconds: float = 4, rate: int = 16000) -> str:
    """Record a few seconds of microphone audio to a .wav file."""
    import sounddevice as sd
    import soundfile as sf

    print(f"Listening... ({seconds:.0f}s)")
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1)
    sd.wait()
    sf.write(path, audio, rate)
    return path


def play_audio(path: str) -> None:
    """Play back an audio file (wav or mp3) through the default speaker."""
    import sounddevice as sd
    import soundfile as sf

    data, rate = sf.read(path)
    sd.play(data, rate)
    sd.wait()


if __name__ == "__main__":
    # python voice.py -> round-trips text -> speech -> playback, no mic needed.
    require_groq_key()
    path = text_to_speech("Hello, sir. JARVIS voice systems are online.")
    print("Wrote", path)
    play_audio(path)