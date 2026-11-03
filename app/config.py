"""Settings, all overridable with environment variables (or a .env file)."""
import os
from pathlib import Path


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _ensure_ffmpeg() -> None:
    """Use the system ffmpeg/ffprobe if installed; otherwise download a private copy once
    (about 80 MB, via the static-ffmpeg package) so users don't have to install it themselves."""
    import shutil

    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    try:
        import static_ffmpeg

        print("ffmpeg not found; downloading it once (this can take a minute)...", flush=True)
        static_ffmpeg.add_paths(weak=True)
    except Exception as exc:
        raise SystemExit(
            "Could not find or download ffmpeg. Install it yourself and try again "
            f"(Mac: brew install ffmpeg, Windows: winget install ffmpeg). Details: {exc}"
        )


_ensure_ffmpeg()

DATA_DIR = Path(os.environ.get("VID_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
VIDEOS_DIR = DATA_DIR / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# Which AI watches the videos: "gemini" (default, has a free tier) or "claude" (paid API)
PROVIDER = os.environ.get("VID_PROVIDER", "gemini").strip().lower()

# Gemini (free key at https://aistudio.google.com/apikey)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = os.environ.get("VID_GEMINI_MODEL", "gemini-flash-latest")
GEMINI_BASE_URL = os.environ.get("VID_GEMINI_BASE_URL")  # only for testing against a mock server

# Claude (needs ANTHROPIC_API_KEY)
CLAUDE_MODEL = os.environ.get("VID_CLAUDE_MODEL") or os.environ.get("VID_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("VID_EFFORT", "medium")  # low | medium | high | xhigh | max
USE_FALLBACKS = os.environ.get("VID_USE_FALLBACKS", "1") == "1"

# Chunks analyzed at the same time. The Gemini free tier only allows a few requests a minute.
PARALLEL_SEGMENTS = int(os.environ.get("VID_PARALLEL_SEGMENTS", "2" if PROVIDER == "gemini" else "4"))

# Video chunking
# Gemini's free tier allows very few requests per day (as few as 20), so with Gemini the video is
# watched in 30 minute chunks (one request each) instead of 5 minute ones. Same frame density.
_GEMINI = PROVIDER == "gemini"
SEGMENT_SECONDS = int(os.environ.get("VID_SEGMENT_SECONDS", "1800" if _GEMINI else "300"))
FRAMES_PER_SEGMENT = int(os.environ.get("VID_FRAMES_PER_SEGMENT", "150" if _GEMINI else "30"))
MIN_FRAME_GAP = float(os.environ.get("VID_MIN_FRAME_GAP", "2"))  # seconds
FRAME_WIDTH = int(os.environ.get("VID_FRAME_WIDTH", "768"))

# Whisper (faster-whisper). tiny | base | small | medium | large-v3
WHISPER_MODEL = os.environ.get("VID_WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("VID_WHISPER_DEVICE", "auto")  # auto | cpu | cuda
