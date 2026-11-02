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

DATA_DIR = Path(os.environ.get("VID_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
VIDEOS_DIR = DATA_DIR / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# Claude
MODEL = os.environ.get("VID_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("VID_EFFORT", "medium")  # low | medium | high | xhigh | max
USE_FALLBACKS = os.environ.get("VID_USE_FALLBACKS", "1") == "1"
PARALLEL_SEGMENTS = int(os.environ.get("VID_PARALLEL_SEGMENTS", "4"))

# Video chunking
SEGMENT_SECONDS = int(os.environ.get("VID_SEGMENT_SECONDS", "300"))  # 5 minute chunks
FRAMES_PER_SEGMENT = int(os.environ.get("VID_FRAMES_PER_SEGMENT", "20"))
MIN_FRAME_GAP = float(os.environ.get("VID_MIN_FRAME_GAP", "2"))  # seconds
FRAME_WIDTH = int(os.environ.get("VID_FRAME_WIDTH", "768"))

# Whisper (faster-whisper). tiny | base | small | medium | large-v3
WHISPER_MODEL = os.environ.get("VID_WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("VID_WHISPER_DEVICE", "auto")  # auto | cpu | cuda
