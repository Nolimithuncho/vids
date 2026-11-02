# Vid Explains

Upload a video (any length) and get a full breakdown of it: a summary, a numbered step-by-step
procedure of what is done in the video, chapters, a scene-by-scene description with thumbnails, the full
transcript, and a chat where you can ask anything about it. Every timestamp is clickable and jumps the
player to that moment.

You can also cut the video up: every step, chapter and scene has a ✂ Clip button, the Clips tab cuts any
custom range, and "Split the whole video" turns it into one file per chapter, step or scene, as a zip.

## How it works

1. **Upload.** The file streams straight to disk, so multi-GB videos are fine.
2. **Transcribe.** ffmpeg pulls out the audio and [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   transcribes it locally (free, no upload).
3. **Sample frames.** The video is cut into 5 minute chunks and 30 frames are sampled from each one (one every 10 seconds).
4. **Watch.** Each chunk's frames plus its slice of the transcript go to the AI (Google Gemini by default,
   or Claude), which
   writes structured notes: every action performed (with exact values, settings and quantities), scenes,
   key moments, people, objects, on-screen text. Chunks run 4 at a time.
5. **Report.** The AI reads all the chunk notes and the full transcript and writes the final report,
   including one merged, ordered step-by-step procedure someone could follow without the video.
6. **Ask.** Questions are answered from the report, the chunk notes, and the transcript, with timestamps.

Each stage saves its output, so if something fails (or you restart the server) **Retry** continues where it
stopped instead of starting over.

## Setup

Requirements: Python 3.10+, and ffmpeg on your PATH
(`brew install ffmpeg`, `sudo apt install ffmpeg`, or `winget install ffmpeg`).

```bash
cd vid-explains
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then put your Gemini API key in .env
uvicorn app.main:app --port 8000
```

Open http://localhost:8000 and drop in a video.

### Getting a free Gemini API key

Go to https://aistudio.google.com/apikey, sign in with a Google account, click **Create API key**, and
copy it. No payment is needed for the free tier.

### In VS Code

1. Command Palette (Ctrl/Cmd+Shift+P) → **Git: Clone** → paste `https://github.com/Nolimithuncho/vids`, then open it.
2. Install the **Python** extension when VS Code suggests it.
3. Command Palette → **Python: Create Environment** → Venv → tick `requirements.txt`.
4. Create a file named `.env` containing one line: `GEMINI_API_KEY=your-key`.
5. Press **F5** (Run and Debug → "Run Vid Explains"), then open http://localhost:8000.

The first run downloads the Whisper model (about 500 MB for `small`). With an NVIDIA GPU, transcription is
much faster; on CPU, expect very roughly 1 minute of processing per 5 to 10 minutes of audio with `small`.
Set `VID_WHISPER_MODEL=base` for speed or `large-v3` for accuracy.

## Cutting clips

- **Precise** (default) re-encodes the clip so it starts on the exact frame. It takes roughly as long as
  the clip is, faster on a good CPU.
- **Fast cut** copies the video without re-encoding, so it is near-instant even on huge files, but it can
  start a few seconds early (at the nearest keyframe).

Clips and split zips are saved under `data/videos/<id>/clips/`, so downloading the same one again is
instant. Delete that folder to free disk space.

## Which AI, and cost

**Gemini (default, `VID_PROVIDER=gemini`).** Free tier, no card needed. The free tier has per-minute and
per-day request limits, so the app analyzes 2 chunks at a time and waits and retries automatically when it
hits the per-minute limit. A long video can still use up the daily limit; if that happens the job stops
with a message, and pressing **Retry** the next day continues where it left off. Note that Google may use
data sent on the free tier to improve its products, so don't upload anything private. Change the model with
`VID_GEMINI_MODEL`.

**Claude (`VID_PROVIDER=claude`).** Paid API (a Claude Pro subscription does not include API access). Set
`ANTHROPIC_API_KEY`. As a rough estimate, about $1 to $3 per hour of video with `claude-opus-5-5`.

## Settings

All in `.env` (see `.env.example`): model, effort level, chunk length, frames per chunk, parallelism,
Whisper model and device, data folder.

## Layout

```
app/main.py       FastAPI server: upload, status, results, Q&A, serves the UI
app/pipeline.py   probe -> transcribe -> sample frames -> analyze chunks -> report
app/llm.py        Claude calls (structured output, refusal fallbacks, cached Q&A context)
app/schemas.py    the shape of chunk notes and the final report
static/index.html the whole web UI (no build step)
data/videos/<id>/ everything produced for one video
```

## Known limits

- Claude sees sampled frames, not continuous motion: fast action between frames can be missed.
  Raise `VID_FRAMES_PER_SEGMENT` for action-heavy footage.
- Speaker names come from context (introductions, captions); Whisper does not label speakers.
- This is a single-user app meant to run on your own machine. Add authentication before putting it online.
