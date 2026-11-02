# Vid Explains

Upload a video (any length) and get a full breakdown of it: a summary, chapters, a scene-by-scene
description with thumbnails, the full transcript, and a chat where you can ask anything about it.
Every timestamp is clickable and jumps the player to that moment.

## How it works

1. **Upload.** The file streams straight to disk, so multi-GB videos are fine.
2. **Transcribe.** ffmpeg pulls out the audio and [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   transcribes it locally (free, no upload).
3. **Sample frames.** The video is cut into 5 minute chunks and 20 frames are sampled from each one.
4. **Watch.** Each chunk's frames plus its slice of the transcript go to Claude (`claude-opus-5-5`), which
   writes structured notes: scenes, key moments, people, objects, on-screen text. Chunks run 4 at a time.
5. **Report.** Claude reads all the chunk notes and the full transcript and writes the final report.
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
cp .env.example .env               # then put your Anthropic API key in .env
uvicorn app.main:app --port 8000
```

Open http://localhost:8000 and drop in a video.

The first run downloads the Whisper model (about 500 MB for `small`). With an NVIDIA GPU, transcription is
much faster; on CPU, expect very roughly 1 minute of processing per 5 to 10 minutes of audio with `small`.
Set `VID_WHISPER_MODEL=base` for speed or `large-v3` for accuracy.

## Cost

You pay for Claude API usage. As a rough estimate, a 5 minute chunk is about 10k input tokens (mostly
images) plus a few thousand output tokens, which comes to roughly **$1 to $3 per hour of video** with
`claude-opus-5-5` at the default settings. Ways to lower it: `VID_MODEL=claude-sonnet-5-5`, fewer
`VID_FRAMES_PER_SEGMENT`, or longer `VID_SEGMENT_SECONDS`. Q&A reuses a cached copy of the video's notes,
so follow-up questions are cheap.

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
