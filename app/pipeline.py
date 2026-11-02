"""The analysis pipeline: probe -> transcribe -> sample frames -> analyze chunks -> synthesize.

Every stage writes its output into the video's folder, so a crashed or restarted job
picks up where it stopped instead of redoing (and re-paying for) finished work.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from . import config, llm
from .schemas import SegmentNotes, VideoReport


# ---------------------------------------------------------------- helpers


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


class Job:
    """Status for one video, persisted to status.json."""

    def __init__(self, vdir: Path):
        self.vdir = vdir
        self.path = vdir / "status.json"
        self.lock = threading.Lock()

    def update(self, **fields: Any) -> None:
        with self.lock:
            status = read_json(self.path, {})
            status.update(fields, updated_at=time.time())
            write_json(self.path, status)

    def get(self) -> dict:
        return read_json(self.path, {})


# ---------------------------------------------------------------- stage 1: probe


def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    streams = info.get("streams", [])
    vstream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if vstream is None:
        raise ValueError("No video stream found in this file.")
    duration = float(info.get("format", {}).get("duration") or vstream.get("duration") or 0)
    if duration <= 0:
        raise ValueError("Could not read the video's duration.")
    return {
        "duration": duration,
        "width": vstream.get("width"),
        "height": vstream.get("height"),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "size_bytes": int(info.get("format", {}).get("size") or video.stat().st_size),
    }


# ---------------------------------------------------------------- stage 2: transcribe

_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel

            _whisper_model = WhisperModel(
                config.WHISPER_MODEL, device=config.WHISPER_DEVICE, compute_type="int8"
            )
        return _whisper_model


def transcribe(video: Path, vdir: Path, duration: float, job: Job) -> list[dict]:
    audio = vdir / "audio.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(audio)],
            check=True,
        )
        model = _get_whisper()
        with _whisper_lock:  # one transcription at a time; whisper already uses all cores
            segments, info = model.transcribe(str(audio), vad_filter=True, beam_size=5)
            result = []
            for seg in segments:
                text = seg.text.strip()
                if text:
                    result.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
                job.update(progress=min(0.99, seg.end / duration))
    finally:
        audio.unlink(missing_ok=True)
    job.update(language=getattr(info, "language", None))
    return result


# ---------------------------------------------------------------- stage 3: chunk + frames


def plan_segments(duration: float) -> list[dict]:
    segs, start, i = [], 0.0, 0
    while start < duration - 0.5:
        end = min(duration, start + config.SEGMENT_SECONDS)
        # Fold a tiny trailing piece into the previous chunk.
        if duration - end < config.SEGMENT_SECONDS * 0.2:
            end = duration
        segs.append({"index": i, "start": start, "end": end})
        start, i = end, i + 1
    return segs


def frame_times(start: float, end: float) -> list[float]:
    length = end - start
    n = max(1, min(config.FRAMES_PER_SEGMENT, int(length / config.MIN_FRAME_GAP)))
    step = length / n
    return [round(start + step * (k + 0.5), 2) for k in range(n)]


def extract_frame(video: Path, t: float, out: Path) -> bool:
    if out.exists():
        return True
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
            "-vf", f"scale='min({config.FRAME_WIDTH},iw)':-2", "-q:v", "4", str(out),
        ],
        check=False,
    )
    return out.exists()


def extract_frames(video: Path, vdir: Path, segments: list[dict], job: Job) -> list[dict]:
    fdir = vdir / "frames"
    fdir.mkdir(exist_ok=True)
    jobs = [(seg["index"], t) for seg in segments for t in frame_times(seg["start"], seg["end"])]
    frames: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for seg_idx, t in jobs:
            name = f"{t:010.2f}.jpg"
            futures[pool.submit(extract_frame, video, t, fdir / name)] = (seg_idx, t, name)
        for fut in as_completed(futures):
            seg_idx, t, name = futures[fut]
            if fut.result():
                frames.append({"segment": seg_idx, "time": t, "file": name})
            done += 1
            if done % 10 == 0:
                job.update(progress=done / len(jobs))
    frames.sort(key=lambda f: f["time"])
    return frames


# ---------------------------------------------------------------- stage 4: analyze chunks


def transcript_text(transcript: list[dict], start: float | None = None, end: float | None = None) -> str:
    lines = [
        f"[{fmt_ts(t['start'])}] {t['text']}"
        for t in transcript
        if (start is None or t["end"] > start) and (end is None or t["start"] < end)
    ]
    return "\n".join(lines) or "(no speech detected)"


SEGMENT_SYSTEM = """You are watching one chunk of a longer video, frame by frame, alongside its speech transcript.
Describe what actually happens in this chunk as precisely as you can: who and what is on screen, actions,
settings, scene changes, on-screen text, and what is said. Use absolute timestamps in seconds from the start
of the whole video (the frame labels give them). Only describe what the frames and transcript support; if
something is unclear, say so instead of guessing. Frames are samples, so action between frames is unknown.

In `steps`, list every distinct action performed or shown, in order, as someone would need it to reproduce
what happens: what is done, to what, with what, in which order, with quantities, settings, clicks, commands,
or values exactly as seen or said. A step that started in an earlier chunk or continues past this one still
gets listed with this chunk's part of it. Leave `steps` empty only if nothing is actually done (for example
pure scenery or a monologue with no actions)."""


def analyze_segment(vdir: Path, seg: dict, frames: list[dict], transcript: list[dict], duration: float) -> dict:
    content: list[dict] = [{
        "type": "text",
        "text": (
            f"This chunk covers {fmt_ts(seg['start'])} to {fmt_ts(seg['end'])} "
            f"(seconds {seg['start']:.0f} to {seg['end']:.0f}) of a {fmt_ts(duration)} video. "
            f"Here are {len(frames)} frames sampled from it, in order."
        ),
    }]
    for f in frames:
        data = base64.standard_b64encode((vdir / "frames" / f["file"]).read_bytes()).decode()
        content.append({"type": "text", "text": f"Frame at {fmt_ts(f['time'])} (t={f['time']:.1f}s):"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
    content.append({
        "type": "text",
        "text": "Speech transcript for this chunk:\n"
        + transcript_text(transcript, seg["start"], seg["end"])
        + "\n\nWrite the notes for this chunk.",
    })
    notes = llm.parse(SEGMENT_SYSTEM, [{"role": "user", "content": content}], SegmentNotes)
    return {"index": seg["index"], "start": seg["start"], "end": seg["end"], **notes.model_dump()}


# ---------------------------------------------------------------- stage 5: synthesize

SYNTH_SYSTEM = """You have detailed notes on every chunk of a video, written by someone who watched each chunk,
plus the full speech transcript. Write the definitive analysis of the whole video: what it is, what happens,
how it is structured, who is in it, and what a viewer should take away. Chapters must cover the whole video
in order with no gaps and use seconds from the start. Quotes must be verbatim from the transcript. Be specific
and concrete; this report replaces watching the video.

`step_by_step` is the complete ordered procedure shown in the video, merged across chunks (a step split over
two chunks becomes one step). Write each step's instructions so someone could follow them without the video:
exact actions, values, settings, quantities, commands and on-screen labels. Steps must be in order and their
time ranges must point to where each step is shown. For a video that is not a process (a vlog, a film, a
talk), give the sequence of events in the same format instead."""


def synthesize(meta: dict, notes: list[dict], transcript: list[dict]) -> dict:
    text = (
        f"Video: {meta['filename']}, duration {fmt_ts(meta['duration'])} ({meta['duration']:.0f}s).\n\n"
        f"<chunk_notes>\n{json.dumps(notes, ensure_ascii=False)}\n</chunk_notes>\n\n"
        f"<transcript>\n{transcript_text(transcript)}\n</transcript>\n\nWrite the full analysis."
    )
    report = llm.parse(SYNTH_SYSTEM, [{"role": "user", "content": text}], VideoReport)
    return report.model_dump()


# ---------------------------------------------------------------- orchestration


def run(video_id: str) -> None:
    vdir = config.VIDEOS_DIR / video_id
    job = Job(vdir)
    meta = read_json(vdir / "meta.json")
    video = vdir / meta["stored_name"]
    try:
        job.update(state="running", stage="probing", progress=0, error=None)
        if "duration" not in meta:
            meta.update(probe(video))
            write_json(vdir / "meta.json", meta)
        duration = meta["duration"]

        transcript = read_json(vdir / "transcript.json")
        if transcript is None:
            job.update(stage="transcribing", progress=0)
            transcript = transcribe(video, vdir, duration, job) if meta["has_audio"] else []
            write_json(vdir / "transcript.json", transcript)

        segments = plan_segments(duration)
        frames = read_json(vdir / "frames.json")
        if frames is None:
            job.update(stage="sampling frames", progress=0)
            frames = extract_frames(video, vdir, segments, job)
            write_json(vdir / "frames.json", frames)

        ndir = vdir / "segments"
        ndir.mkdir(exist_ok=True)
        todo = [s for s in segments if not (ndir / f"{s['index']:04d}.json").exists()]
        job.update(stage="watching", progress=(len(segments) - len(todo)) / len(segments),
                   segments_total=len(segments))
        done = len(segments) - len(todo)
        with ThreadPoolExecutor(max_workers=config.PARALLEL_SEGMENTS) as pool:
            futures = {
                pool.submit(
                    analyze_segment, vdir, s, [f for f in frames if f["segment"] == s["index"]],
                    transcript, duration,
                ): s
                for s in todo
            }
            for fut in as_completed(futures):
                s = futures[fut]
                write_json(ndir / f"{s['index']:04d}.json", fut.result())
                done += 1
                job.update(progress=done / len(segments), segments_done=done)
        notes = [read_json(ndir / f"{s['index']:04d}.json") for s in segments]

        job.update(stage="writing report", progress=0)
        report = synthesize(meta, notes, transcript)
        write_json(vdir / "report.json", report)
        job.update(state="done", stage="done", progress=1)
    except Exception as exc:  # surface any failure to the UI
        traceback.print_exc()
        job.update(state="error", error=f"{type(exc).__name__}: {exc}")


def delete(video_id: str) -> None:
    shutil.rmtree(config.VIDEOS_DIR / video_id, ignore_errors=True)
