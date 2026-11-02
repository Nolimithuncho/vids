"""Cutting clips out of a video and splitting it into sections."""
from __future__ import annotations

import re
import subprocess
import zipfile
from pathlib import Path

from . import pipeline
from .pipeline import read_json


def safe_name(text: str, limit: int = 60) -> str:
    text = re.sub(r"[^\w\s.-]", "", text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit].strip() or "clip"


def cut(video: Path, start: float, end: float, out: Path, precise: bool = True) -> Path:
    """Cut [start, end) seconds into `out`.

    precise=True re-encodes so the clip starts on the exact frame (slower).
    precise=False copies the streams without re-encoding (near-instant even on huge files),
    but the start snaps to the nearest keyframe before `start`, often a few seconds early.
    """
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end - start)
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    if precise:
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k"]
    else:
        codec = ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{duration:.3f}",
         "-map", "0:v:0", "-map", "0:a:0?", *codec, "-movflags", "+faststart", str(tmp)],
        check=True,
    )
    tmp.replace(out)
    return out


def clip_path(vdir: Path, start: float, end: float, precise: bool) -> Path:
    return vdir / "clips" / f"{start:010.2f}-{end:010.2f}-{'p' if precise else 'f'}.mp4"


def sections(vdir: Path, by: str) -> list[dict]:
    """The list of sections to split by: chapters, steps, or scenes."""
    report = read_json(vdir / "report.json") or {}
    if by == "chapters":
        return [{"start": c["start_seconds"], "end": c["end_seconds"], "title": c["title"]}
                for c in report.get("chapters", [])]
    if by == "steps":
        return [{"start": s["start_seconds"], "end": s["end_seconds"], "title": f"Step {s['number']} {s['title']}"}
                for s in report.get("step_by_step", [])]
    if by == "scenes":
        out = []
        for p in sorted((vdir / "segments").glob("*.json")):
            out += [{"start": s["start_seconds"], "end": s["end_seconds"], "title": s["title"]}
                    for s in read_json(p)["scenes"]]
        return out
    raise ValueError(f"Unknown section type {by}")


def split_all(video_id: str, by: str, precise: bool) -> None:
    """Cut every section and bundle them into one zip. Progress goes to status.json under `split`."""
    from . import config

    vdir = config.VIDEOS_DIR / video_id
    job = pipeline.Job(vdir)
    meta = read_json(vdir / "meta.json")
    video = vdir / meta["stored_name"]
    items = [s for s in sections(vdir, by) if s["end"] > s["start"]]
    zip_name = f"{safe_name(Path(meta['filename']).stem)} - {by}{'' if precise else ' (fast)'}.zip"
    state = {"by": by, "precise": precise, "state": "running", "done": 0, "total": len(items), "file": None, "error": None}
    job.update(split=state)
    try:
        zpath = vdir / "clips" / zip_name
        zpath.parent.mkdir(exist_ok=True)
        tmp = zpath.with_suffix(".part")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for i, s in enumerate(items, 1):
                clip = cut(video, s["start"], min(s["end"], meta["duration"]), clip_path(vdir, s["start"], s["end"], precise), precise)
                z.write(clip, f"{i:02d} - {safe_name(s['title'])} [{pipeline.fmt_ts(s['start']).replace(':', '.')}].mp4")
                state["done"] = i
                job.update(split=state)
        tmp.replace(zpath)
        state.update(state="done", file=zip_name)
    except Exception as exc:
        state.update(state="error", error=f"{type(exc).__name__}: {exc}")
    job.update(split=state)
