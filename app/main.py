"""FastAPI server: upload, job status, results, Q&A, and the web UI."""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import clips, config, export, llm, pipeline
from .pipeline import fmt_ts, read_json, write_json

app = FastAPI(title="Vid Explains")
STATIC = Path(__file__).resolve().parent.parent / "static"
ALLOWED = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ts"}

_running: set[str] = set()
_running_lock = threading.Lock()


def _start(video_id: str) -> None:
    with _running_lock:
        if video_id in _running:
            return
        _running.add(video_id)

    def work():
        try:
            pipeline.run(video_id)
        finally:
            with _running_lock:
                _running.discard(video_id)

    threading.Thread(target=work, daemon=True).start()


def _vdir(video_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12}", video_id):
        raise HTTPException(404, "Unknown video")
    vdir = config.VIDEOS_DIR / video_id
    if not (vdir / "meta.json").exists():
        raise HTTPException(404, "Unknown video")
    return vdir


@app.on_event("startup")
def resume_unfinished() -> None:
    """Pick interrupted jobs back up after a restart; finished stages are skipped."""
    for vdir in config.VIDEOS_DIR.iterdir():
        status = read_json(vdir / "status.json", {})
        if status.get("state") in ("queued", "running"):
            _start(vdir.name)


# ---------------------------------------------------------------- upload


@app.post("/api/videos")
async def upload(request: Request, filename: str):
    """Streams the raw request body straight to disk, so multi-GB files never sit in memory."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type {ext or '(none)'}")
    video_id = uuid.uuid4().hex[:12]
    vdir = config.VIDEOS_DIR / video_id
    vdir.mkdir(parents=True)
    stored = f"video{ext}"
    size = 0
    try:
        with open(vdir / stored, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                size += len(chunk)
    except Exception:
        pipeline.delete(video_id)
        raise HTTPException(400, "Upload was interrupted")
    if size == 0:
        pipeline.delete(video_id)
        raise HTTPException(400, "Empty file")
    write_json(vdir / "meta.json", {
        "id": video_id, "filename": Path(filename).name, "stored_name": stored, "uploaded_at": time.time(),
    })
    pipeline.Job(vdir).update(state="queued", stage="queued", progress=0)
    _start(video_id)
    return {"id": video_id}


# ---------------------------------------------------------------- read


@app.get("/api/videos")
def list_videos():
    items = []
    for vdir in config.VIDEOS_DIR.iterdir():
        meta = read_json(vdir / "meta.json")
        if not meta:
            continue
        report = read_json(vdir / "report.json") or {}
        items.append({**meta, "status": read_json(vdir / "status.json", {}), "title": report.get("title")})
    items.sort(key=lambda m: m.get("uploaded_at", 0), reverse=True)
    return items


@app.get("/api/videos/{video_id}")
def get_video(video_id: str):
    vdir = _vdir(video_id)
    seg_dir = vdir / "segments"
    segments = sorted(
        (read_json(p) for p in seg_dir.glob("*.json")), key=lambda s: s["index"]
    ) if seg_dir.exists() else []
    return {
        "meta": read_json(vdir / "meta.json"),
        "status": read_json(vdir / "status.json", {}),
        "report": read_json(vdir / "report.json"),
        "segments": segments,
        "transcript": read_json(vdir / "transcript.json"),
        "frames": read_json(vdir / "frames.json"),
    }


@app.get("/api/videos/{video_id}/file")
def video_file(video_id: str):
    vdir = _vdir(video_id)
    return FileResponse(vdir / read_json(vdir / "meta.json")["stored_name"])  # supports range requests


@app.get("/api/videos/{video_id}/frames/{name}")
def frame(video_id: str, name: str):
    if not re.fullmatch(r"[0-9.]+\.jpg", name):
        raise HTTPException(404)
    path = _vdir(video_id) / "frames" / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/videos/{video_id}/retry")
def retry(video_id: str):
    vdir = _vdir(video_id)
    pipeline.Job(vdir).update(state="queued", stage="queued", error=None)
    _start(video_id)
    return {"ok": True}


@app.post("/api/videos/{video_id}/reanalyze")
def reanalyze(video_id: str):
    """Redo the Claude part (chunk notes + report), keeping the transcript and frames."""
    vdir = _vdir(video_id)
    if video_id in _running:
        raise HTTPException(409, "Still processing")
    shutil.rmtree(vdir / "segments", ignore_errors=True)
    (vdir / "report.json").unlink(missing_ok=True)
    pipeline.Job(vdir).update(state="queued", stage="queued", error=None)
    _start(video_id)
    return {"ok": True}


@app.delete("/api/videos/{video_id}")
def delete(video_id: str):
    _vdir(video_id)
    if video_id in _running:
        raise HTTPException(409, "Still processing")
    pipeline.delete(video_id)
    return {"ok": True}


# ---------------------------------------------------------------- export


@app.get("/api/videos/{video_id}/export")
def export_for_ai(video_id: str):
    """Zip with transcript + labeled frame sheets + prompt, to hand to claude.ai or another chat AI."""
    vdir = _vdir(video_id)
    try:
        name, data = export.build(vdir)
    except ValueError as e:
        raise HTTPException(409, str(e))
    from urllib.parse import quote
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename*=utf-8''{quote(name)}"})


# ---------------------------------------------------------------- clips


@app.get("/api/videos/{video_id}/clip")
def clip(video_id: str, start: float, end: float, name: str = "clip", precise: bool = True):
    """Cut [start, end] out of the video and download it as an MP4."""
    vdir = _vdir(video_id)
    meta = read_json(vdir / "meta.json")
    duration = meta.get("duration") or pipeline.probe(vdir / meta["stored_name"])["duration"]
    start, end = max(0.0, start), min(end, duration)
    if end - start < 0.1:
        raise HTTPException(400, "The clip's end must be after its start")
    path = clips.cut(vdir / meta["stored_name"], start, end, clips.clip_path(vdir, start, end, precise), precise)
    filename = f"{clips.safe_name(name)} [{fmt_ts(start).replace(':', '.')}-{fmt_ts(end).replace(':', '.')}].mp4"
    return FileResponse(path, media_type="video/mp4", filename=filename)


class SplitBody(BaseModel):
    by: str = "chapters"  # chapters | steps | scenes
    precise: bool = True


@app.post("/api/videos/{video_id}/split")
def split(video_id: str, body: SplitBody):
    """Start cutting the whole video into its chapters, steps or scenes, bundled as a zip."""
    vdir = _vdir(video_id)
    if body.by not in ("chapters", "steps", "scenes"):
        raise HTTPException(400, "by must be chapters, steps or scenes")
    if not (vdir / "report.json").exists():
        raise HTTPException(409, "The video is still being analyzed")
    if (read_json(vdir / "status.json", {}).get("split") or {}).get("state") == "running":
        raise HTTPException(409, "A split is already running for this video")
    pipeline.Job(vdir).update(split={"by": body.by, "state": "running", "done": 0, "total": 0})
    threading.Thread(target=clips.split_all, args=(video_id, body.by, body.precise), daemon=True).start()
    return {"ok": True}


@app.get("/api/videos/{video_id}/split/{name}")
def split_download(video_id: str, name: str):
    path = _vdir(video_id) / "clips" / name
    if "/" in name or ".." in name or not name.endswith(".zip") or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="application/zip", filename=name)


# ---------------------------------------------------------------- Q&A


class AskBody(BaseModel):
    question: str


def _qa_context(vdir: Path) -> str:
    meta = read_json(vdir / "meta.json")
    report = read_json(vdir / "report.json")
    notes = [read_json(p) for p in sorted((vdir / "segments").glob("*.json"))]
    transcript = read_json(vdir / "transcript.json", [])
    return (
        f"Video file: {meta['filename']} ({fmt_ts(meta['duration'])})\n\n"
        f"<report>\n{json.dumps(report, ensure_ascii=False)}\n</report>\n\n"
        f"<chunk_notes>\n{json.dumps(notes, ensure_ascii=False)}\n</chunk_notes>\n\n"
        f"<transcript>\n{pipeline.transcript_text(transcript)}\n</transcript>"
    )


@app.get("/api/videos/{video_id}/chat")
def chat_history(video_id: str):
    return read_json(_vdir(video_id) / "chat.json", [])


@app.post("/api/videos/{video_id}/ask")
def ask(video_id: str, body: AskBody):
    vdir = _vdir(video_id)
    if not (vdir / "report.json").exists():
        raise HTTPException(409, "The video is still being analyzed")
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Empty question")
    history = read_json(vdir / "chat.json", [])
    try:
        answer = llm.ask(_qa_context(vdir), history[-20:], question)
    except llm.RefusalError as e:
        raise HTTPException(422, str(e))
    except llm.AIError as e:
        raise HTTPException(502, str(e))
    except Exception as e:  # provider SDK errors (network, auth, ...)
        raise HTTPException(502, f"AI error: {type(e).__name__}: {e}")
    history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    write_json(vdir / "chat.json", history)
    return {"answer": answer}


@app.delete("/api/videos/{video_id}/chat")
def clear_chat(video_id: str):
    (_vdir(video_id) / "chat.json").unlink(missing_ok=True)
    return {"ok": True}


# ---------------------------------------------------------------- UI


@app.get("/")
def index():
    # no-store so the browser always picks up a new version of the page after an update
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")
