"""'Download for Claude': a small zip you can drop into claude.ai (or any chat AI) instead of the video.

Contains the timestamped transcript, the sampled frames laid out on labeled picture sheets, the
analysis if one exists, and a ready-made prompt. Needs only the transcript and frames, so it works
even when the AI step hasn't run.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .pipeline import fmt_ts, read_json, transcript_text

COLS, ROWS = 3, 4          # 12 frames per sheet
CELL_W = 512               # 3 x 512 = 1536 px wide, under Claude's ~1568 px resize limit
LABEL_H = 34
MAX_SHEETS = 15            # chat apps accept about 20 files per message


def _pick(frames: list[dict], limit: int) -> list[dict]:
    if len(frames) <= limit:
        return frames
    step = len(frames) / limit
    return [frames[int(i * step)] for i in range(limit)]


def _sheets(vdir: Path, frames: list[dict]) -> list[tuple[str, bytes]]:
    per_sheet = COLS * ROWS
    frames = _pick(frames, per_sheet * MAX_SHEETS)
    font = ImageFont.load_default(size=24)
    out = []
    for n in range(0, len(frames), per_sheet):
        batch = frames[n:n + per_sheet]
        thumbs = []
        for f in batch:
            img = Image.open(vdir / "frames" / f["file"]).convert("RGB")
            img.thumbnail((CELL_W, CELL_W))
            thumbs.append((f, img))
        cell_h = max(img.height for _, img in thumbs) + LABEL_H
        rows = (len(thumbs) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * CELL_W, rows * cell_h), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (f, img) in enumerate(thumbs):
            x, y = (i % COLS) * CELL_W, (i // COLS) * cell_h
            draw.rectangle([x, y, x + CELL_W - 1, y + LABEL_H - 1], fill="black")
            draw.text((x + 8, y + 4), f"[{fmt_ts(f['time'])}]", fill="yellow", font=font)
            sheet.paste(img, (x + (CELL_W - img.width) // 2, y + LABEL_H))
        buf = io.BytesIO()
        sheet.save(buf, "JPEG", quality=82)
        first, last = batch[0]["time"], batch[-1]["time"]
        name = f"frames-{len(out) + 1:02d} ({fmt_ts(first).replace(':', '.')} to {fmt_ts(last).replace(':', '.')}).jpg"
        out.append((name, buf.getvalue()))
    return out


PROMPT = """I want you to fully understand a video I can't upload directly. I'm giving you:

- transcript.txt: everything said in the video, with [minutes:seconds] timestamps.
- {n_sheets} picture sheets (frames-01.jpg, frames-02.jpg, ...): still frames taken about every {gap} seconds,
  in order, each labeled with its [timestamp] in yellow.{report_line}

The video is "{name}" and is {duration} long.

Please go through the sheets in order alongside the transcript, as if you were watching the video, and give me:
1. A short summary of what the video is and what happens.
2. A detailed step-by-step breakdown of everything that is done or shown, in order, with timestamps and the
   exact values, settings, words on screen and actions, so I could follow it without the video.
3. Chapters (start time, title, one line each).
4. Anything important that is visible but not said, or said but not shown.

Then answer my questions about the video using this material. If something happens between frames and you
can't tell, say so instead of guessing.
"""


def build(vdir: Path) -> tuple[str, bytes]:
    meta = read_json(vdir / "meta.json")
    transcript = read_json(vdir / "transcript.json")
    frames = read_json(vdir / "frames.json")
    if transcript is None or frames is None:
        raise ValueError("The transcript and frames aren't ready yet. Wait until the 'watching' step starts.")
    report = read_json(vdir / "report.json")
    sheets = _sheets(vdir, frames)
    gap = round(meta["duration"] / max(1, len(frames)))
    stem = Path(meta["filename"]).stem
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("PROMPT - paste this into Claude.txt", PROMPT.format(
            n_sheets=len(sheets), gap=gap, name=meta["filename"], duration=fmt_ts(meta["duration"]),
            report_line="\n- analysis.json: an automatic analysis made earlier (chapters, steps, summary)." if report else "",
        ))
        z.writestr("transcript.txt", transcript_text(transcript) + "\n")
        if report:
            z.writestr("analysis.json", json.dumps(report, indent=2, ensure_ascii=False))
        for name, data in sheets:
            z.writestr(name, data)
    return f"{stem[:40]} - for Claude.zip", buf.getvalue()
