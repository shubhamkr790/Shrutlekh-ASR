"""Export a transcript as TXT, SRT, VTT, JSON or a review report (Markdown)."""

from __future__ import annotations

import json

from .models import Transcript

FORMATS = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "json": "application/json",
    "md": "text/markdown; charset=utf-8",
}


def _clock(seconds: float, sep: str) -> str:
    ms = int(round(max(seconds, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _segment_text(seg) -> str:
    if seg.words:
        return "".join(w.text for w in seg.words).strip()
    return seg.text.strip()


def to_txt(t: Transcript) -> str:
    return "\n".join(_segment_text(s) for s in t.segments if _segment_text(s)) + "\n"


def to_srt(t: Transcript) -> str:
    blocks = []
    for n, s in enumerate((s for s in t.segments if _segment_text(s)), start=1):
        blocks.append(f"{n}\n{_clock(s.start, ',')} --> {_clock(s.end, ',')}\n{_segment_text(s)}\n")
    return "\n".join(blocks)


def to_vtt(t: Transcript) -> str:
    out = ["WEBVTT", ""]
    for s in t.segments:
        if _segment_text(s):
            out += [f"{_clock(s.start, '.')} --> {_clock(s.end, '.')}", _segment_text(s), ""]
    return "\n".join(out)


def to_json(t: Transcript) -> str:
    return json.dumps(t.to_dict(), ensure_ascii=False, indent=2)


def to_review_md(t: Transcript) -> str:
    """A checklist of the words worth listening to again, with timestamps."""
    sm = t.summary
    lines = [
        "# Shrutlekh review report",
        "",
        f"- Model: `{t.model}`  |  Language: `{t.language}`  |  Duration: {t.duration:.1f} s",
        f"- Words: {sm.get('words', 0)}  |  To check: {sm.get('medium', 0)}  |  Likely wrong: {sm.get('low', 0)}",
        "",
        "| Time | Word | Confidence | Level | Context |",
        "|---|---|---|---|---|",
    ]
    for s in t.segments:
        for i, w in enumerate(s.words):
            if w.level == "high":
                continue
            ctx = "".join(x.text for x in s.words[max(0, i - 3): i + 4]).strip()
            lines.append(f"| {_clock(w.start, '.')[3:]} | {w.text.strip()} | {w.probability:.0%} | {w.level} | {ctx} |")
    if lines[-1].startswith("|---"):
        lines.append("| - | nothing to review | - | - | - |")
    flagged = [s for s in t.segments if s.flags]
    if flagged:
        lines += ["", "## Segments with warnings", ""]
        lines += [f"- {_clock(s.start, '.')[3:]}: {', '.join(s.flags)} - \"{_segment_text(s)}\"" for s in flagged]
    return "\n".join(lines) + "\n"


def export(t: Transcript, fmt: str) -> str:
    fmt = fmt.lower()
    writers = {"txt": to_txt, "srt": to_srt, "vtt": to_vtt, "json": to_json, "md": to_review_md}
    if fmt not in writers:
        raise ValueError(f"unknown format {fmt!r}; choose from {sorted(writers)}")
    return writers[fmt](t)
