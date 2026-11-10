"""FastAPI backend: serves the web app and the transcription API.

Endpoints
  GET  /api/health                  status, models and languages
  GET  /api/samples                 the bundled sample clips
  GET  /api/samples/{name}          a sample's audio
  POST /api/transcribe              upload audio (multipart) -> transcript JSON
  POST /api/transcribe/sample/{n}   transcribe a bundled sample
  POST /api/export?format=srt       transcript JSON (possibly edited) -> file
"""

from __future__ import annotations

import json
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .confidence import ConfidenceThresholds, classify
from .exporters import FORMATS, export
from .metrics import cer, word_errors
from .models import Transcript
from .transcriber import LANGUAGES, MODEL_SIZES, Transcriber

PACKAGE_DIR = Path(__file__).parent
WEB_DIR = PACKAGE_DIR / "web"
SAMPLES_DIR = PACKAGE_DIR.parent / "samples"
ASSETS_DIR = PACKAGE_DIR.parent / "assets"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".webm", ".flac", ".aac", ".mp4"}


class TranscriberLike(Protocol):
    def transcribe(self, audio: Any, model: str = ..., language: str = ..., thresholds: ConfidenceThresholds = ...) -> Transcript: ...


def _thresholds(high: float, low: float) -> ConfidenceThresholds:
    try:
        return ConfidenceThresholds(high=high, low=low)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


def _check_options(model: str, language: str) -> None:
    if model not in MODEL_SIZES:
        raise HTTPException(422, f"model must be one of {list(MODEL_SIZES)}")
    if language not in LANGUAGES:
        raise HTTPException(422, f"language must be one of {list(LANGUAGES)}")


def _sample_meta(samples_dir: Path) -> list[dict[str, Any]]:
    ref = samples_dir / "references.json"
    if not ref.exists():
        return []
    clips = json.loads(ref.read_text(encoding="utf-8"))["clips"]
    return [{"file": c["file"], "title": c.get("title", c["file"]), "reference": c["text"]}
            for c in clips if (samples_dir / c["file"]).exists()]


def create_app(transcriber: TranscriberLike | None = None, samples_dir: Path = SAMPLES_DIR) -> FastAPI:
    app = FastAPI(title="Shrutlekh", version=__version__, description="Confidence-aware speech-to-text")
    engine = transcriber or Transcriber()
    recent: OrderedDict[str, dict[str, Any]] = OrderedDict()  # last 20 results, for re-download

    def remember(t: Transcript) -> dict[str, Any]:
        d = t.to_dict()
        recent[t.id] = d
        while len(recent) > 20:
            recent.popitem(last=False)
        return d

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "models": list(MODEL_SIZES), "languages": list(LANGUAGES),
                "formats": list(FORMATS)}

    @app.get("/api/samples")
    def samples() -> list[dict[str, Any]]:
        return _sample_meta(samples_dir)

    @app.get("/api/samples/{name}")
    def sample_audio(name: str) -> FileResponse:
        names = {s["file"] for s in _sample_meta(samples_dir)}
        if name not in names:
            raise HTTPException(404, "no such sample")
        return FileResponse(samples_dir / name, media_type="audio/wav")

    @app.post("/api/transcribe")
    async def transcribe(
        file: UploadFile = File(...),
        model: str = Form("base"),
        language: str = Form("auto"),
        high: float = Form(0.80),
        low: float = Form(0.50),
    ) -> dict[str, Any]:
        _check_options(model, language)
        thresholds = _thresholds(high, low)
        suffix = Path(file.filename or "audio.webm").suffix.lower() or ".webm"
        if suffix not in AUDIO_SUFFIXES:
            raise HTTPException(415, f"unsupported file type {suffix}")
        data = await file.read()
        if not data:
            raise HTTPException(400, "the uploaded file is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file is larger than 50 MB")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(data)
            try:
                result = engine.transcribe(str(path), model=model, language=language, thresholds=thresholds)
            except (ValueError, RuntimeError, OSError) as e:
                raise HTTPException(422, f"could not process this audio: {e}") from e
        return remember(result)

    @app.post("/api/transcribe/sample/{name}")
    def transcribe_sample(
        name: str, model: str = Query("base"), language: str = Query("auto"),
        high: float = Query(0.80), low: float = Query(0.50),
    ) -> dict[str, Any]:
        _check_options(model, language)
        meta = {s["file"]: s for s in _sample_meta(samples_dir)}
        if name not in meta:
            raise HTTPException(404, "no such sample")
        result = engine.transcribe(str(samples_dir / name), model=model, language=language,
                                   thresholds=_thresholds(high, low))
        d = remember(result)
        ref = meta[name]["reference"]
        errs = word_errors(ref, result.text)
        d["reference"] = ref
        d["evaluation"] = {
            "wer": round(errs.rate, 4), "cer": round(cer(ref, result.text), 4),
            "substitutions": errs.substitutions, "deletions": errs.deletions,
            "insertions": errs.insertions, "reference_words": errs.reference_length,
        }
        return d

    @app.post("/api/export")
    def export_transcript(body: dict[str, Any], format: str = Query("txt")) -> Response:
        fmt = format.lower()
        if fmt not in FORMATS:
            raise HTTPException(422, f"format must be one of {list(FORMATS)}")
        try:
            t = classify(Transcript.from_dict(body))  # re-classify so edited words count as reviewed
        except (TypeError, ValueError) as e:
            raise HTTPException(422, f"invalid transcript: {e}") from e
        name = f"shrutlekh-{t.id or 'transcript'}.{fmt}"
        return Response(export(t, fmt), media_type=FORMATS[fmt],
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/transcripts/{tid}")
    def get_transcript(tid: str) -> dict[str, Any]:
        if tid not in recent:
            raise HTTPException(404, "transcript not found (only the last 20 are kept)")
        return recent[tid]

    if ASSETS_DIR.exists():
        app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
