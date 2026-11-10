"""Plain data classes shared by the transcriber, exporters, server and CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float
    level: str = "high"  # high | medium | low (set by confidence.classify)
    edited: bool = False  # true when a person corrected the word in the UI

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["words"] = [w.to_dict() for w in self.words]
        return d


@dataclass
class Transcript:
    segments: list[Segment]
    language: str = "en"
    language_probability: float = 1.0
    duration: float = 0.0
    model: str = ""
    processing_seconds: float = 0.0
    summary: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "language": self.language,
            "language_probability": round(self.language_probability, 4),
            "duration": round(self.duration, 3),
            "model": self.model,
            "processing_seconds": round(self.processing_seconds, 3),
            "summary": self.summary,
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Transcript":
        segments = []
        for i, s in enumerate(d.get("segments", [])):
            words = [
                Word(
                    text=str(w.get("text", "")),
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    probability=float(w.get("probability", 1.0)),
                    level=str(w.get("level", "high")),
                    edited=bool(w.get("edited", False)),
                )
                for w in s.get("words", [])
            ]
            text = "".join(w.text for w in words).strip() if words else str(s.get("text", ""))
            segments.append(
                Segment(
                    id=int(s.get("id", i)),
                    start=float(s.get("start", 0.0)),
                    end=float(s.get("end", 0.0)),
                    text=text,
                    words=words,
                    avg_logprob=float(s.get("avg_logprob", 0.0)),
                    no_speech_prob=float(s.get("no_speech_prob", 0.0)),
                    compression_ratio=float(s.get("compression_ratio", 1.0)),
                    flags=list(s.get("flags", [])),
                )
            )
        return cls(
            segments=segments,
            language=str(d.get("language", "en")),
            language_probability=float(d.get("language_probability", 1.0)),
            duration=float(d.get("duration", 0.0)),
            model=str(d.get("model", "")),
            processing_seconds=float(d.get("processing_seconds", 0.0)),
            summary=dict(d.get("summary", {})),
            id=str(d.get("id", "")),
        )
