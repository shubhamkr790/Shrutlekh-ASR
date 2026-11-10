"""Turns Whisper's raw numbers into something a person can act on.

Every word gets a level:
  high    - probably right, no need to look
  medium  - worth a quick check
  low     - likely wrong, listen again

Every segment can also get warning flags based on Whisper's own segment
statistics (very repetitive text, low average log-probability, or audio that
looks like silence).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Segment, Transcript


@dataclass(frozen=True)
class ConfidenceThresholds:
    high: float = 0.80  # word probability at or above this is "high"
    low: float = 0.50  # word probability below this is "low"; in between is "medium"
    max_compression_ratio: float = 2.4  # Whisper's own repetition / hallucination signal
    min_avg_logprob: float = -1.0
    max_no_speech_prob: float = 0.6

    def __post_init__(self) -> None:
        if not 0.0 <= self.low <= self.high <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= low <= high <= 1")


DEFAULT_THRESHOLDS = ConfidenceThresholds()


def word_level(probability: float, t: ConfidenceThresholds = DEFAULT_THRESHOLDS) -> str:
    if probability >= t.high:
        return "high"
    if probability >= t.low:
        return "medium"
    return "low"


def segment_flags(segment: Segment, t: ConfidenceThresholds = DEFAULT_THRESHOLDS) -> list[str]:
    flags = []
    if segment.compression_ratio > t.max_compression_ratio:
        flags.append("repetitive_text")
    if segment.avg_logprob < t.min_avg_logprob:
        flags.append("low_average_confidence")
    if segment.no_speech_prob > t.max_no_speech_prob:
        flags.append("possibly_no_speech")
    return flags


def summarize(transcript: Transcript) -> dict[str, Any]:
    words = transcript.words
    counts = {"high": 0, "medium": 0, "low": 0}
    for w in words:
        counts[w.level] = counts.get(w.level, 0) + 1
    total = len(words)
    to_review = counts["medium"] + counts["low"]
    mean_prob = sum(w.probability for w in words) / total if total else 0.0
    flagged_segments = sum(1 for s in transcript.segments if s.flags)
    return {
        "words": total,
        "high": counts["high"],
        "medium": counts["medium"],
        "low": counts["low"],
        "review_ratio": round(to_review / total, 4) if total else 0.0,
        "mean_probability": round(mean_prob, 4),
        "flagged_segments": flagged_segments,
        "edited_words": sum(1 for w in words if w.edited),
    }


def classify(transcript: Transcript, t: ConfidenceThresholds = DEFAULT_THRESHOLDS) -> Transcript:
    """Set a level on every word and flags on every segment, then fill the summary."""
    for seg in transcript.segments:
        for w in seg.words:
            # a word a person has corrected is trusted from then on
            w.level = "high" if w.edited else word_level(w.probability, t)
        seg.flags = segment_flags(seg, t)
    transcript.summary = summarize(transcript)
    return transcript
