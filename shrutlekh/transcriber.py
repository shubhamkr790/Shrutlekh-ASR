"""Speech-to-text with faster-whisper, keeping per-word probabilities.

faster-whisper runs OpenAI's Whisper models through CTranslate2, which is
about 4x faster than the reference implementation on CPU and needs far less
memory when weights are stored as 8-bit integers (int8).
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np

from .confidence import DEFAULT_THRESHOLDS, ConfidenceThresholds, classify
from .models import Segment, Transcript, Word

SAMPLE_RATE = 16_000
MODEL_SIZES = ("tiny", "base", "small")
LANGUAGES = {"auto": None, "en": "en", "hi": "hi"}

AudioInput = Union[str, Path, BinaryIO, np.ndarray]


def load_audio(source: AudioInput) -> np.ndarray:
    """Decode any format FFmpeg understands (wav, mp3, m4a, ogg, webm...) to 16 kHz mono float32."""
    if isinstance(source, np.ndarray):
        return source.astype(np.float32)
    from faster_whisper import decode_audio

    return decode_audio(str(source) if isinstance(source, Path) else source, sampling_rate=SAMPLE_RATE)


class Transcriber:
    """Loads each Whisper model once and reuses it. Safe to share between requests."""

    def __init__(self, device: str = "cpu", compute_type: str = "int8") -> None:
        self.device = device
        self.compute_type = compute_type
        self._models: dict[str, object] = {}
        self._lock = threading.Lock()

    def _model(self, size: str):
        if size not in MODEL_SIZES:
            raise ValueError(f"unknown model {size!r}; choose from {MODEL_SIZES}")
        with self._lock:
            if size not in self._models:
                from faster_whisper import WhisperModel

                self._models[size] = WhisperModel(size, device=self.device, compute_type=self.compute_type)
            return self._models[size]

    def transcribe(
        self,
        audio: AudioInput,
        model: str = "base",
        language: str = "auto",
        thresholds: ConfidenceThresholds = DEFAULT_THRESHOLDS,
        vad: bool = True,
        beam_size: int = 5,
    ) -> Transcript:
        if language not in LANGUAGES:
            raise ValueError(f"unknown language {language!r}; choose from {list(LANGUAGES)}")
        samples = load_audio(audio)
        whisper = self._model(model)
        started = time.perf_counter()
        seg_iter, info = whisper.transcribe(
            samples,
            language=LANGUAGES[language],
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=vad,
            condition_on_previous_text=False,  # stops one mistake from repeating through the rest
        )
        segments = []
        for i, s in enumerate(seg_iter):  # the iterator is lazy: decoding happens here
            words = [
                Word(text=w.word, start=float(w.start), end=float(w.end), probability=float(w.probability))
                for w in (s.words or [])
            ]
            segments.append(
                Segment(
                    id=i,
                    start=float(s.start),
                    end=float(s.end),
                    text=s.text.strip(),
                    words=words,
                    avg_logprob=float(s.avg_logprob),
                    no_speech_prob=float(s.no_speech_prob),
                    compression_ratio=float(s.compression_ratio),
                )
            )
        transcript = Transcript(
            segments=segments,
            language=info.language,
            language_probability=float(info.language_probability),
            duration=float(len(samples) / SAMPLE_RATE),
            model=f"whisper-{model} ({self.compute_type})",
            processing_seconds=time.perf_counter() - started,
            id=uuid.uuid4().hex[:12],
        )
        return classify(transcript, thresholds)
