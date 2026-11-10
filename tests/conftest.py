import os
from pathlib import Path

import pytest

from shrutlekh.confidence import classify
from shrutlekh.models import Segment, Transcript, Word

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def make_transcript() -> Transcript:
    """A small hand-made transcript with one medium and one low word."""
    seg1 = Segment(
        id=0, start=0.0, end=2.5, text="The doctor prescribed a moxicillin.",
        words=[
            Word(" The", 0.0, 0.2, 0.99), Word(" doctor", 0.2, 0.6, 0.95),
            Word(" prescribed", 0.6, 1.2, 0.91), Word(" a", 1.2, 1.3, 0.62),
            Word(" moxicillin.", 1.3, 2.5, 0.31),
        ],
        avg_logprob=-0.3, no_speech_prob=0.01, compression_ratio=1.1,
    )
    seg2 = Segment(
        id=1, start=2.6, end=4.0, text="Twice a day.",
        words=[Word(" Twice", 2.6, 3.0, 0.97), Word(" a", 3.0, 3.1, 0.98), Word(" day.", 3.1, 4.0, 0.99)],
        avg_logprob=-0.2, no_speech_prob=0.02, compression_ratio=1.0,
    )
    return classify(Transcript(segments=[seg1, seg2], language="en", duration=4.0, model="fake", id="abc123"))


class FakeTranscriber:
    """Stands in for Whisper so API tests run in milliseconds without downloading a model."""

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, model="base", language="auto", thresholds=None, **_):
        self.calls.append({"audio": audio, "model": model, "language": language, "thresholds": thresholds})
        t = make_transcript()
        return classify(t, thresholds) if thresholds else t


@pytest.fixture
def transcript() -> Transcript:
    return make_transcript()


@pytest.fixture
def fake() -> FakeTranscriber:
    return FakeTranscriber()
