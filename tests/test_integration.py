"""End-to-end tests with the real Whisper "tiny" model.

These download the model (~75 MB) on first run, so they are marked slow:
    pytest -m slow
"""

import pytest

from shrutlekh.benchmark import flag_quality, load_clips
from shrutlekh.metrics import wer
from shrutlekh.noise import add_noise
from shrutlekh.transcriber import Transcriber, load_audio
from tests.conftest import SAMPLES

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def engine():
    return Transcriber()


def test_clean_clip_is_transcribed_accurately(engine):
    clip = load_clips(SAMPLES)[0]
    t = engine.transcribe(str(clip.file), model="tiny", language="en")
    assert wer(clip.text, t.text) <= 0.10
    assert t.language == "en"
    assert all(w.end >= w.start for w in t.words)


def test_language_is_auto_detected(engine):
    t = engine.transcribe(str(SAMPLES / "02_asr.wav"), model="tiny")
    assert t.language == "en" and t.language_probability > 0.5


def test_heavy_noise_raises_wer_and_flags(engine):
    clip = load_clips(SAMPLES)[3]
    clean = load_audio(clip.file)
    t_clean = engine.transcribe(clean, model="tiny", language="en")
    t_noisy = engine.transcribe(add_noise(clean, -5.0, "white", seed=3), model="tiny", language="en")
    assert t_noisy.summary["review_ratio"] >= t_clean.summary["review_ratio"]
    assert wer(clip.text, t_noisy.text) >= wer(clip.text, t_clean.text)


def test_flag_quality_counts_every_word(engine):
    clip = load_clips(SAMPLES)[3]
    t = engine.transcribe(str(clip.file), model="tiny", language="en")
    counts = flag_quality(clip.text, t)
    assert sum(counts.values()) == len([w for w in t.words if w.text.strip(" .,")])
