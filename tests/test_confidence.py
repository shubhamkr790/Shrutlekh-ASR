import pytest

from shrutlekh.confidence import ConfidenceThresholds, classify, segment_flags, word_level
from shrutlekh.models import Segment


@pytest.mark.parametrize("p,level", [(0.99, "high"), (0.80, "high"), (0.79, "medium"), (0.50, "medium"), (0.49, "low"), (0.0, "low")])
def test_word_levels_use_default_thresholds(p, level):
    assert word_level(p) == level


def test_custom_thresholds():
    strict = ConfidenceThresholds(high=0.95, low=0.7)
    assert word_level(0.9, strict) == "medium"
    assert word_level(0.6, strict) == "low"


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        ConfidenceThresholds(high=0.4, low=0.6)


def test_summary_counts(transcript):
    s = transcript.summary
    assert s["words"] == 8
    assert (s["high"], s["medium"], s["low"]) == (6, 1, 1)
    assert s["review_ratio"] == pytest.approx(0.25)


def test_edited_word_counts_as_reviewed(transcript):
    w = transcript.segments[0].words[4]
    w.text, w.edited = " amoxicillin.", True
    classify(transcript)
    assert w.level == "high"
    assert transcript.summary["low"] == 0
    assert transcript.summary["edited_words"] == 1


def test_segment_flags():
    seg = Segment(id=0, start=0, end=1, text="la la la la la", compression_ratio=3.1, avg_logprob=-1.4, no_speech_prob=0.8)
    assert segment_flags(seg) == ["repetitive_text", "low_average_confidence", "possibly_no_speech"]
    ok = Segment(id=1, start=0, end=1, text="fine", compression_ratio=1.2, avg_logprob=-0.2, no_speech_prob=0.01)
    assert segment_flags(ok) == []
