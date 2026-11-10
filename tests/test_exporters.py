import json

import pytest

from shrutlekh.exporters import export
from shrutlekh.models import Transcript


def test_txt(transcript):
    assert export(transcript, "txt") == "The doctor prescribed a moxicillin.\nTwice a day.\n"


def test_srt_numbering_and_comma_timestamps(transcript):
    srt = export(transcript, "srt")
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,500\nThe doctor prescribed a moxicillin.\n")
    assert "2\n00:00:02,600 --> 00:00:04,000\nTwice a day." in srt


def test_vtt_header_and_dot_timestamps(transcript):
    vtt = export(transcript, "vtt")
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:02.500\n")


def test_json_round_trip(transcript):
    data = json.loads(export(transcript, "json"))
    again = Transcript.from_dict(data)
    assert again.text == transcript.text
    assert [w.level for w in again.words] == [w.level for w in transcript.words]


def test_review_report_lists_only_flagged_words(transcript):
    md = export(transcript, "md")
    assert "| a | 62% | medium |" in md
    assert "| moxicillin. | 31% | low |" in md
    assert "| doctor |" not in md


def test_edited_text_is_exported(transcript):
    transcript.segments[0].words[4].text = " amoxicillin."
    assert "amoxicillin." in export(transcript, "txt").split()[4]


def test_unknown_format(transcript):
    with pytest.raises(ValueError):
        export(transcript, "docx")
