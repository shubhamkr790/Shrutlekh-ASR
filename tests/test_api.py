import io

import pytest
from fastapi.testclient import TestClient

from shrutlekh.server import create_app
from tests.conftest import SAMPLES, make_transcript


@pytest.fixture
def client(fake):
    return TestClient(create_app(transcriber=fake, samples_dir=SAMPLES))


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["models"] == ["tiny", "base", "small"]


def test_web_app_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Shrutlekh" in r.text


def test_samples_listed(client):
    names = [s["file"] for s in client.get("/api/samples").json()]
    assert "04_medicine.wav" in names and len(names) == 9


def test_upload_transcribes_and_passes_options(client, fake):
    r = client.post(
        "/api/transcribe",
        files={"file": ("talk.wav", io.BytesIO(b"RIFF....fake"), "audio/wav")},
        data={"model": "tiny", "language": "hi", "high": "0.9", "low": "0.4"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["words"] == 8
    call = fake.calls[-1]
    assert (call["model"], call["language"]) == ("tiny", "hi")
    assert call["thresholds"].high == 0.9 and call["thresholds"].low == 0.4


@pytest.mark.parametrize("data,code", [
    ({"model": "huge"}, 422),
    ({"language": "fr"}, 422),
    ({"high": "0.3", "low": "0.6"}, 422),
])
def test_bad_options_rejected(client, data, code):
    r = client.post("/api/transcribe", files={"file": ("a.wav", io.BytesIO(b"x"), "audio/wav")}, data=data)
    assert r.status_code == code


def test_unsupported_and_empty_files(client):
    assert client.post("/api/transcribe", files={"file": ("notes.pdf", io.BytesIO(b"x"))}).status_code == 415
    assert client.post("/api/transcribe", files={"file": ("a.wav", io.BytesIO(b""))}).status_code == 400


def test_sample_transcription_includes_accuracy(client):
    r = client.post("/api/transcribe/sample/04_medicine.wav?model=tiny")
    body = r.json()
    assert r.status_code == 200
    assert body["reference"].startswith("The doctor prescribed amoxicillin")
    assert 0 < body["evaluation"]["wer"] < 1


def test_sample_not_found_and_no_path_escape(client):
    assert client.post("/api/transcribe/sample/missing.wav").status_code == 404
    assert client.get("/api/samples/references.json").status_code == 404  # only listed clips are served
    assert client.get("/api/samples/..%2Fpyproject.toml").status_code == 404


def test_export_uses_edits(client):
    data = make_transcript().to_dict()
    data["segments"][0]["words"][4].update(text=" amoxicillin.", edited=True)
    r = client.post("/api/export?format=srt", json=data)
    assert r.status_code == 200
    assert "amoxicillin." in r.text and "moxicillin." not in r.text.replace("amoxicillin.", "")
    assert r.headers["content-disposition"].endswith('.srt"')


def test_export_rejects_unknown_format(client):
    assert client.post("/api/export?format=exe", json=make_transcript().to_dict()).status_code == 422


def test_recent_transcript_can_be_fetched(client):
    tid = client.post("/api/transcribe", files={"file": ("a.wav", io.BytesIO(b"x"))}).json()["id"]
    assert client.get(f"/api/transcripts/{tid}").status_code == 200
    assert client.get("/api/transcripts/nope").status_code == 404
