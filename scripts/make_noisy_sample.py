"""Builds samples/09_noisy_mix.wav: four clips joined together with pink noise at 5 dB SNR.

Run after scripts/make_samples.ps1:
    python scripts/make_noisy_sample.py
"""

import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shrutlekh.noise import add_noise  # noqa: E402
from shrutlekh.transcriber import SAMPLE_RATE, load_audio  # noqa: E402

PARTS = ["05_bengaluru.wav", "04_medicine.wav", "08_hyderabad.wav", "06_seashells.wav"]
OUT = "09_noisy_mix.wav"
SNR_DB = 5.0


def main() -> None:
    samples = ROOT / "samples"
    refs_path = samples / "references.json"
    refs = json.loads(refs_path.read_text(encoding="utf-8"))
    by_file = {c["file"]: c for c in refs["clips"]}

    gap = np.zeros(int(0.7 * SAMPLE_RATE), dtype=np.float32)
    pieces = []
    for name in PARTS:
        pieces += [load_audio(samples / name), gap]
    mixed = add_noise(np.concatenate(pieces), SNR_DB, "pink", seed=9)

    pcm = (np.clip(mixed, -1, 1) * 32767).astype("<i2")
    with wave.open(str(samples / OUT), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())

    text = " ".join(by_file[name]["text"] for name in PARTS)
    refs["clips"] = [c for c in refs["clips"] if c["file"] != OUT]
    refs["clips"].append({"file": OUT, "voice": "mixed", "rate": 0,
                          "title": "Noisy mix: 4 sentences, 5 dB noise", "text": text, "generated": True})
    refs_path.write_text(json.dumps(refs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote samples/{OUT} ({len(mixed) / SAMPLE_RATE:.1f} s)")


if __name__ == "__main__":
    main()
