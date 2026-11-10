"""Benchmark: how accurate is the transcript, and are the flagged words really the wrong ones?

For every sample clip, every model and every noise level we:
  1. add noise at an exact SNR (or leave the clip clean),
  2. transcribe it,
  3. compute WER and CER against the reference text,
  4. align the recognised words with the reference to find which ones are wrong,
  5. compare that with the confidence levels Shrutlekh gave each word.

Step 5 answers the question the tool depends on: if you only check the
flagged words, how many of the real mistakes do you catch?
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .metrics import cer, tokenize, word_errors, wrong_hypothesis_tokens
from .models import Transcript
from .noise import add_noise
from .transcriber import Transcriber, load_audio

DEFAULT_SNRS: tuple[float | None, ...] = (None, 20.0, 10.0, 5.0, 0.0)


@dataclass
class Clip:
    file: Path
    text: str
    title: str = ""


def load_clips(samples_dir: Path) -> list[Clip]:
    meta = json.loads((samples_dir / "references.json").read_text(encoding="utf-8"))
    # generated clips (e.g. the noisy mix) reuse other clips' speech, so they would be counted twice
    return [Clip(samples_dir / c["file"], c["text"], c.get("title", "")) for c in meta["clips"] if not c.get("generated")]


def flag_quality(reference: str, transcript: Transcript) -> dict[str, int]:
    """Count recognised words by (flagged or not) x (wrong or not)."""
    tokens: list[str] = []
    owner: list[int] = []  # which Whisper word each normalised token came from
    words = transcript.words
    for wi, w in enumerate(words):
        for tok in tokenize(w.text):
            tokens.append(tok)
            owner.append(wi)
    wrong_words = {owner[i] for i in wrong_hypothesis_tokens(reference, tokens)}
    counts = {"flagged_wrong": 0, "flagged_right": 0, "unflagged_wrong": 0, "unflagged_right": 0}
    for wi, w in enumerate(words):
        if not tokenize(w.text):
            continue
        flagged = w.level != "high"
        wrong = wi in wrong_words
        key = ("flagged_" if flagged else "unflagged_") + ("wrong" if wrong else "right")
        counts[key] += 1
    return counts


def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def run(
    samples_dir: Path,
    models: Iterable[str] = ("tiny", "base"),
    snrs: Iterable[float | None] = DEFAULT_SNRS,
    noise_kind: str = "pink",
    transcriber: Transcriber | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    transcriber = transcriber or Transcriber()
    clips = load_clips(samples_dir)
    rows = []
    for model in models:
        for snr in snrs:
            for ci, clip in enumerate(clips):
                clean = load_audio(clip.file)
                audio = clean if snr is None else add_noise(clean, snr, noise_kind, seed=ci)
                started = time.perf_counter()
                t = transcriber.transcribe(audio, model=model, language="en")
                elapsed = time.perf_counter() - started
                errs = word_errors(clip.text, t.text)
                rows.append(
                    {
                        "model": model,
                        "snr_db": snr,
                        "clip": clip.file.name,
                        "reference": clip.text,
                        "hypothesis": t.text,
                        "wer": round(errs.rate, 4),
                        "cer": round(cer(clip.text, t.text), 4),
                        "substitutions": errs.substitutions,
                        "deletions": errs.deletions,
                        "insertions": errs.insertions,
                        "reference_words": errs.reference_length,
                        "real_time_factor": round(elapsed / max(t.duration, 1e-6), 3),
                        **flag_quality(clip.text, t),
                    }
                )
                label = "clean" if snr is None else f"{snr:g} dB"
                progress(f"{model:5s} {label:>6s} {clip.file.name:18s} WER {errs.rate:6.1%}")
    return {"meta": _meta(noise_kind), "rows": rows, "summary": summarize(rows)}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["snr_db"]), []).append(r)
    out = []
    for (model, snr), rs in groups.items():
        ref_words = sum(r["reference_words"] for r in rs)
        errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in rs)
        fw = sum(r["flagged_wrong"] for r in rs)
        fr = sum(r["flagged_right"] for r in rs)
        uw = sum(r["unflagged_wrong"] for r in rs)
        ur = sum(r["unflagged_right"] for r in rs)
        out.append(
            {
                "model": model,
                "snr_db": snr,
                "clips": len(rs),
                "wer": round(errors / ref_words, 4) if ref_words else 0.0,  # pooled over all words
                "cer": round(sum(r["cer"] for r in rs) / len(rs), 4),
                "real_time_factor": round(sum(r["real_time_factor"] for r in rs) / len(rs), 3),
                "flagged_share": _ratio(fw + fr, fw + fr + uw + ur),
                "flag_precision": _ratio(fw, fw + fr),  # of the flagged words, how many were wrong
                "flag_recall": _ratio(fw, fw + uw),  # of the wrong words, how many were flagged
                "error_rate_unflagged": _ratio(uw, uw + ur),
                "error_rate_flagged": _ratio(fw, fw + fr),
            }
        )
    return out


def _meta(noise_kind: str) -> dict[str, Any]:
    import ctranslate2
    import faster_whisper

    return {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "noise": noise_kind,
        "machine": f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}",
        "python": platform.python_version(),
        "faster_whisper": faster_whisper.__version__,
        "ctranslate2": ctranslate2.__version__,
    }


def to_markdown(result: dict[str, Any]) -> str:
    m = result["meta"]
    pct = lambda v: "-" if v is None else f"{v:.1%}"  # noqa: E731
    lines = [
        "# Benchmark results",
        "",
        f"Generated by `python -m shrutlekh benchmark` on {m['date']} ({m['machine']}, Python {m['python']}, "
        f"faster-whisper {m['faster_whisper']}). Noise type: {m['noise']}. Device: CPU, int8.",
        "",
        "**WER** is pooled over all words of all clips. **Flagged** = words marked medium or low. "
        "**Precision** = share of flagged words that were really wrong. "
        "**Recall** = share of all wrong words that were flagged. "
        "**RTF** (real-time factor) = processing time / audio length; below 1 means faster than real time.",
        "",
        "| Model | Noise (SNR) | WER | CER | Flagged | Precision | Recall | Error rate, unflagged | Error rate, flagged | RTF |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in result["summary"]:
        snr = "clean" if s["snr_db"] is None else f"{s['snr_db']:g} dB"
        lines.append(
            f"| {s['model']} | {snr} | {pct(s['wer'])} | {pct(s['cer'])} | {pct(s['flagged_share'])} | "
            f"{pct(s['flag_precision'])} | {pct(s['flag_recall'])} | {pct(s['error_rate_unflagged'])} | "
            f"{pct(s['error_rate_flagged'])} | {s['real_time_factor']:.2f} |"
        )
    lines += ["", "## Per-clip results", "", "| Model | SNR | Clip | WER | Recognised text |", "|---|---|---|---|---|"]
    for r in result["rows"]:
        snr = "clean" if r["snr_db"] is None else f"{r['snr_db']:g} dB"
        lines.append(f"| {r['model']} | {snr} | {r['clip']} | {pct(r['wer'])} | {r['hypothesis']} |")
    return "\n".join(lines) + "\n"
