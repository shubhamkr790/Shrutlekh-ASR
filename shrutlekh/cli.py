"""Command-line interface.

  python -m shrutlekh serve                       start the web app
  python -m shrutlekh transcribe talk.mp3         print a colour-coded transcript
  python -m shrutlekh transcribe talk.mp3 -f srt -o talk.srt
  python -m shrutlekh benchmark                   WER + flag quality on the sample clips
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from .confidence import ConfidenceThresholds  # noqa: E402
from .exporters import FORMATS, export  # noqa: E402
from .transcriber import LANGUAGES, MODEL_SIZES  # noqa: E402

COLOURS = {"high": "", "medium": "\033[33m", "low": "\033[31;4m"}
RESET = "\033[0m"


def _coloured(transcript, colour: bool) -> str:
    out = []
    for seg in transcript.segments:
        line = []
        for w in seg.words:
            line.append(f"{COLOURS[w.level]}{w.text}{RESET}" if colour and w.level != "high" else w.text)
        out.append(f"[{seg.start:6.1f}s] " + "".join(line).strip())
    return "\n".join(out)


def cmd_transcribe(a: argparse.Namespace) -> int:
    from .transcriber import Transcriber

    if not Path(a.audio).exists():
        print(f"error: file not found: {a.audio}", file=sys.stderr)
        return 2
    t = Transcriber().transcribe(a.audio, model=a.model, language=a.language,
                                 thresholds=ConfidenceThresholds(high=a.high, low=a.low), vad=not a.no_vad)
    if a.output or a.format != "console":
        text = export(t, "txt" if a.format == "console" else a.format)
        if a.output:
            Path(a.output).write_text(text, encoding="utf-8")
            print(f"saved {a.output}", file=sys.stderr)
        else:
            sys.stdout.write(text)
        return 0
    s = t.summary
    print(_coloured(t, colour=sys.stdout.isatty() and not a.no_colour))
    print(f"\n{s['words']} words | {s['medium']} to check | {s['low']} likely wrong | "
          f"language {t.language} ({t.language_probability:.0%}) | {t.duration:.1f}s audio in {t.processing_seconds:.1f}s",
          file=sys.stderr)
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    print(f"Shrutlekh running at http://{a.host}:{a.port}", file=sys.stderr)
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")
    return 0


def cmd_benchmark(a: argparse.Namespace) -> int:
    from . import benchmark

    snrs = [None if s == "clean" else float(s) for s in a.snr]
    result = benchmark.run(Path(a.samples), models=a.models, snrs=snrs, noise_kind=a.noise)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "benchmark.md").write_text(benchmark.to_markdown(result), encoding="utf-8")
    print(f"\nsaved {out / 'benchmark.md'} and {out / 'benchmark.json'}")
    for s in result["summary"]:
        snr = "clean" if s["snr_db"] is None else f"{s['snr_db']:g} dB"
        print(f"{s['model']:5s} {snr:>6s}  WER {s['wer']:.1%}  flag recall {s['flag_recall'] if s['flag_recall'] is not None else '-'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shrutlekh", description="Confidence-aware speech-to-text")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="transcribe an audio file")
    t.add_argument("audio")
    t.add_argument("-m", "--model", choices=MODEL_SIZES, default="base")
    t.add_argument("-l", "--language", choices=list(LANGUAGES), default="auto")
    t.add_argument("-f", "--format", choices=["console", *FORMATS], default="console")
    t.add_argument("-o", "--output", help="write to this file instead of the screen")
    t.add_argument("--high", type=float, default=0.80, help="probability for 'high' confidence (default 0.80)")
    t.add_argument("--low", type=float, default=0.50, help="probability below which a word is 'low' (default 0.50)")
    t.add_argument("--no-vad", action="store_true", help="do not skip silence")
    t.add_argument("--no-colour", action="store_true")
    t.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("serve", help="start the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    b = sub.add_parser("benchmark", help="measure WER and flag quality on the sample clips")
    b.add_argument("--samples", default="samples")
    b.add_argument("--models", nargs="+", choices=MODEL_SIZES, default=["tiny", "base"])
    b.add_argument("--snr", nargs="+", default=["clean", "20", "10", "5", "0"],
                   help="noise levels in dB, or 'clean'")
    b.add_argument("--noise", choices=["white", "pink"], default="pink")
    b.add_argument("--out", default="docs")
    b.set_defaults(func=cmd_benchmark)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Hindi output on Windows consoles
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
