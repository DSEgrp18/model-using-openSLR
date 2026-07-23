#!/usr/bin/env python
"""Synthesize Sinhala speech from a trained Coqui VITS checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="checkpoint.pth")
    parser.add_argument("--config-path", type=Path, required=True, help="config.json")
    parser.add_argument("--text", required=True, help="Sinhala text to synthesize")
    parser.add_argument("--speaker", required=True, help="Speaker id, e.g. sin_2241")
    parser.add_argument("--output", type=Path, required=True, help="Output WAV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from TTS.utils.synthesizer import Synthesizer
    except ImportError as exc:
        print(f"error: install Coqui TTS first ({exc})", file=sys.stderr)
        return 2

    synthesizer = Synthesizer(
        tts_checkpoint=str(args.model_path),
        tts_config_path=str(args.config_path),
        use_cuda=True,
    )
    wav = synthesizer.tts(text=args.text, speaker_name=args.speaker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    synthesizer.save_wav(wav, str(args.output))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
