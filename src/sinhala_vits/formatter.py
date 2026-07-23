"""Coqui TTS dataset formatter for prepared OpenSLR Sinhala metadata."""

from __future__ import annotations

from pathlib import Path


def sinhala_openslr(root_path: str, meta_file: str, ignored_speakers=None):
    """Parse ``utt|text|speaker`` metadata used by this project.

    Returns Coqui sample dicts with ``text``, ``audio_file``, ``speaker_name``.
    """
    ignored_speakers = set(ignored_speakers or [])
    root = Path(root_path)
    items: list[dict[str, str]] = []
    with (root / meta_file).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) != 3:
                raise ValueError(
                    f"{meta_file}:{line_number}: expected utt|text|speaker, got {line!r}"
                )
            utterance_id, text, speaker = (part.strip() for part in parts)
            if speaker in ignored_speakers:
                continue
            wav_path = root / "wavs" / f"{utterance_id}.wav"
            if not wav_path.is_file():
                raise FileNotFoundError(f"Missing audio for {utterance_id}: {wav_path}")
            items.append(
                {
                    "text": text,
                    "audio_file": str(wav_path),
                    "speaker_name": speaker,
                    "root_path": str(root),
                }
            )
    return items
