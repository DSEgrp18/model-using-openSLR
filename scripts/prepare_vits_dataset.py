#!/usr/bin/env python
"""Prepare OpenSLR Sinhala cleaned WAVs for Coqui VITS training.

Creates:
  <output>/wavs/*.wav          resampled mono 22050 Hz
  <output>/metadata.csv        utt|text|speaker  (all matched rows)
  <output>/metadata_train.csv
  <output>/metadata_val.csv
  <output>/preparation_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
import unicodedata
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

RE_PAREN = re.compile(r'^\(\s*([^\s()]+)\s+"(.*)"\s*\)$')


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200d", "").replace("\u200c", "")
    return re.sub(r"\s+", " ", text).strip()


def parse_transcripts(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = RE_PAREN.match(line)
            if match:
                utt_id, text = match.groups()
            elif "\t" in line:
                utt_id, text = line.split("\t", 1)
            elif "|" in line:
                utt_id, text = line.split("|", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"{path}:{line_number}: cannot parse {line!r}")
                utt_id, text = parts
            utt_id = Path(utt_id.strip()).stem
            text = normalize_text(text)
            if not utt_id or not text:
                raise ValueError(f"{path}:{line_number}: empty id/text")
            if utt_id in entries:
                raise ValueError(f"Duplicate transcript id: {utt_id}")
            entries[utt_id] = text
    return entries


def speaker_from_id(utt_id: str) -> str:
    speaker, sep, _ = utt_id.rpartition("_")
    if not sep or not speaker:
        raise ValueError(f"Cannot infer speaker from {utt_id!r}")
    return speaker


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM is supported (got width={width})")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio.astype(np.float32)
    try:
        import librosa
    except ImportError as exc:
        raise ImportError("Install librosa to resample audio: pip install librosa") from exc
    return librosa.resample(audio, orig_sr=src_rate, target_sr=dst_rate).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def split_by_speaker(
    rows: list[dict],
    seed: int,
    val_ratio: float,
) -> tuple[list[dict], list[dict]]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)

    train: list[dict] = []
    val: list[dict] = []
    for speaker in sorted(by_speaker):
        items = sorted(by_speaker[speaker], key=lambda item: item["utt_id"])
        digest = hashlib.sha256(f"{seed}\0{speaker}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio))) if len(items) >= 5 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    train.sort(key=lambda item: item["utt_id"])
    val.sort(key=lambda item: item["utt_id"])
    return train, val


def write_metadata(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(f"{row['utt_id']}|{row['text']}|{row['speaker']}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path(r"C:\Users\PCland\Desktop\5th sem project\dataset_clean"),
        help="Cleaned WAV folder",
    )
    parser.add_argument(
        "--transcript-file",
        type=Path,
        default=Path("data/metadata/si_lk.lines.txt"),
        help="OpenSLR si_lk.lines.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/prepared/sinhala_vits"),
        help="Prepared dataset output directory",
    )
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Copy/link WAVs without resampling (must already be target rate)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink WAVs instead of rewriting when sample rates already match",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.audio_dir.is_dir():
        print(f"error: audio dir not found: {args.audio_dir}", file=sys.stderr)
        return 2
    if not args.transcript_file.is_file():
        print(f"error: transcript not found: {args.transcript_file}", file=sys.stderr)
        return 2

    transcripts = parse_transcripts(args.transcript_file)
    wavs = {
        path.stem: path
        for path in sorted(args.audio_dir.glob("*.wav"))
    }
    matched_ids = sorted(set(transcripts) & set(wavs))
    unmatched_transcripts = sorted(set(transcripts) - set(wavs))
    unmatched_audio = sorted(set(wavs) - set(transcripts))

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    wav_out = args.output_dir / "wavs"
    wav_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped: list[dict] = []
    for utt_id in matched_ids:
        src = wavs[utt_id]
        text = transcripts[utt_id]
        try:
            speaker = speaker_from_id(utt_id)
            audio, rate = load_mono(src)
            duration = len(audio) / float(rate)
            if duration < 0.5 or duration > 20.0:
                skipped.append(
                    {"utt_id": utt_id, "reason": f"duration={duration:.2f}s out of range"}
                )
                continue
            dst = wav_out / f"{utt_id}.wav"
            if args.copy_only or (args.symlink and rate == args.sample_rate):
                if args.symlink and rate == args.sample_rate:
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    dst.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dst)
            else:
                audio = resample(audio, rate, args.sample_rate)
                write_wav(dst, audio, args.sample_rate)
            rows.append(
                {
                    "utt_id": utt_id,
                    "text": text,
                    "speaker": speaker,
                    "duration": duration,
                    "src_sample_rate": rate,
                }
            )
        except Exception as exc:  # noqa: BLE001 - collect per-file failures
            skipped.append({"utt_id": utt_id, "reason": str(exc)})

    train_rows, val_rows = split_by_speaker(rows, args.seed, args.val_ratio)
    write_metadata(args.output_dir / "metadata.csv", rows)
    write_metadata(args.output_dir / "metadata_train.csv", train_rows)
    write_metadata(args.output_dir / "metadata_val.csv", val_rows)

    report = {
        "audio_dir": str(args.audio_dir),
        "transcript_file": str(args.transcript_file),
        "output_dir": str(args.output_dir),
        "sample_rate": args.sample_rate,
        "transcript_rows": len(transcripts),
        "discovered_wavs": len(wavs),
        "matched": len(matched_ids),
        "prepared": len(rows),
        "train": len(train_rows),
        "validation": len(val_rows),
        "speakers": sorted({row["speaker"] for row in rows}),
        "unmatched_transcript_ids": unmatched_transcripts,
        "unmatched_audio_ids": unmatched_audio,
        "skipped": skipped,
        "note": (
            "Official OpenSLR si_lk.lines.txt currently has fewer rows than the "
            "cleaned WAV folder. Only uniquely matched utterance IDs are used."
        ),
    }
    (args.output_dir / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"prepared {len(rows)} utterances "
        f"(train={len(train_rows)}, val={len(val_rows)}, "
        f"speakers={len(report['speakers'])}) -> {args.output_dir}"
    )
    print(
        f"unmatched audio={len(unmatched_audio)}, "
        f"unmatched transcripts={len(unmatched_transcripts)}, "
        f"skipped={len(skipped)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
