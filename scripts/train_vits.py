#!/usr/bin/env python
"""Train a multi-speaker Sinhala VITS model with Coqui TTS.

Designed for Google Colab GPU runtimes.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--run-name", default="sinhala_vits_openslr30")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-loader-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.add_argument("--restore-path", type=Path, default=None)
    parser.add_argument("--continue-path", type=Path, default=None)
    return parser.parse_args()


def _patch_transformers() -> None:
    """Keep Coqui importable when transformers removed isin_mps_friendly."""
    import torch
    import transformers.pytorch_utils as pytorch_utils

    if not hasattr(pytorch_utils, "isin_mps_friendly"):
        pytorch_utils.isin_mps_friendly = torch.isin


def _register_formatter(sinhala_openslr) -> None:
    try:
        from TTS.tts.datasets import register_formatter

        register_formatter("sinhala_openslr", sinhala_openslr)
        return
    except Exception:  # noqa: BLE001 - fall back for older coqui builds
        pass

    import TTS.tts.datasets.formatters as formatters_module

    formatters_module.sinhala_openslr = sinhala_openslr
    for attr in ("FORMATTERS", "_FORMATTERS", "formatters"):
        table = getattr(formatters_module, attr, None)
        if isinstance(table, dict):
            table["sinhala_openslr"] = sinhala_openslr
    original_get = getattr(formatters_module, "get_formatter", None)
    if callable(original_get):

        def get_formatter(name: str):
            if name == "sinhala_openslr":
                return sinhala_openslr
            return original_get(name)

        formatters_module.get_formatter = get_formatter


def main() -> int:
    args = parse_args()
    repo_src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(repo_src))

    try:
        _patch_transformers()
        from sinhala_vits.formatter import sinhala_openslr
        from sinhala_vits.text import SINHALA_CHARACTERS

        from trainer import Trainer, TrainerArgs
        from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
        from TTS.tts.utils.speakers import SpeakerManager
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.utils.audio import AudioProcessor
    except Exception as exc:  # noqa: BLE001 - show full Colab-facing traceback
        print("error: failed to import training dependencies", file=sys.stderr)
        traceback.print_exc()
        print(
            "\nIf this mentions isin_mps_friendly, run:\n"
            '  pip install -U "transformers>=4.57.0,<5"\n'
            "then Runtime → Restart session.",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        return 2

    dataset_path = args.dataset_path
    train_meta = dataset_path / "metadata_train.csv"
    val_meta = dataset_path / "metadata_val.csv"
    if not train_meta.is_file():
        print(f"error: missing {train_meta}", file=sys.stderr)
        return 2
    if not (dataset_path / "wavs").is_dir():
        print(f"error: missing wavs folder: {dataset_path / 'wavs'}", file=sys.stderr)
        return 2

    args.output_path.mkdir(parents=True, exist_ok=True)
    _register_formatter(sinhala_openslr)

    dataset_config = BaseDatasetConfig(
        formatter="sinhala_openslr",
        meta_file_train=train_meta.name,
        meta_file_val=val_meta.name if val_meta.is_file() else None,
        path=str(dataset_path),
        language="si",
    )

    audio_config = VitsAudioConfig(
        sample_rate=args.sample_rate,
        hop_length=256,
        win_length=1024,
        fft_size=1024,
        mel_fmin=0,
        mel_fmax=None,
        num_mels=80,
    )

    model_args = VitsArgs(
        use_speaker_embedding=True,
        speaker_embedding_channels=256,
        num_speakers=0,
    )

    characters_config = CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=SINHALA_CHARACTERS,
        punctuations=" !,.?",
        phonemes="",
        is_unique=True,
        is_sorted=True,
    )

    # Use first speaker ids present in metadata for test sentences.
    first_speakers: list[str] = []
    with train_meta.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("|")
            if len(parts) == 3 and parts[2] not in first_speakers:
                first_speakers.append(parts[2])
            if len(first_speakers) >= 3:
                break
    while len(first_speakers) < 3:
        first_speakers.append(first_speakers[0] if first_speakers else "sin_2241")

    config = VitsConfig(
        model_args=model_args,
        audio=audio_config,
        characters=characters_config,
        run_name=args.run_name,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        batch_group_size=4,
        num_loader_workers=args.num_loader_workers,
        num_eval_loader_workers=max(1, args.num_loader_workers // 2),
        run_eval=True,
        test_delay_epochs=-1,
        epochs=args.epochs,
        text_cleaner="basic_cleaners",
        use_phonemes=False,
        phoneme_language=None,
        compute_input_seq_cache=True,
        print_step=50,
        print_eval=True,
        mixed_precision=args.mixed_precision,
        output_path=str(args.output_path),
        datasets=[dataset_config],
        cudnn_benchmark=False,
        max_audio_len=args.sample_rate * 12,
        max_text_len=220,
        lr=args.lr,
        lr_gen=args.lr,
        lr_disc=args.lr,
        test_sentences=[
            ["ආයුබෝවන්", first_speakers[0]],
            ["මම සිංහලෙන් කතා කරනවා", first_speakers[1]],
            ["ඔබට සුභ දවසක් වේවා", first_speakers[2]],
        ],
    )

    print("Loading audio processor / tokenizer...")
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    print("Loading samples...")
    # meta_file_val is set, so keep eval_split True for Coqui's split helper.
    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=True,
    )
    if not train_samples:
        print("error: no training samples loaded", file=sys.stderr)
        return 2
    if not eval_samples:
        print("warning: no eval samples; using a tiny train slice for eval", file=sys.stderr)
        eval_samples = train_samples[: max(1, min(16, len(train_samples) // 20))]

    print(f"train_samples={len(train_samples)} eval_samples={len(eval_samples)}")

    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(
        train_samples + eval_samples, parse_key="speaker_name"
    )
    config.model_args.num_speakers = speaker_manager.num_speakers
    config.num_speakers = speaker_manager.num_speakers
    print(f"num_speakers={speaker_manager.num_speakers}")

    model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)

    trainer = Trainer(
        TrainerArgs(
            restore_path=str(args.restore_path) if args.restore_path else "",
            continue_path=str(args.continue_path) if args.continue_path else "",
            grad_accum_steps=1,
        ),
        config,
        output_path=str(args.output_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    print("Starting trainer.fit() ...")
    trainer.fit()
    print(f"training finished; outputs in {args.output_path}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1)
