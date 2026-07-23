# Sinhala VITS TTS (OpenSLR 30)

Train a **multi-speaker VITS** Sinhala text-to-speech model on the cleaned OpenSLR Resource 30 dataset using **Google Colab**.

Your local Windows PC (no CUDA GPU / low RAM) is used for **data preparation**. Training runs in Colab.

## Dataset

| Item | Path |
|------|------|
| Cleaned WAVs | `C:\Users\PCland\Desktop\5th sem project\dataset_clean` |
| Official transcripts | `data/metadata/si_lk.lines.txt` (OpenSLR 30) |
| License | CC BY-SA 4.0 (Google) |

**Data note:** the official transcript file currently has **1251** rows while the cleaned folder has **2061** WAVs. Preparation keeps only uniquely matched IDs (~1248) and reports the rest. No transcripts are invented.

## Project layout

```text
configs/vits_sinhala.json          # training defaults (documentation)
data/metadata/si_lk.lines.txt      # OpenSLR transcripts
data/prepared/sinhala_vits/        # prepared 22.05 kHz dataset (local)
notebooks/train_sinhala_vits.ipynb # full Colab guide
scripts/prepare_vits_dataset.py    # WAV+transcript → Coqui format
scripts/train_vits.py              # multi-speaker VITS training
scripts/infer_vits.py              # synthesize WAV from checkpoint
src/sinhala_vits/                  # Sinhala characters + Coqui formatter
```

## 1. Prepare data on Windows

```powershell
cd C:\Users\PCland\Desktop\model-using-openSLR
python -m pip install -r requirements.txt
python scripts/prepare_vits_dataset.py `
  --audio-dir "C:\Users\PCland\Desktop\5th sem project\dataset_clean" `
  --transcript-file "data\metadata\si_lk.lines.txt" `
  --output-dir "data\prepared\sinhala_vits" `
  --sample-rate 22050 `
  --seed 42
```

This creates:

```text
data/prepared/sinhala_vits/
  wavs/*.wav
  metadata.csv
  metadata_train.csv
  metadata_val.csv
  preparation_report.json
```

Metadata format:

```text
utterance_id|sinhala_text|speaker_id
```

## 2. Upload to Google Drive

Upload `data/prepared/sinhala_vits` to:

```text
MyDrive/sinhala_vits/dataset/
```

## 3. Train in Google Colab

1. Open [`notebooks/train_sinhala_vits.ipynb`](notebooks/train_sinhala_vits.ipynb) in Colab.
2. Set runtime to **GPU**.
3. Run all cells in order:
   - mount Drive
   - install `coqui-tts` (not the old `TTS` package)
   - verify dataset
   - train VITS
   - synthesize a sample

If `pip install TTS` fails on Colab Python 3.12, use:

```bash
pip uninstall -y TTS trainer
pip install -U "coqui-tts[notebooks]" "transformers>=4.57.0"
```

Imports stay the same (`from TTS...`, `from trainer import Trainer`).
If you see `isin_mps_friendly` ImportError, upgrade transformers with the command above, then restart the runtime and re-run the install cell.

### Suggested Colab settings

| GPU | `batch_size` |
|-----|--------------|
| T4 16GB | 8–12 |
| L4 24GB | 16 |
| A100 | 24–32 |

If Colab disconnects, remount Drive and set `CONTINUE_PATH` in the notebook to the previous run folder under `MyDrive/sinhala-vits/outputs/`.

## 4. Inference

In Colab (after training):

```bash
python scripts/infer_vits.py \
  --model-path /content/drive/MyDrive/sinhala-vits/outputs/<run>/best_model.pth \
  --config-path /content/drive/MyDrive/sinhala-vits/outputs/<run>/config.json \
  --text "ආයුබෝවන්" \
  --speaker sin_2241 \
  --output hello.wav
```

Valid speaker IDs come from your metadata (for example `sin_2241`, `sin_4191`, `sin_6314`).

## Architecture

```text
Sinhala text + speaker id
        │
        ▼
   VITS (Coqui)
   ├── text encoder
   ├── posterior encoder
   ├── flow / duration predictor
   ├── speaker embedding
   └── HiFi-GAN-style decoder
        │
        ▼
   22.05 kHz WAV
```

This is a practical multi-speaker VITS setup for ~1.2k OpenSLR utterances, not a giant industrial TTS system.

## Limitations

- Only matched transcript↔audio pairs are used (~1248 clips).
- Quality will be limited by dataset size (~2 hours total matched audio).
- Colab free GPUs may throttle or disconnect; always save checkpoints to Drive.
- Sinhala is trained with character-level text (no phonemizer dependency).
