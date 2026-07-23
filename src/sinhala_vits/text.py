"""Sinhala text helpers for VITS training."""

from __future__ import annotations

import re
import unicodedata

# OpenSLR / Sinhala orthography plus punctuation used in transcripts.
SINHALA_CHARACTERS = (
    " !,.?abcdefghijklmnopqrstuvwxyz"
    "ංඃඅආඇඈඉඊඋඌඍඎඏඐඑඒඓඔඕඖ"
    "කඛගඝඞඟචඡජඣඤඥඦටඨඩඪණඬතථදධනඳපඵබභමඹ"
    "යරලවශෂසහළෆ"
    "්ාැෑිීුූෘෙේෛොෝෞෟෲෳ"
)


def normalize_sinhala(text: str) -> str:
    """NFC-normalize and lightly clean Sinhala transcript text."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200d", "")  # ZWJ
    text = text.replace("\u200c", "")  # ZWNJ
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()
