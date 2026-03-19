from __future__ import annotations

import re


FLORENCE_DETAIL_PROMPT = "<MORE_DETAILED_CAPTION>"

CAPTION_NOISE_TERMS = {
    "girl",
    "woman",
    "person",
    "anime",
    "character",
    "standing",
    "posing",
    "pose",
    "background",
    "white background",
    "close up",
    "close-up",
    "illustration",
}

LOW_SIGNAL_VISUAL_PATTERN = re.compile(
    r"\b(?:playful|feminine|dreamlike|sophisticated|summery|detailed|delicate|"
    r"stand out|overall look|adding a touch|touch of sparkle|subtle sheen|"
    r"soft|silky|sparkle|elegance|elegant|relaxed fit)\b"
)

STOP_VISUAL_PATTERN = re.compile(
    r"\b(?:expression|eyes?|camera|mood|serene|peaceful|dreamy|whimsical|"
    r"ethereal|minimalistic|background|landscape|mountains?|trees?|snowy|"
    r"looking directly|sleeping)\b"
)

TOP_GARMENT_PATTERNS: tuple[str, ...] = (
    r"\btank top\b",
    r"\bcamisole\b",
    r"\bblouse\b",
    r"\bshirt\b",
    r"\btop\b",
    r"\bbodice\b",
    r"\bneckline\b",
)

BOTTOM_GARMENT_PATTERNS: tuple[str, ...] = (
    r"\bshorts?\b",
    r"\bskirt\b",
    r"\bpants\b",
    r"\btrousers\b",
    r"\bwaistband\b",
    r"\bhigh-waisted\b",
)

OUTERWEAR_GARMENT_PATTERNS: tuple[str, ...] = (
    r"\bjacket\b",
    r"\bcoat\b",
    r"\bcloak\b",
    r"\bcape\b",
    r"\bshawl\b",
    r"\bblazer\b",
)

DRESS_GARMENT_PATTERNS: tuple[str, ...] = (
    r"\bdress\b",
    r"\bgown\b",
    r"\bonesie\b",
)
