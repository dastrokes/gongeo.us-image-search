from __future__ import annotations

import re


QWEN_STRUCTURED_DETAIL_PROMPT = """
Describe only the main wearable item from the provided images.
Focus on {focus}.
Return JSON only with these keys:
{{
  "icon": ["short lowercase phrases from the icon image"],
  "overview": ["short lowercase phrases from the overview image"],
  "merged": ["combined item-level phrases"]
}}
Rules:
- Use arrays of short lowercase phrases.
- Use [] when unsure.
- Be exhaustive about visible item attributes, but omit anything not clearly visible.
- Prefer concrete visual facts such as subcategory, length, silhouette, shape, placement, material, pattern, motif, trim, closure, and ornament.
- Use pattern for repeated surface graphics, motif for symbolic imagery, and ornament for attached decoration.
- Put the most identifying phrases first.
- Do not mention the character, pose, face, body, hands, background, or framing.
- Do not write prose outside the JSON object.
""".strip()

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
    r"soft|silky|sparkle|elegance|elegant|relaxed fit|tight-fitting)\b"
)

STOP_VISUAL_PATTERN = re.compile(
    r"\b(?:expression|eyes?|camera|mood|serene|peaceful|dreamy|whimsical|"
    r"ethereal|minimalistic|background|landscape|mountains?|trees?|snowy|"
    r"looking directly|sleeping)\b"
)

TOP_GARMENT_PATTERNS: tuple[str, ...] = (
    r"\btank top\b",
    r"\bsports bra\b",
    r"\bbra top\b",
    r"\bcamisole\b",
    r"\bblouse\b",
    r"\bshirt\b",
    r"\btop\b",
    r"\bbodice\b",
    r"\bneckline\b",
    r"\bsquare neckline\b",
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
