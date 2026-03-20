from __future__ import annotations

import re

NEUTRAL_COLOR_LABELS = {"white", "gray", "silver", "black"}
COLOR_ALIASES = {"grey": "gray"}
COLOR_WORDS = (
    "aqua",
    "beige",
    "black",
    "blue",
    "blonde",
    "bronze",
    "purple",
    "brown",
    "burgundy",
    "cream",
    "cyan",
    "gold",
    "gradient",
    "gray",
    "green",
    "grey",
    "indigo",
    "iridescent",
    "ivory",
    "lavender",
    "maroon",
    "multicolor",
    "navy",
    "ombre",
    "orange",
    "peach",
    "pink",
    "rainbow",
    "red",
    "rose",
    "silver",
    "teal",
    "turquoise",
    "two-tone",
    "violet",
    "white",
    "yellow",
)
COLOR_PATTERN = r"(?:(?:light|dark|pale)\s+)?" + rf"(?:{'|'.join(COLOR_WORDS)})"
COLOR_ONLY_PATTERN = re.compile(rf"^{COLOR_PATTERN}$")
MATERIAL_WORDS = (
    "silk",
    "silken",
    "satin",
    "lace",
    "velvet",
    "denim",
    "knit",
    "knitted",
    "leather",
    "fur",
    "furry",
    "tulle",
    "mesh",
    "sheer",
    "metallic",
    "sequined",
    "embroidered",
)
LEADING_COLOR_OR_MATERIAL_PATTERN = re.compile(
    r"^(?:(?:light|dark|pale)\s+)?"
    r"(?:"
    + "|".join(sorted(set(COLOR_WORDS) | set(MATERIAL_WORDS), key=len, reverse=True))
    + r")\b\s+"
)


def normalize_color_label(label: str) -> str:
    normalized = re.sub(r"\s+", " ", label.lower()).strip()
    if not normalized:
        return normalized
    normalized = COLOR_ALIASES.get(normalized, normalized)
    if normalized.startswith(("light ", "dark ", "pale ")):
        _, _, normalized = normalized.partition(" ")
    return COLOR_ALIASES.get(normalized, normalized)


def strip_leading_color_or_material_phrase(value: str) -> str | None:
    normalized = re.sub(r"\s+", " ", str(value).strip().lower()).strip(" ,")
    if not normalized:
        return None

    stripped = normalized
    while True:
        updated = LEADING_COLOR_OR_MATERIAL_PATTERN.sub("", stripped, count=1).strip()
        if not updated or updated == stripped:
            break
        stripped = updated

    if stripped == normalized:
        return None
    return stripped or None
