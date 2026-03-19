from __future__ import annotations

import re


NEUTRAL_COLOR_LABELS = {"white", "gray", "silver", "black"}
COLOR_ALIASES = {"grey": "gray"}
COLOR_WORDS = (
    "blue",
    "purple",
    "pink",
    "red",
    "green",
    "gold",
    "silver",
    "gray",
    "grey",
    "white",
    "black",
    "brown",
    "blonde",
    "yellow",
    "orange",
)
COLOR_PATTERN = (
    r"(?:(?:light|dark|pale)\s+)?"
    + rf"(?:{'|'.join(COLOR_WORDS)})"
)
COLOR_ONLY_PATTERN = re.compile(rf"^{COLOR_PATTERN}$")

COLOR_DETAIL_NOUNS = (
    "border",
    "trim",
    "collar",
    "cuff",
    "hem",
    "lining",
    "lace",
    "ribbon",
    "sash",
    "fur",
    "hood",
    "mask",
    "cape",
    "cloak",
    "robe",
    "armor",
)

ACCENT_DETAIL_NOUNS = (
    "button",
    "buttons",
    "zipper",
    "buckle",
    "trim",
    "lining",
    "collar",
    "cuff",
    "hem",
    "fur",
    "lace",
    "ribbon",
    "bow",
    "hood",
    "sash",
    "embroidery",
    "emblem",
)

APPAREL_BODY_NOUNS = (
    "jacket",
    "coat",
    "cloak",
    "cape",
    "blazer",
    "shawl",
    "hoodie",
    "top",
    "blouse",
    "shirt",
    "bodice",
    "corset",
    "vest",
    "camisole",
    "skirt",
    "pants",
    "trousers",
    "shorts",
    "sock",
    "stocking",
    "tights",
    "shoe",
    "boot",
    "heel",
    "sandal",
    "dress",
    "gown",
)

HAIR_BODY_PATTERNS = (
    ("light blue hair", "blue"),
    ("grey hair", "gray"),
    ("gray hair", "gray"),
    ("silver hair", "silver"),
    ("white hair", "white"),
    ("black hair", "black"),
    ("brown hair", "brown"),
    ("blonde hair", "blonde"),
    ("gold hair", "gold"),
    ("red hair", "red"),
    ("pink hair", "pink"),
    ("purple hair", "purple"),
    ("blue hair", "blue"),
    ("green hair", "green"),
    ("orange hair", "orange"),
    ("yellow hair", "yellow"),
)


def normalize_color_label(label: str) -> str:
    normalized = re.sub(r"\s+", " ", label.lower()).strip()
    if not normalized:
        return normalized
    normalized = COLOR_ALIASES.get(normalized, normalized)
    if normalized.startswith(("light ", "dark ", "pale ")):
        _, _, normalized = normalized.partition(" ")
    return COLOR_ALIASES.get(normalized, normalized)
