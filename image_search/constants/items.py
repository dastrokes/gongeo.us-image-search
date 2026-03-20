from __future__ import annotations

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

# Base-item ID ranges, mirrored from the tracker's
# server/api/items/index.get.ts BASE_ITEM_PREFIX_RANGES.
# Prefixes 1022–1026 are variation tiers (glowup / evo1–evo3) and are excluded.
BASE_ITEM_PREFIX_RANGES: tuple[tuple[int, int], ...] = (
    (1020_000_000, 1020_999_999),
    (1021_000_000, 1021_999_999),
    (1027_000_000, 1027_999_999),
    (1028_000_000, 1028_999_999),
    (1029_000_000, 1029_999_999),
)

TYPE_KEY_MAP = {
    "LGS03_1": "hair",
    "LGS03_2": "outerwear",
    "LGS03_3": "tops",
    "LGS03_4": "bottoms",
    "LGS03_5": "bottoms",
    "LGS03_6": "socks",
    "LGS03_7": "shoes",
    "LGS03_8": "hairAccessories",
    "LGS03_9": "headwear",
    "LGS03_10": "earrings",
    "LGS03_11": "neckwear",
    "LGS03_12": "bracelets",
    "LGS03_13": "chokers",
    "LGS03_14": "gloves",
    "LGS03_15": "handhelds",
    "LGS03_16": "bodyPaint",
    "LGS03_17": "baseMakeup",
    "LGS03_18": "eyebrows",
    "LGS03_19": "eyelashes",
    "LGS03_20": "contactLenses",
    "LGS03_21": "lips",
    "LGS03_22": "skinTones",
    "LGS03_23": "dresses",
    "LGS03_25": "faceDecorations",
    "LGS03_26": "chestAccessories",
    "LGS03_27": "pendants",
    "LGS03_28": "backpieces",
    "LGS03_29": "rings",
    "LGS03_30": "armDecorations",
    "LGS03_31": "fullMakeup",
    "LGS03_32": "abilityHandhelds",
}

UPPER_FOCUS_TYPES = {
    "hair",
    "hairAccessories",
    "headwear",
    "earrings",
    "neckwear",
    "bracelets",
    "chokers",
    "faceDecorations",
    "pendants",
}
