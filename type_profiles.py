from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ItemTypeProfile:
    keywords: tuple[str, ...]
    style_patterns: tuple[tuple[str, str], ...] = ()
    prefer_icon: bool = False
    extract_hair_color: bool = False


KNOWN_ITEM_TYPES = {
    "hair",
    "outerwear",
    "tops",
    "bottoms",
    "socks",
    "shoes",
    "hairAccessories",
    "headwear",
    "earrings",
    "neckwear",
    "bracelets",
    "chokers",
    "gloves",
    "handhelds",
    "bodyPaint",
    "baseMakeup",
    "eyebrows",
    "eyelashes",
    "contactLenses",
    "lips",
    "skinTones",
    "dresses",
    "faceDecorations",
    "chestAccessories",
    "pendants",
    "backpieces",
    "rings",
    "armDecorations",
    "fullMakeup",
    "abilityHandhelds",
    "unknown",
}

HAIR_STYLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"twin tails?|two (?:large )?pigtails", "twin tails"),
    (r"pigtails?", "pigtails"),
    (r"half-up,? half-down", "half-up half-down"),
    (r"high ponytail", "high ponytail"),
    (r"ponytail", "ponytail"),
    (r"loose waves?|wavy hair", "wavy"),
    (r"loose curls?|curly hair", "curly"),
    (r"straight hair", "straight"),
    (r"braid(?:ed)?", "braid"),
    (r"bangs?", "bangs"),
    (r"bun", "bun"),
    (r"middle part|parted in the middle", "center part"),
    (r"side part|parted to the side", "side part"),
    (r"long hair|falls over her shoulders|cascad(?:es|ing) down", "long"),
    (r"short hair|bob cut", "short"),
)

DECOR_STYLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bbow\b", "bow"),
    (r"ribbon", "ribbon"),
    (r"ruffled?", "ruffled"),
    (r"lace", "lace"),
    (r"floral|flowers?", "floral"),
    (r"embroider(?:ed|y)", "embroidery"),
)

HEAD_STYLE_PATTERNS: tuple[tuple[str, str], ...] = DECOR_STYLE_PATTERNS + (
    (r"headband", "headband"),
    (r"hat|bonnet|beret|cap", "hat"),
    (r"crown|tiara", "crown"),
    (r"veil", "veil"),
)


TYPE_PROFILES: dict[str, ItemTypeProfile] = {
    "hair": ItemTypeProfile(
        keywords=(
            "hair",
            "ponytail",
            "pigtail",
            "twin tail",
            "half-up",
            "braid",
            "bang",
            "bun",
            "curl",
            "wave",
            "straight",
            "part",
            "bob",
            "long",
            "short",
            "headband",
            "bow",
            "ribbon",
        ),
        style_patterns=HAIR_STYLE_PATTERNS + DECOR_STYLE_PATTERNS[:3],
        extract_hair_color=True,
    ),
    "outerwear": ItemTypeProfile(
        keywords=(
            "coat",
            "jacket",
            "cloak",
            "cape",
            "blazer",
            "hood",
            "shawl",
            "fur",
            "sleeve",
            "collar",
            "cuff",
            "button",
            "zipper",
            "trim",
            "lace",
            "bow",
            "ribbon",
        ),
        style_patterns=DECOR_STYLE_PATTERNS,
    ),
    "tops": ItemTypeProfile(
        keywords=(
            "top",
            "blouse",
            "shirt",
            "bodice",
            "corset",
            "vest",
            "camisole",
            "sleeve",
            "collar",
            "cuff",
            "button",
            "trim",
            "lace",
            "bow",
            "ribbon",
            "floral",
        ),
        style_patterns=DECOR_STYLE_PATTERNS,
    ),
    "bottoms": ItemTypeProfile(
        keywords=(
            "skirt",
            "pants",
            "trousers",
            "shorts",
            "hem",
            "waist",
            "waistband",
            "pleat",
            "ruffled",
            "lace",
            "bow",
            "ribbon",
            "floral",
        ),
        style_patterns=DECOR_STYLE_PATTERNS,
    ),
    "socks": ItemTypeProfile(
        keywords=(
            "sock",
            "stocking",
            "tights",
            "thigh",
            "knee",
            "ankle",
            "stripe",
            "trim",
            "lace",
            "bow",
            "ribbon",
        ),
        style_patterns=DECOR_STYLE_PATTERNS,
    ),
    "shoes": ItemTypeProfile(
        keywords=(
            "shoe",
            "boot",
            "heel",
            "sandal",
            "toe",
            "sole",
            "ankle",
            "strap",
            "buckle",
            "lace",
            "ribbon",
            "bow",
        ),
        style_patterns=DECOR_STYLE_PATTERNS[:3],
    ),
    "hairAccessories": ItemTypeProfile(
        keywords=(
            "headband",
            "hairpin",
            "clip",
            "barrette",
            "scrunchie",
            "bow",
            "ribbon",
            "flower",
            "floral",
            "crown",
            "tiara",
            "veil",
            "feather",
        ),
        style_patterns=HEAD_STYLE_PATTERNS,
        prefer_icon=True,
    ),
    "headwear": ItemTypeProfile(
        keywords=(
            "hat",
            "cap",
            "bonnet",
            "beret",
            "hood",
            "crown",
            "tiara",
            "veil",
            "headband",
            "bow",
            "ribbon",
            "flower",
            "feather",
        ),
        style_patterns=HEAD_STYLE_PATTERNS,
        prefer_icon=True,
    ),
    "earrings": ItemTypeProfile(
        keywords=("earring", "hoop", "stud", "drop", "tassel", "gem", "pearl", "star"),
        style_patterns=((r"pearl", "pearl"), (r"gem|crystal", "gem")),
        prefer_icon=True,
    ),
    "neckwear": ItemTypeProfile(
        keywords=("necklace", "scarf", "collar", "tie", "ribbon", "bow", "pendant", "gem", "pearl", "lace"),
        style_patterns=DECOR_STYLE_PATTERNS,
        prefer_icon=True,
    ),
    "bracelets": ItemTypeProfile(
        keywords=("bracelet", "bangle", "cuff", "chain", "charm", "gem", "pearl", "ribbon"),
        style_patterns=((r"pearl", "pearl"), (r"gem|crystal", "gem")),
        prefer_icon=True,
    ),
    "chokers": ItemTypeProfile(
        keywords=("choker", "lace", "ribbon", "bow", "gem", "pearl", "collar"),
        style_patterns=DECOR_STYLE_PATTERNS,
        prefer_icon=True,
    ),
    "gloves": ItemTypeProfile(
        keywords=("glove", "mitten", "finger", "cuff", "sleeve", "lace", "bow", "ribbon"),
        style_patterns=DECOR_STYLE_PATTERNS,
        prefer_icon=True,
    ),
    "handhelds": ItemTypeProfile(
        keywords=("bag", "purse", "umbrella", "parasol", "fan", "book", "basket", "lantern", "bouquet", "plush", "staff", "wand"),
        style_patterns=DECOR_STYLE_PATTERNS[:3],
        prefer_icon=True,
    ),
    "bodyPaint": ItemTypeProfile(
        keywords=("body paint", "paint", "tattoo", "mark", "pattern", "glow", "shimmer"),
    ),
    "baseMakeup": ItemTypeProfile(
        keywords=("makeup", "blush", "eyeshadow", "eyeliner", "contour", "freckles", "highlight"),
        prefer_icon=True,
    ),
    "eyebrows": ItemTypeProfile(
        keywords=("eyebrow", "eyebrows", "brow", "brows", "arch"),
        prefer_icon=True,
    ),
    "eyelashes": ItemTypeProfile(
        keywords=("eyelash", "eyelashes", "lash", "lashes", "mascara"),
        prefer_icon=True,
    ),
    "contactLenses": ItemTypeProfile(
        keywords=("contact", "contacts", "lens", "lenses", "iris", "pupil", "eye", "eyes"),
        prefer_icon=True,
    ),
    "lips": ItemTypeProfile(
        keywords=("lip", "lips", "lipstick", "gloss"),
        prefer_icon=True,
    ),
    "skinTones": ItemTypeProfile(
        keywords=("skin", "tone", "complexion"),
    ),
    "dresses": ItemTypeProfile(
        keywords=(
            "dress",
            "gown",
            "skirt",
            "bodice",
            "sleeve",
            "collar",
            "hem",
            "waist",
            "waistband",
            "train",
            "strap",
            "tulle",
            "lace",
            "bow",
            "ribbon",
            "ruffled",
            "floral",
            "embroidery",
        ),
        style_patterns=DECOR_STYLE_PATTERNS,
    ),
    "faceDecorations": ItemTypeProfile(
        keywords=("face", "cheek", "forehead", "sticker", "decal", "jewel", "heart", "star", "freckle", "tattoo"),
        prefer_icon=True,
    ),
    "chestAccessories": ItemTypeProfile(
        keywords=("brooch", "corsage", "pin", "badge", "flower", "bow", "ribbon", "chest", "gem", "pearl"),
        style_patterns=DECOR_STYLE_PATTERNS,
        prefer_icon=True,
    ),
    "pendants": ItemTypeProfile(
        keywords=("pendant", "charm", "necklace", "chain", "gem", "pearl"),
        style_patterns=((r"pearl", "pearl"), (r"gem|crystal", "gem")),
        prefer_icon=True,
    ),
    "backpieces": ItemTypeProfile(
        keywords=("back", "backpack", "pack", "wing", "wings", "tail", "cape", "bow", "ribbon"),
        style_patterns=DECOR_STYLE_PATTERNS[:3],
        prefer_icon=True,
    ),
    "rings": ItemTypeProfile(
        keywords=("ring", "rings", "band", "gem", "pearl"),
        style_patterns=((r"pearl", "pearl"), (r"gem|crystal", "gem")),
        prefer_icon=True,
    ),
    "armDecorations": ItemTypeProfile(
        keywords=("armlet", "armband", "arm", "bracelet", "cuff", "ribbon", "bow", "lace"),
        style_patterns=DECOR_STYLE_PATTERNS,
        prefer_icon=True,
    ),
    "fullMakeup": ItemTypeProfile(
        keywords=("makeup", "blush", "eyeshadow", "eyeliner", "lip", "freckles", "highlight", "face paint"),
        prefer_icon=True,
    ),
    "abilityHandhelds": ItemTypeProfile(
        keywords=("staff", "wand", "scepter", "lantern", "umbrella", "parasol", "fan", "sword", "instrument", "bag", "handheld"),
        style_patterns=DECOR_STYLE_PATTERNS[:3],
        prefer_icon=True,
    ),
    "unknown": ItemTypeProfile(
        keywords=(
            "hair",
            "dress",
            "skirt",
            "shoe",
            "hat",
            "bow",
            "ribbon",
            "lace",
            "floral",
            "glove",
            "necklace",
        ),
        style_patterns=DECOR_STYLE_PATTERNS,
    ),
}

APPAREL_TYPES = {
    item_type
    for item_type, profile in TYPE_PROFILES.items()
    if item_type
    in {"outerwear", "tops", "bottoms", "socks", "shoes", "dresses"}
}
ACCESSORY_TYPES = {
    item_type
    for item_type, profile in TYPE_PROFILES.items()
    if profile.prefer_icon and item_type not in {"baseMakeup", "eyebrows", "eyelashes", "contactLenses", "lips", "fullMakeup", "faceDecorations"}
}
FACE_DETAIL_TYPES = {
    "baseMakeup",
    "eyebrows",
    "eyelashes",
    "contactLenses",
    "lips",
    "faceDecorations",
    "fullMakeup",
}

GENERIC_NARRATIVE_PATTERNS: tuple[str, ...] = (
    r"\b(?:there is|there are|image is|image shows|pictured is)\b",
    r"\b(?:standing|posing|looking|sleeping|smiling|holding|wearing)\b",
    r"\b(?:background|landscape|mountain|tree|snow|camera|expression|mood)\b",
)


def get_type_profile(item_type: str) -> ItemTypeProfile:
    return TYPE_PROFILES.get(item_type, TYPE_PROFILES["unknown"])


def is_visual_term_relevant(term: str, item_type: str) -> bool:
    normalized = term.strip().lower()
    if not normalized:
        return False
    profile = get_type_profile(item_type)
    if any(keyword in normalized for keyword in profile.keywords):
        return True
    if any(re.search(pattern, normalized) for pattern in GENERIC_NARRATIVE_PATTERNS):
        return False
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", normalized)
    return 1 <= len(words) <= 6


_missing_profiles = KNOWN_ITEM_TYPES.difference(TYPE_PROFILES)
if _missing_profiles:
    raise RuntimeError(f"Missing item type profiles: {sorted(_missing_profiles)}")
