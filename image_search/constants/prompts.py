from __future__ import annotations

from image_search.constants.structured import (
    ACCESSORY_ITEM_TYPES,
    shape_definition_for_item_type,
)

STRUCTURED_EXTRACTION_SYSTEM_PROMPT = (
    "You are a deterministic structured extraction engine for Infinity Nikki items.\n"
    "Output exactly one JSON object. No markdown, no code fences, no prose.\n"
    "Use only the schema keys exactly as written.\n"
    "All non-null values must be short lowercase underscore_tokens.\n"
    "Scalar fields: one token or null. Array fields: zero or more unique tokens, or [].\n"
    "Describe only the target item in the provided images using only directly visible evidence.\n"
    "Do not infer hidden, back-side, off-frame, gameplay, or lore details.\n"
    "category: broadest stable visible class of the item.\n"
    "subcategory: narrower visible refinement of category; null if category is already maximally specific.\n"
    "Do not encode the same concept in more than one field.\n"
    "Do not use the slot name as category or subcategory.\n"
    "subcategory must be a valid refinement of category.\n"
    "Never output an incompatible category/subcategory pair.\n"
    "Prefer fewer high-confidence attributes over many uncertain ones.\n"
    "When unsure between a specific and a generic label, use the generic label.\n"
    "When unsure between two specific labels, prefer the more general one.\n"
    "If a field is unclear or inapplicable, output null or [].\n"
)

ATTRIBUTE_GUIDANCE = (
    "Field boundaries: pattern=repeat motifs; material=fabric/material terms; ornament=decorative details; "
    "closure=functional fastening; style=visible aesthetic mood; theme=costume archetype; occasion=functional context.\n"
    "Common misplacements:\n"
    "  lace      -> material (the fabric). lace_trim -> ornament (the applied edging).\n"
    "  sheer     -> material (not pattern).\n"
    "  bow       -> ornament (not closure).\n"
    "  star      -> pattern if a repeat print; ornament if a single applique or charm.\n"
    "  fairy     -> theme (not style). ethereal -> style (not theme).\n"
)

SLOT_GUIDANCE: dict[str, dict[str, str]] = {
    "outerwear": {
        "category": (
            "Choose category from visible outer-layer forms: "
            "jacket, coat, cape, cardigan, bolero, shrug, cloak, capelet, or vest."
        ),
        "subcategory": (
            "Refine outerwear category when clearly visible. "
            "Examples include trench_coat, fur_coat, or capelet. "
            "Null if unclear."
        ),
    },
    "tops": {
        "category": (
            "Choose category from visible upper-body forms: "
            "blouse, shirt, sweater, vest, camisole, corset, tunic, or crop_top."
        ),
        "subcategory": (
            "Refine top category when clearly visible. "
            "Examples include sailor_blouse, puff_sleeve_blouse, or bustier. "
            "Null if unclear."
        ),
    },
    "bottoms": {
        "category": (
            "Choose category from visible lower-body forms: "
            "skirt, pants, shorts, leggings, bloomers, or overalls."
        ),
        "subcategory": (
            "Refine bottom category when clearly visible. "
            "Examples include tiered_skirt, bubble_skirt, or pleated_skirt. "
            "Null if unclear."
        ),
    },
    "dresses": {
        "category": (
            "Choose category from visible one-piece forms: "
            "dress, gown, jumpsuit, romper, pinafore, overall_dress, qipao, or sundress."
        ),
        "subcategory": (
            "Refine one-piece category when clearly visible. "
            "Examples include lolita_dress, ball_gown, or tiered_dress. "
            "Null if unclear."
        ),
    },
    "hair": {
        "category": (
            "Choose category from broad visible hairstyle classes: "
            "long_hair, short_hair, bob, ponytail, twin_tails, pigtails, "
            "bun, braid, updo, or half_up."
        ),
        "subcategory": (
            "Refine hairstyle category when clearly visible. "
            "Examples include drill_hair, twin_braids, or hime_cut. "
            "Null if unclear."
        ),
    },
    "shoes": {
        "category": (
            "Choose category from broad visible footwear classes: "
            "boots, heels, sandals, flats, loafers, sneakers, "
            "pumps, mary_janes, or platform_shoes."
        ),
        "subcategory": (
            "Refine footwear category when clearly visible. "
            "Examples include ankle_boots, platform_heels, or ballet_flats. "
            "Null if unclear."
        ),
    },
    "socks": {
        "category": (
            "Choose category from broad visible legwear classes: "
            "socks, stockings, tights, thigh_highs, "
            "over_knee_socks, knee_socks, ankle_socks, or leg_warmers."
        ),
        "subcategory": (
            "Refine legwear category when clearly visible. "
            "Examples include fishnet_tights, lace_stockings, or ruffled_ankle_socks. "
            "Null if unclear."
        ),
    },
    "accessory": {
        "category": (
            "The slot label is a placement hint, not the category. "
            "Choose category from the visible object form: "
            "bow, ribbon, flower, clip, hat, bonnet, crown, veil, "
            "earring, necklace, brooch, bracelet, cuff, choker, "
            "gloves, fan, bag, book, lantern, wings, ring, or armlet."
        ),
        "subcategory": (
            "Refine accessory category only when a narrower visible form is unambiguous. "
            "Null if unclear."
        ),
    },
    "pendants": {
        "category": (
            "Pendants is an exception slot that may contain non-neckwear items. "
            "Choose category from the visible object form: "
            "pendant, locket, charm, medallion, tassel, "
            "bag, crossbody_bag, shoulder_bag, satchel, or garter. "
            "Do not default to neckwear just because the slot is named pendants."
        ),
        "subcategory": (
            "Refine pendant category only when a narrower visible form is unambiguous. "
            "Null if unclear."
        ),
    },
}

_OUTERWEAR_CATEGORY_TOKENS: tuple[str, ...] = (
    "jacket",
    "coat",
    "cape",
    "cardigan",
    "bolero",
    "shrug",
    "cloak",
    "capelet",
    "vest",
)

_TOP_CATEGORY_TOKENS: tuple[str, ...] = (
    "blouse",
    "shirt",
    "sweater",
    "vest",
    "camisole",
    "corset",
    "tunic",
    "crop_top",
)

_BOTTOM_CATEGORY_TOKENS: tuple[str, ...] = (
    "skirt",
    "pants",
    "shorts",
    "leggings",
    "bloomers",
    "overalls",
)

_DRESS_CATEGORY_TOKENS: tuple[str, ...] = (
    "dress",
    "gown",
    "jumpsuit",
    "romper",
    "pinafore",
    "overall_dress",
    "qipao",
    "sundress",
)

_HAIR_CATEGORY_TOKENS: tuple[str, ...] = (
    "long_hair",
    "short_hair",
    "bob",
    "ponytail",
    "twin_tails",
    "pigtails",
    "bun",
    "braid",
    "updo",
    "half_up",
)

_SHOE_CATEGORY_TOKENS: tuple[str, ...] = (
    "boots",
    "heels",
    "sandals",
    "flats",
    "loafers",
    "sneakers",
    "pumps",
    "mary_janes",
    "platform_shoes",
)

_SOCK_CATEGORY_TOKENS: tuple[str, ...] = (
    "socks",
    "stockings",
    "tights",
    "thigh_highs",
    "over_knee_socks",
    "knee_socks",
    "ankle_socks",
    "leg_warmers",
)

_ACCESSORY_CATEGORY_TOKENS: tuple[str, ...] = (
    "bow",
    "ribbon",
    "flower",
    "clip",
    "hat",
    "bonnet",
    "crown",
    "veil",
    "earring",
    "necklace",
    "brooch",
    "bracelet",
    "cuff",
    "choker",
    "gloves",
    "fan",
    "bag",
    "book",
    "lantern",
    "wings",
    "ring",
    "armlet",
)

_PENDANT_CATEGORY_TOKENS: tuple[str, ...] = (
    "pendant",
    "locket",
    "charm",
    "medallion",
    "tassel",
    "bag",
    "crossbody_bag",
    "shoulder_bag",
    "satchel",
    "garter",
)

CANONICAL_CATEGORY_TOKENS: dict[str, tuple[str, ...]] = {
    "outerwear": _OUTERWEAR_CATEGORY_TOKENS,
    "tops": _TOP_CATEGORY_TOKENS,
    "bottoms": _BOTTOM_CATEGORY_TOKENS,
    "dresses": _DRESS_CATEGORY_TOKENS,
    "hair": _HAIR_CATEGORY_TOKENS,
    "shoes": _SHOE_CATEGORY_TOKENS,
    "socks": _SOCK_CATEGORY_TOKENS,
    "accessory": _ACCESSORY_CATEGORY_TOKENS,
    **{
        item_type: _ACCESSORY_CATEGORY_TOKENS
        for item_type in ACCESSORY_ITEM_TYPES
        if item_type != "pendants"
    },
    "pendants": _PENDANT_CATEGORY_TOKENS,
}

CANONICAL_SUBCATEGORY_TOKENS: dict[str, tuple[str, ...]] = {
    "outerwear": (
        "trench_coat",
        "fur_coat",
        "hooded_cape",
        "lace_jacket",
        "cropped_jacket",
        "capelet",
    ),
    "tops": (
        "puff_sleeve_blouse",
        "sailor_blouse",
        "ruffled_blouse",
        "off_shoulder_top",
        "tube_top",
        "bustier",
        "hoodie",
        "waistcoat",
    ),
    "bottoms": (
        "pleated_skirt",
        "tiered_skirt",
        "bubble_skirt",
        "gathered_skirt",
        "layered_skirt",
        "a_line_skirt",
        "mini_skirt",
        "midi_skirt",
        "maxi_skirt",
        "wide_leg_pants",
    ),
    "dresses": (
        "lolita_dress",
        "ball_gown",
        "tiered_dress",
        "a_line_dress",
        "slip_dress",
        "pinafore_dress",
        "wrap_dress",
    ),
    "hair": (
        "side_ponytail",
        "low_ponytail",
        "twin_braids",
        "french_braid",
        "space_buns",
        "drill_hair",
        "hime_cut",
        "curled_bob",
        "ahoge",
    ),
    "shoes": (
        "ankle_boots",
        "knee_boots",
        "over_knee_boots",
        "platform_boots",
        "platform_heels",
        "block_heels",
        "t_strap_heels",
        "wedge_heels",
        "ballet_flats",
        "lace_up_shoes",
    ),
    "socks": (
        "fishnet_tights",
        "sheer_tights",
        "lace_stockings",
        "garter_stockings",
        "ruffled_ankle_socks",
        "ribbed_socks",
    ),
    "accessory": (),
    "pendants": (),
}

CANONICAL_ATTRIBUTE_TOKENS: dict[str, tuple[str, ...]] = {
    "pattern": (
        "floral",
        "stripe",
        "plaid",
        "checkered",
        "polka_dot",
        "star",
        "heart",
        "cherry",
        "moon",
        "mushroom",
        "rose_print",
        "butterfly",
        "cross",
        "gingham",
        "toile",
        "damask",
        "animal_print",
    ),
    "material": (
        "lace",
        "tulle",
        "chiffon",
        "velvet",
        "organza",
        "satin",
        "silk",
        "voile",
        "brocade",
        "jacquard",
        "mesh",
        "fur",
        "sheer",
        "cotton",
        "knit",
        "leather",
        "denim",
        "wool",
        "taffeta",
    ),
    "ornament": (
        "bow",
        "ribbon",
        "ruffle",
        "frill",
        "sequin",
        "embroidery",
        "bead",
        "tassel",
        "fringe",
        "pearl",
        "crystal",
        "flower",
        "rosette",
        "lace_trim",
        "pom_pom",
        "feather",
        "corsage",
        "applique",
    ),
    "closure": (
        "button",
        "button_front",
        "zipper",
        "lace_up",
        "corset_lacing",
        "tie",
        "belt",
        "buckle",
        "drawstring",
    ),
    "style": (
        "gothic",
        "elegant",
        "cute",
        "ethereal",
        "pastoral",
        "vintage",
        "military",
        "punk",
        "sporty",
    ),
    "theme": (
        "witch",
        "maid",
        "fairy",
        "shrine_maiden",
        "magical_girl",
        "cowgirl",
        "pirate",
        "knight",
        "nurse",
        "idol",
        "angel",
        "demon",
    ),
    "occasion": (
        "bridal",
        "swimwear",
        "pajamas",
        "homewear",
        "uniform",
        "festival",
        "party",
    ),
    "primary_color": (
        "white",
        "black",
        "gray",
        "cream",
        "beige",
        "pink",
        "red",
        "burgundy",
        "orange",
        "yellow",
        "green",
        "mint",
        "teal",
        "blue",
        "navy",
        "purple",
        "lavender",
        "brown",
        "gold",
        "silver",
    ),
    "secondary_color": (
        "white",
        "black",
        "gray",
        "cream",
        "beige",
        "pink",
        "red",
        "burgundy",
        "orange",
        "yellow",
        "green",
        "mint",
        "teal",
        "blue",
        "navy",
        "purple",
        "lavender",
        "brown",
        "gold",
        "silver",
    ),
    "silhouette": (
        "a_line",
        "ball_gown",
        "fitted",
        "shift",
        "wrap",
        "tiered",
        "mermaid",
        "empire",
        "dropped_waist",
    ),
    "neckline": (
        "round",
        "v_neck",
        "square",
        "sweetheart",
        "off_shoulder",
        "halter",
        "boat",
        "high_neck",
        "strapless",
        "scoop",
    ),
    "collar": (
        "peter_pan",
        "sailor",
        "mandarin",
        "ruffle_collar",
        "lace_collar",
        "bow_collar",
        "flat_collar",
    ),
    "sleeve_length": (
        "sleeveless",
        "cap",
        "short",
        "elbow",
        "three_quarter",
        "long",
    ),
    "sleeve_shape": (
        "puff",
        "bell",
        "bishop",
        "fitted",
        "flutter",
        "lantern",
        "leg_of_mutton",
        "ruffle_cuff",
    ),
}


def _build_schema_template(fields: tuple[str, ...]) -> str:
    lines = ["{"]
    last = len(fields) - 1
    for i, field in enumerate(fields):
        comma = "," if i < last else ""
        value = "null"
        lines.append(f'  "{field}": {value}{comma}')
    lines.append("}")
    return "\n".join(lines)


SUBCATEGORY_HIERARCHY: dict[str, str] = {
    # "platform_boots": "boots",
}

TOKEN_ALIASES: dict[str, str] = {
    # "tee": "shirt",
}

FUZZY_OVERLAP_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"ethereal", "fairy"}),
        frozenset({"pastoral", "cottagecore"}),
        frozenset({"ruffle", "frill"}),
        frozenset({"elegant", "vintage"}),
        frozenset({"magical_girl", "fairy"}),
        frozenset({"idol", "cute"}),
    }
)

_EXHAUSTIVE_CANONICAL_FIELDS: frozenset[str] = frozenset({"sleeve_length"})

_EXAMPLE_VOCAB_FIELDS: frozenset[str] = frozenset(
    {
        "pattern",
        "material",
        "ornament",
        "style",
        "theme",
        "occasion",
        "outerwear_length",
        "top_length",
        "bottom_length",
        "dress_length",
        "hair_length",
        "silhouette",
        "neckline",
        "collar",
        "closure",
        "sleeve_shape",
        "texture",
        "parting",
        "bangs",
        "hairstyle",
        "adornment",
        "shaft_height",
        "heel_height",
        "toe_shape",
        "platform",
        "sock_height",
        "opacity",
        "trim",
        "placement",
        "attachment",
        "shape",
    }
)

EXAMPLE_ATTRIBUTE_TOKENS: dict[str, tuple[str, ...]] = {
    "pattern": ("floral", "plaid", "polka_dot", "star", "gingham"),
    "material": ("lace", "tulle", "velvet", "sheer", "knit"),
    "ornament": ("bow", "ruffle", "embroidery", "pearl", "lace_trim"),
    "style": ("cute", "gothic", "ethereal", "elegant", "pastoral"),
    "theme": ("fairy", "maid", "magical_girl", "witch", "cowgirl"),
    "occasion": ("bridal", "uniform", "party", "festival", "pajamas"),
    "outerwear_length": ("waist_length", "hip_length", "thigh_length", "knee_length"),
    "top_length": ("crop", "waist_length", "hip_length", "tunic_length"),
    "bottom_length": ("mini", "knee_length", "midi", "maxi"),
    "dress_length": ("mini", "knee_length", "midi", "floor_length"),
    "hair_length": ("short", "medium", "long", "very_long"),
    "silhouette": ("a_line", "fitted", "wrap", "tiered", "empire"),
    "neckline": ("sweetheart", "v_neck", "off_shoulder", "halter", "high_neck"),
    "collar": ("peter_pan", "sailor", "mandarin", "lace_collar"),
    "closure": ("button_front", "zipper", "lace_up", "belt", "tie"),
    "sleeve_shape": ("puff", "bell", "bishop", "flutter", "ruffle_cuff"),
    "texture": ("straight", "wavy", "curly"),
    "parting": ("center_part", "side_part", "no_part"),
    "bangs": ("blunt_bangs", "side_swept_bangs", "curtain_bangs"),
    "hairstyle": ("twin_braids", "side_ponytail", "space_buns", "half_up"),
    "adornment": ("bow", "ribbon", "flower", "clip"),
    "shaft_height": ("ankle", "mid_calf", "knee_high", "over_knee"),
    "heel_height": ("flat", "low", "mid", "high"),
    "toe_shape": ("round_toe", "almond_toe", "pointed_toe", "square_toe"),
    "platform": ("none", "low_platform", "high_platform"),
    "sock_height": ("ankle", "knee", "over_knee", "thigh_high"),
    "opacity": ("sheer", "semi_sheer", "opaque"),
    "trim": ("lace_trim", "ruffle_trim", "ribbed_trim"),
    "placement": ("head", "neck", "chest", "back", "arm", "hand"),
    "attachment": ("clip_on", "tie_on", "wrap", "dangle"),
    "shape": ("heart", "star", "flower", "cross", "wing"),
}


def _example_tokens_for(field_name: str) -> tuple[str, ...] | None:
    return EXAMPLE_ATTRIBUTE_TOKENS.get(
        field_name,
        CANONICAL_ATTRIBUTE_TOKENS.get(field_name),
    )


def _slot_guidance_for(slot: str) -> dict[str, str]:
    return SLOT_GUIDANCE.get(slot, SLOT_GUIDANCE["accessory"])


def _field_names_for(slot: str) -> tuple[str, ...]:
    return tuple(
        field_definition.name
        for field_definition in shape_definition_for_item_type(slot).fields
    )


def _schema_template_for(slot: str) -> str:
    shape_definition = shape_definition_for_item_type(slot)
    lines = ["{"]
    last = len(shape_definition.fields) - 1
    for index, field_definition in enumerate(shape_definition.fields):
        comma = "," if index < last else ""
        value = "[]" if field_definition.kind == "array" else "null"
        lines.append(f'  "{field_definition.name}": {value}{comma}')
    lines.append("}")
    return "\n".join(lines)


def _canonical_block(slot: str, field_names: tuple[str, ...]) -> str:
    lines: list[str] = []
    category_tokens = CANONICAL_CATEGORY_TOKENS.get(slot)
    if "category" in field_names and category_tokens:
        lines.append(f"  category: {', '.join(category_tokens)}")
    subcategory_tokens = CANONICAL_SUBCATEGORY_TOKENS.get(slot)
    if "subcategory" in field_names and subcategory_tokens:
        lines.append(f"  subcategory examples: {', '.join(subcategory_tokens)}")
    for name in field_names:
        if name in {"primary_color", "secondary_color"}:
            continue
        if name in _EXHAUSTIVE_CANONICAL_FIELDS:
            tokens = CANONICAL_ATTRIBUTE_TOKENS.get(name)
            if tokens:
                lines.append(f"  {name}: {', '.join(tokens)}")
            continue
        if name in _EXAMPLE_VOCAB_FIELDS:
            tokens = _example_tokens_for(name)
            if tokens:
                lines.append(f"  {name} examples: {', '.join(tokens)}")
            continue
    if not lines:
        return ""
    return "Canonical token vocabulary:\n" + "\n".join(lines)


def build_extraction_user_message(
    slot: str,
    visual_tags: str | None = None,
    *,
    field_guidance: str | None = None,
    field_names: tuple[str, ...] | None = None,
) -> str:
    guidance = _slot_guidance_for(slot)
    schema = _schema_template_for(slot)
    resolved_field_names = field_names or _field_names_for(slot)
    canonical = _canonical_block(slot, resolved_field_names)
    sections: list[str] = [
        f"Slot: {slot}",
        f"Category guidance: {guidance['category']}",
        f"Subcategory guidance: {guidance['subcategory']}",
        ATTRIBUTE_GUIDANCE,
    ]
    if canonical:
        sections.append(canonical)
    if field_guidance:
        sections.append(field_guidance)
    if visual_tags and visual_tags.strip():
        sections.append(f"Visual context from tagger: {visual_tags}")
    sections.append(
        "Extract the structured JSON for the item in the provided image(s).\n"
        "Image order: overview first, icon second when both are present.\n"
        f"Output exactly this JSON:\n{schema}"
    )
    return "\n\n".join(sections)


def normalise_token(token: str) -> str:
    return TOKEN_ALIASES.get(token, token)


def get_subcategory_ancestors(subcategory: str) -> list[str]:
    ancestors: list[str] = []
    current = subcategory
    while current in SUBCATEGORY_HIERARCHY:
        current = SUBCATEGORY_HIERARCHY[current]
        ancestors.append(current)
    return ancestors


__all__ = [
    "ATTRIBUTE_GUIDANCE",
    "CANONICAL_ATTRIBUTE_TOKENS",
    "CANONICAL_CATEGORY_TOKENS",
    "CANONICAL_SUBCATEGORY_TOKENS",
    "EXAMPLE_ATTRIBUTE_TOKENS",
    "FUZZY_OVERLAP_PAIRS",
    "SLOT_GUIDANCE",
    "STRUCTURED_EXTRACTION_SYSTEM_PROMPT",
    "SUBCATEGORY_HIERARCHY",
    "TOKEN_ALIASES",
    "build_extraction_user_message",
    "get_subcategory_ancestors",
    "normalise_token",
]
