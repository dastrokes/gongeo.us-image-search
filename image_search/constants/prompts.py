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
)

# slot: (category_phrase, subcategory_label, strict_subcategory)
# strict_subcategory=True  → "only when a narrower visible form is unambiguous"
# strict_subcategory=False → "when clearly visible"
_SLOT_META: dict[str, tuple[str, str, bool]] = {
    "outerwear": ("visible outer-layer forms", "outerwear", False),
    "tops": ("visible upper-body forms", "top", False),
    "bottoms": ("visible lower-body forms", "bottom", False),
    "dresses": ("visible one-piece forms", "one-piece", False),
    "hair": ("broad visible hairstyle classes", "hairstyle", False),
    "shoes": ("broad visible footwear classes", "footwear", False),
    "socks": ("broad visible legwear classes", "legwear", False),
    "accessory": ("the visible object form", "accessory", True),
    "pendants": ("the visible object form", "pendant", True),
}

_SLOT_CATEGORY_PREFIXES: dict[str, str] = {
    slot: f"Choose category from {phrase}"
    for slot, (phrase, _, _) in _SLOT_META.items()
}

_SLOT_SUBCATEGORY_PREFIXES: dict[str, str] = {
    slot: (
        f"Refine {label} category only when a narrower visible form is unambiguous."
        if strict
        else f"Refine {label} category when clearly visible."
    )
    for slot, (_, label, strict) in _SLOT_META.items()
}

_SLOT_CATEGORY_PREAMBLES: dict[str, str] = {
    "accessory": "The slot label is a placement hint, not the category.",
    "pendants": (
        "Pendants is an exception slot that may contain non-neckwear items. "
        "Do not default to neckwear just because the slot is named pendants."
    ),
}

_GARMENT_SLOTS: frozenset[str] = frozenset({"outerwear", "tops", "bottoms", "dresses"})

_SLOT_SPECS: dict[str, dict[str, tuple[str, ...] | str]] = {
    "outerwear": {
        "category_tokens": (
            # True structural outerwear silhouettes only
            "jacket",
            "coat",
            "cape",
            "cardigan",
            "vest",
            "robe",
            "stole",
        ),
        "subcategory_tokens": (
            # blazer/bolero/shrug are cropped or tailored jacket variants
            # cloak/poncho are cape variants
            # kimono as outerwear is a robe variant
            "blazer",
            "bolero",
            "shrug",
            "cropped_jacket",
            "lace_jacket",
            "leather_jacket",
            "trench_coat",
            "fur_coat",
            "pea_coat",
            "duster_coat",
            "hooded_cape",
            "capelet",
            "cloak",
            "poncho",
            "kimono",
            "windbreaker",
        ),
    },
    "tops": {
        "category_tokens": (
            # Distinct garment types by structure and silhouette
            # crop_top/halter_top are length/neckline modifiers, not garment types
            "blouse",
            "shirt",
            "sweater",
            "camisole",
            "corset",
            "bustier",
            "tunic",
            "tank_top",
            "tube_top",
            "bodysuit",
        ),
        "subcategory_tokens": (
            # Length and neckline modifiers, plus specific cut variants
            "crop_top",
            "halter_top",
            "off_shoulder_top",
            "wrap_top",
            "puff_sleeve_blouse",
            "sailor_blouse",
            "ruffled_blouse",
            "knit_sweater",
            "turtleneck",
        ),
    },
    "bottoms": {
        "category_tokens": (
            # Fundamental lower-body silhouette types
            # bloomers/culottes/harem_pants/palazzo are pants variants
            "skirt",
            "pants",
            "shorts",
            "leggings",
            "overalls",
        ),
        "subcategory_tokens": (
            # pants variants
            "bloomers",
            "culottes",
            "harem_pants",
            "palazzo_pants",
            "wide_leg_pants",
            "straight_leg_pants",
            "tapered_pants",
            "cropped_pants",
            # skirt variants by cut and length
            "pleated_skirt",
            "tiered_skirt",
            "bubble_skirt",
            "circle_skirt",
            "pencil_skirt",
            "wrap_skirt",
            "gathered_skirt",
            "layered_skirt",
            "a_line_skirt",
            "mini_skirt",
            "midi_skirt",
            "maxi_skirt",
        ),
    },
    "dresses": {
        "category_tokens": (
            # dress: all one-piece skirted garments
            # jumpsuit: one-piece with legs, kept here until a dedicated slot exists
            "dress",
            "jumpsuit",
        ),
        "subcategory_tokens": (
            # jumpsuit variants — romper/playsuit are short; onesie is full-body hooded
            "romper",
            "playsuit",
            "onesie",
            # dress variants — culturally specific silhouettes must be listed
            # explicitly or the model will collapse them into generic dress
            "gown",
            "ball_gown",
            "sundress",
            "qipao",
            "hanbok",
            "yukata",
            "pinafore_dress",
            "overall_dress",
            "shirt_dress",
            "wrap_dress",
            "slip_dress",
            "bodycon_dress",
            "babydoll_dress",
            "a_line_dress",
            "sheath_dress",
            "lolita_dress",
            "tiered_dress",
            "mini_dress",
            "midi_dress",
            "maxi_dress",
        ),
    },
    "hair": {
        "category_tokens": (
            # Organised by style silhouette, not length
            # Length is a separate attribute field (hair_length)
            "loose",
            "ponytail",
            "twin_tails",
            "bun",
            "braid",
            "updo",
            "half_up",
            "bob",
        ),
        "subcategory_tokens": (
            # ponytail variants by position
            "high_ponytail",
            "low_ponytail",
            "side_ponytail",
            # bun variants
            "space_buns",
            "top_knot",
            # braid variants
            "twin_braids",
            "french_braid",
            "fishtail_braid",
            # style-specific
            "drill_hair",
            "hime_cut",
            "curled_bob",
            "ahoge",
        ),
    },
    "shoes": {
        "category_tokens": (
            # Fundamental footwear types by sole/structure
            # pumps = heel variant; mary_janes = flat variant
            # platform is a modifier, not a shoe type
            "boots",
            "heels",
            "sandals",
            "flats",
            "loafers",
            "sneakers",
            "mules",
            "slippers",
        ),
        "subcategory_tokens": (
            # boot variants by shaft height
            "ankle_boots",
            "knee_boots",
            "over_knee_boots",
            "platform_boots",
            # heel variants by shape
            "pumps",
            "stiletto_heels",
            "block_heels",
            "kitten_heels",
            "wedge_heels",
            "platform_heels",
            "t_strap_heels",
            # flat variants
            "mary_janes",
            "ballet_flats",
            "lace_up_flats",
            # cultural footwear with distinct silhouette
            "geta",
        ),
    },
    "socks": {
        "category_tokens": (
            # Fundamental legwear types by construction
            # thigh_highs = stockings variant; pantyhose = tights variant
            # height-specific socks (ankle/knee/over_knee) are length modifiers
            "socks",
            "stockings",
            "tights",
            "leg_warmers",
        ),
        "subcategory_tokens": (
            # socks by height
            "ankle_socks",
            "knee_socks",
            "over_knee_socks",
            # stockings variants
            "thigh_highs",
            "lace_stockings",
            "garter_stockings",
            # tights variants
            "pantyhose",
            "fishnet_tights",
            "sheer_tights",
            "printed_tights",
            # detail variants
            "ruffled_socks",
            "ribbed_socks",
        ),
    },
    "accessory": {
        "category_tokens": (
            # Visually distinct object forms
            # Items with unique silhouettes must be listed explicitly
            # or the model will default to a generic nearby token
            "headband",
            "bow",
            "ribbon",
            "flower",
            "clip",
            "hat",
            "bonnet",
            "crown",
            "tiara",
            "veil",
            "mask",
            "earring",
            "necklace",
            "pendant",
            "choker",
            "collar",
            "brooch",
            "bracelet",
            "cuff",
            "ring",
            "armlet",
            "gloves",
            "scarf",
            "fan",
            "parasol",
            "wand",
            "staff",
            "lantern",
            "book",
            "wings",
            "pin",
            "badge",
        ),
        "subcategory_tokens": (),
    },
    "pendants": {
        "category_tokens": (
            # Decorative hanging/attached items
            # Bags retained here until a dedicated bags slot is added
            # garter refers to decorative leg garter pieces, not legwear
            "pendant",
            "locket",
            "charm",
            "medallion",
            "tassel",
            "keychain",
            "garter",
            "bag",
            "crossbody_bag",
            "shoulder_bag",
            "satchel",
        ),
        "subcategory_tokens": (),
    },
}


def _format_token_list(tokens: tuple[str, ...]) -> str:
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    if len(tokens) == 2:
        return f"{tokens[0]} or {tokens[1]}"
    return f"{', '.join(tokens[:-1])}, or {tokens[-1]}"


def _build_slot_guidance(
    slot: str,
    *,
    category_tokens: tuple[str, ...],
    subcategory_tokens: tuple[str, ...],
) -> dict[str, str]:
    preamble = _SLOT_CATEGORY_PREAMBLES.get(slot)
    category_parts = []
    if preamble:
        category_parts.append(preamble)
    category_parts.append(
        f"{_SLOT_CATEGORY_PREFIXES[slot]}: {_format_token_list(category_tokens)}."
    )
    subcategory = _SLOT_SUBCATEGORY_PREFIXES[slot]
    if subcategory_tokens:
        preview = _format_token_list(subcategory_tokens[:3])
        subcategory = f"{subcategory} Examples include {preview}."
    else:
        subcategory = str(subcategory)
    if slot in _GARMENT_SLOTS:
        subcategory = (
            f"{subcategory} Garment rule: always output a non-null subcategory token. "
        )
    else:
        subcategory = f"{subcategory} Null if unclear."
    return {
        "category": " ".join(category_parts),
        "subcategory": subcategory,
    }


CANONICAL_CATEGORY_TOKENS: dict[str, tuple[str, ...]] = {
    slot: spec["category_tokens"] for slot, spec in _SLOT_SPECS.items()
}
CANONICAL_CATEGORY_TOKENS.update(
    {
        item_type: CANONICAL_CATEGORY_TOKENS["accessory"]
        for item_type in ACCESSORY_ITEM_TYPES
        if item_type != "pendants"
    }
)

CANONICAL_SUBCATEGORY_TOKENS: dict[str, tuple[str, ...]] = {
    slot: spec["subcategory_tokens"] for slot, spec in _SLOT_SPECS.items()
}

SLOT_GUIDANCE: dict[str, dict[str, str]] = {
    slot: _build_slot_guidance(
        slot,
        category_tokens=CANONICAL_CATEGORY_TOKENS[slot],
        subcategory_tokens=CANONICAL_SUBCATEGORY_TOKENS[slot],
    )
    for slot in _SLOT_SPECS
}

# Closed-list fields only — model must pick from these tokens exclusively.
# Contains only fields present in FILTERED_CANONICAL_ATTRIBUTE_FIELDS
# plus genuinely closed enumerations (colors).
# Must not overlap with EXAMPLE_ATTRIBUTE_TOKENS.
CANONICAL_ATTRIBUTE_TOKENS: dict[str, tuple[str, ...]] = {
    # garment length fields (FILTERED_CANONICAL_ATTRIBUTE_FIELDS)
    "outerwear_length": (
        "waist_length",
        "hip_length",
        "thigh_length",
        "knee_length",
    ),
    "top_length": (
        "crop",
        "waist_length",
        "hip_length",
        "tunic_length",
    ),
    "bottom_length": (
        "mini",
        "knee_length",
        "midi",
        "maxi",
    ),
    "dress_length": (
        "mini",
        "knee_length",
        "midi",
        "floor_length",
    ),
    "hair_length": (
        "short",
        "medium",
        "long",
        "very_long",
    ),
    # sleeve length (FILTERED_CANONICAL_ATTRIBUTE_FIELDS)
    "sleeve_length": (
        "sleeveless",
        "cap",
        "short",
        "elbow",
        "three_quarter",
        "long",
    ),
    # shoe/boot measurement fields (FILTERED_CANONICAL_ATTRIBUTE_FIELDS)
    "shaft_height": (
        "ankle",
        "mid_calf",
        "knee_high",
        "over_knee",
    ),
    "heel_height": (
        "flat",
        "low",
        "mid",
        "high",
    ),
    # legwear measurement field (FILTERED_CANONICAL_ATTRIBUTE_FIELDS)
    "sock_height": (
        "ankle",
        "knee",
        "over_knee",
        "thigh_high",
    ),
    # color enumerations (genuinely closed)
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
        "rose_gold",
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
        "rose_gold",
    ),
}

# Illustrative examples only — not exhaustive, model may go beyond these.
# Must not overlap with CANONICAL_ATTRIBUTE_TOKENS.
EXAMPLE_ATTRIBUTE_TOKENS: dict[str, tuple[str, ...]] = {
    # surface appearance
    "pattern": ("floral", "plaid", "polka_dot", "star", "gingham", "gradient"),
    "material": ("lace", "tulle", "velvet", "sheer", "knit", "metallic"),
    "ornament": ("bow", "ruffle", "embroidery", "pearl", "lace_trim", "star"),
    # construction
    "closure": ("button_front", "zipper", "lace_up", "belt", "corset_lacing"),
    "silhouette": ("a_line", "fitted", "wrap", "tiered", "empire"),
    "neckline": ("sweetheart", "v_neck", "off_shoulder", "halter", "high_neck"),
    "collar": (
        "peter_pan",
        "sailor",
        "mandarin",
        "lace_collar",
        "bertha_collar",
        "ruffle_collar",
    ),
    "sleeve_shape": ("puff", "bell", "bishop", "flutter", "detached"),
    # aesthetic
    "style": (
        "gothic",
        "ethereal",
        "elegant",
        "romantic",
        "cottagecore",
        "pastoral",
    ),
    "theme": ("fairy", "maid", "magical_girl", "witch", "princess"),
    "occasion": ("bridal", "uniform", "party", "festival", "stage"),
    # hair-specific
    "texture": ("straight", "wavy", "curly"),
    "parting": ("center_part", "side_part", "no_part"),
    "bangs": ("blunt_bangs", "side_swept_bangs", "curtain_bangs"),
    "hairstyle": (
        "high_ponytail",
        "side_ponytail",
        "twin_braids",
        "space_buns",
        "half_up",
    ),
    "adornment": ("bow", "ribbon", "flower", "clip", "pin"),
    # shoes-specific
    "toe_shape": ("round_toe", "almond_toe", "pointed_toe", "square_toe"),
    "platform": ("none", "low_platform", "high_platform"),
    # legwear-specific
    "opacity": ("sheer", "semi_sheer", "opaque"),
    "trim": ("lace_trim", "ruffle_trim", "ribbed_trim"),
    # accessory-specific
    "placement": ("head", "neck", "chest", "waist", "back", "arm", "hand", "leg"),
    "attachment": ("clip_on", "tie_on", "wrap", "dangle", "pin_on"),
    "shape": ("heart", "star", "flower", "cross", "wing", "moon"),
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
    filtered_lines: list[str] = []
    example_lines: list[str] = []
    category_tokens = CANONICAL_CATEGORY_TOKENS.get(slot)
    if "category" in field_names and category_tokens:
        filtered_lines.append(f"  category: {', '.join(category_tokens)}")
    subcategory_tokens = CANONICAL_SUBCATEGORY_TOKENS.get(slot)
    if "subcategory" in field_names and subcategory_tokens:
        example_lines.append(f"  subcategory examples: {', '.join(subcategory_tokens)}")
    for name in field_names:
        if name in FILTERED_CANONICAL_ATTRIBUTE_FIELDS:
            tokens = CANONICAL_ATTRIBUTE_TOKENS.get(name)
            if tokens:
                filtered_lines.append(f"  {name}: {', '.join(tokens)}")
            continue
        if name in _EXAMPLE_VOCAB_FIELDS:
            tokens = _example_tokens_for(name)
            if tokens:
                example_lines.append(f"  {name} examples: {', '.join(tokens)}")
            continue
    sections: list[str] = []
    if filtered_lines:
        sections.append(
            "Filtered token vocabulary (treat as closed lists):\n"
            + "\n".join(filtered_lines)
        )
    if example_lines:
        sections.append(
            "Example vocabulary (illustrative, not exhaustive):\n"
            + "\n".join(example_lines)
        )
    if not sections:
        return ""
    return "\n\n".join(sections)


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
    if slot in _GARMENT_SLOTS:
        sections.append(
            "Garment extraction rule: `subcategory` is required and must never be null."
        )
    if canonical:
        sections.append(canonical)
    if field_guidance:
        sections.append(field_guidance)
    if visual_tags and visual_tags.strip():
        sections.append(f"Visual context from tagger: {visual_tags}")
    sections.append(
        "Extract the structured JSON for the item in the provided image(s).\n"
        "Image order: overview first, icon second when both are present.\n"
        "If the images conflict, prefer the overview image.\n"
        f"Output only the following JSON, nothing else:\n{schema}"
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

FILTERED_CANONICAL_ATTRIBUTE_FIELDS: frozenset[str] = frozenset(
    {
        "outerwear_length",
        "top_length",
        "bottom_length",
        "dress_length",
        "hair_length",
        "sleeve_length",
        "shaft_height",
        "heel_height",
        "sock_height",
    }
)

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


__all__ = [
    "ATTRIBUTE_GUIDANCE",
    "CANONICAL_ATTRIBUTE_TOKENS",
    "CANONICAL_CATEGORY_TOKENS",
    "CANONICAL_SUBCATEGORY_TOKENS",
    "EXAMPLE_ATTRIBUTE_TOKENS",
    "FILTERED_CANONICAL_ATTRIBUTE_FIELDS",
    "SLOT_GUIDANCE",
    "STRUCTURED_EXTRACTION_SYSTEM_PROMPT",
    "SUBCATEGORY_HIERARCHY",
    "TOKEN_ALIASES",
    "build_extraction_user_message",
    "get_subcategory_ancestors",
    "normalise_token",
]
