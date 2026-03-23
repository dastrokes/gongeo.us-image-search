from __future__ import annotations

from image_search.constants.structured import (
    ACCESSORY_ITEM_TYPES,
    shape_definition_for_item_type,
)

STRUCTURED_EXTRACTION_SYSTEM_PROMPT = (
    "You are a deterministic structured extraction engine for Infinity Nikki items.\n"
    "Output exactly one JSON object. No markdown, no code fences, no prose.\n"
    "Use only the schema keys exactly as written; do not add, rename, or omit keys.\n"
    "All non-null values must be short lowercase underscore_tokens.\n"
    "Scalar fields must be one token or null. Array fields must contain zero or more unique tokens, or [].\n"
    "Describe only the target item from the provided images using directly visible evidence only.\n"
    "Do not infer hidden, back-side, off-frame, gameplay, or lore details.\n"
    "category is the broadest stable visible class of the item.\n"
    "subcategory is a narrower visible refinement of category, or null if no valid refinement is visible.\n"
    "Never use the slot name as category or subcategory.\n"
    "Never encode the same concept in more than one field or output an incompatible category/subcategory pair.\n"
    "Prefer fewer high-confidence attributes over many uncertain ones.\n"
    "When uncertain, choose the more general label.\n"
    "If a field is unclear or inapplicable, output null or [].\n"
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
            # jacket variants — blazer/bolero/shrug are cropped or tailored
            "blazer",
            "bolero",
            "shrug",
            "cropped_jacket",
            "lace_jacket",
            "leather_jacket",
            "haori",
            "windbreaker",
            # coat variants
            "trench_coat",
            "fur_coat",
            "pea_coat",
            "duster_coat",
            "tailcoat",
            # cape variants — cloak/poncho are structural cape silhouettes
            "hooded_cape",
            "capelet",
            "mantle",
            "cloak",
            "poncho",
            # robe variants — kimono is a robe-silhouette outerwear
            "kimono",
        ),
    },
    "tops": {
        "category_tokens": (
            # Distinct garment types by structure and silhouette
            # crop_top/halter_top are length/neckline modifiers, not garment types
            # bustier is a boned bodice variant of corset
            "blouse",
            "shirt",
            "sweater",
            "camisole",
            "corset",
            "tunic",
            "tank_top",
            "tube_top",
            "bodysuit",
        ),
        "subcategory_tokens": (
            # length and neckline modifiers
            "crop_top",
            "halter_top",
            "off_shoulder_top",
            "bardot_top",
            # structural cut variants
            "wrap_top",
            "smock_top",
            # corset/bodice variants
            "bustier",
            # blouse variants
            "puff_sleeve_blouse",
            "sailor_blouse",
            "ruffled_blouse",
            # sweater variants
            "knit_sweater",
            "turtleneck_sweater",
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
            "hakama",
            "skort",
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
            # skirt variants by cut — length is handled by skirt_length field
            "pleated_skirt",
            "tiered_skirt",
            "bubble_skirt",
            "circle_skirt",
            "pencil_skirt",
            "wrap_skirt",
            "gathered_skirt",
            "layered_skirt",
            "a_line_skirt",
        ),
    },
    "dresses": {
        "category_tokens": (
            # dress: all one-piece skirted garments
            # jumpsuit: one-piece with legs, kept here until a dedicated slot exists
            "dress",
            "jumpsuit",
            "onesie",
        ),
        "subcategory_tokens": (
            # jumpsuit variants — romper/playsuit are short
            "romper",
            "playsuit",
            # dress variants — keep garment types here; silhouette-like cuts belong
            # in dress_silhouette instead
            "gown",
            "sundress",
            "kaftan",
            "chemise",
            "qipao",
            "hanbok",
            "yukata",
            "pinafore_dress",
            "overall_dress",
            "shirt_dress",
            "slip_dress",
            "lolita_dress",
            "sweater_dress",
            "sailor_dress",
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
            "side_bun",
            "chignon",
            "odango",
            # braid variants
            "twin_braids",
            "french_braid",
            "fishtail_braid",
            # twin_tails variants
            "pigtails",
            # style-specific
            "drill_hair",
            "hime_cut",
            "curled_bob",
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
            "oxfords",
            "sneakers",
            "mules",
            "slippers",
        ),
        "subcategory_tokens": (
            # boot variants by shaft height
            "ankle_boots",
            "knee_boots",
            "over_knee_boots",
            "thigh_boots",
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
            "zori",
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


def _validate_slot_config() -> None:
    slot_meta_keys = set(_SLOT_META)
    slot_spec_keys = set(_SLOT_SPECS)
    if slot_meta_keys != slot_spec_keys:
        missing_specs = sorted(slot_meta_keys - slot_spec_keys)
        missing_meta = sorted(slot_spec_keys - slot_meta_keys)
        raise RuntimeError(
            "slot config mismatch: "
            f"missing specs={missing_specs}, missing meta={missing_meta}"
        )


_validate_slot_config()


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

_BOTTOMS_LENGTH_TOKENS: tuple[str, ...] = (
    "micro",
    "upper_thigh",
    "mid_thigh",
    "knee_length",
    "mid_calf",
    "ankle_length",
    "floor_length",
)

_COLOR_ENUM_TOKENS: tuple[str, ...] = (
    "white",
    "black",
    "gray",
    "multicolor",
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
)

# Closed-list fields only — model must pick from these tokens exclusively.
# Contains only fields present in FILTERED_CANONICAL_ATTRIBUTE_FIELDS
# plus genuinely closed enumerations (colors).
# Must not overlap with EXAMPLE_ATTRIBUTE_TOKENS.
CANONICAL_ATTRIBUTE_TOKENS: dict[str, tuple[str, ...]] = {
    # garment length fields (FILTERED_CANONICAL_ATTRIBUTE_FIELDS)
    "top_length": (
        "ultra_crop",
        "crop",
        "waist_length",
        "hip_length",
        "thigh_length",
        "knee_length",
        "ankle_length",
        "floor_length",
    ),
    "bottom_length": _BOTTOMS_LENGTH_TOKENS,
    "dress_length": (
        "mini",
        "knee_length",
        "midi",
        "maxi",
        "ankle_length",
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
        "extra_long",
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
    "primary_color": _COLOR_ENUM_TOKENS,
    "secondary_color": _COLOR_ENUM_TOKENS,
}

# Illustrative examples only — not exhaustive, model may go beyond these.
# Shared fields apply to every slot that includes the field in its schema.
# Must not overlap with CANONICAL_ATTRIBUTE_TOKENS.
EXAMPLE_ATTRIBUTE_TOKENS: dict[str, tuple[str, ...]] = {
    "pattern": ("floral", "plaid", "polka_dot", "star", "gingham", "gradient"),
    "material": ("lace", "tulle", "velvet", "sheer", "knit", "metallic"),
    "ornament": ("bow", "ruffle", "embroidery", "pearl", "lace_trim", "star", "pleated"),
}

_UPPER_BODY_NECKLINE_TOKENS: tuple[str, ...] = (
    "sweetheart",
    "v_neck",
    "high_neck",
    "square_neck",
    "scoop_neck",
    "boat_neck",
)
_UPPER_BODY_SHOULDER_TOKENS: tuple[str, ...] = (
    "off_shoulder",
    "one_shoulder",
    "cold_shoulder",
    "halter",
    "strapless",
)
_GARMENT_FIT_TOKENS: tuple[str, ...] = (
    "oversized",
    "relaxed",
    "regular",
    "tailored",
    "slim",
    "fitted",
)
_UPPER_BODY_COLLAR_TOKENS: tuple[str, ...] = (
    "peter_pan",
    "sailor",
    "mandarin",
    "lace_collar",
    "bertha_collar",
    "ruffle_collar",
)
_UPPER_BODY_SLEEVE_SHAPE_TOKENS: tuple[str, ...] = (
    "puff",
    "bell",
    "bishop",
    "flutter",
    "detached",
)

_OUTERWEAR_CLOSURE_TOKENS: tuple[str, ...] = (
    "button_front",
    "zipper_front",
    "toggle_front",
    "belted",
    "open_front",
)
_TOP_CLOSURE_TOKENS: tuple[str, ...] = (
    "button_front",
    "zipper_back",
    "lace_up",
    "tie_front",
    "corset_lacing",
)
_BOTTOM_CLOSURE_TOKENS: tuple[str, ...] = (
    "elastic_waist",
    "drawstring_waist",
    "button_fly",
    "zip_fly",
    "side_zip",
)
_DRESS_CLOSURE_TOKENS: tuple[str, ...] = (
    "zipper_back",
    "button_front",
    "lace_up_back",
    "wrap_tie",
    "corset_lacing",
)
_SHOE_CLOSURE_TOKENS: tuple[str, ...] = (
    "lace_up",
    "buckle_strap",
    "ankle_strap",
    "slip_on",
    "zipper_side",
)

SLOT_EXAMPLE_ATTRIBUTE_TOKENS: dict[str, dict[str, tuple[str, ...]]] = {
    "outerwear": {
        "neckline": _UPPER_BODY_NECKLINE_TOKENS,
        "fit": _GARMENT_FIT_TOKENS,
        "shoulder": _UPPER_BODY_SHOULDER_TOKENS,
        "collar": _UPPER_BODY_COLLAR_TOKENS,
        "closure": _OUTERWEAR_CLOSURE_TOKENS,
        "sleeve_shape": _UPPER_BODY_SLEEVE_SHAPE_TOKENS,
    },
    "tops": {
        "fit": _GARMENT_FIT_TOKENS,
        "neckline": _UPPER_BODY_NECKLINE_TOKENS,
        "shoulder": _UPPER_BODY_SHOULDER_TOKENS,
        "collar": _UPPER_BODY_COLLAR_TOKENS,
        "closure": _TOP_CLOSURE_TOKENS,
        "sleeve_shape": _UPPER_BODY_SLEEVE_SHAPE_TOKENS,
    },
    "bottoms": {
        "fit": _GARMENT_FIT_TOKENS,
        "closure": _BOTTOM_CLOSURE_TOKENS,
    },
    "dresses": {
        "dress_silhouette": (
            # straight/minimal shaping
            "shift",
            "fitted",
            # flared — ordered by where the flare originates
            "a_line",
            "fit_and_flare",
            "trumpet",
            "mermaid",
            # waist-defined
            "empire",
            "wrap",
            # volume-based
            "tiered",
            "balloon",
            "trapeze",
        ),
        "neckline": _UPPER_BODY_NECKLINE_TOKENS,
        "shoulder": _UPPER_BODY_SHOULDER_TOKENS,
        "collar": _UPPER_BODY_COLLAR_TOKENS,
        "closure": _DRESS_CLOSURE_TOKENS,
        "sleeve_shape": _UPPER_BODY_SLEEVE_SHAPE_TOKENS,
    },
    "hair": {
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
    },
    "shoes": {
        "toe_shape": ("round_toe", "almond_toe", "pointed_toe", "square_toe"),
        "platform": ("none", "low_platform", "high_platform"),
        "closure": _SHOE_CLOSURE_TOKENS,
    },
    "socks": {
        "opacity": ("sheer", "semi_sheer", "opaque"),
        "trim": ("lace_trim", "ruffle_trim", "ribbed_trim"),
    },
    "accessory": {
        "placement": (
            "head",
            "neck",
            "chest",
            "waist",
            "back",
            "arm",
            "hand",
            "leg",
        ),
        "attachment": ("clip_on", "tie_on", "wrap", "dangle", "pin_on"),
        "shape": ("heart", "star", "flower", "cross", "wing", "moon"),
    },
}


def _slot_example_vocab(slot: str) -> dict[str, tuple[str, ...]]:
    if slot in SLOT_EXAMPLE_ATTRIBUTE_TOKENS:
        return SLOT_EXAMPLE_ATTRIBUTE_TOKENS[slot]
    if slot in ACCESSORY_ITEM_TYPES:
        return SLOT_EXAMPLE_ATTRIBUTE_TOKENS["accessory"]
    return {}


def _example_tokens_for(slot: str, field_name: str) -> tuple[str, ...] | None:
    slot_tokens = _slot_example_vocab(slot).get(field_name)
    if slot_tokens:
        return slot_tokens
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
            tokens = _example_tokens_for(slot, name)
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
    ]
    if canonical:
        sections.append(canonical)
    slot_field_guidance = _slot_field_guidance(slot)
    if field_guidance and slot_field_guidance:
        field_guidance = f"{field_guidance}\n\n{slot_field_guidance}"
    elif slot_field_guidance:
        field_guidance = slot_field_guidance
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


def _slot_field_guidance(slot: str) -> str:
    if slot != "bottoms":
        return ""
    return (
        "Length guidance: use bottom_length for visible lower-body coverage only, "
        "independent of garment type. Choose the closest coverage token from hem "
        "or leg extent rather than a fashion-specific term."
    )


SUBCATEGORY_HIERARCHY: dict[str, str] = {
    # "platform_boots": "boots",
}

TOKEN_ALIASES: dict[str, str] = {
    # "tee": "shirt",
}

FILTERED_CANONICAL_ATTRIBUTE_FIELDS: frozenset[str] = frozenset(
    {
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
    set(EXAMPLE_ATTRIBUTE_TOKENS).union(
        field_name
        for slot_fields in SLOT_EXAMPLE_ATTRIBUTE_TOKENS.values()
        for field_name in slot_fields
    )
)


__all__ = [
    "CANONICAL_ATTRIBUTE_TOKENS",
    "CANONICAL_CATEGORY_TOKENS",
    "CANONICAL_SUBCATEGORY_TOKENS",
    "EXAMPLE_ATTRIBUTE_TOKENS",
    "FILTERED_CANONICAL_ATTRIBUTE_FIELDS",
    "SLOT_EXAMPLE_ATTRIBUTE_TOKENS",
    "SLOT_GUIDANCE",
    "STRUCTURED_EXTRACTION_SYSTEM_PROMPT",
    "SUBCATEGORY_HIERARCHY",
    "TOKEN_ALIASES",
    "build_extraction_user_message",
    "get_subcategory_ancestors",
    "normalise_token",
]
