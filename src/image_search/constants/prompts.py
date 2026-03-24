from __future__ import annotations


from image_search.constants.structured import (
    ACCESSORY_ITEM_TYPES,
    schema_definition_for_item_type,
)

STRUCTURED_EXTRACTION_SYSTEM_PROMPT = (
    "You are a deterministic structured extraction engine for Infinity Nikki items.\n\n"
    "Output exactly one raw JSON object. No markdown, no code fences, no prose.\n"
    "Use schema keys exactly as written. Do not add, rename, omit, or reorder keys.\n\n"
    "VALUE RULES\n"
    "- Non-null values must be lowercase underscore_tokens.\n"
    "- Scalars: one token or null.\n"
    "- Arrays: unique tokens or [].\n\n"
    "EVIDENCE\n"
    "- Use only directly visible details from the provided images.\n"
    "- Do not infer hidden, back-side, off-frame, gameplay, or lore details.\n"
    "- If unclear or inapplicable, output null or [].\n"
    "- If the images conflict, trust the overview image over the icon.\n\n"
    "TAXONOMY\n"
    "- category = broadest stable visible class.\n"
    "- subcategory = valid refinement of category, or null.\n"
    "- subcategory should name a type refinement, not a silhouette, material, pattern, color, length, closure, or other dedicated-field concept.\n"
    "- Do not repeat category in subcategory.\n"
    "- Never output incompatible category/subcategory pairs.\n\n"
    "FIELD RULES\n"
    "- pattern = repeated surface motif or print.\n"
    "- material = visible fabric or surface type.\n"
    "- structure = built-in fabric shaping, formation, or visible surface texture.\n"
    "- front_style = visible front construction or opening style.\n"
    "- ornament = attached or applied decorative detail, not an edge or band finish.\n"
    "- Do not encode the same concept in multiple fields.\n"
)

_ITEM_TYPE_FIELD_GUIDANCE: dict[str, str] = {
    "bottoms": (
        "BOTTOMS FIELD RULES\n"
        "- bottom_length = visible length only.\n"
        "- waist_height = visible waist position only.\n"
        "- closure_style only when clearly visible; otherwise null.\n"
        "- skirt_silhouette only when category = skirt or skort; otherwise null.\n"
        "- pant_shape only when category = pants or overalls; otherwise null."
    ),
}

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

_SLOT_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
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
            # jacket variants
            "blazer",
            "bolero",
            "shrug",
            "windbreaker",
            "bomber_jacket",
            "puffer_jacket",
            "military_jacket",
            # coat variants
            "trench_coat",
            "fur_coat",
            "pea_coat",
            "duster_coat",
            "tailcoat",
            "frock_coat",
            "opera_coat",
            "duffle_coat",
            "cape_coat",
            # cape variants
            "hooded_cape",
            "capelet",
            "mantle",
            "cloak",
            "poncho",
            "cape_jacket",
            # cardigan / robe variants
            "long_cardigan",
            # Chinese outer-layer forms
            "beizi",
            "pifeng",
            "zhaoshan",
            "daxiushan",
        ),
    },
    "tops": {
        "category_tokens": (
            # Distinct garment types by structure and silhouette
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
            # corset / bodice variants
            "bustier",
            # shirt / blouse variants
            "sailor_blouse",
            "polo_shirt",
            "henley_shirt",
            "peasant_blouse",
            "milkmaid_top",
            "school_uniform_blouse",
            "maid_blouse",
            "officer_shirt",
            # stable layered variant
            "sweater_vest",
        ),
    },
    "bottoms": {
        "category_tokens": (
            # Fundamental lower-body silhouette types
            "skirt",
            "pants",
            "shorts",
            "leggings",
            "overalls",
            "skort",
        ),
        "subcategory_tokens": (
            # --- pants / structured lowers ---
            "bloomers",
            "culottes",
            "harem_pants",
            "palazzo_pants",
            "jodhpurs",
            "cargo_pants",
            # --- shorts ---
            "bermuda_shorts",
            "suspender_shorts",
            "ruffle_shorts",
            "tailored_shorts",
            # --- skirts / special ---
            "wrap_skirt",
            "sarong",
            # Chinese lower-body forms
            "ma_mian_skirt",
            "bijia_skirt",
        ),
    },
    "dresses": {
        "category_tokens": (
            # dress: one-piece skirted garments
            # jumpsuit: one-piece with legs
            "dress",
            "jumpsuit",
        ),
        "subcategory_tokens": (
            # jumpsuit variants
            "romper",
            "playsuit",
            # dress variants with stable or highly recognizable form
            "gown",
            "sundress",
            "kaftan",
            "chemise",
            "qipao",
            "pinafore_dress",
            "overall_dress",
            "shirt_dress",
            "slip_dress",
            "lolita_dress",
            "sweater_dress",
            "sailor_dress",
            "apron_dress",
            "coat_dress",
            "babydoll_dress",
            "maid_dress",
            # Chinese one-piece forms
            "hanfu",
            "ruqun",
            "shenyi",
            "qiyao_ruqun",
            "panxi_ruqun",
            "duijin_dress",
        ),
    },
    "hair": {
        "category_tokens": (
            # Organized by visible arrangement, not length
            "loose",
            "ponytail",
            "twin_tails",
            "bun",
            "braid",
            "updo",
            "half_up",
        ),
        "subcategory_tokens": (
            # ponytail variants
            "high_ponytail",
            "low_ponytail",
            "side_ponytail",
            # bun variants
            "space_buns",
            "top_knot",
            "side_bun",
            "chignon",
            "braided_bun",
            # braid variants
            "twin_braids",
            "french_braid",
            "fishtail_braid",
            "side_braid",
            "crown_braid",
            # half-up variants
            "half_up_bun",
            "half_up_ponytail",
            # Chinese hair arrangements
            "ji_hair",
            "liangbatou",
            "double_ring_bun",
            "flying_immortal_bun",
        ),
    },
    "shoes": {
        "category_tokens": (
            # Fundamental footwear types by structure
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
            # named shoe forms
            "pumps",
            "mary_janes",
            "ballet_flats",
            "lace_up_flats",
            # heel / sandal forms with stable recognizable construction
            "t_strap_heels",
            "ankle_strap_heels",
            "slingback_heels",
            "dorsay_heels",
            "gladiator_sandals",
            "platform_sandals",
            # boot forms
            "cowboy_boots",
            "combat_boots",
            "riding_boots",
            # Chinese traditional footwear
            "embroidered_shoes",
            "cloud_tip_shoes",
            "silk_platform_shoes",
        ),
    },
    "socks": {
        "category_tokens": (
            # Fundamental legwear types by construction
            "socks",
            "stockings",
            "tights",
            "leg_warmers",
        ),
        "subcategory_tokens": (
            # structurally distinct / high-signal variants
            "garter_stockings",
            "lace_stockings",
            "pantyhose",
            "fishnet_tights",
            "printed_tights",
            "ruffled_socks",
        ),
    },
    "accessory": {
        "category_tokens": (
            # Broad accessory slot; visually distinct forms only
            "headband",
            "bow",
            "ribbon",
            "flower",
            "hairpin",
            "hairclip",
            "hat",
            "bonnet",
            "beret",
            "crown",
            "tiara",
            "veil",
            "mask",
            "monocle",
            "eyepatch",
            "goggles",
            "earring",
            "ear_cuff",
            "necklace",
            "choker",
            "collar",
            "brooch",
            "bracelet",
            "cuff",
            "ring",
            "armlet",
            "anklet",
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
            "headpiece",
            "headdress",
            "corsage",
            "cape_pin",
        ),
        "subcategory_tokens": (),
    },
    "pendants": {
        "category_tokens": (
            # Game-specific mixed slot; intentionally includes hanging ornaments,
            # decorative garters, and bag-like accessories
            "pendant",
            "locket",
            "charm",
            "medallion",
            "tassel",
            "keychain",
            "garter",
            "leg_garter",
            "thigh_garter",
            "bag",
            "handbag",
            "mini_bag",
            "pouch",
            "belt_bag",
            "crossbody_bag",
            "shoulder_bag",
            "satchel",
        ),
        "subcategory_tokens": (),
    },
}


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
        "hair_length",
        "sleeve_length",
        "shaft_height",
        "heel_height",
        "sole_height",
        "sock_height",
        "waist_height",
    }
)


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

_UPPER_LENGTH_TOKENS: tuple[str, ...] = (
    "ultra_crop",
    "crop",
    "waist_length",
    "hip_length",
    "upper_thigh",
    "mid_thigh",
    "knee_length",
    "midi",
    "maxi",
    "ankle_length",
    "floor_length",
)
_LOWER_LENGTH_TOKENS: tuple[str, ...] = (
    "micro",
    "mini",
    "upper_thigh",
    "mid_thigh",
    "knee_length",
    "midi",
    "maxi",
    "ankle_length",
    "floor_length",
)
_HAIR_LENGTH_TOKENS: tuple[str, ...] = (
    "short",
    "medium",
    "long",
    "very_long",
)
_SLEEVE_LENGTH_TOKENS: tuple[str, ...] = (
    "sleeveless",
    "short",
    "elbow",
    "three_quarter",
    "long",
    "extra_long",
)

_SOCK_HEIGHT_TOKENS: tuple[str, ...] = (
    "ankle",
    "knee",
    "over_knee",
    "thigh_high",
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

# Illustrative examples only — not exhaustive, model may go beyond these.
# Shared fields apply to every slot that includes the field in its schema.
# Must not overlap with CANONICAL_ATTRIBUTE_TOKENS.
EXAMPLE_ATTRIBUTE_TOKENS: dict[str, tuple[str, ...]] = {
    "pattern": (
        "solid",
        "floral",
        "plaid",
        "striped",
        "polka_dot",
        "checkered",
        "gradient",
        "star",
        "animal_print",
        "camouflage",
    ),
    "material": (
        "denim",
        "knit",
        "lace",
        "tulle",
        "velvet",
        "leather",
        "sheer",
        "satin",
        "chiffon",
        "mesh",
        "fur",
    ),
    "ornament": (
        "bow",
        "ruffle",
        "frill",
        "embroidery",
        "applique",
        "bead_detail",
        "pearl_detail",
        "sequin",
        "button_detail",
        "buckle",
        "chain_detail",
        "star",
    ),
    "structure": (
        "pleated",
        "gathered",
        "ruched",
        "smocked",
        "peplum",
        "quilted",
        "draped",
        "ribbed",
        "cable_knit",
    ),
}

_NECKLINE_TOKENS: tuple[str, ...] = (
    "crew_neck",
    "v_neck",
    "square_neck",
    "sweetheart_neck",
    "boat_neck",
    "scoop_neck",
    "keyhole_neck",
    "halter_neck",
    "turtleneck",
    "mock_neck",
    "collared",
    # collar variants folded into neckline
    "peter_pan_collar",
    "sailor_collar",
    "mandarin_collar",
    # Chinese neckline forms
    "cross_collar",
    "right_cross_collar",
    "duijin_collar",
)
_SHOULDER_STYLE_TOKENS: tuple[str, ...] = (
    "off_shoulder",
    "one_shoulder",
    "cold_shoulder",
    "halter",
    "strapless",
)
_GARMENT_FIT_TOKENS: tuple[str, ...] = (
    "bodycon",
    "fitted",
    "slim",
    "tailored",
    "regular",
    "relaxed",
    "loose",
    "oversized",
)
_SLEEVE_STYLE_TOKENS: tuple[str, ...] = (
    # --- construction (very distinct) ---
    "raglan_sleeve",
    "dolman_sleeve",
    # --- volume / shape (core set) ---
    "puff_sleeve",
    "bishop_sleeve",
    "bell_sleeve",
    "flare_sleeve",
    # --- structure / notable design ---
    "slit_sleeve",
    "split_sleeve",
    "detached_sleeves",
    # --- shoulder-specific ---
    "cap_sleeve",
    # --- Chinese traditional sleeve forms ---
    "shuixiu_sleeve",
    "daxiu_sleeve",
    "pipa_sleeve",
    "zhishou_sleeve",
)
_HEM_TOKENS: tuple[str, ...] = (
    # basic structure
    "straight_hem",
    "curved_hem",
    "asymmetrical_hem",
    "high_low_hem",
    # decorative edges
    "scalloped_hem",
    "ruffled_hem",
    "frilled_hem",
    # layered / volume
    "tiered_hem",
    "layered_hem",
    # pants-specific finishes
    "cuffed_hem",
    "rolled_hem",
    "split_hem",
)

_UPPER_CLOSURE_STYLE_TOKENS: tuple[str, ...] = (
    "button",
    "zipper",
    "toggle",
    "hook",
    "snap",
    "lace_up",
    "corset_back",
    "pull_on",
)
_FRONT_STYLE_TOKENS: tuple[str, ...] = (
    "single_breasted",
    "double_breasted",
    "wrap",
    "asymmetric_front",
    "off_center",
    "open_front",
    "concealed_placket",
    "exposed_placket",
)
_LOWER_CLOSURE_STYLE_TOKENS: tuple[str, ...] = (
    "zipper",
    "button",
    "hook",
    "elastic_waist",
    "drawstring_waist",
    "belted_waist",
    "pull_on",
)

_DRESS_SILHOUETTE_TOKENS: tuple[str, ...] = (
    "shift",
    "sheath",
    "a_line",
    "fit_and_flare",
    "trumpet",
    "mermaid",
    "trapeze",
    "balloon",
    "tiered",
)
_SKIRT_SILHOUETTE_TOKENS: tuple[str, ...] = (
    "straight",
    "a_line",
    "circle",
    "pencil",
    "layered",
    "tiered",
    "bubble",
    "trumpet",
    "asymmetric",
)
_PANT_SHAPE_TOKENS: tuple[str, ...] = (
    "straight",
    "slim",
    "wide_leg",
    "flared",
    "tapered",
)
_WAIST_HEIGHT_TOKENS: tuple[str, ...] = (
    "low_rise",
    "mid_rise",
    "high_rise",
    "ultra_high_rise",
)
_WAISTLINE_TOKENS: tuple[str, ...] = (
    "natural_waist",
    "empire_waist",
    "drop_waist",
    "no_waist",
    "princess_seam",
    "basque_waist",
    "wrap_waist",
)
_HEEL_TYPE_TOKENS: tuple[str, ...] = (
    "flat",
    "block",
    "stiletto",
    "kitten",
    "wedge",
    "cone",
    "platform",
)
_HEEL_HEIGHT_TOKENS: tuple[str, ...] = (
    "flat",
    "low",
    "mid",
    "high",
)
_SOLE_HEIGHT_TOKENS: tuple[str, ...] = (
    "flat",
    "low_platform",
    "high_platform",
)
_SHAFT_HEIGHT_TOKENS: tuple[str, ...] = (
    "ankle",
    "mid_calf",
    "knee_high",
    "over_knee",
)

_HAIR_TEXTURE_TOKENS: tuple[str, ...] = (
    "straight",
    "wavy",
    "curly",
    "coiled",
    "drill",
)
_HAIRCUT_TOKENS: tuple[str, ...] = (
    "bob",
    "lob",
    "pixie_cut",
    "wolf_cut",
    "shag",
    "hime_cut",
    "bun_cut",
    "layered_cut",
    "blunt_cut",
)
_HAIR_BANGS_TOKENS: tuple[str, ...] = (
    "no_bangs",
    "blunt_bangs",
    "side_swept_bangs",
    "curtain_bangs",
    "wispy_bangs",
)
_HAIR_ADORNMENT_TOKENS: tuple[str, ...] = ("bow", "ribbon", "flower", "clip", "pin")
_SHOE_TOE_SHAPE_TOKENS: tuple[str, ...] = (
    "round_toe",
    "almond_toe",
    "pointed_toe",
    "square_toe",
)
_SOCK_OPACITY_TOKENS: tuple[str, ...] = ("sheer", "semi_sheer", "opaque")
_SOCK_TRIM_TOKENS: tuple[str, ...] = (
    "rolled_edge",
    "scalloped_edge",
    "picot_edge",
    "elastic_band",
)

EXAMPLE_ATTRIBUTE_TOKENS.update(
    {
        "closure_style": tuple(
            dict.fromkeys((*_UPPER_CLOSURE_STYLE_TOKENS, *_LOWER_CLOSURE_STYLE_TOKENS))
        ),
        "hem": _HEM_TOKENS,
        "haircut": _HAIRCUT_TOKENS,
        "texture": _HAIR_TEXTURE_TOKENS,
        "adornment": _HAIR_ADORNMENT_TOKENS,
        "trim": _SOCK_TRIM_TOKENS,
    }
)

# Closed-list fields only — model must pick from these tokens exclusively.
# Includes filtered measurement-style fields plus other genuinely closed enumerations.
# Must not overlap with EXAMPLE_ATTRIBUTE_TOKENS.
CANONICAL_ATTRIBUTE_TOKENS: dict[str, tuple[str, ...]] = {
    # measurement fields (FILTERED_CANONICAL_ATTRIBUTE_FIELDS)
    "outerwear_length": _UPPER_LENGTH_TOKENS,
    "top_length": _UPPER_LENGTH_TOKENS,
    "bottom_length": _LOWER_LENGTH_TOKENS,
    "hair_length": _HAIR_LENGTH_TOKENS,
    "sleeve_length": _SLEEVE_LENGTH_TOKENS,
    "shaft_height": _SHAFT_HEIGHT_TOKENS,
    "heel_height": _HEEL_HEIGHT_TOKENS,
    "sole_height": _SOLE_HEIGHT_TOKENS,
    "sock_height": _SOCK_HEIGHT_TOKENS,
    # closed enumerations
    "opacity": _SOCK_OPACITY_TOKENS,
    "waistline": _WAISTLINE_TOKENS,
    "waist_height": _WAIST_HEIGHT_TOKENS,
    "fit": _GARMENT_FIT_TOKENS,
    "neckline": _NECKLINE_TOKENS,
    "shoulder_style": _SHOULDER_STYLE_TOKENS,
    "sleeve_style": _SLEEVE_STYLE_TOKENS,
    "front_style": _FRONT_STYLE_TOKENS,
    "skirt_silhouette": _SKIRT_SILHOUETTE_TOKENS,
    "pant_shape": _PANT_SHAPE_TOKENS,
    "dress_silhouette": _DRESS_SILHOUETTE_TOKENS,
    "bangs": _HAIR_BANGS_TOKENS,
    "heel_type": _HEEL_TYPE_TOKENS,
    "toe_shape": _SHOE_TOE_SHAPE_TOKENS,
    # color enumerations
    "primary_color": _COLOR_ENUM_TOKENS,
    "secondary_color": _COLOR_ENUM_TOKENS,
}


def _field_names_for(slot: str) -> tuple[str, ...]:
    return tuple(
        field_definition.name
        for field_definition in schema_definition_for_item_type(slot).fields
    )


def _schema_template_for(slot: str) -> str:
    schema_definition = schema_definition_for_item_type(slot)
    lines = ["{"]
    last = len(schema_definition.fields) - 1
    for index, field_definition in enumerate(schema_definition.fields):
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
        category_set = set(category_tokens or ())
        filtered_subcategory = tuple(
            t for t in subcategory_tokens if t not in category_set
        )
        if filtered_subcategory:
            example_lines.append(
                "  subcategory: prefer these refinements when they fit; other clear type refinements are allowed, "
                "but do not invent tokens that repeat dedicated-field concepts such as silhouette, material, "
                f"pattern, color, length, or closure: {', '.join(filtered_subcategory)}"
            )
    for name in field_names:
        canonical_tokens = CANONICAL_ATTRIBUTE_TOKENS.get(name)
        if canonical_tokens:
            filtered_lines.append(f"  {name}: {', '.join(canonical_tokens)}")
            continue
        example_tokens = EXAMPLE_ATTRIBUTE_TOKENS.get(name)
        if example_tokens:
            example_lines.append(f"  {name} examples: {', '.join(example_tokens)}")
            continue
    sections: list[str] = []
    if filtered_lines:
        sections.append(
            "CLOSED LISTS — only these values are valid for these fields:\n"
            + "\n".join(filtered_lines)
        )
    if example_lines:
        sections.append(
            "OPEN VOCABULARY — illustrative examples, other tokens are allowed:\n"
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
    schema = _schema_template_for(slot)
    resolved_field_names = field_names or _field_names_for(slot)
    canonical = _canonical_block(slot, resolved_field_names)
    resolved_field_guidance = field_guidance or _ITEM_TYPE_FIELD_GUIDANCE.get(slot)
    sections: list[str] = []
    if canonical:
        sections.append(canonical)
    if resolved_field_guidance:
        sections.append(resolved_field_guidance)
    if visual_tags and visual_tags.strip():
        sections.append(f"Visual context from tagger: {visual_tags}")
    sections.append(
        "Two images are provided: the overview image and the icon.\n"
        "The overview image takes priority over the icon.\n"
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


def _format_token_list(tokens: tuple[str, ...]) -> str:
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    if len(tokens) == 2:
        return f"{tokens[0]} or {tokens[1]}"

    # Avoid slicing to satisfy broken type checkers:
    parts = []
    for i in range(len(tokens) - 1):
        parts.append(tokens[i])
    prefix = ", ".join(parts)
    last = tokens[-1]
    return f"{prefix}, or {last}"


_validate_slot_config()


__all__ = [
    "CANONICAL_ATTRIBUTE_TOKENS",
    "CANONICAL_CATEGORY_TOKENS",
    "CANONICAL_SUBCATEGORY_TOKENS",
    "EXAMPLE_ATTRIBUTE_TOKENS",
    "FILTERED_CANONICAL_ATTRIBUTE_FIELDS",
    "STRUCTURED_EXTRACTION_SYSTEM_PROMPT",
    "SUBCATEGORY_HIERARCHY",
    "TOKEN_ALIASES",
    "build_extraction_user_message",
    "get_subcategory_ancestors",
    "normalise_token",
]
