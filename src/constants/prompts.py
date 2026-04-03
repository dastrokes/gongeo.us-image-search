from __future__ import annotations


from constants.structured import (
    WEARABLE_ITEM_TYPES,
    schema_definition_for_item_type,
)

STRUCTURED_EXTRACTION_SYSTEM_PROMPT = (
    "You are a deterministic structured extraction engine for Infinity Nikki items.\n\n"
    "Output exactly one raw JSON object. No markdown, no code fences, no prose.\n"
    "Use schema keys exactly as written. Do not add, rename, omit, or reorder keys.\n\n"
    "VALUE RULES\n"
    "- Non-null values must be lowercase underscore_tokens.\n"
    "- Scalars: one token or null.\n"
    "- Arrays: unique tokens or [].\n"
    "- Each field must represent a single concept only.\n"
    "- Do not combine multiple attributes into one token.\n\n"
    "TAXONOMY\n"
    "- category = the main visible item class, chosen at the schema root level rather than a more specific child type.\n"
    "- category must be one of the values in CLOSED LISTS.\n"
    "- subcategory must be a concise, canonical direct child refinement of the chosen category, or null.\n"
    "- Prefer the listed child examples. If no close match exists, use null.\n"
    "- subcategory must be a valid child of the selected category. If unsure, set subcategory = null.\n"
    "- subcategory should name a type refinement, not a silhouette, material, pattern, color, length, haircut, texture, or other dedicated-field concept.\n"
    "- Do not repeat category in subcategory.\n"
    "- Do not compose multiple attributes into subcategory.\n\n"
)

STRUCTURED_EXTRACTION_EVIDENCE_PROMPT = (
    "EVIDENCE\n"
    "- Use only directly visible details from the provided images.\n"
    "- Do not infer hidden, back-side, off-frame, gameplay, or lore details.\n"
    "- If unclear, weak, or inapplicable, output null or [].\n"
    "- Prefer null over uncertain or weak signals.\n"
    "- If the images conflict, trust the overview image over the icon."
)

_SHARED_FIELD_GUIDANCE_LINES: dict[str, str] = {
    "pattern": "- pattern = repeated surface motif or print.",
    "material": "- material = visible fabric or surface type.",
    "structure": "- structure = built-in fabric shaping, formation, or visible surface texture.",
    "ornament": "- ornament = attached or applied decorative detail, not an edge or band finish.",
}

_ITEM_TYPE_FIELD_GUIDANCE: dict[str, str] = {
    "bottoms": (
        "BOTTOMS FIELD RULES\n"
        "- skirt_silhouette only when category = skirt or skort; otherwise null.\n"
        "- pant_shape only when category = pants or overalls; otherwise null.\n"
        "- waist_height only when the waistband or rise is visible; otherwise null."
    ),
    "dresses": (
        "DRESSES FIELD RULES\n"
        "- Stable ensemble-derived or style-family terms are allowed when they are the clearest visible type.\n"
        "- dress_silhouette and waistline only describe the main dress body; use null when category = jumpsuit."
    ),
    "hair": (
        "HAIR FIELD RULES\n"
        "- category/subcategory describe the visible arrangement only.\n"
        "- haircut belongs only in haircut.\n"
        "- texture belongs only in texture.\n"
        "- bangs belongs only in bangs.\n"
        "- If no clear fringe crosses the forehead, use no_bangs.\n"
        "- Do not infer bangs from parting alone."
    ),
    "shoes": (
        "SHOES FIELD RULES\n"
        "- heel_type and heel_height only when a distinct heel is present; otherwise null.\n"
        "- sole_height only for visibly thick, platform, or elevated soles.\n"
        "- shaft_height only for boots; otherwise null."
    ),
}

_TaxonomyDefinition = tuple[tuple[str, tuple[str, ...]], ...]

_OUTERWEAR_TAXONOMY: _TaxonomyDefinition = (
    (
        "jacket",
        (
            "blazer",
            "bolero",
            "bomber_jacket",
            "military_jacket",
            "puffer_jacket",
            "shirt_jacket",
            "windbreaker",
        ),
    ),
    (
        "coat",
        (
            "cape_coat",
            "duster_coat",
            "duffle_coat",
            "fur_coat",
            "pea_coat",
            "tailcoat",
            "trench_coat",
        ),
    ),
    (
        "cape",
        (
            "capelet",
            "cloak",
            "hooded_cape",
            "pifeng",
            "poncho",
        ),
    ),
    ("cardigan", ("long_cardigan", "shrug")),
    ("vest", ("bijia", "hooded_vest", "waistcoat")),
    ("robe", ("beizi", "daxiushan")),
    ("shawl", ("stole",)),
)

_TOPS_TAXONOMY: _TaxonomyDefinition = (
    (
        "blouse",
        (
            "maid_blouse",
            "milkmaid_top",
            "peasant_blouse",
            "sailor_blouse",
            "school_uniform_blouse",
        ),
    ),
    (
        "shirt",
        (
            "button_up_shirt",
            "henley_shirt",
            "officer_shirt",
            "polo_shirt",
        ),
    ),
    ("t_shirt", ()),
    ("sweater", ("sweater_vest",)),
    ("sweatshirt", ("hoodie",)),
    ("camisole", ("bralette",)),
    ("corset", ("bustier",)),
    ("tunic", ()),
    ("tank_top", ()),
    ("tube_top", ("bandeau_top",)),
    ("bodysuit", ("leotard",)),
)

_BOTTOMS_TAXONOMY: _TaxonomyDefinition = (
    ("skirt", ("ma_mian_skirt", "sarong", "wrap_skirt")),
    (
        "pants",
        (
            "cargo_pants",
            "culottes",
            "harem_pants",
            "jodhpurs",
            "palazzo_pants",
        ),
    ),
    (
        "shorts",
        (
            "bermuda_shorts",
            "bloomers",
            "ruffle_shorts",
            "tailored_shorts",
        ),
    ),
    ("leggings", ()),
    ("overalls", ("bib_overalls", "overall_shorts")),
    ("skort", ()),
)

_DRESSES_TAXONOMY: _TaxonomyDefinition = (
    (
        "dress",
        (
            "babydoll_dress",
            "coat_dress",
            "gown",
            "kaftan",
            "lolita_dress",
            "maid_dress",
            "pinafore_dress",
            "qipao",
            "ruqun",
            "sailor_dress",
            "shirt_dress",
            "shenyi",
            "slip_dress",
            "sundress",
            "sweater_dress",
            "wrap_dress",
            "corset_dress",
            "tea_dress",
            "tunic_dress",
        ),
    ),
    ("jumpsuit", ("romper", "onesie")),
)

_HAIR_TAXONOMY: _TaxonomyDefinition = (
    ("loose", ()),
    ("ponytail", ("high_ponytail", "low_ponytail", "side_ponytail")),
    (
        "twin_tails",
        (
            "drill_twin_tails",
            "high_twin_tails",
            "low_twin_tails",
            "side_twin_tails",
        ),
    ),
    (
        "bun",
        (
            "high_bun",
            "low_bun",
            "messy_bun",
            "side_bun",
            "space_buns",
            "top_knot",
        ),
    ),
    (
        "braid",
        (
            "crown_braid",
            "fishtail_braid",
            "french_braid",
            "side_braid",
            "twin_braids",
        ),
    ),
    ("half_up", ("half_up_braid", "half_up_bun", "half_up_ponytail")),
    (
        "updo",
        (
            "chignon",
            "double_ring_bun",
            "flying_immortal_bun",
            "liangbatou",
        ),
    ),
)

_SHOES_TAXONOMY: _TaxonomyDefinition = (
    ("boots", ("combat_boots", "cowboy_boots", "riding_boots")),
    (
        "heels",
        (
            "ankle_strap_heels",
            "dorsay_heels",
            "pumps",
            "slingback_heels",
            "t_strap_heels",
        ),
    ),
    ("flats", ("ballet_flats", "embroidered_shoes", "lace_up_flats")),
    ("sandals", ("gladiator_sandals", "platform_sandals")),
    ("sneakers", ("high_top_sneakers",)),
    ("loafers", ("penny_loafers",)),
    ("mary_janes", ()),
    ("mules", ()),
    ("slippers", ()),
    ("barefoot", ()),
)

_SOCKS_TAXONOMY: _TaxonomyDefinition = (
    ("socks", ("ruffled_socks",)),
    ("stockings", ("garter_stockings", "lace_stockings")),
    ("tights", ("fishnet_tights", "pantyhose", "printed_tights")),
    ("leg_warmers", ()),
)

_HAIR_ACCESSORIES_TAXONOMY: _TaxonomyDefinition = (
    (
        "hair_ornament",
        (
            "bow",
            "flower",
            "hairclip",
            "hairpin",
            "ribbon",
        ),
    ),
    ("headband", ()),
    ("hair_comb", ()),
    ("headpiece", ()),
)

_HEADWEAR_TAXONOMY: _TaxonomyDefinition = (
    ("hat", ("mini_hat", "top_hat", "wide_brim_hat", "witch_hat")),
    ("bonnet", ()),
    ("beret", ()),
    ("crown", ("coronet", "tiara")),
    ("veil", ()),
    ("headdress", ()),
    ("headpiece", ()),
)

_EARRINGS_TAXONOMY: _TaxonomyDefinition = (
    (
        "earring",
        (
            "chandelier_earrings",
            "drop_earrings",
            "hoop_earrings",
            "stud_earrings",
        ),
    ),
    ("ear_cuff", ()),
)

_NECKWEAR_TAXONOMY: _TaxonomyDefinition = (
    (
        "necklace",
        (
            "lariat_necklace",
            "pendant_necklace",
            "strand_necklace",
        ),
    ),
    ("choker", ("lace_choker", "ribbon_choker")),
    ("pendant", ("charm", "locket", "medallion")),
    ("collar", ()),
    ("scarf", ()),
)

_BRACELETS_TAXONOMY: _TaxonomyDefinition = (
    ("bracelet", ("bangle", "beaded_bracelet", "charm_bracelet")),
    ("cuff", ()),
)

_CHOKERS_TAXONOMY: _TaxonomyDefinition = (
    (
        "necklace",
        (
            "lariat_necklace",
            "pendant_necklace",
            "strand_necklace",
        ),
    ),
    ("choker", ("lace_choker", "ribbon_choker")),
    ("pendant", ("charm", "locket", "medallion", "tassel")),
    ("collar", ()),
    ("scarf", ()),
    ("headphone", ()),
)

_GLOVES_TAXONOMY: _TaxonomyDefinition = (
    (
        "gloves",
        (
            "fingerless_gloves",
            "mittens",
            "opera_gloves",
        ),
    ),
)

_HANDHELDS_TAXONOMY: _TaxonomyDefinition = (
    ("fan", ()),
    ("handbag", ("purse",)),
    ("parasol", ()),
    ("wand", ()),
    ("staff", ()),
    ("lantern", ()),
    ("book", ("spellbook",)),
    ("bouquet", ()),
    ("basket", ()),
    ("instrument", ()),
    ("plush", ()),
    ("weapon", ()),
    ("tool", ()),
    ("handheld", ()),
)

_CHEST_ACCESSORIES_TAXONOMY: _TaxonomyDefinition = (
    ("brooch", ("bow_brooch", "cape_pin", "pin")),
    ("corsage", ()),
    ("sash", ()),
)

_PENDANTS_TAXONOMY: _TaxonomyDefinition = (
    ("garter", ("leg_garter", "thigh_garter")),
    (
        "bag",
        (
            "belt_bag",
            "crossbody_bag",
            "handbag",
            "mini_bag",
            "pouch",
            "satchel",
            "shoulder_bag",
        ),
    ),
)

_BACKPIECES_TAXONOMY: _TaxonomyDefinition = (
    ("wings", ()),
    ("backpack", ()),
    ("back_bow", ()),
    ("tail", ()),
    ("drape", ()),
    ("backpiece", ()),
)

_RINGS_TAXONOMY: _TaxonomyDefinition = (("ring", ()),)

_BODY_PAINT_TAXONOMY: _TaxonomyDefinition = (("body_paint", ()),)

_ARM_DECORATIONS_TAXONOMY: _TaxonomyDefinition = (
    ("armlet", ("arm_cuff",)),
    ("sleeve_garter", ()),
    ("wrist_corsage", ()),
)

_FACE_DECORATIONS_TAXONOMY: _TaxonomyDefinition = (
    (
        "mask",
        (
            "full_mask",
            "half_mask",
            "lower_face_mask",
            "eye_mask",
            "ornamental_mask",
        ),
    ),
    (
        "blindfold",
        (
            "cloth_blindfold",
            "decorative_blindfold",
            "ornamental_blindfold",
        ),
    ),
    (
        "eyewear",
        (
            "glasses",
            "sunglasses",
            "goggles",
            "monocle",
        ),
    ),
    (
        "eyepatch",
        (
            "medical_eyepatch",
            "decorative_eyepatch",
        ),
    ),
    (
        "facial_cover",
        (
            "mouth_cover",
            "nose_cover",
            "bandage",
            "face_shield",
        ),
    ),
    ("face_paint", ()),
    ("face_veil", ()),
    ("face_marking", ()),
    ("face_jewelry", ()),
    ("face_prop", ()),
)

_ABILITY_HANDHELDS_TAXONOMY: _TaxonomyDefinition = (
    ("wand", ()),
    ("staff", ()),
    ("lantern", ()),
    ("book", ("spellbook",)),
    ("orb", ()),
    ("handheld", ()),
)

_ITEM_TYPE_TAXONOMIES: dict[str, _TaxonomyDefinition] = {
    "outerwear": _OUTERWEAR_TAXONOMY,
    "tops": _TOPS_TAXONOMY,
    "bottoms": _BOTTOMS_TAXONOMY,
    "dresses": _DRESSES_TAXONOMY,
    "hair": _HAIR_TAXONOMY,
    "shoes": _SHOES_TAXONOMY,
    "socks": _SOCKS_TAXONOMY,
    "hairAccessories": _HAIR_ACCESSORIES_TAXONOMY,
    "headwear": _HEADWEAR_TAXONOMY,
    "earrings": _EARRINGS_TAXONOMY,
    "neckwear": _NECKWEAR_TAXONOMY,
    "bracelets": _BRACELETS_TAXONOMY,
    "chokers": _CHOKERS_TAXONOMY,
    "gloves": _GLOVES_TAXONOMY,
    "handhelds": _HANDHELDS_TAXONOMY,
    "chestAccessories": _CHEST_ACCESSORIES_TAXONOMY,
    "pendants": _PENDANTS_TAXONOMY,
    "backpieces": _BACKPIECES_TAXONOMY,
    "rings": _RINGS_TAXONOMY,
    "armDecorations": _ARM_DECORATIONS_TAXONOMY,
    "faceDecorations": _FACE_DECORATIONS_TAXONOMY,
    "bodyPaint": _BODY_PAINT_TAXONOMY,
    "abilityHandhelds": _ABILITY_HANDHELDS_TAXONOMY,
}


def _category_tokens_for_taxonomy(
    taxonomy: _TaxonomyDefinition,
) -> tuple[str, ...]:
    return tuple(category for category, _children in taxonomy)


def _subcategory_tokens_for_taxonomy(
    taxonomy: _TaxonomyDefinition,
) -> tuple[str, ...]:
    return tuple(child for _category, children in taxonomy for child in children)


CANONICAL_CATEGORY_TOKENS: dict[str, tuple[str, ...]] = {
    item_type: _category_tokens_for_taxonomy(taxonomy)
    for item_type, taxonomy in _ITEM_TYPE_TAXONOMIES.items()
}

CANONICAL_SUBCATEGORY_TOKENS: dict[str, tuple[str, ...]] = {
    item_type: _subcategory_tokens_for_taxonomy(taxonomy)
    for item_type, taxonomy in _ITEM_TYPE_TAXONOMIES.items()
}

SUBCATEGORY_HIERARCHY: dict[str, str] = {}
for _item_type, _taxonomy in _ITEM_TYPE_TAXONOMIES.items():
    for _category, _children in _taxonomy:
        for _child in _children:
            _existing_parent = SUBCATEGORY_HIERARCHY.get(_child)
            if _existing_parent is not None and _existing_parent != _category:
                raise RuntimeError(
                    "subcategory parent mismatch: "
                    f"{_child} -> {_existing_parent} vs {_category}"
                )
            SUBCATEGORY_HIERARCHY[_child] = _category

TOKEN_ALIASES: dict[str, str] = {}

FILTERED_CANONICAL_ATTRIBUTE_FIELDS: frozenset[str] = frozenset(
    {
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
    "crew",
    "knee_high",
    "over_knee",
    "thigh_high",
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
    "regular",
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
    "basque_waist",
    "wrap_waist",
)
_HEEL_TYPE_TOKENS: tuple[str, ...] = (
    "block",
    "stiletto",
    "kitten",
    "wedge",
    "cone",
)
_HEEL_HEIGHT_TOKENS: tuple[str, ...] = (
    "flat",
    "low",
    "high",
)
_SOLE_HEIGHT_TOKENS: tuple[str, ...] = (
    "flat",
    "low",
    "high",
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

EXAMPLE_ATTRIBUTE_TOKENS.update(
    {
        "haircut": _HAIRCUT_TOKENS,
        "texture": _HAIR_TEXTURE_TOKENS,
    }
)

# Closed-list fields only — model must pick from these tokens exclusively.
# Includes filtered measurement-style fields plus other genuinely closed enumerations.
# Must not overlap with EXAMPLE_ATTRIBUTE_TOKENS.
CANONICAL_ATTRIBUTE_TOKENS: dict[str, tuple[str, ...]] = {
    # measurement fields (FILTERED_CANONICAL_ATTRIBUTE_FIELDS)
    "top_length": _UPPER_LENGTH_TOKENS,
    "bottom_length": _LOWER_LENGTH_TOKENS,
    "hair_length": _HAIR_LENGTH_TOKENS,
    "sleeve_length": _SLEEVE_LENGTH_TOKENS,
    "shaft_height": _SHAFT_HEIGHT_TOKENS,
    "heel_height": _HEEL_HEIGHT_TOKENS,
    "sole_height": _SOLE_HEIGHT_TOKENS,
    "sock_height": _SOCK_HEIGHT_TOKENS,
    # closed enumerations
    "waistline": _WAISTLINE_TOKENS,
    "waist_height": _WAIST_HEIGHT_TOKENS,
    "fit": _GARMENT_FIT_TOKENS,
    "neckline": _NECKLINE_TOKENS,
    "shoulder_style": _SHOULDER_STYLE_TOKENS,
    "sleeve_style": _SLEEVE_STYLE_TOKENS,
    "skirt_silhouette": _SKIRT_SILHOUETTE_TOKENS,
    "pant_shape": _PANT_SHAPE_TOKENS,
    "dress_silhouette": _DRESS_SILHOUETTE_TOKENS,
    "bangs": _HAIR_BANGS_TOKENS,
    "heel_type": _HEEL_TYPE_TOKENS,
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


def _category_lines_for(slot: str, field_names: tuple[str, ...]) -> list[str]:
    taxonomy = _ITEM_TYPE_TAXONOMIES.get(slot)
    if taxonomy is None or "category" not in field_names:
        return []

    lines = [f"  category: {', '.join(CANONICAL_CATEGORY_TOKENS[slot])}"]
    return lines


def _subcategory_guidance_lines_for(
    slot: str, field_names: tuple[str, ...]
) -> list[str]:
    taxonomy = _ITEM_TYPE_TAXONOMIES.get(slot)
    if taxonomy is None or "subcategory" not in field_names:
        return []

    hierarchy_lines = [
        f"    {category} -> {', '.join(children)}"
        for category, children in taxonomy
        if children
    ]
    if not hierarchy_lines:
        return []

    lines = ["  subcategory children by category:"]
    lines.extend(hierarchy_lines)
    if any(not children for _category, children in taxonomy):
        lines.append(
            "    categories without listed children -> usually use null unless a clear direct child refinement is visible"
        )
    return lines


def _vocabulary_blocks(slot: str, field_names: tuple[str, ...]) -> tuple[str, str]:
    filtered_lines: list[str] = []
    example_lines: list[str] = []
    filtered_lines.extend(_category_lines_for(slot, field_names))
    example_lines.extend(_subcategory_guidance_lines_for(slot, field_names))
    for name in field_names:
        if name in {"category", "subcategory"}:
            continue
        canonical_tokens = CANONICAL_ATTRIBUTE_TOKENS.get(name)
        if canonical_tokens:
            filtered_lines.append(f"  {name}: {', '.join(canonical_tokens)}")
            continue
        example_tokens = EXAMPLE_ATTRIBUTE_TOKENS.get(name)
        if example_tokens:
            example_lines.append(f"  {name} examples: {', '.join(example_tokens)}")
            continue
    closed_lists = (
        "CLOSED LISTS — only these values are valid for these fields:\n"
        + "\n".join(filtered_lines)
        if filtered_lines
        else ""
    )
    open_vocab = (
        "OPEN VOCABULARY — preferred canonical examples:\n" + "\n".join(example_lines)
        if example_lines
        else ""
    )
    return closed_lists, open_vocab


def build_extraction_user_message(
    slot: str,
    visual_tags: str | None = None,
    *,
    field_guidance: str | None = None,
    field_names: tuple[str, ...] | None = None,
) -> str:
    schema = _schema_template_for(slot)
    resolved_field_names = field_names or _field_names_for(slot)
    closed_lists, open_vocab = _vocabulary_blocks(slot, resolved_field_names)
    shared_field_guidance_lines = [
        _SHARED_FIELD_GUIDANCE_LINES[name]
        for name in resolved_field_names
        if name in _SHARED_FIELD_GUIDANCE_LINES
    ]
    resolved_field_guidance = field_guidance or _ITEM_TYPE_FIELD_GUIDANCE.get(slot)
    sections: list[str] = []
    if closed_lists:
        sections.append(closed_lists)
    if resolved_field_guidance:
        sections.append(resolved_field_guidance)
    if shared_field_guidance_lines:
        sections.append(
            "FIELD RULES\n"
            + "\n".join(shared_field_guidance_lines)
            + "\n- Do not encode the same concept in multiple fields."
        )
    sections.append(STRUCTURED_EXTRACTION_EVIDENCE_PROMPT)
    if open_vocab:
        sections.append(open_vocab)
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


def _validate_taxonomy_config() -> None:
    expected_item_types = set(WEARABLE_ITEM_TYPES)
    configured_item_types = set(_ITEM_TYPE_TAXONOMIES)
    if expected_item_types != configured_item_types:
        missing_taxonomies = sorted(expected_item_types - configured_item_types)
        extra_taxonomies = sorted(configured_item_types - expected_item_types)
        raise RuntimeError(
            "taxonomy config mismatch: "
            f"missing={missing_taxonomies}, extra={extra_taxonomies}"
        )

    for item_type, taxonomy in _ITEM_TYPE_TAXONOMIES.items():
        categories = [category for category, _children in taxonomy]
        if len(categories) != len(set(categories)):
            raise RuntimeError(f"duplicate categories configured for {item_type}")

        category_set = set(categories)
        seen_children: set[str] = set()
        for category, children in taxonomy:
            for child in children:
                if child in category_set:
                    raise RuntimeError(
                        f"subcategory duplicates category token: {item_type}:{child}"
                    )
                if child in seen_children:
                    raise RuntimeError(
                        f"duplicate subcategory configured for {item_type}:{child}"
                    )
                seen_children.add(child)


_validate_taxonomy_config()


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
