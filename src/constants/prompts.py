from __future__ import annotations

import json

from constants.structured import WEARABLE_ITEM_TYPES, schema_definition_for_item_type
from constants.tracker_export import load_tracker_export, normalize_supported_item_type

STRUCTURED_EXTRACTION_SYSTEM_PROMPT = (
    "You are a deterministic structured extraction engine for Infinity Nikki items.\n\n"
    "Output exactly one raw JSON object. No markdown, no code fences, no prose.\n"
    "Use schema keys exactly as written. Do not add, rename, omit, or reorder keys.\n\n"
    "VALUE RULES\n"
    "- Non-null values must be lowercase underscore_tokens.\n"
    "- Scalars: one token or null.\n"
    "- Arrays: unique tokens or [].\n"
    "- Array fields are open-list: prefer known tokens when they fit, otherwise use a new concise token.\n"
    "- Each field must represent a single concept only.\n"
    "- Do not combine multiple attributes into one token.\n\n"
    "TAXONOMY\n"
    "- category = the main visible item class, chosen at the schema root level rather than a more specific child type.\n"
    "- category must be one of the values in CLOSED LISTS.\n"
    "- subcategory should be a concise direct child refinement of the chosen category, or null.\n"
    "- Prefer the listed child examples when they fit, but subcategory is open-list.\n"
    "- If no listed child term fits well, you may use a new concise child refinement or null.\n"
    "- subcategory should stay compatible with the selected category. If unsure, set subcategory = null.\n"
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
    "garment_feature": "- garment_feature = visible functional garment features or closures such as hooded, belted, button_up, lace_up, or zip_up; do not use purely decorative buttons, bows, or lacing.",
}
_OPEN_LIST_FIELD_EXAMPLES: dict[str, tuple[str, ...]] = {
    "garment_feature": ("hooded", "belted", "button_up", "lace_up", "zip_up"),
    "material": ("lace", "leather", "knit", "satin", "sheer"),
    "ornament": ("bow", "button", "chain", "embroidery", "ruffle"),
    "pattern": ("floral", "plaid", "solid", "star", "striped"),
    "structure": ("cutout", "draped", "gathered", "pleated", "ribbed"),
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
        "- heel_type only when a distinct heel is present; heel_height is flat otherwise, or null when the heel component is absent.\n"
        "- sole_height is low or high for visibly thick, platform, or elevated soles; otherwise flat, or null when the sole component is absent.\n"
        "- shaft_height only for boots; otherwise null."
    ),
}

_tracker_export = load_tracker_export()
_shared_terms = _tracker_export.get("terms", {}).get("sharedFieldValues", {})
_scoped_terms = _tracker_export.get("terms", {}).get("scopedFieldValues", {})
_field_kind_by_name = _tracker_export.get("fieldKindByName", {})
_subcategory_parent_by_type = _tracker_export.get("taxonomy", {}).get(
    "subcategoryParentByType", {}
)

CANONICAL_ATTRIBUTE_TOKENS: dict[str, tuple[str, ...]] = {
    str(field_name): tuple(str(value) for value in values or [])
    for field_name, values in _shared_terms.items()
}
FILTERED_CANONICAL_ATTRIBUTE_FIELDS: frozenset[str] = frozenset(
    str(field_name)
    for field_name, kind in (_field_kind_by_name or {}).items()
    if str(kind) == "scalar" and str(field_name) not in {"category", "subcategory"}
)
CANONICAL_CATEGORY_TOKENS: dict[str, tuple[str, ...]] = {
    normalize_supported_item_type(str(item_type)): tuple(
        str(value) for value in values or []
    )
    for item_type, values in (_scoped_terms.get("category") or {}).items()
}
CANONICAL_SUBCATEGORY_TOKENS: dict[str, tuple[str, ...]] = {
    normalize_supported_item_type(str(item_type)): tuple(
        str(value) for value in values or []
    )
    for item_type, values in (_scoped_terms.get("subcategory") or {}).items()
}
SUBCATEGORY_HIERARCHY: dict[str, dict[str, str]] = {
    normalize_supported_item_type(str(item_type)): {
        str(child): str(parent) for child, parent in (parent_map or {}).items()
    }
    for item_type, parent_map in _subcategory_parent_by_type.items()
}
TOKEN_ALIASES: dict[str, str] = {}
for item_type in WEARABLE_ITEM_TYPES:
    for field_definition in schema_definition_for_item_type(item_type).fields:
        for source, target in field_definition.alias_pairs:
            TOKEN_ALIASES[source] = target


def normalise_token(token: str) -> str:
    return TOKEN_ALIASES.get(token, token)


def get_subcategory_ancestors(item_type: str, subcategory: str) -> list[str]:
    ancestors: list[str] = []
    current = subcategory
    hierarchy = SUBCATEGORY_HIERARCHY.get(normalize_supported_item_type(item_type), {})
    while current in hierarchy:
        current = hierarchy[current]
        ancestors.append(current)
    return ancestors


def _field_names_for(slot: str) -> tuple[str, ...]:
    return tuple(
        field_definition.name
        for field_definition in schema_definition_for_item_type(
            normalize_supported_item_type(slot)
        ).fields
    )


def _schema_template_for(
    slot: str,
    field_names: tuple[str, ...] | None = None,
) -> dict[str, object]:
    normalized_slot = normalize_supported_item_type(slot)
    allowed_fields = set(field_names or _field_names_for(normalized_slot))
    template: dict[str, object] = {}
    for field_definition in schema_definition_for_item_type(normalized_slot).fields:
        if field_definition.name not in allowed_fields:
            continue
        template[field_definition.name] = (
            [] if field_definition.kind == "array" else None
        )
    return template


def _category_guidance_lines_for(slot: str) -> list[str]:
    normalized_slot = normalize_supported_item_type(slot)
    categories = CANONICAL_CATEGORY_TOKENS.get(normalized_slot, ())
    if not categories:
        return []
    return [f"  category: {', '.join(categories)}"]


def _subcategory_guidance_lines_for(
    slot: str,
    field_names: tuple[str, ...],
) -> list[str]:
    if "subcategory" not in field_names:
        return []

    normalized_slot = normalize_supported_item_type(slot)
    hierarchy = SUBCATEGORY_HIERARCHY.get(normalized_slot, {})
    categories = CANONICAL_CATEGORY_TOKENS.get(normalized_slot, ())
    lines: list[str] = []

    for category in categories:
        children = sorted(
            child for child, parent in hierarchy.items() if parent == category
        )
        if not children:
            lines.append(f"  {category}: null")
            continue
        lines.append(f"  {category}: {', '.join(children)}")

    return lines


def _attribute_guidance_lines_for(
    slot: str,
    field_names: tuple[str, ...],
) -> list[str]:
    lines: list[str] = []
    for field_name in field_names:
        if field_name not in FILTERED_CANONICAL_ATTRIBUTE_FIELDS:
            continue
        canonical_tokens = CANONICAL_ATTRIBUTE_TOKENS.get(field_name, ())
        if canonical_tokens:
            lines.append(f"  {field_name}: {', '.join(canonical_tokens)}")
    return lines


def _shared_field_guidance_line(field_name: str) -> str | None:
    base_line = _SHARED_FIELD_GUIDANCE_LINES.get(field_name)
    if not base_line:
        return None

    example_tokens = _OPEN_LIST_FIELD_EXAMPLES.get(field_name)
    if not example_tokens:
        return base_line

    return f"{base_line} Examples: {', '.join(example_tokens)}."


def build_extraction_user_message(
    slot: str,
    visual_tags: str | None = None,
    *,
    field_guidance: str | None = None,
    field_names: tuple[str, ...] | None = None,
) -> str:
    normalized_slot = normalize_supported_item_type(slot)
    resolved_field_names = field_names or _field_names_for(normalized_slot)
    schema = _schema_template_for(normalized_slot, resolved_field_names)
    preferred_subcategory_lines = _subcategory_guidance_lines_for(
        normalized_slot, resolved_field_names
    )
    closed_list_lines = [
        *_category_guidance_lines_for(normalized_slot),
        *_attribute_guidance_lines_for(normalized_slot, resolved_field_names),
    ]

    parts = [
        f"ITEM TYPE\n- slot = {normalized_slot}",
        "SCHEMA TEMPLATE\n" + json.dumps(schema, ensure_ascii=False, indent=2),
        STRUCTURED_EXTRACTION_EVIDENCE_PROMPT,
    ]

    if visual_tags:
        parts.append(f"VISUAL CONTEXT\n- hint = {visual_tags}")

    guidance_blocks = [
        _ITEM_TYPE_FIELD_GUIDANCE.get(normalized_slot),
        field_guidance,
        *(
            _shared_field_guidance_line(field_name)
            for field_name in resolved_field_names
        ),
    ]
    filtered_guidance_blocks = [block for block in guidance_blocks if block]
    if filtered_guidance_blocks:
        parts.append("\n".join(filtered_guidance_blocks))

    if closed_list_lines:
        parts.append("CLOSED LISTS\n" + "\n".join(closed_list_lines))
    if preferred_subcategory_lines:
        parts.append(
            "PREFERRED SUBCATEGORY EXAMPLES\n" + "\n".join(preferred_subcategory_lines)
        )

    parts.append(
        "OUTPUT\n"
        "- Return exactly one JSON object matching the schema template.\n"
        "- Preserve the field order shown in SCHEMA TEMPLATE.\n"
        "- Use null or [] when unsupported, invisible, or uncertain."
    )
    return "\n\n".join(parts)


def _validate_taxonomy_config() -> None:
    expected_item_types = set(WEARABLE_ITEM_TYPES)
    actual_item_types = set(CANONICAL_CATEGORY_TOKENS) | set(
        CANONICAL_SUBCATEGORY_TOKENS
    )
    if expected_item_types != actual_item_types:
        missing = ", ".join(sorted(expected_item_types.difference(actual_item_types)))
        extra = ", ".join(sorted(actual_item_types.difference(expected_item_types)))
        problems = []
        if missing:
            problems.append(f"missing={missing}")
        if extra:
            problems.append(f"extra={extra}")
        joined = "; ".join(problems)
        raise RuntimeError(f"tracker export taxonomy mismatch: {joined}")


_validate_taxonomy_config()

__all__ = [
    "CANONICAL_ATTRIBUTE_TOKENS",
    "CANONICAL_CATEGORY_TOKENS",
    "CANONICAL_SUBCATEGORY_TOKENS",
    "FILTERED_CANONICAL_ATTRIBUTE_FIELDS",
    "STRUCTURED_EXTRACTION_SYSTEM_PROMPT",
    "SUBCATEGORY_HIERARCHY",
    "TOKEN_ALIASES",
    "build_extraction_user_message",
    "get_subcategory_ancestors",
    "normalise_token",
]
