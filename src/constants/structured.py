from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from constants.tracker_export import load_tracker_export, normalize_supported_item_type


@dataclass(frozen=True, slots=True)
class StructuredFieldDefinition:
    name: str
    kind: str
    alias_pairs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class StructuredSchemaDefinition:
    name: str
    item_types: tuple[str, ...]
    fields: tuple[StructuredFieldDefinition, ...]


def _field(
    name: str,
    kind: str,
    *,
    aliases: tuple[tuple[str, str], ...] = (),
) -> StructuredFieldDefinition:
    return StructuredFieldDefinition(
        name=name,
        kind=kind,
        alias_pairs=aliases,
    )


_COLOR_ALIASES: tuple[tuple[str, str], ...] = (("grey", "gray"),)
_STRUCTURE_ALIASES: tuple[tuple[str, str], ...] = (
    ("drape", "draped"),
    ("draping", "draped"),
    ("gather", "gathered"),
    ("gathers", "gathered"),
    ("gathering", "gathered"),
    ("pleat", "pleated"),
    ("pleats", "pleated"),
    ("pleating", "pleated"),
    ("quilt", "quilted"),
    ("quilting", "quilted"),
    ("rib", "ribbed"),
    ("ribbing", "ribbed"),
    ("ruche", "ruched"),
    ("ruching", "ruched"),
    ("ruffle", "ruffled"),
    ("ruffles", "ruffled"),
    ("ruffled_hem", "ruffled"),
    ("smock", "smocked"),
    ("smocking", "smocked"),
    ("strapped", "strap"),
    ("straps", "strap"),
)
_PATTERN_ALIASES: tuple[tuple[str, str], ...] = (
    ("butterfly_wing", "butterfly"),
    ("leopard_print", "leopard"),
)
_ORNAMENT_ALIASES: tuple[tuple[str, str], ...] = (
    ("binder_clip", "clip"),
    ("bird_figurine", "bird"),
    ("beads", "bead"),
    ("feathers", "feather"),
    ("flowers", "flower"),
    ("gem", "gemstone"),
    ("lightbulb", "light_bulb"),
    ("pearls", "pearl"),
    ("plush_animal", "plushie"),
    ("pompom", "pom_pom"),
    ("pom_poms", "pom_pom"),
    ("pompoms", "pom_pom"),
    ("studs", "stud"),
    ("wings", "wing"),
)
_CATEGORY_ALIASES: tuple[tuple[str, str], ...] = (
    ("apron_dress", "pinafore_dress"),
    ("boot", "boots"),
    ("flat", "flats"),
    ("fur_stole", "stole"),
    ("glove", "gloves"),
    ("heel", "heels"),
    ("loafer", "loafers"),
    ("mary_jane", "mary_janes"),
    ("overall_dress", "pinafore_dress"),
    ("playsuit", "romper"),
    ("sandal", "sandals"),
    ("shawl_wrap", "shawl"),
    ("sneaker", "sneakers"),
    ("sock", "socks"),
    ("stocking", "stockings"),
    ("stole", "shawl"),
    ("tee", "t_shirt"),
    ("tshirt", "t_shirt"),
    ("twin_tail", "twin_tails"),
)
_SUBCATEGORY_ALIASES: tuple[tuple[str, str], ...] = (
    ("apron_dress", "pinafore_dress"),
    ("chandelier_earring", "chandelier_earrings"),
    ("drop_earring", "drop_earrings"),
    ("fingerless_glove", "fingerless_gloves"),
    ("fur_stole", "stole"),
    ("half_up_braids", "half_up_braid"),
    ("hoop_earring", "hoop_earrings"),
    ("lace_glove", "lace_gloves"),
    ("opera_glove", "opera_gloves"),
    ("overall_dress", "pinafore_dress"),
    ("playsuit", "romper"),
    ("shawl", "stole"),
    ("side_buns", "side_bun"),
    ("stud_earring", "stud_earrings"),
)
_GARMENT_FEATURE_ALIASES: tuple[tuple[str, str], ...] = (
    ("belt", "belted"),
    ("buttoned", "button_up"),
    ("buttoned_up", "button_up"),
    ("button_front", "button_up"),
    ("hood", "hooded"),
    ("hoodie", "hooded"),
    ("laced", "lace_up"),
    ("laced_up", "lace_up"),
    ("zip_front", "zip_up"),
    ("zipped", "zip_up"),
    ("zippered", "zip_up"),
)

_FIELD_ALIASES: dict[str, tuple[tuple[str, str], ...]] = {
    "category": _CATEGORY_ALIASES,
    "subcategory": _SUBCATEGORY_ALIASES,
    "garment_feature": _GARMENT_FEATURE_ALIASES,
    "ornament": _ORNAMENT_ALIASES,
    "pattern": _PATTERN_ALIASES,
    "primary_color": _COLOR_ALIASES,
    "secondary_color": _COLOR_ALIASES,
    "structure": _STRUCTURE_ALIASES,
}

_tracker_export = load_tracker_export()
_field_kind_by_name = {
    str(name): str(kind)
    for name, kind in (_tracker_export.get("fieldKindByName") or {}).items()
}
_schema_key_by_item_type = {
    normalize_supported_item_type(str(item_type)): str(schema_key)
    for item_type, schema_key in (
        _tracker_export.get("schemaKeyByItemType") or {}
    ).items()
}
_field_names_by_item_type = {
    normalize_supported_item_type(str(item_type)): tuple(
        str(field_name) for field_name in (field_names or [])
    )
    for item_type, field_names in (
        _tracker_export.get("fieldNamesByItemType") or {}
    ).items()
}

WEARABLE_ITEM_TYPES: tuple[str, ...] = tuple(
    normalize_supported_item_type(str(item_type))
    for item_type in (_tracker_export.get("supportedItemTypes") or [])
)

GARMENT_ITEM_TYPES: tuple[str, ...] = tuple(
    item_type
    for item_type in WEARABLE_ITEM_TYPES
    if _schema_key_by_item_type.get(item_type) == "garment"
)

ACCESSORY_ITEM_TYPES: tuple[str, ...] = tuple(
    item_type
    for item_type in WEARABLE_ITEM_TYPES
    if _schema_key_by_item_type.get(item_type) == "accessory"
)

CLOTHING_ITEM_TYPES: tuple[str, ...] = tuple(
    item_type
    for item_type in (
        "hair",
        "dresses",
        "tops",
        "bottoms",
        "outerwear",
        "socks",
        "shoes",
    )
    if item_type in WEARABLE_ITEM_TYPES
)

SPECIAL_ITEM_TYPE_FILTERS: dict[str, tuple[str, ...]] = {
    "clothing": CLOTHING_ITEM_TYPES,
    "accessories": tuple(
        item_type
        for item_type in WEARABLE_ITEM_TYPES
        if item_type not in CLOTHING_ITEM_TYPES
    ),
}


def expand_item_type_filters(item_types: Iterable[str] | None) -> set[str]:
    expanded: set[str] = set()
    for item_type in item_types or ():
        normalized = normalize_supported_item_type(item_type)
        expanded.update(SPECIAL_ITEM_TYPE_FILTERS.get(normalized, (normalized,)))
    unknown = expanded.difference(WEARABLE_ITEM_TYPES)
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown item type filter(s): {joined}")
    return expanded


_FIELD_LIBRARY: dict[str, StructuredFieldDefinition] = {
    field_name: _field(
        field_name,
        field_kind,
        aliases=_FIELD_ALIASES.get(field_name, ()),
    )
    for field_name, field_kind in _field_kind_by_name.items()
}


def _fields(*field_names: str) -> tuple[StructuredFieldDefinition, ...]:
    return tuple(_FIELD_LIBRARY[name] for name in field_names)


STRUCTURED_SCHEMA_ITEM_TYPES: dict[str, tuple[str, ...]] = {
    schema_key: tuple(
        item_type
        for item_type in WEARABLE_ITEM_TYPES
        if _schema_key_by_item_type.get(item_type) == schema_key
    )
    for schema_key in sorted(set(_schema_key_by_item_type.values()))
}


def schema_definition_for_item_type(item_type: str) -> StructuredSchemaDefinition:
    normalized = normalize_supported_item_type(item_type)
    schema_key = _schema_key_by_item_type.get(normalized, "accessory")
    schema_item_types = STRUCTURED_SCHEMA_ITEM_TYPES.get(schema_key, ())
    fallback_item_type = schema_item_types[0] if schema_item_types else None
    field_names = _field_names_by_item_type.get(
        normalized,
        _field_names_by_item_type.get(fallback_item_type or "", ()),
    )
    return StructuredSchemaDefinition(
        name=schema_key,
        item_types=schema_item_types,
        fields=_fields(*field_names),
    )


def is_supported_item_type(item_type: str) -> bool:
    return normalize_supported_item_type(item_type) in WEARABLE_ITEM_TYPES

