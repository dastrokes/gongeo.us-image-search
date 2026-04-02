from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


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
    ("stud_earring", "stud_earrings"),
)

GARMENT_ITEM_TYPES: tuple[str, ...] = ("outerwear", "tops", "dresses", "bottoms")

ACCESSORY_ITEM_TYPES: tuple[str, ...] = (
    "hairAccessories",
    "headwear",
    "earrings",
    "neckwear",
    "bracelets",
    "chokers",
    "gloves",
    "handhelds",
    "chestAccessories",
    "pendants",
    "backpieces",
    "rings",
    "armDecorations",
    "faceDecorations",
    "bodyPaint",
    "abilityHandhelds",
)

CLOTHING_ITEM_TYPES: tuple[str, ...] = (
    "hair",
    "dresses",
    "tops",
    "bottoms",
    "outerwear",
    "socks",
    "shoes",
)

WEARABLE_ITEM_TYPES: tuple[str, ...] = (
    "hair",
    *GARMENT_ITEM_TYPES,
    "socks",
    "shoes",
    *ACCESSORY_ITEM_TYPES,
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
        expanded.update(SPECIAL_ITEM_TYPE_FILTERS.get(item_type, (item_type,)))
    unknown = expanded.difference(WEARABLE_ITEM_TYPES)
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown item type filter(s): {joined}")
    return expanded


_SHARED_COLOR_FIELDS: tuple[StructuredFieldDefinition, ...] = (
    _field("primary_color", "scalar", aliases=_COLOR_ALIASES),
    _field("secondary_color", "scalar", aliases=_COLOR_ALIASES),
)

_SHARED_VISUAL_FIELDS: tuple[StructuredFieldDefinition, ...] = (
    _field("pattern", "array"),
    _field("material", "array"),
    _field("structure", "array"),
    _field("ornament", "array"),
)


_FIELD_LIBRARY: dict[str, StructuredFieldDefinition] = {
    "category": _field("category", "scalar", aliases=_CATEGORY_ALIASES),
    "subcategory": _field("subcategory", "scalar", aliases=_SUBCATEGORY_ALIASES),
    "primary_color": _SHARED_COLOR_FIELDS[0],
    "secondary_color": _SHARED_COLOR_FIELDS[1],
    "pattern": _SHARED_VISUAL_FIELDS[0],
    "material": _SHARED_VISUAL_FIELDS[1],
    "structure": _SHARED_VISUAL_FIELDS[2],
    "ornament": _SHARED_VISUAL_FIELDS[3],
    # length / height fields
    "top_length": _field("top_length", "scalar"),
    "bottom_length": _field("bottom_length", "scalar"),
    "hair_length": _field("hair_length", "scalar"),
    # upper-body structure
    "fit": _field("fit", "scalar"),
    "neckline": _field("neckline", "scalar"),
    "shoulder_style": _field("shoulder_style", "scalar"),
    "sleeve_length": _field("sleeve_length", "scalar"),
    "sleeve_style": _field("sleeve_style", "scalar"),
    # bottoms-specific
    "skirt_silhouette": _field("skirt_silhouette", "scalar"),
    "pant_shape": _field("pant_shape", "scalar"),
    "waist_height": _field("waist_height", "scalar"),
    # dresses-specific
    "dress_silhouette": _field("dress_silhouette", "scalar"),
    "waistline": _field("waistline", "scalar"),
    # hair
    "haircut": _field("haircut", "scalar"),
    "texture": _field("texture", "scalar"),
    "bangs": _field("bangs", "scalar"),
    # shoes
    "heel_type": _field("heel_type", "scalar"),
    "heel_height": _field("heel_height", "scalar"),
    "sole_height": _field("sole_height", "scalar"),
    "shaft_height": _field("shaft_height", "scalar"),
    # socks
    "sock_height": _field("sock_height", "scalar"),
}

_SHARED_COLOR_FIELD_NAMES: tuple[str, ...] = ("primary_color", "secondary_color")
_SHARED_VISUAL_FIELD_NAMES: tuple[str, ...] = (
    "pattern",
    "material",
    "structure",
    "ornament",
)
# Kept as separate stored fields for compatibility, but treated as a parent/child
# taxonomy pair by the prompt and extractor layers.
_TAXONOMY_FIELD_NAMES: tuple[str, ...] = ("category", "subcategory")
# Shared upper-body structure fields (outerwear, tops, dresses)
_UPPER_BODY_FIELD_NAMES: tuple[str, ...] = (
    "top_length",
    "fit",
    "neckline",
    "shoulder_style",
    "sleeve_length",
    "sleeve_style",
)

OUTERWEAR_FIELDS: tuple[str, ...] = (*_UPPER_BODY_FIELD_NAMES,)
TOP_FIELDS: tuple[str, ...] = (*_UPPER_BODY_FIELD_NAMES,)
BOTTOM_FIELDS: tuple[str, ...] = (
    "bottom_length",
    "skirt_silhouette",
    "pant_shape",
    "waist_height",
)
DRESS_FIELDS: tuple[str, ...] = (
    "bottom_length",
    "dress_silhouette",
    "fit",
    "waistline",
    "neckline",
    "shoulder_style",
    "sleeve_length",
    "sleeve_style",
)
SHOE_FIELDS: tuple[str, ...] = (
    "heel_type",
    "heel_height",
    "sole_height",
    "shaft_height",
)
SOCK_FIELDS: tuple[str, ...] = ("sock_height",)

_FIELD_NAMES_BY_ITEM_TYPE: dict[str, tuple[str, ...]] = {
    "outerwear": (
        *_TAXONOMY_FIELD_NAMES,
        *OUTERWEAR_FIELDS,
        *_SHARED_VISUAL_FIELD_NAMES,
    ),
    "tops": (
        *_TAXONOMY_FIELD_NAMES,
        *TOP_FIELDS,
        *_SHARED_VISUAL_FIELD_NAMES,
    ),
    "bottoms": (
        *_TAXONOMY_FIELD_NAMES,
        *BOTTOM_FIELDS,
        *_SHARED_VISUAL_FIELD_NAMES,
    ),
    "dresses": (
        *_TAXONOMY_FIELD_NAMES,
        *DRESS_FIELDS,
        *_SHARED_VISUAL_FIELD_NAMES,
    ),
}


def _fields(*field_names: str) -> tuple[StructuredFieldDefinition, ...]:
    return tuple(_FIELD_LIBRARY[name] for name in field_names)


STRUCTURED_SCHEMAS: dict[str, StructuredSchemaDefinition] = {
    "outerwear": StructuredSchemaDefinition(
        name="garment",
        item_types=("outerwear",),
        fields=_fields(*_FIELD_NAMES_BY_ITEM_TYPE["outerwear"]),
    ),
    "tops": StructuredSchemaDefinition(
        name="garment",
        item_types=("tops",),
        fields=_fields(*_FIELD_NAMES_BY_ITEM_TYPE["tops"]),
    ),
    "bottoms": StructuredSchemaDefinition(
        name="garment",
        item_types=("bottoms",),
        fields=_fields(*_FIELD_NAMES_BY_ITEM_TYPE["bottoms"]),
    ),
    "dresses": StructuredSchemaDefinition(
        name="garment",
        item_types=("dresses",),
        fields=_fields(*_FIELD_NAMES_BY_ITEM_TYPE["dresses"]),
    ),
    "hair": StructuredSchemaDefinition(
        name="hair",
        item_types=("hair",),
        fields=_fields(
            *_TAXONOMY_FIELD_NAMES,
            "hair_length",
            "haircut",
            "texture",
            "bangs",
        ),
    ),
    "shoes": StructuredSchemaDefinition(
        name="shoes",
        item_types=("shoes",),
        fields=_fields(
            *_TAXONOMY_FIELD_NAMES,
            *SHOE_FIELDS,
            *_SHARED_VISUAL_FIELD_NAMES,
        ),
    ),
    "socks": StructuredSchemaDefinition(
        name="socks",
        item_types=("socks",),
        fields=_fields(
            *_TAXONOMY_FIELD_NAMES,
            *SOCK_FIELDS,
            *_SHARED_VISUAL_FIELD_NAMES,
        ),
    ),
    "accessory": StructuredSchemaDefinition(
        name="accessory",
        item_types=ACCESSORY_ITEM_TYPES,
        fields=_fields(
            *_TAXONOMY_FIELD_NAMES,
            *_SHARED_VISUAL_FIELD_NAMES,
        ),
    ),
}

ITEM_TYPE_TO_SCHEMA_KEY: dict[str, str] = {
    item_type: schema_key
    for schema_key, schema in STRUCTURED_SCHEMAS.items()
    for item_type in schema.item_types
}


def schema_definition_for_item_type(item_type: str) -> StructuredSchemaDefinition:
    return STRUCTURED_SCHEMAS[ITEM_TYPE_TO_SCHEMA_KEY.get(item_type, "accessory")]


def is_supported_item_type(item_type: str) -> bool:
    return item_type in WEARABLE_ITEM_TYPES
