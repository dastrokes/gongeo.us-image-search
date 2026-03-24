from __future__ import annotations

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
    "abilityHandhelds",
)

WEARABLE_ITEM_TYPES: tuple[str, ...] = (
    "hair",
    *GARMENT_ITEM_TYPES,
    "socks",
    "shoes",
    *ACCESSORY_ITEM_TYPES,
)


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
    "category": _field("category", "scalar"),
    "subcategory": _field("subcategory", "scalar"),
    "primary_color": _SHARED_COLOR_FIELDS[0],
    "secondary_color": _SHARED_COLOR_FIELDS[1],
    "pattern": _SHARED_VISUAL_FIELDS[0],
    "material": _SHARED_VISUAL_FIELDS[1],
    "structure": _SHARED_VISUAL_FIELDS[2],
    "ornament": _SHARED_VISUAL_FIELDS[3],
    # length / height fields
    "outerwear_length": _field("outerwear_length", "scalar"),
    "top_length": _field("top_length", "scalar"),
    "bottom_length": _field("bottom_length", "scalar"),
    "hair_length": _field("hair_length", "scalar"),
    "haircut": _field("haircut", "scalar"),
    # upper-body structure
    "fit": _field("fit", "scalar"),
    "neckline": _field("neckline", "scalar"),
    "shoulder_style": _field("shoulder_style", "scalar"),
    "sleeve_length": _field("sleeve_length", "scalar"),
    "sleeve_style": _field("sleeve_style", "scalar"),
    "closure_style": _field("closure_style", "scalar"),
    "front_style": _field("front_style", "scalar"),
    "hem": _field("hem", "scalar"),
    # bottoms-specific
    "skirt_silhouette": _field("skirt_silhouette", "scalar"),
    "pant_shape": _field("pant_shape", "scalar"),
    "waist_height": _field("waist_height", "scalar"),
    # dresses-specific
    "dress_silhouette": _field("dress_silhouette", "scalar"),
    "waistline": _field("waistline", "scalar"),
    # hair
    "texture": _field("texture", "scalar"),
    "bangs": _field("bangs", "scalar"),
    "adornment": _field("adornment", "array"),
    # shoes
    "heel_type": _field("heel_type", "scalar"),
    "heel_height": _field("heel_height", "scalar"),
    "sole_height": _field("sole_height", "scalar"),
    "shaft_height": _field("shaft_height", "scalar"),
    "toe_shape": _field("toe_shape", "scalar"),
    # socks
    "sock_height": _field("sock_height", "scalar"),
    "opacity": _field("opacity", "scalar"),
    "trim": _field("trim", "scalar"),
}

_SHARED_COLOR_FIELD_NAMES: tuple[str, ...] = ("primary_color", "secondary_color")
_SHARED_VISUAL_FIELD_NAMES: tuple[str, ...] = (
    "pattern",
    "material",
    "structure",
    "ornament",
)
# Shared upper-body structure fields (outerwear, tops, dresses)
_UPPER_BODY_FIELD_NAMES: tuple[str, ...] = (
    "fit",
    "neckline",
    "shoulder_style",
    "sleeve_length",
    "sleeve_style",
    "closure_style",
    "front_style",
    "hem",
)

OUTERWEAR_FIELDS: tuple[str, ...] = (
    "outerwear_length",
    *_UPPER_BODY_FIELD_NAMES,
)
TOP_FIELDS: tuple[str, ...] = (
    "top_length",
    *_UPPER_BODY_FIELD_NAMES,
)
BOTTOM_FIELDS: tuple[str, ...] = (
    "bottom_length",
    "skirt_silhouette",
    "pant_shape",
    "waist_height",
    "closure_style",
    "hem",
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
    "closure_style",
    "hem",
)
SHOE_FIELDS: tuple[str, ...] = (
    "heel_type",
    "heel_height",
    "sole_height",
    "shaft_height",
    "toe_shape",
    "closure_style",
)
SOCK_FIELDS: tuple[str, ...] = (
    "sock_height",
    "opacity",
    "trim",
)

_FIELD_NAMES_BY_ITEM_TYPE: dict[str, tuple[str, ...]] = {
    "outerwear": (
        "category",
        "subcategory",
        *_SHARED_COLOR_FIELD_NAMES,
        *OUTERWEAR_FIELDS,
        *_SHARED_VISUAL_FIELD_NAMES,
    ),
    "tops": (
        "category",
        "subcategory",
        *_SHARED_COLOR_FIELD_NAMES,
        *TOP_FIELDS,
        *_SHARED_VISUAL_FIELD_NAMES,
    ),
    "bottoms": (
        "category",
        "subcategory",
        *_SHARED_COLOR_FIELD_NAMES,
        *BOTTOM_FIELDS,
        *_SHARED_VISUAL_FIELD_NAMES,
    ),
    "dresses": (
        "category",
        "subcategory",
        *_SHARED_COLOR_FIELD_NAMES,
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
            "category",
            "subcategory",
            *_SHARED_COLOR_FIELD_NAMES,
            "hair_length",
            "haircut",
            "texture",
            "bangs",
            "adornment",
        ),
    ),
    "shoes": StructuredSchemaDefinition(
        name="shoes",
        item_types=("shoes",),
        fields=_fields(
            "category",
            "subcategory",
            *_SHARED_COLOR_FIELD_NAMES,
            *SHOE_FIELDS,
            *_SHARED_VISUAL_FIELD_NAMES,
        ),
    ),
    "socks": StructuredSchemaDefinition(
        name="socks",
        item_types=("socks",),
        fields=_fields(
            "category",
            "subcategory",
            *_SHARED_COLOR_FIELD_NAMES,
            *SOCK_FIELDS,
            *_SHARED_VISUAL_FIELD_NAMES,
        ),
    ),
    "accessory": StructuredSchemaDefinition(
        name="accessory",
        item_types=ACCESSORY_ITEM_TYPES,
        fields=_fields(
            "category",
            "subcategory",
            *_SHARED_COLOR_FIELD_NAMES,
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
