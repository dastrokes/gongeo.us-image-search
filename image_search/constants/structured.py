from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StructuredFieldDefinition:
    name: str
    kind: str
    alias_pairs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class StructuredShapeDefinition:
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


_COLOR_ALIASES: tuple[tuple[str, str], ...] = (
    ("grey", "gray"),
    ("multi_color", "multicolor"),
    ("multi_colored", "multicolor"),
    ("multi_colour", "multicolor"),
    ("multi_coloured", "multicolor"),
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
    _field("ornament", "array"),
)


_FIELD_LIBRARY: dict[str, StructuredFieldDefinition] = {
    "category": _field("category", "scalar"),
    "subcategory": _field("subcategory", "scalar"),
    "primary_color": _SHARED_COLOR_FIELDS[0],
    "secondary_color": _SHARED_COLOR_FIELDS[1],
    "pattern": _SHARED_VISUAL_FIELDS[0],
    "material": _SHARED_VISUAL_FIELDS[1],
    "ornament": _SHARED_VISUAL_FIELDS[2],
    "top_length": _field("top_length", "scalar"),
    "bottom_length": _field("bottom_length", "scalar"),
    "dress_length": _field("dress_length", "scalar"),
    "fit": _field("fit", "scalar"),
    "dress_silhouette": _field("dress_silhouette", "scalar"),
    "neckline": _field("neckline", "scalar"),
    "shoulder": _field("shoulder", "scalar"),
    "collar": _field("collar", "array"),
    "closure": _field("closure", "array"),
    "sleeve_length": _field("sleeve_length", "scalar"),
    "sleeve_shape": _field("sleeve_shape", "array"),
    "hair_length": _field("hair_length", "scalar"),
    "texture": _field("texture", "scalar"),
    "parting": _field("parting", "scalar"),
    "bangs": _field("bangs", "scalar"),
    "hairstyle": _field("hairstyle", "array"),
    "adornment": _field("adornment", "array"),
    "shaft_height": _field("shaft_height", "scalar"),
    "heel_height": _field("heel_height", "scalar"),
    "toe_shape": _field("toe_shape", "scalar"),
    "platform": _field("platform", "scalar"),
    "sock_height": _field("sock_height", "scalar"),
    "opacity": _field("opacity", "scalar"),
    "trim": _field("trim", "array"),
    "placement": _field("placement", "scalar"),
    "attachment": _field("attachment", "array"),
    "shape": _field("shape", "array"),
}

_SHARED_COLOR_FIELD_NAMES: tuple[str, ...] = ("primary_color", "secondary_color")
_SHARED_VISUAL_FIELD_NAMES: tuple[str, ...] = (
    "pattern",
    "material",
    "ornament",
)
_UPPER_BODY_FIELD_NAMES: tuple[str, ...] = (
    "neckline",
    "shoulder",
    "collar",
    "closure",
    "fit",
    "sleeve_length",
    "sleeve_shape",
)

_FIELD_NAMES_BY_ITEM_TYPE: dict[str, tuple[str, ...]] = {
    "outerwear": (
        "category",
        "subcategory",
        *_SHARED_COLOR_FIELD_NAMES,
        *_SHARED_VISUAL_FIELD_NAMES,
        "top_length",
        *_UPPER_BODY_FIELD_NAMES,
    ),
    "tops": (
        "category",
        "subcategory",
        *_SHARED_COLOR_FIELD_NAMES,
        *_SHARED_VISUAL_FIELD_NAMES,
        "top_length",
        *_UPPER_BODY_FIELD_NAMES,
    ),
    "bottoms": (
        "category",
        "subcategory",
        *_SHARED_COLOR_FIELD_NAMES,
        *_SHARED_VISUAL_FIELD_NAMES,
        "bottom_length",
        "closure",
        "fit",
    ),
    "dresses": (
        "category",
        "subcategory",
        *_SHARED_COLOR_FIELD_NAMES,
        *_SHARED_VISUAL_FIELD_NAMES,
        "dress_length",
        "dress_silhouette",
        *_UPPER_BODY_FIELD_NAMES,
    ),
}


def _fields(*field_names: str) -> tuple[StructuredFieldDefinition, ...]:
    return tuple(_FIELD_LIBRARY[name] for name in field_names)


STRUCTURED_SHAPES: dict[str, StructuredShapeDefinition] = {
    "outerwear": StructuredShapeDefinition(
        name="garment",
        item_types=("outerwear",),
        fields=_fields(*_FIELD_NAMES_BY_ITEM_TYPE["outerwear"]),
    ),
    "tops": StructuredShapeDefinition(
        name="garment",
        item_types=("tops",),
        fields=_fields(*_FIELD_NAMES_BY_ITEM_TYPE["tops"]),
    ),
    "bottoms": StructuredShapeDefinition(
        name="garment",
        item_types=("bottoms",),
        fields=_fields(*_FIELD_NAMES_BY_ITEM_TYPE["bottoms"]),
    ),
    "dresses": StructuredShapeDefinition(
        name="garment",
        item_types=("dresses",),
        fields=_fields(*_FIELD_NAMES_BY_ITEM_TYPE["dresses"]),
    ),
    "hair": StructuredShapeDefinition(
        name="hair",
        item_types=("hair",),
        fields=_fields(
            "category",
            "subcategory",
            *_SHARED_COLOR_FIELD_NAMES,
            "hair_length",
            "texture",
            "parting",
            "bangs",
            "hairstyle",
            "adornment",
        ),
    ),
    "shoes": StructuredShapeDefinition(
        name="shoes",
        item_types=("shoes",),
        fields=_fields(
            "category",
            "subcategory",
            *_SHARED_COLOR_FIELD_NAMES,
            *_SHARED_VISUAL_FIELD_NAMES,
            "shaft_height",
            "heel_height",
            "toe_shape",
            "platform",
            "closure",
        ),
    ),
    "socks": StructuredShapeDefinition(
        name="socks",
        item_types=("socks",),
        fields=_fields(
            "category",
            "subcategory",
            *_SHARED_COLOR_FIELD_NAMES,
            *_SHARED_VISUAL_FIELD_NAMES,
            "sock_height",
            "opacity",
            "trim",
        ),
    ),
    "accessory": StructuredShapeDefinition(
        name="accessory",
        item_types=ACCESSORY_ITEM_TYPES,
        fields=_fields(
            "category",
            "subcategory",
            *_SHARED_COLOR_FIELD_NAMES,
            *_SHARED_VISUAL_FIELD_NAMES,
            "placement",
            "attachment",
            "shape",
        ),
    ),
}

ITEM_TYPE_TO_SHAPE_KEY: dict[str, str] = {
    item_type: shape_key
    for shape_key, shape in STRUCTURED_SHAPES.items()
    for item_type in shape.item_types
}


def shape_definition_for_item_type(item_type: str) -> StructuredShapeDefinition:
    return STRUCTURED_SHAPES[ITEM_TYPE_TO_SHAPE_KEY.get(item_type, "accessory")]


def is_supported_item_type(item_type: str) -> bool:
    return item_type in WEARABLE_ITEM_TYPES
