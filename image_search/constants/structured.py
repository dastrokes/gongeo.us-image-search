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


def _shared_color_fields() -> tuple[StructuredFieldDefinition, ...]:
    return (
        _field("primary_color", "scalar", aliases=_COLOR_ALIASES),
        _field(
            "secondary_color",
            "scalar",
            aliases=_COLOR_ALIASES,
        ),
    )


def _shared_visual_fields() -> tuple[StructuredFieldDefinition, ...]:
    return (
        _field("pattern", "array"),
        _field("material", "array"),
        _field("ornament", "array"),
        _field("style", "array"),
        _field("theme", "array"),
        _field("occasion", "array"),
    )


def _garment_fields(
    length_name: str,
) -> tuple[StructuredFieldDefinition, ...]:
    return (
        _field("category", "scalar"),
        _field("subcategory", "scalar"),
        *_shared_color_fields(),
        *_shared_visual_fields(),
        _field(length_name, "scalar"),
        _field("silhouette", "scalar"),
        _field("neckline", "scalar"),
        _field("collar", "array"),
        _field("closure", "array"),
        _field("sleeve_length", "scalar"),
        _field("sleeve_shape", "array"),
    )


STRUCTURED_SHAPES: dict[str, StructuredShapeDefinition] = {
    "outerwear": StructuredShapeDefinition(
        name="garment",
        item_types=("outerwear",),
        fields=_garment_fields("outerwear_length"),
    ),
    "tops": StructuredShapeDefinition(
        name="garment",
        item_types=("tops",),
        fields=_garment_fields("top_length"),
    ),
    "bottoms": StructuredShapeDefinition(
        name="garment",
        item_types=("bottoms",),
        fields=_garment_fields("bottom_length"),
    ),
    "dresses": StructuredShapeDefinition(
        name="garment",
        item_types=("dresses",),
        fields=_garment_fields("dress_length"),
    ),
    "hair": StructuredShapeDefinition(
        name="hair",
        item_types=("hair",),
        fields=(
            _field("category", "scalar"),
            _field("subcategory", "scalar"),
            *_shared_color_fields(),
            _field("hair_length", "scalar"),
            _field("texture", "scalar"),
            _field("parting", "scalar"),
            _field("bangs", "scalar"),
            _field("hairstyle", "array"),
            _field("adornment", "array"),
            _field("style", "array"),
            _field("theme", "array"),
            _field("occasion", "array"),
        ),
    ),
    "shoes": StructuredShapeDefinition(
        name="shoes",
        item_types=("shoes",),
        fields=(
            _field("category", "scalar"),
            _field("subcategory", "scalar"),
            *_shared_color_fields(),
            *_shared_visual_fields(),
            _field("shaft_height", "scalar"),
            _field("heel_height", "scalar"),
            _field("toe_shape", "scalar"),
            _field("platform", "scalar"),
            _field("closure", "array"),
        ),
    ),
    "socks": StructuredShapeDefinition(
        name="socks",
        item_types=("socks",),
        fields=(
            _field("category", "scalar"),
            _field("subcategory", "scalar"),
            *_shared_color_fields(),
            *_shared_visual_fields(),
            _field("sock_height", "scalar"),
            _field("opacity", "scalar"),
            _field("trim", "array"),
        ),
    ),
    "accessory": StructuredShapeDefinition(
        name="accessory",
        item_types=ACCESSORY_ITEM_TYPES,
        fields=(
            _field("category", "scalar"),
            _field("subcategory", "scalar"),
            *_shared_color_fields(),
            *_shared_visual_fields(),
            _field("placement", "scalar"),
            _field("attachment", "array"),
            _field("shape", "array"),
        ),
    ),
}

ITEM_TYPE_TO_SHAPE_KEY: dict[str, str] = {
    item_type: shape_key
    for shape_key, shape in STRUCTURED_SHAPES.items()
    for item_type in shape.item_types
}


def shape_for_item_type(item_type: str) -> str:
    return STRUCTURED_SHAPES[ITEM_TYPE_TO_SHAPE_KEY.get(item_type, "accessory")].name


def shape_definition_for_item_type(item_type: str) -> StructuredShapeDefinition:
    return STRUCTURED_SHAPES[ITEM_TYPE_TO_SHAPE_KEY.get(item_type, "accessory")]


def is_supported_item_type(item_type: str) -> bool:
    return item_type in WEARABLE_ITEM_TYPES
