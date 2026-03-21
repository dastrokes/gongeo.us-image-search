from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StructuredFieldDefinition:
    name: str
    kind: str
    description: str
    alias_pairs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class StructuredShapeDefinition:
    name: str
    item_types: tuple[str, ...]
    fields: tuple[StructuredFieldDefinition, ...]


def _field(
    name: str,
    kind: str,
    description: str,
    *,
    aliases: tuple[tuple[str, str], ...] = (),
) -> StructuredFieldDefinition:
    return StructuredFieldDefinition(
        name=name,
        kind=kind,
        description=description,
        alias_pairs=aliases,
    )


_COLOR_ALIASES: tuple[tuple[str, str], ...] = (
    ("grey", "gray"),
    ("multi_color", "multicolor"),
    ("multi_colored", "multicolor"),
    ("multi_colour", "multicolor"),
    ("multi_coloured", "multicolor"),
)

FACE_ITEM_TYPES: tuple[str, ...] = (
    "baseMakeup",
    "eyebrows",
    "eyelashes",
    "contactLenses",
    "lips",
    "skinTones",
    "faceDecorations",
    "fullMakeup",
)

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


STRUCTURED_SHAPES: dict[str, StructuredShapeDefinition] = {
    "garment": StructuredShapeDefinition(
        name="garment",
        item_types=("outerwear", "tops", "dresses", "bottoms"),
        fields=(
            _field("subtype", "scalar", "specific garment subtype when visible"),
            _field("primary_color", "scalar", "main visible color", aliases=_COLOR_ALIASES),
            _field("secondary_color", "scalar", "secondary visible color", aliases=_COLOR_ALIASES),
            _field("pattern", "array", "visible surface patterns"),
            _field("material", "array", "visible materials or surface construction"),
            _field("length", "scalar", "overall length or coverage"),
            _field("silhouette", "scalar", "overall shape or cut"),
            _field("neckline", "scalar", "neck opening shape"),
            _field("collar", "array", "visible collar details"),
            _field("closure", "array", "visible closures or front openings"),
            _field("sleeve_length", "scalar", "sleeve length when present"),
            _field("sleeve_shape", "array", "distinct sleeve shapes or cuff treatments"),
        ),
    ),
    "hair": StructuredShapeDefinition(
        name="hair",
        item_types=("hair",),
        fields=(
            _field("primary_color", "scalar", "main hair color", aliases=_COLOR_ALIASES),
            _field("secondary_color", "scalar", "secondary hair color", aliases=_COLOR_ALIASES),
            _field("length", "scalar", "overall hair length"),
            _field("texture", "scalar", "hair texture"),
            _field("parting", "scalar", "hair parting"),
            _field("bangs", "scalar", "bangs or fringe treatment"),
            _field("hairstyle", "array", "visible arrangement or styling"),
            _field("adornment", "array", "hair-attached decorations"),
        ),
    ),
    "shoes": StructuredShapeDefinition(
        name="shoes",
        item_types=("shoes",),
        fields=(
            _field("subtype", "scalar", "specific shoe subtype when visible"),
            _field("primary_color", "scalar", "main visible color", aliases=_COLOR_ALIASES),
            _field("secondary_color", "scalar", "secondary visible color", aliases=_COLOR_ALIASES),
            _field("pattern", "array", "visible surface patterns"),
            _field("material", "array", "visible materials"),
            _field("shaft_height", "scalar", "boot or upper height"),
            _field("heel_height", "scalar", "heel height"),
            _field("toe_shape", "scalar", "toe shape"),
            _field("platform", "scalar", "platform presence or size"),
            _field("closure", "array", "straps, buckles, laces, or other closures"),
        ),
    ),
    "socks": StructuredShapeDefinition(
        name="socks",
        item_types=("socks",),
        fields=(
            _field("subtype", "scalar", "specific legwear subtype when visible"),
            _field("primary_color", "scalar", "main visible color", aliases=_COLOR_ALIASES),
            _field("secondary_color", "scalar", "secondary visible color", aliases=_COLOR_ALIASES),
            _field("pattern", "array", "visible surface patterns"),
            _field("material", "array", "visible materials"),
            _field("height", "scalar", "overall height"),
            _field("opacity", "scalar", "sheer, opaque, or similar visibility"),
            _field("trim", "array", "visible trim or edge details"),
        ),
    ),
    "accessory": StructuredShapeDefinition(
        name="accessory",
        item_types=ACCESSORY_ITEM_TYPES,
        fields=(
            _field("subtype", "scalar", "specific accessory subtype when visible"),
            _field("primary_color", "scalar", "main visible color", aliases=_COLOR_ALIASES),
            _field("secondary_color", "scalar", "secondary visible color", aliases=_COLOR_ALIASES),
            _field("pattern", "array", "visible surface patterns"),
            _field("material", "array", "visible materials"),
            _field("placement", "scalar", "where the accessory sits on the body"),
            _field("attachment", "array", "how it attaches or hangs"),
            _field("shape", "array", "distinct overall shapes or decorative forms"),
        ),
    ),
    "face": StructuredShapeDefinition(
        name="face",
        item_types=FACE_ITEM_TYPES,
        fields=(
            _field("subtype", "scalar", "specific cosmetic subtype when visible"),
            _field("primary_color", "scalar", "main visible color", aliases=_COLOR_ALIASES),
            _field("secondary_color", "scalar", "secondary visible color", aliases=_COLOR_ALIASES),
            _field("pattern", "array", "visible marks or motifs"),
            _field("placement", "scalar", "where the effect appears"),
            _field("finish", "scalar", "finish such as glossy or matte"),
            _field("effect", "scalar", "main cosmetic effect"),
            _field("shape", "array", "distinct visible shapes"),
        ),
    ),
    "body_paint": StructuredShapeDefinition(
        name="body_paint",
        item_types=("bodyPaint",),
        fields=(
            _field("primary_color", "scalar", "main visible color", aliases=_COLOR_ALIASES),
            _field("secondary_color", "scalar", "secondary visible color", aliases=_COLOR_ALIASES),
            _field("pattern", "array", "visible marks or motifs"),
            _field("placement", "scalar", "where the effect appears"),
            _field("coverage", "scalar", "overall spread or extent"),
            _field("shape", "array", "distinct visible shapes"),
        ),
    ),
}

ITEM_TYPE_TO_SHAPE: dict[str, str] = {
    item_type: shape.name
    for shape in STRUCTURED_SHAPES.values()
    for item_type in shape.item_types
}


def shape_for_item_type(item_type: str) -> str:
    return ITEM_TYPE_TO_SHAPE.get(item_type, "accessory")


def shape_definition_for_item_type(item_type: str) -> StructuredShapeDefinition:
    return STRUCTURED_SHAPES[shape_for_item_type(item_type)]
