from __future__ import annotations

import math
from colorsys import rgb_to_hsv
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from PIL import Image

from constants.colors import COLOR_PALETTE, MULTICOLOR_LABEL, ColorPaletteEntry
from models.schemas import (
    ColorDebugRecord,
    ColorManifestRecord,
    ColorSwatch,
    ItemColorsRecord,
)

_ALPHA_THRESHOLD = 32
_WHITE_LOW_SATURATION_MAX = 28.0
_NEUTRAL_SATURATION_MAX = 24
_SEARCH_HUE_MIN_CHROMA = 12
_SEARCH_HUE_MIN_SATURATION = 0.045
_WHITE_LUMA_MIN = 220
_HAIR_CENTER_FACE_BIAS = 12.0
_PRIMARY_COLOR_WEIGHT = 0.18
_SECONDARY_COLOR_WEIGHT = 0.10
_MULTICOLOR_TAG_COUNT = 3
_MAX_PRIMARY_COLORS = 3
_MAX_SECONDARY_COLORS = 3
_HAIR_SECONDARY_NOISE_LABELS = frozenset({"white", "cream", "beige", "olive"})
_HAIR_TAUPE_BLONDE_LABELS = frozenset({"gray", "brown", "olive", "cream", "beige"})
_HAIR_TAUPE_LIGHT_LABELS = frozenset({"cream", "beige"})
_HAIR_TAUPE_SHADOW_LABELS = frozenset({"gray", "brown", "olive"})


@dataclass(frozen=True, slots=True)
class _Sample:
    rgb: tuple[int, int, int]
    weight: float


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    return (
        int(hex_color[1:3], 16),
        int(hex_color[3:5], 16),
        int(hex_color[5:7], 16),
    )


def _distance(
    rgb: tuple[int, int, int],
    palette_entry: ColorPaletteEntry,
) -> float:
    r, g, b = rgb
    pr, pg, pb = palette_entry.rgb
    return math.sqrt(
        ((r - pr) * 0.30) ** 2 + ((g - pg) * 0.59) ** 2 + ((b - pb) * 0.45) ** 2
    )


def _is_low_information_highlight(rgb: tuple[int, int, int]) -> bool:
    return min(rgb) >= 238 and max(rgb) - min(rgb) <= _WHITE_LOW_SATURATION_MAX


def _neutral_color_label(rgb: tuple[int, int, int]) -> str | None:
    if max(rgb) - min(rgb) > _NEUTRAL_SATURATION_MAX:
        return None
    r, g, b = rgb
    luma = (r * 0.30) + (g * 0.59) + (b * 0.11)
    if luma < 85:
        return "black"
    if luma < 175:
        return "gray"
    if luma < _WHITE_LUMA_MIN:
        return "silver"
    return "white"


def _search_hue_label(rgb: tuple[int, int, int]) -> str | None:
    high = max(rgb)
    low = min(rgb)
    chroma = high - low
    if chroma < _SEARCH_HUE_MIN_CHROMA:
        return None

    hue, saturation, _value = rgb_to_hsv(
        rgb[0] / 255.0,
        rgb[1] / 255.0,
        rgb[2] / 255.0,
    )
    if saturation < _SEARCH_HUE_MIN_SATURATION:
        return None

    hue_degrees = hue * 360.0
    luma = (rgb[0] * 0.30) + (rgb[1] * 0.59) + (rgb[2] * 0.11)

    if 330 <= hue_degrees or hue_degrees < 10:
        if luma >= 150 and saturation <= 0.40:
            return "pink"
        return None
    if 300 <= hue_degrees < 330 and luma >= 140:
        return "pink"
    if 250 <= hue_degrees < 300:
        return "purple"
    if 195 <= hue_degrees < 250:
        if luma < 90 and saturation >= 0.18:
            return "navy"
        return "blue"
    if 165 <= hue_degrees < 195:
        return "teal"
    if 45 <= hue_degrees < 95 and luma < 175 and saturation >= 0.18:
        return "olive"
    if 80 <= hue_degrees < 165 and chroma >= 16:
        return "green"
    if 15 <= hue_degrees < 80 and luma >= 200 and saturation <= 0.22:
        return "cream"
    if 45 <= hue_degrees < 80 and luma >= 185 and saturation <= 0.34:
        return "yellow"
    if 15 <= hue_degrees < 45 and luma >= 185 and saturation <= 0.34:
        if luma >= 210 and saturation <= 0.10:
            return "cream"
        return "beige"
    return None


def _classify_color(rgb: tuple[int, int, int]) -> str:
    search_hue_label = _search_hue_label(rgb)
    if search_hue_label is not None:
        return search_hue_label
    neutral_label = _neutral_color_label(rgb)
    if neutral_label is not None:
        return neutral_label
    if _is_low_information_highlight(rgb):
        return "white"
    return min(COLOR_PALETTE, key=lambda entry: _distance(rgb, entry)).label


def _hair_pixel_multiplier(
    *,
    rgb: tuple[int, int, int],
    x: int,
    y: int,
    width: int,
    height: int,
) -> float:
    r, g, b = rgb
    horizontal_center = abs((x / max(1, width - 1)) - 0.5)
    vertical = y / max(1, height - 1)
    in_avatar_core = horizontal_center < 0.32 and 0.14 < vertical < 0.82
    likely_skin = (
        r > g * 1.03
        and g > b * 1.05
        and r > 135
    )
    likely_face_highlight = min(rgb) >= 220 and max(rgb) - min(rgb) <= 48
    if in_avatar_core and (likely_skin or likely_face_highlight):
        return 1.0 / _HAIR_CENTER_FACE_BIAS
    return 1.0


def _iter_samples(
    image: Image.Image,
    *,
    item_type: str,
) -> tuple[list[_Sample], int, int]:
    rgba = image.convert("RGBA")
    width, height = rgba.size
    visible_pixel_count = 0
    ignored_pixel_count = 0
    samples: list[_Sample] = []

    for y in range(height):
        for x in range(width):
            r, g, b, alpha = rgba.getpixel((x, y))
            if alpha < _ALPHA_THRESHOLD:
                ignored_pixel_count += 1
                continue
            visible_pixel_count += 1
            if _is_low_information_highlight((r, g, b)) and alpha < 192:
                ignored_pixel_count += 1
                continue
            multiplier = (
                _hair_pixel_multiplier(
                    rgb=(r, g, b),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                )
                if item_type == "hair"
                else 1.0
            )
            samples.append(_Sample((r, g, b), (alpha / 255.0) * multiplier))
    return samples, visible_pixel_count, ignored_pixel_count


def _weighted_average_hex(samples: Iterable[_Sample]) -> str:
    total_weight = 0.0
    red = 0.0
    green = 0.0
    blue = 0.0
    for sample in samples:
        total_weight += sample.weight
        red += sample.rgb[0] * sample.weight
        green += sample.rgb[1] * sample.weight
        blue += sample.rgb[2] * sample.weight
    if total_weight <= 0:
        return "#000000"
    return _rgb_to_hex(
        (
            round(red / total_weight),
            round(green / total_weight),
            round(blue / total_weight),
        )
    )


def _build_swatches(samples_by_label: dict[str, list[_Sample]]) -> list[ColorSwatch]:
    total_weight = sum(
        sample.weight for samples in samples_by_label.values() for sample in samples
    )
    if total_weight <= 0:
        return []
    swatches = [
        ColorSwatch(
            label=label,
            hex=_weighted_average_hex(samples),
            weight=round(sum(sample.weight for sample in samples) / total_weight, 4),
        )
        for label, samples in samples_by_label.items()
    ]
    return sorted(swatches, key=lambda swatch: (-swatch.weight, swatch.label))[:5]


def _weighted_average_swatch_hex(swatches: Iterable[ColorSwatch]) -> str:
    total_weight = 0.0
    red = 0.0
    green = 0.0
    blue = 0.0
    for swatch in swatches:
        rgb = _hex_to_rgb(swatch.hex)
        total_weight += swatch.weight
        red += rgb[0] * swatch.weight
        green += rgb[1] * swatch.weight
        blue += rgb[2] * swatch.weight
    if total_weight <= 0:
        return "#000000"
    return _rgb_to_hex(
        (
            round(red / total_weight),
            round(green / total_weight),
            round(blue / total_weight),
        )
    )


def _normalize_hair_swatches(swatches: list[ColorSwatch]) -> list[ColorSwatch]:
    if not swatches:
        return []

    taupe_swatches = [
        swatch for swatch in swatches if swatch.label in _HAIR_TAUPE_BLONDE_LABELS
    ]
    taupe_weight = sum(swatch.weight for swatch in taupe_swatches)
    light_weight = sum(
        swatch.weight
        for swatch in taupe_swatches
        if swatch.label in _HAIR_TAUPE_LIGHT_LABELS
    )
    shadow_weight = sum(
        swatch.weight
        for swatch in taupe_swatches
        if swatch.label in _HAIR_TAUPE_SHADOW_LABELS
    )
    has_strong_chromatic_primary = any(
        swatch.weight >= _PRIMARY_COLOR_WEIGHT
        for swatch in swatches
        if swatch.label not in _HAIR_TAUPE_BLONDE_LABELS
    )

    if (
        taupe_weight < 0.65
        or light_weight < 0.18
        or shadow_weight < 0.35
        or swatches[0].label == "brown"
        or has_strong_chromatic_primary
    ):
        return swatches

    normalized = [
        ColorSwatch(
            label="beige",
            hex=_weighted_average_swatch_hex(taupe_swatches),
            weight=round(taupe_weight, 4),
        ),
        *[
            swatch
            for swatch in swatches
            if swatch.label not in _HAIR_TAUPE_BLONDE_LABELS
        ],
    ]
    return sorted(normalized, key=lambda swatch: (-swatch.weight, swatch.label))[:5]


def _assign_color_roles(
    swatches: list[ColorSwatch],
    *,
    item_type: str,
) -> tuple[list[ColorSwatch], list[ColorSwatch]]:
    if not swatches:
        return [], []

    top_label = swatches[0].label
    primary_swatches: list[ColorSwatch] = []
    for swatch in swatches:
        if swatch.weight < _PRIMARY_COLOR_WEIGHT:
            continue
        if (
            item_type == "hair"
            and swatch.label in _HAIR_SECONDARY_NOISE_LABELS
            and swatch.label != top_label
        ):
            continue
        primary_swatches.append(swatch)
        if len(primary_swatches) >= _MAX_PRIMARY_COLORS:
            break
    if not primary_swatches:
        primary_swatches = swatches[:1]

    primary_labels = {swatch.label for swatch in primary_swatches}
    secondary_swatches: list[ColorSwatch] = []
    for swatch in swatches:
        if swatch.label in primary_labels:
            continue
        if swatch.weight < _SECONDARY_COLOR_WEIGHT:
            continue
        if item_type == "hair" and swatch.label in _HAIR_SECONDARY_NOISE_LABELS:
            continue
        secondary_swatches.append(swatch)
        if len(secondary_swatches) >= _MAX_SECONDARY_COLORS:
            break

    return primary_swatches, secondary_swatches


def extract_color_record(
    record: ColorManifestRecord,
) -> tuple[ItemColorsRecord, ColorDebugRecord]:
    with Image.open(record.icon_path) as image:
        samples, visible_pixel_count, ignored_pixel_count = _iter_samples(
            image,
            item_type=record.item_type,
        )

    samples_by_label: dict[str, list[_Sample]] = defaultdict(list)
    for sample in samples:
        samples_by_label[_classify_color(sample.rgb)].append(sample)

    swatches = _build_swatches(samples_by_label)
    if record.item_type == "hair":
        swatches = _normalize_hair_swatches(swatches)
    color_weights = {swatch.label: swatch.weight for swatch in swatches}
    primary_swatches, secondary_swatches = _assign_color_roles(
        swatches,
        item_type=record.item_type,
    )
    primary_colors = [swatch.label for swatch in primary_swatches]
    secondary_colors = [swatch.label for swatch in secondary_swatches]
    canonical_swatches = [*primary_swatches, *secondary_swatches]
    color_tags = [*primary_colors, *secondary_colors]
    if len(color_tags) >= _MULTICOLOR_TAG_COUNT:
        color_tags.append(MULTICOLOR_LABEL)

    review_reasons: list[str] = []
    if record.item_type == "hair":
        review_reasons.append("hair_icon_composite")
    if not color_tags:
        review_reasons.append("no_color_tags")

    colors_record = ItemColorsRecord(
        item_id=record.item_id,
        item_type=record.item_type,
        primary_colors=primary_colors,
        secondary_colors=secondary_colors,
        color_tags=color_tags,
        swatches=canonical_swatches,
        needs_review=bool(review_reasons),
        review_reasons=review_reasons,
    )
    debug_record = ColorDebugRecord(
        item_id=record.item_id,
        item_type=record.item_type,
        image_path=record.icon_path,
        visible_pixel_count=visible_pixel_count,
        sampled_pixel_count=len(samples),
        ignored_pixel_count=ignored_pixel_count,
        color_weights=color_weights,
        swatches=swatches,
        needs_review=bool(review_reasons),
        review_reasons=review_reasons,
    )
    return colors_record, debug_record


def color_manifest_record_from_payload(
    payload: dict[str, object],
) -> ColorManifestRecord:
    return ColorManifestRecord(
        item_id=int(payload["item_id"]),
        item_type=str(payload["item_type"]),
        icon_path=str(payload["icon_path"]),
    )
