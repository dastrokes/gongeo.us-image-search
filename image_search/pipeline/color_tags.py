from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from image_search.constants.colors import (
    ACCENT_DETAIL_NOUNS,
    APPAREL_BODY_NOUNS,
    COLOR_PATTERN,
    HAIR_BODY_PATTERNS,
    NEUTRAL_COLOR_LABELS,
    normalize_color_label,
)
from image_search.models.type_profiles import APPAREL_TYPES, get_type_profile
from image_search.vision.palette import (
    _is_skin_like,
    _prepare_image,
    _rgb_to_label,
    extract_dominant_colors,
)


@dataclass(slots=True)
class ImageColorStats:
    weighted_counts: Counter[str]
    raw_counts: Counter[str]
    core_weighted_counts: Counter[str]
    total_weight: float
    core_total_weight: float
    has_transparency: bool


@dataclass(slots=True)
class ColorTaggingResult:
    dominant_colors: list[str]
    accent_colors: list[str]


def _image_has_transparency(image: Image.Image) -> bool:
    alpha_min, alpha_max = image.getchannel("A").getextrema()
    return alpha_min < 255 and alpha_max > 0


def _core_box(width: int, height: int) -> tuple[int, int, int, int]:
    core_width = max(1, int(round(width * 0.6)))
    core_height = max(1, int(round(height * 0.55)))
    left = max(0, (width - core_width) // 2)
    top = max(0, (height - core_height) // 2)
    right = min(width, left + core_width)
    bottom = min(height, top + core_height)
    return left, top, right, bottom


def extract_image_color_stats(
    image_path: str | Path | None,
    item_type: str | None,
    *,
    core_weighting: bool,
) -> ImageColorStats:
    empty = ImageColorStats(Counter(), Counter(), Counter(), 0.0, 0.0, False)
    if not image_path:
        return empty

    image = _prepare_image(Image.open(image_path), item_type=item_type)
    if image.getbbox() is None:
        return empty

    has_transparency = _image_has_transparency(image)
    width, height = image.size
    core_left, core_top, core_right, core_bottom = _core_box(width, height)

    weighted_counts: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    core_weighted_counts: Counter[str] = Counter()
    total_weight = 0.0
    core_total_weight = 0.0

    for y in range(height):
        for x in range(width):
            red, green, blue, alpha = image.getpixel((x, y))
            if alpha < 40 or _is_skin_like(red, green, blue):
                continue

            label = _rgb_to_label((red, green, blue), item_type=item_type)
            if (
                not has_transparency
                and label in {"white", "gray"}
                and max(red, green, blue) > 220
            ):
                continue

            raw_counts[label] += 1
            in_core = core_left <= x < core_right and core_top <= y < core_bottom
            weight = 3.0 if core_weighting and in_core else 1.0
            weighted_counts[label] += weight
            total_weight += weight
            if in_core:
                core_weighted_counts[label] += weight
                core_total_weight += weight

    return ImageColorStats(
        weighted_counts=weighted_counts,
        raw_counts=raw_counts,
        core_weighted_counts=core_weighted_counts,
        total_weight=total_weight,
        core_total_weight=core_total_weight,
        has_transparency=has_transparency,
    )


def _scaled_counter(values: Counter[str], scale: float) -> Counter[str]:
    if scale == 1.0:
        return Counter(values)
    return Counter({label: count * scale for label, count in values.items()})


def merge_image_color_stats(
    icon_stats: ImageColorStats | None,
    overview_stats: ImageColorStats | None,
) -> ImageColorStats:
    icon_stats = icon_stats or ImageColorStats(
        Counter(), Counter(), Counter(), 0.0, 0.0, False
    )
    overview_stats = overview_stats or ImageColorStats(
        Counter(), Counter(), Counter(), 0.0, 0.0, False
    )
    return ImageColorStats(
        weighted_counts=_scaled_counter(icon_stats.weighted_counts, 2.0)
        + _scaled_counter(overview_stats.weighted_counts, 1.0),
        raw_counts=Counter(icon_stats.raw_counts) + Counter(overview_stats.raw_counts),
        core_weighted_counts=_scaled_counter(icon_stats.core_weighted_counts, 2.0)
        + _scaled_counter(overview_stats.core_weighted_counts, 1.0),
        total_weight=(icon_stats.total_weight * 2.0) + overview_stats.total_weight,
        core_total_weight=(icon_stats.core_total_weight * 2.0)
        + overview_stats.core_total_weight,
        has_transparency=icon_stats.has_transparency or overview_stats.has_transparency,
    )


def _unique_ordered(colors: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for color in colors:
        if not color or color in seen:
            continue
        seen.add(color)
        ordered.append(color)
    return ordered


def _extract_colors(pattern: re.Pattern[str], text: str) -> list[str]:
    matches: list[str] = []
    for match in pattern.finditer(text):
        matches.append(normalize_color_label(match.group("color")))
    return _unique_ordered(matches)


def parse_caption_color_mentions(
    visual: str, item_type: str
) -> tuple[list[str], list[str]]:
    lowered = visual.lower()

    if item_type == "hair":
        body_colors: list[str] = []
        for pattern, color in HAIR_BODY_PATTERNS:
            if pattern in lowered and color not in body_colors:
                body_colors.append(color)
        return body_colors[:2], []

    accent_pattern = re.compile(
        rf"\b(?P<color>{COLOR_PATTERN})(?:-colored)?\s+(?P<detail>{'|'.join(ACCENT_DETAIL_NOUNS)})\b"
    )
    accent_colors = _unique_ordered(
        [
            normalize_color_label(match.group("color"))
            for match in accent_pattern.finditer(lowered)
        ]
    )

    if item_type in APPAREL_TYPES:
        body_nouns = APPAREL_BODY_NOUNS
    else:
        profile = get_type_profile(item_type)
        noun_candidates = tuple(
            keyword
            for keyword in profile.keywords
            if keyword.isalpha() and keyword not in ACCENT_DETAIL_NOUNS
        )
        body_nouns = noun_candidates or APPAREL_BODY_NOUNS

    noun_pattern = "|".join(sorted(set(body_nouns), key=len, reverse=True))
    body_patterns = [
        re.compile(rf"\b(?P<color>{COLOR_PATTERN})\s+(?P<body>{noun_pattern})s?\b"),
        re.compile(
            rf"\b(?:a|an|the)\s+(?P<body>{noun_pattern})s?\s+(?:is|are)\s+(?P<color>{COLOR_PATTERN})\b"
        ),
        re.compile(
            rf"\b(?P<body>{noun_pattern})s?\s+(?:is|are)\s+(?P<color>{COLOR_PATTERN})\b"
        ),
    ]

    body_colors: list[str] = []
    for pattern in body_patterns:
        for color in _extract_colors(pattern, lowered):
            if color not in body_colors:
                body_colors.append(color)

    return body_colors[:2], accent_colors[:2]


def _share(counter: Counter[str], total: float, label: str) -> float:
    if total <= 0:
        return 0.0
    return float(counter.get(label, 0.0)) / total


def _select_dominant_from_stats(stats: ImageColorStats) -> list[str]:
    if stats.total_weight <= 0 or not stats.weighted_counts:
        return []

    ranked = stats.weighted_counts.most_common()
    non_neutral = [entry for entry in ranked if entry[0] not in NEUTRAL_COLOR_LABELS]
    eligible_non_neutral = [
        label
        for label, _ in non_neutral
        if _share(stats.weighted_counts, stats.total_weight, label) >= 0.18
    ]

    candidate_labels = [
        label
        for label, _ in ranked
        if not eligible_non_neutral or label not in NEUTRAL_COLOR_LABELS
    ]
    if not candidate_labels:
        candidate_labels = [label for label, _ in ranked]

    primary = candidate_labels[0]

    if len(non_neutral) >= 2:
        first_non_neutral, second_non_neutral = non_neutral[0][0], non_neutral[1][0]
        if (
            _share(stats.weighted_counts, stats.total_weight, first_non_neutral) >= 0.25
            and _share(stats.weighted_counts, stats.total_weight, second_non_neutral)
            >= 0.25
            and _share(stats.weighted_counts, stats.total_weight, first_non_neutral)
            <= 0.55
        ):
            return ["multicolor", first_non_neutral]

    secondary: str | None = None
    for label in candidate_labels[1:]:
        share = _share(stats.weighted_counts, stats.total_weight, label)
        if share < 0.22:
            continue
        if (
            label in NEUTRAL_COLOR_LABELS
            and _share(
                stats.core_weighted_counts,
                stats.core_total_weight,
                label,
            )
            < 0.18
        ):
            continue
        secondary = label
        break

    return [primary, secondary] if secondary and secondary != primary else [primary]


def _use_caption_body_fallback(
    dominant_colors: list[str],
    caption_body_colors: list[str],
) -> bool:
    if not caption_body_colors:
        return False
    if not dominant_colors:
        return True
    if len(dominant_colors) == 1 and dominant_colors[0] in NEUTRAL_COLOR_LABELS:
        return True
    return False


def _tag_hair_colors(
    overview_path: str,
    icon_path: str,
    caption_visual: str,
) -> ColorTaggingResult:
    caption_body_colors, _ = parse_caption_color_mentions(caption_visual, "hair")
    palette_colors = extract_dominant_colors(icon_path or overview_path, "hair")
    dominant = caption_body_colors[:]
    for color in palette_colors:
        if color not in dominant and color in {"silver", "white", "gray", "multicolor"}:
            dominant.append(color)
        if len(dominant) >= 2:
            break
    return ColorTaggingResult(
        dominant_colors=dominant[:2] or palette_colors[:2],
        accent_colors=[],
    )


def _tag_apparel_colors(
    overview_path: str,
    icon_path: str,
    item_type: str,
    caption_visual: str,
) -> ColorTaggingResult:
    icon_stats = extract_image_color_stats(icon_path, item_type, core_weighting=True)
    overview_stats = extract_image_color_stats(
        overview_path, item_type, core_weighting=True
    )
    merged_stats = merge_image_color_stats(icon_stats, overview_stats)
    caption_body_colors, caption_accent_colors = parse_caption_color_mentions(
        caption_visual,
        item_type,
    )

    dominant_colors = _select_dominant_from_stats(merged_stats)
    if _use_caption_body_fallback(dominant_colors, caption_body_colors):
        dominant_colors = caption_body_colors[:2]

    accent_colors = [
        color for color in caption_accent_colors if color not in dominant_colors
    ][:2]
    return ColorTaggingResult(
        dominant_colors=dominant_colors,
        accent_colors=accent_colors,
    )


def _tag_non_apparel_colors(
    overview_path: str,
    icon_path: str,
    item_type: str,
    caption_visual: str,
) -> ColorTaggingResult:
    profile = get_type_profile(item_type)
    preferred_path = icon_path if profile.prefer_icon and icon_path else overview_path
    secondary_path = overview_path if preferred_path == icon_path else icon_path

    dominant_colors = extract_dominant_colors(preferred_path, item_type)
    if not dominant_colors:
        dominant_colors = extract_dominant_colors(secondary_path, item_type)

    caption_body_colors, _ = parse_caption_color_mentions(caption_visual, item_type)
    if _use_caption_body_fallback(dominant_colors, caption_body_colors):
        dominant_colors = caption_body_colors[:2]

    return ColorTaggingResult(dominant_colors=dominant_colors, accent_colors=[])


def tag_item_colors(
    overview_path: str,
    icon_path: str,
    item_type: str,
    caption_visual: str,
) -> ColorTaggingResult:
    if item_type == "hair":
        return _tag_hair_colors(overview_path, icon_path, caption_visual)
    if item_type in APPAREL_TYPES:
        return _tag_apparel_colors(overview_path, icon_path, item_type, caption_visual)
    return _tag_non_apparel_colors(overview_path, icon_path, item_type, caption_visual)
