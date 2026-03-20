from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

from image_search.constants.colors import NEUTRAL_COLOR_LABELS
from image_search.constants.items import UPPER_FOCUS_TYPES


def _is_skin_like(red: int, green: int, blue: int) -> bool:
    red_f, green_f, blue_f = red / 255.0, green / 255.0, blue / 255.0
    maximum = max(red_f, green_f, blue_f)
    minimum = min(red_f, green_f, blue_f)
    delta = maximum - minimum
    saturation = 0.0 if maximum == 0 else delta / maximum

    if delta == 0:
        hue = 0.0
    elif maximum == red_f:
        hue = (60 * ((green_f - blue_f) / delta) + 360) % 360
    elif maximum == green_f:
        hue = (60 * ((blue_f - red_f) / delta) + 120) % 360
    else:
        hue = (60 * ((red_f - green_f) / delta) + 240) % 360

    return 8 <= hue <= 45 and 0.12 <= saturation <= 0.68 and red > blue and red > 120


def _rgb_to_label(rgb: tuple[int, int, int]) -> str:
    red, green, blue = [channel / 255.0 for channel in rgb]
    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    delta = maximum - minimum

    value = maximum
    saturation = 0.0 if maximum == 0 else delta / maximum

    if delta == 0:
        hue = 0.0
    elif maximum == red:
        hue = (60 * ((green - blue) / delta) + 360) % 360
    elif maximum == green:
        hue = (60 * ((blue - red) / delta) + 120) % 360
    else:
        hue = (60 * ((red - green) / delta) + 240) % 360

    if value < 0.14:
        return "black"

    if saturation < 0.12:
        if value > 0.92:
            return "white"
        if value > 0.72:
            return "silver" if blue >= red and blue >= green else "gray"
        if value > 0.36:
            return "gray"
        return "brown"

    if 18 <= hue < 40 and value < 0.56:
        return "brown"
    if 38 <= hue < 52 and value > 0.62 and saturation < 0.68:
        return "blonde"
    if 42 <= hue < 56 and value > 0.58:
        return "gold"
    if hue < 12 or hue >= 345:
        return "red"
    if hue < 25:
        return "orange"
    if hue < 38:
        return "pink" if value > 0.7 else "orange"
    if hue < 68:
        return "yellow"
    if hue < 165:
        return "green"
    if hue < 255:
        return "blue"
    if hue < 345:
        return "purple" if value < 0.72 else "pink"
    return "red"


def _prepare_image(image: Image.Image, item_type: str | None = None) -> Image.Image:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    image.thumbnail((128, 128))

    if image.getbbox() is None:
        return image

    alpha_bbox = image.getchannel("A").getbbox()
    if alpha_bbox:
        image = image.crop(alpha_bbox)
    else:
        width, height = image.size
        left = int(width * 0.1)
        right = int(width * 0.9)
        top = 0
        bottom = int(height * 0.72)
        image = image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))

    width, height = image.size
    if item_type in UPPER_FOCUS_TYPES and height > 1:
        bottom = max(1, int(height * 0.62))
        image = image.crop((0, 0, width, bottom))

    image.thumbnail((96, 96))
    return image


def extract_dominant_colors(
    image_path: str | Path | None,
    item_type: str | None = None,
) -> list[str]:
    if not image_path:
        return []

    image = _prepare_image(Image.open(image_path), item_type=item_type)

    labels: Counter[str] = Counter()
    colorful_pixel_count = 0
    alpha_min, alpha_max = image.getchannel("A").getextrema()
    has_transparency = alpha_min < 255 and alpha_max > 0

    for red, green, blue, alpha in image.getdata():
        if alpha < 40 or _is_skin_like(red, green, blue):
            continue
        label = _rgb_to_label((red, green, blue))
        if (
            not has_transparency
            and label in {"white", "gray"}
            and max(red, green, blue) > 220
        ):
            continue
        labels[label] += 1
        if label not in NEUTRAL_COLOR_LABELS:
            colorful_pixel_count += 1

    if not labels:
        return []

    if colorful_pixel_count > 0:
        labels = (
            Counter(
                {
                    label: count
                    for label, count in labels.items()
                    if label not in {"white", "gray"}
                }
            )
            or labels
        )

    most_common = labels.most_common()
    primary_label, primary_count = most_common[0]

    if len(most_common) == 1:
        return [primary_label]

    secondary_label, secondary_count = most_common[1]
    total = sum(labels.values())
    primary_ratio = primary_count / total
    secondary_ratio = secondary_count / total

    if (
        primary_label != secondary_label
        and primary_label not in NEUTRAL_COLOR_LABELS
        and secondary_label not in NEUTRAL_COLOR_LABELS
        and colorful_pixel_count > 0
        and primary_ratio < 0.72
        and secondary_ratio > 0.18
    ):
        return ["multicolor", primary_label]

    if secondary_ratio > 0.22 and secondary_label != primary_label:
        return [primary_label, secondary_label]

    return [primary_label]
