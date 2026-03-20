from __future__ import annotations

import re

STRUCTURED_DETAIL_PROMPT = """
Describe exactly one main wearable item from the provided image(s).
Focus on {focus}.
Return only a comma-separated list of short lowercase phrases describing the item itself. No preamble, no explanation.
Rules:
- If a specific subcategory is visible, use that concrete subcategory instead of a generic bucket like `tops`, `dress`, `bottoms`, `socks`, `shoes`, or `accessory`.
- Exclude the wearer, face, body, hair, hands, pose, expression, background, lighting, framing, and other items.
- Include only directly visible attributes of that item.
- Be exhaustive about clear visual details, but do not infer hidden or ambiguous details.
- Prefer concrete visual facts such as length, cut, silhouette, shape, coverage, placement, material, texture, pattern, motif, trim, closure, ornament, and construction details.
- Do not describe colors, color gradients, or multicolor effects.
- Include smaller visible details when they are clear.
- Use lowercase normalized phrases only.
- Do not write full sentences or repeat near-duplicate phrases.
- Do not return JSON.
""".strip()

JOINT_DETAIL_PROMPT = """
You are given up to two images of the same wearable item.
Image order:
- image 1: overview image
- image 2: icon image

Describe exactly one main wearable item shared across the images.
Return exactly these two lines and nothing else:
overview: <comma-separated short lowercase phrases>
icon: <comma-separated short lowercase phrases>

Rules:
- Start with the visible main item subcategory if it is clear.
- If a specific subcategory is visible, use that concrete subcategory instead of a generic bucket like `tops`, `dress`, `bottoms`, `socks`, `shoes`, or `accessory`.
- Exclude the wearer, face, body, hair, hands, pose, expression, background, lighting, framing, and other items.
- Keep the `overview` and `icon` distinction clear.
- `overview` should focus on what is most visible in the overview image.
- `icon` should focus on what is most visible in the icon image.
- Each value must be a plain comma-separated tag list, not a sentence fragment.
- Include only directly visible attributes of that item.
- Be exhaustive about clear visual details, but do not infer hidden or ambiguous details.
- Prefer concrete visual facts such as length, cut, silhouette, shape, coverage, placement, material, texture, pattern, motif, trim, closure, ornament, and construction details.
- Do not describe colors, color gradients, or multicolor effects.
- Include smaller visible details when they are clear.
- Use lowercase normalized phrases only.
- Do not write full sentences or repeat near-duplicate phrases.
- Do not return JSON.
""".strip()

PLAIN_DETAIL_PROMPT_LINES: tuple[str, ...] = (
    "Describe exactly one main wearable item from the provided image(s).",
    "Focus on {focus}.",
    "Return only a comma-separated list of short lowercase phrases describing the item itself. No preamble, no explanation.",
    "Start with the visible main item subcategory if it is clear.",
    "If a specific subcategory is visible, use that concrete subcategory instead of a generic bucket like `tops`, `dress`, `bottoms`, `socks`, `shoes`, or `accessory`.",
    "Exclude the wearer, face, body, hair, hands, pose, expression, background, lighting, framing, and other items.",
    "Include only directly visible attributes of that item.",
    "Be exhaustive about clear visual details, but do not infer hidden or ambiguous details.",
    "Prefer concrete visual facts such as length, cut, silhouette, shape, coverage, placement, material, texture, pattern, motif, trim, closure, ornament, and construction details.",
    "Do not describe colors, color gradients, or multicolor effects.",
    "Include smaller visible details when they are clear.",
    "Use lowercase normalized phrases only.",
    "Do not write full sentences or repeat near-duplicate phrases.",
    "Do not return JSON.",
)

JOINT_PLAIN_DETAIL_PROMPT_LINES: tuple[str, ...] = (
    "You are given up to two images of the same wearable item.",
    "Image 1 is the overview image.",
    "Image 2 is the icon image.",
    "Describe exactly one main wearable item shared across the images.",
    "Return exactly these two lines and nothing else:",
    "overview: <comma-separated short lowercase phrases>",
    "icon: <comma-separated short lowercase phrases>",
    "Start with the visible main item subcategory if it is clear.",
    "If a specific subcategory is visible, use that concrete subcategory instead of a generic bucket like `tops`, `dress`, `bottoms`, `socks`, `shoes`, or `accessory`.",
    "Exclude the wearer, face, body, hair, hands, pose, expression, background, lighting, framing, and other items.",
    "Keep the `overview` and `icon` distinction clear.",
    "`overview` should focus on what is most visible in the overview image.",
    "`icon` should focus on what is most visible in the icon image.",
    "Each value must be a plain comma-separated tag list, not a sentence fragment.",
    "Include only directly visible attributes of that item.",
    "Be exhaustive about clear visual details, but do not infer hidden or ambiguous details.",
    "Prefer concrete visual facts such as length, cut, silhouette, shape, coverage, placement, material, texture, pattern, motif, trim, closure, ornament, and construction details.",
    "Do not describe colors, color gradients, or multicolor effects.",
    "Include smaller visible details when they are clear.",
    "Use lowercase normalized phrases only.",
    "Do not write full sentences or repeat near-duplicate phrases.",
    "Do not return JSON.",
)

PROMPT_FOCUS_BY_MODALITY: dict[str, str] = {
    "overview": "overall silhouette, length, layering, placement, and major materials",
    "icon": "small motifs, trims, closures, embroidery, ornaments, and accent details",
}

DEFAULT_PROMPT_FOCUS = (
    "the item's visible structure, materials, and distinguishing details"
)

HAIR_ATTRIBUTE_PROMPT_RULE = "Cover visible hair attributes: length, arrangement, bangs, texture, parting, and attached ornaments."

DRESS_ATTRIBUTE_PROMPT_RULE = "Cover visible dress attributes: subcategory, length, silhouette, neckline, sleeve length, sleeve shape, straps, waist, hem, layering, materials, pattern, motif, trim, ornament, and closures. Do not include color descriptions. Include smaller visible motifs and graphics when clear."

APPAREL_ATTRIBUTE_PROMPT_RULE = "Cover visible apparel attributes: subcategory, neckline, collar, sleeve length, sleeve shape, garment length, hem, layering, front opening or closure, materials, pattern, motif, trim, and ornament. Do not include color descriptions. Include smaller visible motifs and graphics when clear."

BOTTOM_ATTRIBUTE_PROMPT_RULE = "Cover visible bottom attributes: subcategory, rise, length, silhouette, pleats or layering, hem, materials, pattern, motif, trim, and ornament. Do not include color descriptions. Include smaller visible motifs and graphics when clear."

SOCKS_ATTRIBUTE_PROMPT_RULE = "Cover visible legwear attributes: subcategory, height, opacity, trim, pattern, motif, and ornament. Do not include color descriptions."

SHOES_ATTRIBUTE_PROMPT_RULE = "Cover visible footwear attributes: subcategory, heel height, shaft height, toe shape, platform, straps, buckles or closures, materials, pattern, trim, and ornament. Do not include color descriptions."

ACCESSORY_ATTRIBUTE_PROMPT_RULE = "Cover visible accessory attributes: subcategory, shape, size, placement, attachment style, materials, pattern, motif, trim, ornament, gems, bows, ribbons, and closures. Do not include color descriptions."

FACE_DETAIL_ATTRIBUTE_PROMPT_RULE = "Cover visible face-detail attributes: placement, shape, finish, intensity, pattern, motif, and decorative accents. Do not include color descriptions."

BODY_PAINT_ATTRIBUTE_PROMPT_RULE = "Cover visible body-paint attributes: placement, coverage, shape, pattern, motif, and finish. Do not include color descriptions."

SKIN_TONE_ATTRIBUTE_PROMPT_RULE = (
    "Describe only visible skin tone or complexion cues of the target cosmetic item."
)

GENERIC_ATTRIBUTE_PROMPT_RULE = "Cover visible attributes such as subcategory, shape, placement, length, materials, pattern, motif, trim, ornament, and closures when applicable. Do not include color descriptions."

GENERIC_SUBCATEGORY_AVOIDANCE_RULE = (
    "When the item's specific subcategory is visible, name that specific kind first. "
    "Avoid generic bucket words like the raw item type unless no more specific subcategory can be determined from the image."
)

HAIR_TYPE_SPECIFIC_PROMPT_RULES = (
    "Describe only the hairstyle or hair-attached decorations.\n"
    "Do not describe clothing, dresses, tops, jewelry, skin, face, or anything below the neck.\n"
    "Mention bows, ribbons, clips, or headbands only when they are attached to the hair."
)

ACCESSORY_TYPE_SPECIFIC_PROMPT_RULES = (
    "Describe only the target accessory or face detail.\n"
    "Do not describe surrounding clothing, adjacent jewelry, hairstyle, body parts, or nearby items unless they are part of the target item."
)

CAPTION_NOISE_TERMS = frozenset(
    {
        "person",
        "background",
        "illustration",
    }
)

CAPTION_NOISE_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(term) for term in sorted(CAPTION_NOISE_TERMS, key=len, reverse=True)
    )
    + r")\b",
    re.IGNORECASE,
)

LOW_SIGNAL_VISUAL_PATTERN = re.compile(r"\b(?:overall look)\b", re.IGNORECASE)

STOP_VISUAL_PATTERN = re.compile(
    r"\b(?:camera|photo background|looking directly at)\b",
    re.IGNORECASE,
)

TOP_GARMENT_PATTERNS: tuple[str, ...] = (
    r"\bblouse\b",
    r"\bshirt\b",
    r"\btop\b",
)

BOTTOM_GARMENT_PATTERNS: tuple[str, ...] = (
    r"\bskirt\b",
    r"\bpants\b",
    r"\btrousers\b",
)

OUTERWEAR_GARMENT_PATTERNS: tuple[str, ...] = (
    r"\bjacket\b",
    r"\bcoat\b",
    r"\bcloak\b",
)

DRESS_GARMENT_PATTERNS: tuple[str, ...] = (
    r"\bdress\b",
    r"\bgown\b",
)

APPAREL_LEAK_PATTERNS: tuple[str, ...] = (
    *TOP_GARMENT_PATTERNS,
    *BOTTOM_GARMENT_PATTERNS,
    *OUTERWEAR_GARMENT_PATTERNS,
    *DRESS_GARMENT_PATTERNS,
)

JEWELRY_LEAK_PATTERNS: tuple[str, ...] = (
    r"\bnecklace\b",
    r"\bearrings?\b",
    r"\bring\b",
)

BODY_LEAK_PATTERNS: tuple[str, ...] = (
    r"\bface\b",
    r"\bhand\b",
    r"\bbody\b",
)

HAIR_LEAK_PATTERNS: tuple[str, ...] = (
    r"\bhair\b",
    r"\bbangs?\b",
    r"\bbraid(?:ed)?\b",
)
