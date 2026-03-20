from __future__ import annotations

import re

QWEN_STRUCTURED_DETAIL_PROMPT = """
Describe exactly one main wearable item from the provided images.
Focus on {focus}.
Return only a comma-separated list of short lowercase phrases describing the item itself.
Rules:
- Use lowercase normalized phrases only.
- Start with the visible main item subcategory if it is clear.
- Include only directly visible attributes of that item.
- Be exhaustive about clear visual details, but do not infer hidden or ambiguous details.
- Prefer concrete visual facts such as length, cut, silhouette, shape, coverage, placement, material, texture, pattern, motif, trim, closure, ornament, and construction details.
- Exclude the wearer, face, body, hair, hands, pose, expression, background, lighting, framing, and other items.
- Do not write full sentences or repeat near-duplicate phrases.
- Do not return JSON.
""".strip()

QWEN_JOINT_DETAIL_PROMPT = """
You are given up to two images of the same wearable item.
Image order:
- image 1: overview image
- image 2: icon image

Describe exactly one main wearable item shared across the images.
Return exactly these three lines and nothing else:
overview: <comma-separated short lowercase phrases>
icon: <comma-separated short lowercase phrases>
visual: <comma-separated short lowercase phrases>

Rules:
- Use lowercase normalized phrases only.
- Start with the visible main item subcategory if it is clear.
- Keep the `overview` and `icon` distinction clear.
- `overview` should focus on what is most visible in the overview image.
- `icon` should focus on what is most visible in the icon image.
- `visual` should describe the item across both images as one combined output.
- Each value must be a plain comma-separated tag list, not a sentence fragment.
- Include only directly visible attributes of that item.
- Be exhaustive about clear visual details, but do not infer hidden or ambiguous details.
- Prefer concrete visual facts such as length, cut, silhouette, shape, coverage, placement, material, texture, pattern, motif, trim, closure, ornament, and construction details.
- Exclude the wearer, face, body, hair, hands, pose, expression, background, lighting, framing, and other items.
- Do not write full sentences or repeat near-duplicate phrases.
- Do not return JSON.
""".strip()

PLAIN_DETAIL_PROMPT_LINES: tuple[str, ...] = (
    "Describe exactly one main wearable item from the provided images.",
    "Focus on {focus}.",
    "Return only a comma-separated list of short lowercase phrases describing the item itself.",
    "Use lowercase normalized phrases only.",
    "Start with the visible main item subcategory if it is clear.",
    "Include only directly visible attributes of that item.",
    "Be exhaustive about clear visual details, but do not infer hidden or ambiguous details.",
    "Prefer concrete visual facts such as length, cut, silhouette, shape, coverage, placement, material, texture, pattern, motif, trim, closure, ornament, and construction details.",
    "Exclude the wearer, face, body, hair, hands, pose, expression, background, lighting, framing, and other items.",
    "Do not write full sentences or repeat near-duplicate phrases.",
    "Do not return JSON.",
)

JOINT_PLAIN_DETAIL_PROMPT_LINES: tuple[str, ...] = (
    "You are given up to two images of the same wearable item.",
    "Image 1 is the overview image.",
    "Image 2 is the icon image.",
    "Describe exactly one main wearable item shared across the images.",
    "Return exactly these three lines and nothing else:",
    "overview: <comma-separated short lowercase phrases>",
    "icon: <comma-separated short lowercase phrases>",
    "visual: <comma-separated short lowercase phrases>",
    "Use lowercase normalized phrases only.",
    "Start with the visible main item subcategory if it is clear.",
    "Keep the `overview` and `icon` distinction clear.",
    "`overview` should focus on what is most visible in the overview image.",
    "`icon` should focus on what is most visible in the icon image.",
    "`visual` should describe the item across both images as one combined output.",
    "Each value must be a plain comma-separated tag list, not a sentence fragment.",
    "Include only directly visible attributes of that item.",
    "Be exhaustive about clear visual details, but do not infer hidden or ambiguous details.",
    "Prefer concrete visual facts such as length, cut, silhouette, shape, coverage, placement, material, texture, pattern, motif, trim, closure, ornament, and construction details.",
    "Exclude the wearer, face, body, hair, hands, pose, expression, background, lighting, framing, and other items.",
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

HAIR_TYPE_SPECIFIC_PROMPT_RULES = (
    "\nDescribe only the hairstyle or hair-attached decorations."
    "\nIgnore clothing, dresses, tops, jewelry, skin, face, and anything below the neck."
    "\nMention bows, ribbons, clips, or headbands only when they are attached to the hair."
)

ACCESSORY_TYPE_SPECIFIC_PROMPT_RULES = (
    "\nDescribe only the target accessory or face detail."
    "\nIgnore surrounding clothing, adjacent jewelry, hairstyle, body parts, and nearby items unless they are part of the target item."
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
    + r")\b"
)

LOW_SIGNAL_VISUAL_PATTERN = re.compile(r"\b(?:overall look)\b")

STOP_VISUAL_PATTERN = re.compile(r"\b(?:camera|background|looking directly)\b")

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
