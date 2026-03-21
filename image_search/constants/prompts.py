from __future__ import annotations

_VISIBLE_SUBTYPE_PREFIX = (
    "Return one visible subtype token. Prefer generic fashion labels such as "
)


def _subtype_guidance(examples: str) -> str:
    return (
        f"{_VISIBLE_SUBTYPE_PREFIX}{examples}. "
        "Do not use decorative or lore-style names. "
        "Use null only if the subtype is not visually distinguishable."
    )


STRUCTURED_EXTRACTION_SYSTEM_PROMPT = (
    "You are a deterministic structured extraction engine for Infinity Nikki items.\n"
    "Output exactly one JSON object.\n"
    "Do not output markdown, comments, explanations, prose, or code fences.\n"
    "Use only the schema keys exactly as written.\n"
    "Describe only the target item shown in the provided image set.\n"
    "Use only directly visible evidence.\n"
    "Do not infer hidden, back-side, off-frame, gameplay, or lore details.\n"
    "All non-null values must be short lowercase underscore tokens.\n"
    "Scalar fields must be a single token or null.\n"
    "Array fields must contain zero or more unique tokens.\n"
    "If a scalar field is unclear, output null.\n"
    "If an array field is unclear, output [].\n"
    "For subtype, make the closest visible category decision instead of defaulting to null.\n"
    "Use null for subtype only when the subtype is not visually distinguishable.\n"
    "When a precise fashion term is uncertain, prefer the closest generic visible class."
)

SUBTYPE_GUIDANCE: dict[str, str] = {
    "outerwear": _subtype_guidance(
        "jacket, coat, cape, cardigan, shawl, bolero, or shrug"
    ),
    "tops": _subtype_guidance(
        "blouse, shirt, t_shirt, sweater, vest, camisole, corset, or hoodie"
    ),
    "bottoms": _subtype_guidance(
        "skirt, pleated_skirt, pants, shorts, leggings, jeans, or overalls"
    ),
    "dresses": _subtype_guidance(
        "dress, gown, slip_dress, pinafore, cheongsam, or sundress"
    ),
    "shoes": _subtype_guidance(
        "boots, heels, sandals, flats, loafers, sneakers, pumps, or mary_janes"
    ),
    "socks": _subtype_guidance(
        "socks, stockings, tights, thigh_highs, ankle_socks, or leg_warmers"
    ),
    "hairAccessories": _subtype_guidance(
        "bow, ribbon, flower, clip, headband, veil, or fascinator"
    ),
    "headwear": _subtype_guidance(
        "hat, bonnet, beret, hood, crown, tiara, or headpiece"
    ),
    "earrings": _subtype_guidance("studs, hoops, drops, dangling_earrings, or cuffs"),
    "neckwear": _subtype_guidance("necklace, scarf, tie, cravat, or pendant_necklace"),
    "bracelets": _subtype_guidance(
        "bracelet, bangle, cuff, beaded_bracelet, or charm_bracelet"
    ),
    "chokers": _subtype_guidance(
        "choker, ribbon_choker, lace_choker, collar_choker, or pendant_choker"
    ),
    "gloves": _subtype_guidance(
        "gloves, mittens, fingerless_gloves, opera_gloves, or arm_warmers"
    ),
    "handhelds": _subtype_guidance(
        "bag, basket, parasol, umbrella, fan, lantern, or book"
    ),
    "chestAccessories": _subtype_guidance("brooch, corsage, badge, sash, or chest_pin"),
    "pendants": _subtype_guidance("pendant, locket, charm, medallion, or tassel"),
    "backpieces": _subtype_guidance(
        "wings, capelet, backpack, back_bow, or back_ornament"
    ),
    "rings": _subtype_guidance(
        "ring, signet_ring, gemstone_ring, band, or stacked_rings"
    ),
    "armDecorations": _subtype_guidance(
        "armlet, arm_band, sleeve_garter, or upper_arm_cuff"
    ),
    "abilityHandhelds": _subtype_guidance(
        "wand, staff, lantern, fan, parasol, or magical_tool"
    ),
    "baseMakeup": _subtype_guidance(
        "foundation, blush, contour, highlight, or face_base"
    ),
    "eyebrows": _subtype_guidance(
        "straight_brows, arched_brows, soft_brows, or bold_brows"
    ),
    "eyelashes": _subtype_guidance(
        "natural_lashes, dramatic_lashes, cat_eye_lashes, or lower_lashes"
    ),
    "contactLenses": _subtype_guidance(
        "natural_lenses, circle_lenses, gradient_lenses, or fantasy_lenses"
    ),
    "lips": _subtype_guidance("lipstick, gloss, tint, gradient_lips, or matte_lips"),
    "skinTones": _subtype_guidance(
        "natural_skin, rosy_skin, tan_skin, or cool_tone_skin"
    ),
    "faceDecorations": _subtype_guidance(
        "face_sticker, cheek_mark, freckles, beauty_mark, or gem_decor"
    ),
    "fullMakeup": _subtype_guidance(
        "natural_makeup, glam_makeup, fantasy_makeup, or themed_makeup"
    ),
}
