from __future__ import annotations

import re

from image_search.constants.text import LOW_SIGNAL_VISUAL_PATTERN
from image_search.models.schemas import CaptionRecord, DocumentRecord, MetadataRecord
from image_search.models.type_profiles import is_visual_term_relevant


def _word_count(term: str) -> int:
    return len(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", term.lower()))


def _term_rank(term: str) -> int:
    words = _word_count(term)
    if words >= 5:
        return 4
    if words >= 3:
        return 3
    if words >= 2:
        return 2
    return 1


def _visual_terms(caption: CaptionRecord, item_type: str) -> list[str]:
    raw_terms: list[str] = []
    seen: set[str] = set()
    for chunk in caption.visual.split(","):
        term = chunk.strip()
        if (
            not term
            or term == item_type
            or term in seen
            or LOW_SIGNAL_VISUAL_PATTERN.search(term)
            or not is_visual_term_relevant(term, item_type)
        ):
            continue
        seen.add(term)
        raw_terms.append(term)

    ranked = sorted(
        enumerate(raw_terms),
        key=lambda pair: (-_term_rank(pair[1]), pair[0]),
    )
    return [term for _, term in ranked]


def build_document_text(metadata: MetadataRecord, caption: CaptionRecord) -> str:
    colors = ", ".join(metadata.dominant_colors) if metadata.dominant_colors else ""
    visual_terms = _visual_terms(caption, metadata.type)
    lines: list[str] = []
    if visual_terms:
        lines.append(f"appearance: {', '.join(visual_terms)}")
    if colors:
        lines.append(f"colors: {colors}")
    lines.append(f"item kind: {metadata.type}")
    return "\n".join(lines).strip()


def build_document_record(
    metadata: MetadataRecord,
    caption: CaptionRecord,
) -> DocumentRecord:
    return DocumentRecord(
        id=metadata.item_id,
        data=build_document_text(metadata, caption),
        metadata={
            "item_id": metadata.item_id,
            "name": metadata.name,
            "type": metadata.type,
            "quality": metadata.quality,
            "obtain_type": metadata.obtain_type,
            "dominant_colors": metadata.dominant_colors,
            "accent_colors": metadata.accent_colors,
        },
    )
