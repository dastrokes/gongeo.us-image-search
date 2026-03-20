from __future__ import annotations

from image_search.constants.taxonomy import CONCEPT_BY_KEY
from image_search.models.schemas import (
    MetadataRecord,
    SearchDocumentRecord,
    TagAssignmentRecord,
    VisualFeatureRecord,
)


def _accepted_assignments(
    assignments: list[TagAssignmentRecord],
) -> list[TagAssignmentRecord]:
    return [assignment for assignment in assignments if assignment.status == "accepted"]


def _display_label(assignment: TagAssignmentRecord) -> str:
    definition = CONCEPT_BY_KEY.get(assignment.concept_key)
    if definition is None:
        return assignment.value_key.replace("_", " ")
    return definition.display


def build_search_text(
    metadata: MetadataRecord,
    visual_features: VisualFeatureRecord,
    assignments: list[TagAssignmentRecord],
) -> str:
    accepted = _accepted_assignments(assignments)
    filterable_terms = [
        _display_label(assignment)
        for assignment in accepted
        if not assignment.search_only
    ]
    search_only_terms = list(metadata.search_terms)

    lines: list[str] = [
        f"item id: {metadata.item_id}",
        f"item type: {metadata.item_type}",
    ]
    if metadata.primary_color:
        secondary = (
            f", secondary {metadata.secondary_color}"
            if metadata.secondary_color
            else ""
        )
        lines.append(f"colors: primary {metadata.primary_color}{secondary}")
    elif metadata.dominant_colors:
        lines.append(f"colors: {', '.join(metadata.dominant_colors)}")
    if filterable_terms:
        lines.append(f"visual tags: {', '.join(filterable_terms)}")
    if search_only_terms:
        lines.append(f"search terms: {', '.join(search_only_terms)}")
    elif visual_features.raw_terms:
        lines.append(f"caption hints: {', '.join(visual_features.raw_terms[:4])}")
    return "\n".join(lines).strip()


def build_document_record(
    metadata: MetadataRecord,
    visual_features: VisualFeatureRecord,
    assignments: list[TagAssignmentRecord],
) -> SearchDocumentRecord:
    return SearchDocumentRecord(
        id=metadata.item_id,
        data=build_search_text(metadata, visual_features, assignments),
        metadata={
            "item_id": metadata.item_id,
            "item_type": metadata.item_type,
            "dominant_colors": metadata.dominant_colors,
            "accent_colors": metadata.accent_colors,
            "primary_color": metadata.primary_color,
            "secondary_color": metadata.secondary_color,
            "facet_values": metadata.facet_values,
            "accepted_facets": metadata.accepted_facets,
            "review_facets": metadata.review_facets,
            "search_terms": metadata.search_terms,
        },
    )
