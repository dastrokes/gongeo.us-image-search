from __future__ import annotations

from image_search.models.schemas import SearchDocumentRecord, StructuredItemRecord


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        flattened: list[str] = []
        for nested in value.values():
            flattened.extend(_flatten_values(nested))
        return flattened
    if isinstance(value, list):
        flattened: list[str] = []
        for nested in value:
            flattened.extend(_flatten_values(nested))
        return flattened
    normalized = str(value or "").strip()
    return [normalized] if normalized else []


def _item_colors(data: dict[str, object]) -> list[str]:
    return _dedupe(
        [
            str(data.get("primary_color") or "").strip(),
            str(data.get("secondary_color") or "").strip(),
        ]
    )


def build_search_text(record: StructuredItemRecord) -> str:
    lines = [
        f"item id: {record.item_id}",
        f"item type: {record.item_type}",
        f"shape: {record.shape}",
    ]
    subtype = str(record.data.get("subtype") or "").strip()
    if subtype:
        lines.append(f"subtype: {subtype}")
    colors = _item_colors(record.data)
    if colors:
        lines.append(f"colors: {', '.join(colors)}")

    attribute_values = _dedupe(
        value
        for field_name, field_value in record.data.items()
        if field_name not in {"subtype", "primary_color", "secondary_color"}
        for value in _flatten_values(field_value)
    )
    if attribute_values:
        lines.append(f"attributes: {', '.join(attribute_values)}")
    return "\n".join(lines).strip()


def build_document_record(record: StructuredItemRecord) -> SearchDocumentRecord:
    colors = _item_colors(record.data)
    metadata = {
        "item_id": record.item_id,
        "item_type": record.item_type,
        "shape": record.shape,
        "colors": colors,
    }
    metadata.update(record.data)
    return SearchDocumentRecord(
        id=record.item_id,
        data=build_search_text(record),
        metadata=metadata,
    )
