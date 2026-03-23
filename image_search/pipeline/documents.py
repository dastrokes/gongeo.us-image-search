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


def _item_tokens(data: dict[str, object], field_name: str) -> list[str]:
    return _dedupe(_flatten_values(data.get(field_name)))


def build_search_text(record: StructuredItemRecord) -> str:
    lines = [
        f"item id: {record.item_id}",
        f"slot: {record.item_type}",
    ]
    category = str(record.data.get("category") or "").strip()
    if category:
        lines.append(f"category: {category}")
    subcategory = str(record.data.get("subcategory") or "").strip()
    if subcategory:
        lines.append(f"subcategory: {subcategory}")
    placement = str(record.data.get("placement") or "").strip()
    if placement:
        lines.append(f"placement: {placement}")
    colors = _item_colors(record.data)
    if colors:
        lines.append(f"colors: {', '.join(colors)}")
    ornament = _item_tokens(record.data, "ornament")
    if ornament:
        lines.append(f"ornament: {', '.join(ornament)}")

    attribute_values = _dedupe(
        value
        for field_name, field_value in record.data.items()
        if field_name
        not in {
            "category",
            "subcategory",
            "placement",
            "primary_color",
            "secondary_color",
            "ornament",
        }
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
        "slot": record.item_type,
        "colors": colors,
    }
    metadata.update(record.data)
    return SearchDocumentRecord(
        id=record.item_id,
        data=build_search_text(record),
        metadata=metadata,
    )
