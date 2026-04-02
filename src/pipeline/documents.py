from __future__ import annotations

from models.schemas import SearchDocumentRecord, StructuredItemRecord


def _has_value(value: object) -> bool:
    return value not in (None, "", [], {})


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


def _item_tokens(data: dict[str, object], field_name: str) -> list[str]:
    return _dedupe(_flatten_values(data.get(field_name)))


def build_search_text(record: StructuredItemRecord) -> str:
    search_terms = [record.item_type]
    category = str(record.data.get("category") or "").strip()
    if category:
        search_terms.append(category)
    subcategory = str(record.data.get("subcategory") or "").strip()
    if subcategory:
        search_terms.append(subcategory)
    placement = str(record.data.get("placement") or "").strip()
    if placement:
        search_terms.append(placement)
    ornament = _item_tokens(record.data, "ornament")

    attribute_values = _dedupe(
        list(
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
    )
    search_terms.extend(ornament)
    search_terms.extend(attribute_values)
    return " ".join(_dedupe(search_terms)).strip()


def build_document_record(record: StructuredItemRecord) -> SearchDocumentRecord:
    metadata = {
        "item_id": record.item_id,
        "item_type": record.item_type,
    }
    metadata.update(
        {
            key: value
            for key, value in record.data.items()
            if key not in {"primary_color", "secondary_color"} and _has_value(value)
        }
    )
    return SearchDocumentRecord(
        id=record.item_id,
        data=build_search_text(record),
        metadata=metadata,
    )
