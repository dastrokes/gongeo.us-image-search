from __future__ import annotations

from typing import Any

from constants.structured import schema_definition_for_item_type
from constants.tracker_export import load_tracker_overrides
from models.schemas import ItemAttributesRecord, StructuredItemRecord


def _has_value(value: object) -> bool:
    return value not in (None, "", [], {})


def _canonical_metadata_from_data(
    item_type: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    schema_definition = schema_definition_for_item_type(item_type)
    for field_definition in schema_definition.fields:
        field_name = field_definition.name
        if field_name in {"category", "subcategory"}:
            continue
        value = data.get(field_name)
        if _has_value(value):
            metadata[field_name] = value
    return metadata


def build_structured_payload_from_item_attributes(
    record: ItemAttributesRecord,
) -> dict[str, Any]:
    payload = {
        "category": record.category,
        "subcategory": record.subcategory,
        **dict(record.metadata or {}),
    }

    from pipeline.extraction import VisionStructuredExtractor

    return VisionStructuredExtractor.normalize_payload(record.item_type, payload)


def build_item_attributes_record(
    record: StructuredItemRecord,
) -> ItemAttributesRecord:
    return ItemAttributesRecord(
        item_id=record.item_id,
        item_type=record.item_type,
        category=(
            str(record.data.get("category")).strip()
            if isinstance(record.data.get("category"), str)
            and str(record.data.get("category")).strip()
            else None
        ),
        subcategory=(
            str(record.data.get("subcategory")).strip()
            if isinstance(record.data.get("subcategory"), str)
            and str(record.data.get("subcategory")).strip()
            else None
        ),
        metadata=_canonical_metadata_from_data(record.item_type, record.data),
    )


def _normalize_override_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items()}


def load_curated_override_map() -> dict[int, dict[str, Any]]:
    payload = load_tracker_overrides()
    items = payload.get("items")
    if not isinstance(items, dict):
        return {}
    override_map: dict[int, dict[str, Any]] = {}
    for item_id, entry in items.items():
        try:
            normalized_item_id = int(item_id)
        except (TypeError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue
        override_map[normalized_item_id] = entry
    return override_map


def apply_curated_override(
    record: StructuredItemRecord,
    override_entry: dict[str, Any] | None,
) -> ItemAttributesRecord:
    base_record = build_item_attributes_record(record)
    if not override_entry:
        return base_record

    from pipeline.extraction import VisionStructuredExtractor

    if "item_id" in override_entry or "item_type" in override_entry:
        override_payload = {
            "category": override_entry.get("category"),
            "subcategory": override_entry.get("subcategory"),
            **_normalize_override_payload(override_entry.get("metadata")),
        }
    else:
        override_payload = {
            "category": base_record.category,
            "subcategory": base_record.subcategory,
            **dict(base_record.metadata or {}),
            **_normalize_override_payload(override_entry.get("metadata")),
        }

    final_data = VisionStructuredExtractor.normalize_payload(
        record.item_type,
        override_payload,
    )

    return build_item_attributes_record(
        StructuredItemRecord(
            item_id=record.item_id,
            item_type=record.item_type,
            data=final_data,
            parse_error=record.parse_error,
        )
    )
