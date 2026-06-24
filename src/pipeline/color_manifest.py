from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from constants.items import TYPE_KEY_MAP
from constants.structured import is_supported_item_type
from constants.tracker_export import normalize_supported_item_type
from models.schemas import ColorManifestRecord
from pipeline.manifest import _find_image_path, resolve_manifest_paths


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_item_type(
    item_payload: dict[str, Any] | None,
    minor_type_info: dict[str, Any],
) -> str:
    if not item_payload:
        return "unknown"
    minor_type = item_payload.get("minor_type")
    if minor_type is None:
        return "unknown"
    type_info = minor_type_info.get(str(minor_type))
    if not type_info:
        return "unknown"
    return normalize_supported_item_type(
        TYPE_KEY_MAP.get(type_info.get("l10nshow_name"), "unknown")
    )


def build_color_manifest(
    *,
    tracker_root: str | None = None,
    config_root: str | None = None,
    sync_report_path: str | None = None,
    limit: int | None = None,
    item_id: int | None = None,
    item_ids: set[int] | None = None,
    item_types: set[str] | None = None,
) -> tuple[list[ColorManifestRecord], dict[str, int]]:
    paths = resolve_manifest_paths(
        tracker_root=tracker_root,
        config_root=config_root,
        sync_report_path=sync_report_path,
    )
    sync_report = _load_json(paths.sync_report_path)
    item_config = _load_json(paths.item_config_path)
    minor_type_info = _load_json(paths.minor_type_path)
    items = sync_report.get("syncedDetails", {}).get("items", [])
    selected_item_ids = set(item_ids or ())
    if item_id is not None:
        selected_item_ids.add(item_id)
    selected_item_types = {
        normalize_supported_item_type(item_type) for item_type in (item_types or ())
    }

    manifest: list[ColorManifestRecord] = []
    stats = {
        "skipped_count": 0,
        "unsupported_type_count": 0,
        "missing_icon_count": 0,
    }

    for raw in items:
        current_item_id = int(raw["id"])
        if selected_item_ids and current_item_id not in selected_item_ids:
            continue

        item_type = _resolve_item_type(
            item_config.get(str(current_item_id)),
            minor_type_info,
        )
        if not is_supported_item_type(item_type):
            stats["unsupported_type_count"] += 1
            stats["skipped_count"] += 1
            continue
        if selected_item_types and item_type not in selected_item_types:
            continue

        icon_path = _find_image_path(paths.item_icon_root, current_item_id)
        if icon_path is None:
            stats["missing_icon_count"] += 1
            stats["skipped_count"] += 1
            continue

        manifest.append(
            ColorManifestRecord(
                item_id=current_item_id,
                item_type=item_type,
                icon_path=str(icon_path),
            )
        )
        if limit is not None and len(manifest) >= limit:
            break

    return manifest, stats
