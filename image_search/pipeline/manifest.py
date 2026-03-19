from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from image_search.constants.items import (
    BASE_ITEM_PREFIX_RANGES,
    IMAGE_EXTENSIONS,
    TYPE_KEY_MAP,
)
from image_search.constants.settings import PROJECT_ROOT
from image_search.models.schemas import ManifestRecord


DEFAULT_TRACKER_ROOT = PROJECT_ROOT.parent / "gongeo.us-nikki-tracker"
DEFAULT_CONFIG_ROOT = PROJECT_ROOT.parent / "gongeo.us-config-decoder" / "cfg" / "config_output"
DEFAULT_SYNC_REPORT = PROJECT_ROOT.parent / "gongeo.us-processor" / "reports" / "database-sync-report.json"


def _is_base_item(item_id: int) -> bool:
    return any(lower <= item_id <= upper for lower, upper in BASE_ITEM_PREFIX_RANGES)


@dataclass(slots=True)
class ManifestPaths:
    tracker_root: Path
    config_root: Path
    sync_report_path: Path

    @property
    def locale_path(self) -> Path:
        return self.tracker_root / "app" / "locales" / "en" / "item.json"

    @property
    def item_image_root(self) -> Path:
        return self.tracker_root / "public" / "images" / "items"

    @property
    def item_icon_root(self) -> Path:
        return self.item_image_root / "icons"

    @property
    def item_config_path(self) -> Path:
        return self.config_root / "item" / "TbItem.json"

    @property
    def item_extra_path(self) -> Path:
        return self.config_root / "item" / "TbItemExtra.json"

    @property
    def minor_type_path(self) -> Path:
        return self.config_root / "clothes" / "TbClothesMinorTypeInfo.json"


def resolve_manifest_paths(
    tracker_root: str | None = None,
    config_root: str | None = None,
    sync_report_path: str | None = None,
) -> ManifestPaths:
    resolved_tracker = Path(tracker_root or os.getenv("TRACKER_ROOT") or DEFAULT_TRACKER_ROOT)
    resolved_config = Path(config_root or os.getenv("CONFIG_DECODER_OUTPUT") or DEFAULT_CONFIG_ROOT)
    resolved_sync_report = Path(sync_report_path or DEFAULT_SYNC_REPORT)
    return ManifestPaths(
        tracker_root=resolved_tracker,
        config_root=resolved_config,
        sync_report_path=resolved_sync_report,
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_image_path(root: Path, item_id: int) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = root / f"{item_id}{extension}"
        if candidate.exists():
            return candidate
    return None


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
    return TYPE_KEY_MAP.get(type_info.get("l10nshow_name"), "unknown")


def build_manifest(
    *,
    tracker_root: str | None = None,
    config_root: str | None = None,
    sync_report_path: str | None = None,
    limit: int | None = None,
    source_version: str | None = None,
) -> tuple[list[ManifestRecord], dict[str, int | str]]:
    paths = resolve_manifest_paths(
        tracker_root=tracker_root,
        config_root=config_root,
        sync_report_path=sync_report_path,
    )

    sync_report = _load_json(paths.sync_report_path)
    locale_data = _load_json(paths.locale_path)
    item_config = _load_json(paths.item_config_path)
    item_extras = _load_json(paths.item_extra_path)
    minor_type_info = _load_json(paths.minor_type_path)
    items = sync_report.get("syncedDetails", {}).get("items", [])

    resolved_source_version = (
        source_version
        or sync_report.get("timestamp")
        or str(paths.sync_report_path.stat().st_mtime_ns)
    )

    manifest: list[ManifestRecord] = []
    stats = {
        "skipped_count": 0,
        "non_base_skipped_count": 0,
        "missing_icon_count": 0,
        "missing_overview_count": 0,
        "source_version": str(resolved_source_version),
    }

    for raw in items:
        item_id = int(raw["id"])

        if not _is_base_item(item_id):
            stats["non_base_skipped_count"] += 1
            continue

        item_payload = item_config.get(str(item_id))
        item_extra = item_extras.get(str(item_id), {})

        locale_key = f"item.{item_id}.name"
        name = (
            locale_data.get(locale_key)
            or raw.get("name")
            or (item_payload or {}).get("name")
            or ""
        ).strip()
        item_type = _resolve_item_type(item_payload, minor_type_info)
        quality = item_payload.get("quality") if item_payload else None
        obtain_type = item_extra.get("obtain_type")

        icon_path = _find_image_path(paths.item_icon_root, item_id)
        overview_path = _find_image_path(paths.item_image_root, item_id)
        has_icon = icon_path is not None
        has_overview = overview_path is not None

        if not has_icon:
            stats["missing_icon_count"] += 1
        if not has_overview:
            stats["missing_overview_count"] += 1

        if not (has_icon or has_overview or name):
            stats["skipped_count"] += 1
            continue

        manifest.append(
            ManifestRecord(
                item_id=item_id,
                name=name,
                type=item_type,
                quality=int(quality) if quality is not None else None,
                obtain_type=int(obtain_type) if obtain_type is not None else None,
                icon_path=str(icon_path or ""),
                overview_path=str(overview_path or ""),
                has_icon=has_icon,
                has_overview=has_overview,
                source_version=str(resolved_source_version),
            )
        )

        if limit is not None and len(manifest) >= limit:
            break

    return manifest, stats
