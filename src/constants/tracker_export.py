from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from constants.settings import PROJECT_ROOT

DEFAULT_TRACKER_ROOT = PROJECT_ROOT.parent / "gongeo.us-nikki-tracker"
DEFAULT_TRACKER_EXPORT = (
    DEFAULT_TRACKER_ROOT
    / "data"
    / "item-search"
    / "generated"
    / "image-search-taxonomy.json"
)
DEFAULT_TRACKER_OVERRIDES = (
    DEFAULT_TRACKER_ROOT / "data" / "item-search" / "overrides.json"
)


def _resolve_path(
    explicit_path: str | None,
    env_name: str,
    default_path: Path,
) -> Path:
    return Path(explicit_path or os.getenv(env_name) or default_path)


def resolve_tracker_export_path(path: str | None = None) -> Path:
    return _resolve_path(path, "TRACKER_ITEM_SEARCH_EXPORT", DEFAULT_TRACKER_EXPORT)


def resolve_tracker_overrides_path(path: str | None = None) -> Path:
    return _resolve_path(
        path, "TRACKER_ITEM_SEARCH_OVERRIDES", DEFAULT_TRACKER_OVERRIDES
    )


@lru_cache(maxsize=1)
def load_tracker_export(path: str | None = None) -> dict[str, Any]:
    export_path = resolve_tracker_export_path(path)
    with export_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_tracker_overrides(path: str | None = None) -> dict[str, Any]:
    overrides_path = resolve_tracker_overrides_path(path)
    with overrides_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_supported_item_type(item_type: str | None) -> str:
    if not item_type:
        return "unknown"
    export = load_tracker_export()
    alias_map = export.get("itemTypeAliasToSupported", {})
    return str(alias_map.get(item_type, item_type)).strip() or "unknown"
