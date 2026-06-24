from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, cast


@dataclass
class ManifestRecord:
    item_id: int
    item_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(cast(Any, self))


@dataclass
class StructuredItemRecord:
    item_id: int
    item_type: str
    data: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(cast(Any, self))


@dataclass
class StructuredDebugRecord:
    item_id: int
    item_type: str
    image_paths: dict[str, str | None] = field(default_factory=dict)
    prompt: str = ""
    raw_response: str = ""
    raw_payload: dict[str, Any] | None = None
    normalized_data: dict[str, Any] = field(default_factory=dict)
    filter_report: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(cast(Any, self))


@dataclass
class ItemAttributesRecord:
    item_id: int
    item_type: str
    category: str | None = None
    subcategory: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(cast(Any, self))


@dataclass
class ColorManifestRecord:
    item_id: int
    item_type: str
    icon_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(cast(Any, self))


@dataclass
class ColorSwatch:
    label: str
    hex: str
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(cast(Any, self))


@dataclass
class ItemColorsRecord:
    item_id: int
    item_type: str
    primary_colors: list[str] = field(default_factory=list)
    secondary_colors: list[str] = field(default_factory=list)
    color_tags: list[str] = field(default_factory=list)
    swatches: list[ColorSwatch] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(cast(Any, self))
        payload["swatches"] = [swatch.to_dict() for swatch in self.swatches]
        return payload


@dataclass
class ColorDebugRecord:
    item_id: int
    item_type: str
    image_path: str
    visible_pixel_count: int = 0
    sampled_pixel_count: int = 0
    ignored_pixel_count: int = 0
    color_weights: dict[str, float] = field(default_factory=dict)
    swatches: list[ColorSwatch] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(cast(Any, self))
        payload["swatches"] = [swatch.to_dict() for swatch in self.swatches]
        return payload


@dataclass
class BuildSummary:
    extraction_model_id: str
    item_count: int
    skipped_count: int
    missing_icon_count: int
    missing_overview_count: int
    structured_parse_fail_count: int
    build_started_at: str
    build_finished_at: str
    duration_seconds: float
    search_report_path: str | None = None
    item_attributes_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(cast(Any, self))
