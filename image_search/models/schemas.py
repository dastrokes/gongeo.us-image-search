from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ManifestRecord:
    item_id: int
    name: str
    type: str
    quality: int | None
    obtain_type: int | None
    icon_path: str
    overview_path: str
    has_icon: bool
    has_overview: bool
    source_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MetadataRecord:
    item_id: int
    name: str
    type: str
    quality: int | None
    obtain_type: int | None
    icon_path: str
    overview_path: str
    dominant_colors: list[str] = field(default_factory=list)
    accent_colors: list[str] = field(default_factory=list)
    has_icon: bool = False
    has_overview: bool = False
    source_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CaptionRecord:
    item_id: int
    icon_caption: str
    overview_caption: str
    visual: str
    failed_modalities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocumentRecord:
    id: int
    data: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BuildSummary:
    caption_model_id: str
    upstash_embedding_model: str
    item_count: int
    skipped_count: int
    missing_icon_count: int
    missing_overview_count: int
    caption_fail_count: int
    build_started_at: str
    build_finished_at: str
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QueryRequest:
    q: str
    limit: int = 20
    type: list[str] = field(default_factory=list)
    quality: list[int] = field(default_factory=list)
    obtain_type: list[int] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryResult:
    item_id: int
    score: float
    name: str
    type: str
    quality: int | None
    obtain_type: int | None
    dominant_colors: list[str] = field(default_factory=list)
    accent_colors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationQuery:
    query: str
    expected_item_ids: list[int]
    filters: dict[str, list[Any]] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationMetrics:
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
