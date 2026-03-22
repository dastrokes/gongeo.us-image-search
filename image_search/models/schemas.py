from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ManifestRecord:
    item_id: int
    type: str
    icon_path: str
    overview_path: str
    has_icon: bool
    has_overview: bool
    source_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StructuredItemRecord:
    item_id: int
    item_type: str
    shape: str
    source_version: str
    data: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StructuredDebugRecord:
    item_id: int
    item_type: str
    shape: str
    image_paths: dict[str, str | None] = field(default_factory=dict)
    prompt: str = ""
    raw_response: str = ""
    raw_payload: dict[str, Any] | None = None
    normalized_data: dict[str, Any] = field(default_factory=dict)
    filter_report: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchDocumentRecord:
    id: int
    data: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BuildSummary:
    extraction_model_id: str
    upstash_embedding_model: str
    item_count: int
    skipped_count: int
    missing_icon_count: int
    missing_overview_count: int
    structured_parse_fail_count: int
    build_started_at: str
    build_finished_at: str
    duration_seconds: float
    filter_report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QueryRequest:
    q: str
    limit: int = 20
    item_type: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryResult:
    item_id: int
    score: float
    item_type: str = ""
    shape: str = ""
    colors: list[str] = field(default_factory=list)
    primary_color: str | None = None
    secondary_color: str | None = None
    structured_data: dict[str, Any] = field(default_factory=dict)


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
