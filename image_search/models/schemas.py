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
class MetadataRecord:
    item_id: int
    item_type: str
    dominant_colors: list[str] = field(default_factory=list)
    accent_colors: list[str] = field(default_factory=list)
    primary_color: str | None = None
    secondary_color: str | None = None
    motifs: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    subtypes: list[str] = field(default_factory=list)
    accepted_facets: list[str] = field(default_factory=list)
    review_facets: list[str] = field(default_factory=list)
    search_terms: list[str] = field(default_factory=list)

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
    taxonomy_concept_count: int = 0
    accepted_tag_count: int = 0
    review_tag_count: int = 0
    suppressed_tag_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ItemInputRecord:
    item_id: int
    item_type: str
    icon_path: str
    overview_path: str
    has_icon: bool = False
    has_overview: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VisualFeatureRecord:
    item_id: int
    item_type: str
    dominant_colors: list[str] = field(default_factory=list)
    accent_colors: list[str] = field(default_factory=list)
    primary_color: str | None = None
    secondary_color: str | None = None
    palette: list[str] = field(default_factory=list)
    icon_caption: str = ""
    overview_caption: str = ""
    caption_visual: str = ""
    icon_terms: list[str] = field(default_factory=list)
    overview_terms: list[str] = field(default_factory=list)
    raw_terms: list[str] = field(default_factory=list)
    search_terms: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaxonomyConceptRecord:
    key: str
    layer: str
    facet_key: str
    value_key: str
    display: str
    allowed_item_types: list[str] = field(default_factory=list)
    multi_value: bool = False
    preferred_evidence: str = "both"
    aliases: list[str] = field(default_factory=list)
    search_only: bool = False
    is_filterable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TagAssignmentRecord:
    item_id: int
    concept_key: str
    facet_key: str
    value_key: str
    layer: str
    source: str
    confidence: float
    status: str
    model_version: str
    search_only: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StructuredCandidateRecord:
    item_id: int
    facet_key: str
    value_key: str
    concept_key: str
    source: str
    source_modalities: list[str] = field(default_factory=list)
    pre_validation_score: float = 0.0
    search_only: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewRecord:
    item_id: int
    concept_key: str
    facet_key: str
    value_key: str
    confidence: float
    status: str
    evidence: dict[str, Any] = field(default_factory=dict)

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
class QueryRequest:
    q: str
    limit: int = 20
    item_type: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    facets: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryResult:
    item_id: int
    score: float
    item_type: str = ""
    dominant_colors: list[str] = field(default_factory=list)
    accent_colors: list[str] = field(default_factory=list)
    accepted_facets: list[str] = field(default_factory=list)
    search_terms: list[str] = field(default_factory=list)


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
