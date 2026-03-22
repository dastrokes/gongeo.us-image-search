from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from upstash_vector import Index

from image_search.constants.settings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_TYPE,
    DEFAULT_MANAGEMENT_URL,
    DEFAULT_SPARSE_EMBEDDING_MODEL,
)
from image_search.constants.structured import (
    is_supported_item_type,
    shape_definition_for_item_type,
)
from image_search.models.schemas import QueryRequest, QueryResult, SearchDocumentRecord

EMBEDDING_MODEL_API_NAMES = {
    "BAAI/bge-small-en-v1.5": "BGE_SMALL_EN_V1_5",
    "BAAI/bge-base-en-v1.5": "BGE_BASE_EN_V1_5",
    "BAAI/bge-large-en-v1.5": "BGE_LARGE_EN_V1_5",
    "BAAI/bge-m3": "BGE_M3",
    "BGE_SMALL_EN_V1_5": "BGE_SMALL_EN_V1_5",
    "BGE_BASE_EN_V1_5": "BGE_BASE_EN_V1_5",
    "BGE_LARGE_EN_V1_5": "BGE_LARGE_EN_V1_5",
    "BGE_M3": "BGE_M3",
}

SPARSE_EMBEDDING_MODEL_API_NAMES = {
    "BM25": "BM25",
    "BGE_M3": "BGE_M3",
    "BAAI/bge-m3": "BGE_M3",
}


def _validate_vector_rest_url(rest_url: str) -> None:
    hostname = (urlparse(rest_url).hostname or "").lower()
    if hostname.endswith("search.upstash.io"):
        raise ValueError(
            "UPSTASH_VECTOR_REST_URL points to an Upstash Search database "
            f"({hostname}). This CLI uses Upstash Vector. Copy the REST URL and "
            "token from the Vector index Connect panel instead."
        )


def _normalize_embedding_model_name(model: str) -> str:
    normalized = EMBEDDING_MODEL_API_NAMES.get(model)
    if normalized is None:
        supported = ", ".join(sorted(EMBEDDING_MODEL_API_NAMES))
        raise ValueError(
            f"Unsupported dense embedding model '{model}'. Supported values: {supported}."
        )
    return normalized


def _normalize_sparse_embedding_model_name(model: str) -> str:
    normalized = SPARSE_EMBEDDING_MODEL_API_NAMES.get(model)
    if normalized is None:
        supported = ", ".join(sorted(SPARSE_EMBEDDING_MODEL_API_NAMES))
        raise ValueError(
            f"Unsupported sparse embedding model '{model}'. Supported values: {supported}."
        )
    return normalized


@dataclass(slots=True)
class UpstashConfig:
    rest_url: str | None
    rest_token: str | None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    management_url: str = DEFAULT_MANAGEMENT_URL
    management_email: str | None = None
    management_api_key: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        rest_url: str | None = None,
        rest_token: str | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        require_rest: bool = True,
    ) -> "UpstashConfig":
        resolved_rest_url = rest_url or os.getenv("UPSTASH_VECTOR_REST_URL") or ""
        resolved_rest_token = rest_token or os.getenv("UPSTASH_VECTOR_REST_TOKEN") or ""
        if require_rest and (not resolved_rest_url or not resolved_rest_token):
            raise ValueError(
                "UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN must be set."
            )
        if resolved_rest_url:
            _validate_vector_rest_url(resolved_rest_url)
        return cls(
            rest_url=resolved_rest_url or None,
            rest_token=resolved_rest_token or None,
            embedding_model=embedding_model,
            management_url=os.getenv(
                "UPSTASH_VECTOR_MANAGEMENT_URL",
                DEFAULT_MANAGEMENT_URL,
            ),
            management_email=os.getenv("UPSTASH_EMAIL"),
            management_api_key=os.getenv("UPSTASH_API_KEY"),
        )


def load_documents(documents_path: str | Path) -> list[SearchDocumentRecord]:
    records: list[SearchDocumentRecord] = []
    with Path(documents_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            raw_id = payload.get("id", payload.get("item_id"))
            if raw_id is None:
                raise ValueError("Document record is missing both 'id' and 'item_id'.")
            if isinstance(raw_id, str) and raw_id.startswith("item:"):
                raw_id = raw_id.split(":", 1)[1]
            records.append(
                SearchDocumentRecord(
                    id=int(raw_id),
                    data=str(payload["data"]),
                    metadata=dict(payload["metadata"]),
                )
            )
    return records


def create_index(
    *,
    name: str,
    region: str,
    dimension_count: int = 1024,
    similarity_function: str = "COSINE",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    sparse_embedding_model: str = DEFAULT_SPARSE_EMBEDDING_MODEL,
    index_type: str = DEFAULT_INDEX_TYPE,
    config: UpstashConfig,
) -> dict[str, Any]:
    if not config.management_email or not config.management_api_key:
        raise ValueError(
            "UPSTASH_EMAIL and UPSTASH_API_KEY are required to create an index."
        )
    normalized_index_type = index_type.upper()
    if normalized_index_type not in {"DENSE", "SPARSE", "HYBRID"}:
        raise ValueError("index_type must be one of DENSE, SPARSE, or HYBRID.")

    auth_value = base64.b64encode(
        f"{config.management_email}:{config.management_api_key}".encode("utf-8")
    ).decode("utf-8")
    payload: dict[str, Any] = {
        "name": name,
        "region": region,
        "similarity_function": similarity_function.upper(),
        "dimension_count": int(dimension_count),
        "embedding_model": _normalize_embedding_model_name(embedding_model),
        "index_type": normalized_index_type,
    }
    if normalized_index_type in {"SPARSE", "HYBRID"}:
        payload["sparse_embedding_model"] = _normalize_sparse_embedding_model_name(
            sparse_embedding_model
        )

    response = requests.post(
        f"{config.management_url.rstrip('/')}/index",
        headers={
            "Authorization": f"Basic {auth_value}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _build_index_client(config: UpstashConfig) -> Index:
    if not config.rest_url or not config.rest_token:
        raise ValueError(
            "UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN are required."
        )
    return Index(url=config.rest_url, token=config.rest_token)


def sync_documents(
    *,
    documents_path: str | Path,
    config: UpstashConfig,
    batch_size: int = 100,
) -> dict[str, int]:
    documents = load_documents(documents_path)
    index = _build_index_client(config)

    upserted = 0
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        payload = [
            {
                "id": record.id,
                "data": record.data,
                "metadata": record.metadata,
            }
            for record in batch
        ]
        index.upsert(payload)
        upserted += len(batch)

    return {"upserted": upserted}


def build_filter_expression(request: QueryRequest) -> str | None:
    clauses: list[str] = []

    if request.item_type:
        escaped = [value.replace("'", "\\'") for value in request.item_type]
        clauses.append(
            "(" + " OR ".join(f"item_type = '{value}'" for value in escaped) + ")"
        )
    if request.colors:
        escaped = [value.replace("'", "\\'") for value in request.colors]
        clauses.append(
            "(" + " OR ".join(f"colors CONTAINS '{value}'" for value in escaped) + ")"
        )

    return " AND ".join(clauses) if clauses else None


def query_upstash(request: QueryRequest, config: UpstashConfig) -> list[QueryResult]:
    index = _build_index_client(config)
    filter_expression = build_filter_expression(request)
    response = index.query(
        data=request.q,
        top_k=request.limit,
        include_metadata=True,
        include_data=False,
        filter=filter_expression,
    )

    results: list[QueryResult] = []
    for match in response:
        metadata = getattr(match, "metadata", {}) or {}
        if "item_id" not in metadata:
            continue
        item_type = str(metadata.get("item_type", "unknown"))
        if not is_supported_item_type(item_type):
            continue
        shape_definition = shape_definition_for_item_type(item_type)
        structured_data = {
            field_definition.name: metadata.get(field_definition.name)
            for field_definition in shape_definition.fields
            if field_definition.name in metadata
        }
        if not structured_data and isinstance(metadata.get("structured_data"), dict):
            structured_data = dict(metadata.get("structured_data", {}) or {})
        results.append(
            QueryResult(
                item_id=int(metadata.get("item_id")),
                score=float(getattr(match, "score", 0.0)),
                item_type=item_type,
                shape=str(metadata.get("shape", "")),
                colors=list(metadata.get("colors", [])),
                primary_color=metadata.get("primary_color"),
                secondary_color=metadata.get("secondary_color"),
                structured_data=structured_data,
            )
        )
    return results
