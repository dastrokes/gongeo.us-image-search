from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from image_search.models.schemas import EvaluationMetrics, EvaluationQuery, QueryRequest
from image_search.search.upstash import UpstashConfig, query_upstash


def load_evaluation_queries(path: str | Path) -> list[EvaluationQuery]:
    queries: list[EvaluationQuery] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            queries.append(
                EvaluationQuery(
                    query=str(payload["query"]),
                    expected_item_ids=[
                        int(value) for value in payload["expected_item_ids"]
                    ],
                    filters=dict(payload.get("filters", {})),
                )
            )
    return queries


def load_search_documents(path: str | Path) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            metadata = dict(payload.get("metadata", {}))
            metadata["item_id"] = int(metadata.get("item_id", payload.get("id", 0)))
            metadata["search_text"] = str(payload.get("data", ""))
            documents.append(metadata)
    return documents


def _tokenize(value: str) -> list[str]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in value)
    return [token for token in normalized.split() if token]


def lexical_baseline(
    query: str,
    documents: list[dict[str, object]],
    limit: int,
    filters: dict[str, list[object]],
) -> list[int]:
    filtered = documents
    if filters.get("item_type"):
        requested = {str(value) for value in filters["item_type"]}
        filtered = [row for row in filtered if str(row.get("item_type")) in requested]
    elif filters.get("type"):
        requested = {str(value) for value in filters["type"]}
        filtered = [row for row in filtered if str(row.get("item_type")) in requested]
    if filters.get("colors"):
        requested_colors = {str(value) for value in filters["colors"]}
        filtered = [
            row
            for row in filtered
            if requested_colors.intersection(
                {str(value) for value in row.get("colors", [])}
            )
        ]

    query_tokens = set(_tokenize(query))
    ranked = sorted(
        (
            (
                int(row["item_id"]),
                len(
                    query_tokens.intersection(
                        _tokenize(
                            " ".join(
                                [
                                    str(row.get("item_type", "")),
                                    str(row.get("subcategory", "")),
                                    " ".join(
                                        str(value)
                                        for value in row.get("colors", []) or []
                                    ),
                                    str(row.get("search_text", "")),
                                ]
                            )
                        )
                    )
                ),
                str(row.get("item_type", "")).lower(),
            )
            for row in filtered
        ),
        key=lambda entry: (entry[1], entry[2]),
        reverse=True,
    )
    return [item_id for item_id, _, _ in ranked[:limit]]


def _recall_at_k(result_ids: list[int], expected: set[int], limit: int) -> float:
    if not expected:
        return 0.0
    return len(expected.intersection(result_ids[:limit])) / len(expected)


def _mrr_at_k(result_ids: list[int], expected: set[int], limit: int) -> float:
    for rank, item_id in enumerate(result_ids[:limit], start=1):
        if item_id in expected:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(result_ids: list[int], expected: set[int], limit: int) -> float:
    dcg = 0.0
    for rank, item_id in enumerate(result_ids[:limit], start=1):
        if item_id in expected:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(limit, len(expected))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _summarize_metrics(rows: list[dict[str, float]]) -> EvaluationMetrics:
    if not rows:
        return EvaluationMetrics(recall_at_10=0.0, mrr_at_10=0.0, ndcg_at_10=0.0)
    return EvaluationMetrics(
        recall_at_10=sum(row["recall_at_10"] for row in rows) / len(rows),
        mrr_at_10=sum(row["mrr_at_10"] for row in rows) / len(rows),
        ndcg_at_10=sum(row["ndcg_at_10"] for row in rows) / len(rows),
    )


def evaluate_queries(
    *,
    queries_path: str | Path,
    metadata_path: str | Path,
    config: UpstashConfig,
    limit: int = 10,
) -> dict[str, object]:
    queries = load_evaluation_queries(queries_path)
    documents = load_search_documents(metadata_path)

    semantic_rows: list[dict[str, float]] = []
    lexical_rows: list[dict[str, float]] = []
    per_query: list[dict[str, object]] = []
    semantic_latencies_ms: list[float] = []

    for evaluation_query in queries:
        filters = evaluation_query.filters
        expected = set(evaluation_query.expected_item_ids)
        request = QueryRequest(
            q=evaluation_query.query,
            limit=limit,
            item_type=[
                str(value)
                for value in filters.get("item_type", filters.get("type", []))
            ],
            colors=[str(value) for value in filters.get("colors", [])],
        )

        semantic_started = time.perf_counter()
        semantic_results = query_upstash(request, config)
        semantic_latency = (time.perf_counter() - semantic_started) * 1000.0
        semantic_latencies_ms.append(semantic_latency)

        semantic_ids = [result.item_id for result in semantic_results]
        lexical_ids = lexical_baseline(
            evaluation_query.query, documents, limit, filters
        )

        semantic_metric_row = {
            "recall_at_10": _recall_at_k(semantic_ids, expected, limit),
            "mrr_at_10": _mrr_at_k(semantic_ids, expected, limit),
            "ndcg_at_10": _ndcg_at_k(semantic_ids, expected, limit),
        }
        lexical_metric_row = {
            "recall_at_10": _recall_at_k(lexical_ids, expected, limit),
            "mrr_at_10": _mrr_at_k(lexical_ids, expected, limit),
            "ndcg_at_10": _ndcg_at_k(lexical_ids, expected, limit),
        }
        semantic_rows.append(semantic_metric_row)
        lexical_rows.append(lexical_metric_row)

        per_query.append(
            {
                "query": evaluation_query.query,
                "filters": filters,
                "expected_item_ids": evaluation_query.expected_item_ids,
                "semantic_result_ids": semantic_ids,
                "lexical_result_ids": lexical_ids,
                "semantic_metrics": semantic_metric_row,
                "lexical_metrics": lexical_metric_row,
                "semantic_latency_ms": semantic_latency,
            }
        )

    return {
        "summary": {
            "query_count": len(queries),
            "limit": limit,
            "semantic": asdict(_summarize_metrics(semantic_rows)),
            "lexical": asdict(_summarize_metrics(lexical_rows)),
            "avg_semantic_latency_ms": (
                sum(semantic_latencies_ms) / len(semantic_latencies_ms)
                if semantic_latencies_ms
                else 0.0
            ),
        },
        "queries": per_query,
    }
