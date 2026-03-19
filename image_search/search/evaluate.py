from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

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
                    expected_item_ids=[int(value) for value in payload["expected_item_ids"]],
                    filters=dict(payload.get("filters", {})),
                )
            )
    return queries


def _tokenize(value: str) -> list[str]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in value)
    return [token for token in normalized.split() if token]


def lexical_baseline(
    query: str,
    metadata: pd.DataFrame,
    limit: int,
    filters: dict[str, list[object]],
) -> list[int]:
    filtered = metadata
    if filters.get("type"):
        filtered = filtered[filtered["type"].isin(filters["type"])]
    if filters.get("quality"):
        filtered = filtered[filtered["quality"].isin(filters["quality"])]
    if filters.get("obtain_type"):
        filtered = filtered[filtered["obtain_type"].isin(filters["obtain_type"])]
    if filters.get("colors"):
        requested_colors = set(str(value) for value in filters["colors"])
        filtered = filtered[
            filtered["dominant_colors"].apply(
                lambda values: bool(requested_colors.intersection(values or []))
            )
        ]

    query_tokens = set(_tokenize(query))
    ranked = sorted(
        (
            (
                int(row["item_id"]),
                len(query_tokens.intersection(_tokenize(str(row["name"])))),
                str(row["name"]).lower(),
            )
            for _, row in filtered.iterrows()
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
    metadata = pd.read_parquet(metadata_path)

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
            type=[str(value) for value in filters.get("type", [])],
            quality=[int(value) for value in filters.get("quality", [])],
            obtain_type=[int(value) for value in filters.get("obtain_type", [])],
            colors=[str(value) for value in filters.get("colors", [])],
        )

        semantic_started = time.perf_counter()
        semantic_results = query_upstash(request, config)
        semantic_latency = (time.perf_counter() - semantic_started) * 1000.0
        semantic_latencies_ms.append(semantic_latency)

        semantic_ids = [result.item_id for result in semantic_results]
        lexical_ids = lexical_baseline(evaluation_query.query, metadata, limit, filters)

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

    semantic_summary = _summarize_metrics(semantic_rows)
    lexical_summary = _summarize_metrics(lexical_rows)

    return {
        "summary": {
            "query_count": len(queries),
            "semantic": asdict(semantic_summary),
            "lexical": asdict(lexical_summary),
            "semantic_latency_ms_avg": (
                sum(semantic_latencies_ms) / len(semantic_latencies_ms)
                if semantic_latencies_ms
                else 0.0
            ),
        },
        "queries": per_query,
    }
