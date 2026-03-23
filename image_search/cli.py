from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from image_search.constants.settings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EXTRACTION_MODEL_ID,
    DEFAULT_INDEX_DIMENSION_COUNT,
    DEFAULT_INDEX_METRIC,
    DEFAULT_INDEX_NAME,
    DEFAULT_INDEX_REGION,
    DEFAULT_INDEX_TYPE,
    DEFAULT_MODEL_QUANTIZATION,
    DEFAULT_SPARSE_EMBEDDING_MODEL,
    PROJECT_ROOT,
)
from image_search.models.schemas import (
    BuildSummary,
    ManifestRecord,
    StructuredDebugRecord,
    StructuredItemRecord,
)


def _load_project_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _default_output_root() -> Path:
    return PROJECT_ROOT / "index"


def _default_manifest_root() -> Path:
    return PROJECT_ROOT / "manifest"


def _cleanup_stale_outputs(output_root: Path) -> None:
    for filename in (
        "item-official-metadata.jsonl",
        "item-documents.jsonl",
        "item-metadata.parquet",
        "taxonomy-concepts.jsonl",
        "item-visual-features.jsonl",
        "item-structured-candidates.jsonl",
        "item-tag-assignments.jsonl",
        "item-review-queue.jsonl",
        "item-unmapped-terms.jsonl",
        "item-captions-debug.jsonl",
        "item-filter-report.jsonl",
    ):
        try:
            (output_root / filename).unlink(missing_ok=True)
        except OSError:
            pass


def _manifest_needs_regen(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return True
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
    except OSError:
        return True
    if not first_line:
        return True
    try:
        payload = json.loads(first_line)
    except json.JSONDecodeError:
        return True
    required_fields = {
        "item_id",
        "type",
        "icon_path",
        "overview_path",
        "has_icon",
        "has_overview",
        "source_version",
    }
    deprecated_fields = {
        "name",
        "quality",
        "obtain_type",
        "style_scores",
        "style_key",
        "label_ids",
        "label_keys",
        "category_tag",
        "version_area",
        "gallery_score",
    }
    return not required_fields.issubset(payload) or bool(
        deprecated_fields.intersection(payload)
    )


def _load_manifest_records(manifest_path: Path, limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def _load_record_cache(path: Path) -> dict[int, dict]:
    cache: dict[int, dict] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                cache[int(payload["item_id"])] = payload
            except Exception:
                pass
    return cache


def _structured_record_from_payload(payload: dict[str, object]) -> StructuredItemRecord:
    from image_search.pipeline.structured import VisionStructuredExtractor

    item_type = str(payload["item_type"])
    data = dict(payload.get("data", {}) or {})
    return StructuredItemRecord(
        item_id=int(payload["item_id"]),
        item_type=item_type,
        shape=str(payload["shape"]),
        source_version=str(payload.get("source_version", "")),
        data=VisionStructuredExtractor.normalize_payload(item_type, data),
        parse_error=(
            str(payload["parse_error"])
            if payload.get("parse_error") is not None
            else None
        ),
    )


def _structured_debug_from_payload(payload: dict[str, object]) -> StructuredDebugRecord:
    return StructuredDebugRecord(
        item_id=int(payload["item_id"]),
        item_type=str(payload["item_type"]),
        shape=str(payload["shape"]),
        image_paths=dict(payload.get("image_paths", {}) or {}),
        prompt=str(payload.get("prompt", "")),
        raw_response=str(payload.get("raw_response", "")),
        raw_payload=(
            dict(payload.get("raw_payload", {}) or {})
            if isinstance(payload.get("raw_payload"), dict)
            else None
        ),
        normalized_data=dict(payload.get("normalized_data", {}) or {}),
        filter_report=dict(payload.get("filter_report", {}) or {}),
        parse_error=(
            str(payload["parse_error"])
            if payload.get("parse_error") is not None
            else None
        ),
    )


def _build_filter_report_rows(
    debug_records: list[StructuredDebugRecord],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for debug_record in debug_records:
        filter_report = debug_record.filter_report or {}
        if not any(
            (
                filter_report.get("non_canonical"),
                filter_report.get("parent_child_mismatch"),
                filter_report.get("unknown_fields"),
            )
        ):
            continue
        rows.append(
            {
                "item_id": debug_record.item_id,
                "item_type": debug_record.item_type,
                "shape": debug_record.shape,
                "non_canonical": list(filter_report.get("non_canonical", []) or []),
                "parent_child_mismatch": filter_report.get("parent_child_mismatch"),
                "unknown_fields": list(filter_report.get("unknown_fields", []) or []),
            }
        )
    return rows


def _is_complete_cached_structured(
    structured_payload: dict | None,
    debug_payload: dict | None,
) -> bool:
    if structured_payload is None or debug_payload is None:
        return False
    if structured_payload.get("parse_error") is not None:
        return False
    data = structured_payload.get("data")
    return isinstance(data, dict) and bool(data)


def _chunk_records(
    records: list[ManifestRecord],
    batch_size: int,
) -> list[list[ManifestRecord]]:
    if batch_size <= 0:
        batch_size = 1
    return [
        records[index : index + batch_size]
        for index in range(0, len(records), batch_size)
    ]


def _run_single_item_debug(
    *,
    record: ManifestRecord,
    extractor: object,
) -> int:
    structured_record, debug_record = extractor.extract_record(record)
    print(
        json.dumps(
            {
                "item_id": record.item_id,
                "item_type": record.type,
                "shape": structured_record.shape,
                "image_paths": debug_record.image_paths,
                "structured_debug": debug_record.to_dict(),
            },
            indent=2,
        )
    )
    return 0


def run_build_index(args: argparse.Namespace) -> int:
    from image_search.constants.structured import is_supported_item_type
    from image_search.pipeline.documents import build_document_record
    from image_search.pipeline.manifest import build_manifest
    from image_search.pipeline.structured import (
        StructuredExtractorConfig,
        VisionStructuredExtractor,
    )

    output_root = Path(args.output_root or _default_output_root())
    manifest_root = Path(args.manifest_root or _default_manifest_root())
    manifest_path = manifest_root / "item-manifest.jsonl"
    structured_data_path = output_root / "item-structured-data.jsonl"
    structured_debug_path = output_root / "item-structured-debug.jsonl"
    search_documents_path = output_root / "item-search-documents.jsonl"
    filter_report_path = output_root / "item-filter-report.jsonl"
    summary_path = output_root / "build-summary.json"
    started_at = datetime.now(timezone.utc)

    if (
        not args.regen_manifest
        and manifest_path.exists()
        and not _manifest_needs_regen(manifest_path)
    ):
        print(f"Loading existing manifest from {manifest_path} ...")
        manifest_records: list[ManifestRecord] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = ManifestRecord(**json.loads(line))
                if args.item_id is not None and record.item_id != args.item_id:
                    continue
                if not is_supported_item_type(record.type):
                    continue
                manifest_records.append(record)
                if args.limit is not None and len(manifest_records) >= args.limit:
                    break
        manifest_stats: dict[str, int | str] = {
            "source": "cached",
            "skipped_count": 0,
            "missing_icon_count": 0,
            "missing_overview_count": 0,
        }
    else:
        print("Building manifest ...")
        manifest_records, manifest_stats = build_manifest(
            tracker_root=args.tracker_root,
            config_root=args.config_root,
            sync_report_path=args.sync_report,
            limit=args.limit,
            item_id=args.item_id,
            source_version=args.source_version,
        )
        if args.item_id is None:
            _write_jsonl(
                manifest_path, [record.to_dict() for record in manifest_records]
            )

    if args.item_id is not None and not manifest_records:
        print(
            json.dumps({"item_id": args.item_id, "error": "item_not_found"}, indent=2)
        )
        return 1

    extractor = VisionStructuredExtractor(
        StructuredExtractorConfig(
            model_id=args.extraction_model,
            device=args.device,
            quantization=args.quantization,
            inference_batch_size=args.extraction_inference_batch_size,
        )
    )

    if args.item_id is not None:
        return _run_single_item_debug(
            record=manifest_records[0],
            extractor=extractor,
        )

    structured_cache = _load_record_cache(structured_data_path)
    debug_cache = _load_record_cache(structured_debug_path)
    already_done = {
        record.item_id
        for record in manifest_records
        if _is_complete_cached_structured(
            structured_cache.get(record.item_id),
            debug_cache.get(record.item_id),
        )
    }
    if already_done:
        print(f"Resuming: {len(already_done)} items already extracted, skipping them.")

    pending = [
        record for record in manifest_records if record.item_id not in already_done
    ]
    print(f"Items to extract: {len(pending)} / {len(manifest_records)}")

    for batch in _chunk_records(pending, args.batch_size):
        extracted = extractor.extract_records_batch(batch)
        for structured_record, debug_record in extracted:
            structured_cache[structured_record.item_id] = structured_record.to_dict()
            debug_cache[debug_record.item_id] = debug_record.to_dict()
        done_so_far = len(
            [
                record
                for record in manifest_records
                if record.item_id in structured_cache
            ]
        )
        print(f"  extracted {done_so_far}/{len(manifest_records)}", flush=True)

    _cleanup_stale_outputs(output_root)

    structured_rows: list[dict[str, object]] = []
    debug_rows: list[dict[str, object]] = []
    debug_records_for_summary: list[StructuredDebugRecord] = []
    search_document_rows: list[dict[str, object]] = []
    structured_parse_fail_count = 0

    for record in manifest_records:
        structured_payload = structured_cache.get(record.item_id)
        debug_payload = debug_cache.get(record.item_id)
        if structured_payload is None or debug_payload is None:
            continue
        structured_record = _structured_record_from_payload(structured_payload)
        debug_record = _structured_debug_from_payload(debug_payload)
        structured_rows.append(structured_record.to_dict())
        debug_rows.append(debug_record.to_dict())
        debug_records_for_summary.append(debug_record)
        search_document_rows.append(build_document_record(structured_record).to_dict())
        if structured_record.parse_error is not None:
            structured_parse_fail_count += 1

    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(structured_data_path, structured_rows)
    _write_jsonl(structured_debug_path, debug_rows)
    _write_jsonl(search_documents_path, search_document_rows)
    filter_report_rows = _build_filter_report_rows(debug_records_for_summary)
    _write_jsonl(filter_report_path, filter_report_rows)

    finished_at = datetime.now(timezone.utc)
    summary = BuildSummary(
        extraction_model_id=extractor.model_id,
        upstash_embedding_model=DEFAULT_EMBEDDING_MODEL,
        item_count=len(search_document_rows),
        skipped_count=int(manifest_stats["skipped_count"]),
        missing_icon_count=int(manifest_stats["missing_icon_count"]),
        missing_overview_count=int(manifest_stats["missing_overview_count"]),
        structured_parse_fail_count=structured_parse_fail_count,
        build_started_at=started_at.isoformat(),
        build_finished_at=finished_at.isoformat(),
        duration_seconds=(finished_at - started_at).total_seconds(),
        filter_report_path=str(filter_report_path),
    )
    _write_json(summary_path, summary.to_dict())

    print(
        json.dumps({"summary_path": str(summary_path), **summary.to_dict()}, indent=2)
    )
    return 0


def run_refresh_derived(args: argparse.Namespace) -> int:
    from image_search.constants.structured import is_supported_item_type
    from image_search.pipeline.documents import build_document_record

    manifest_path = Path(args.manifest_path)
    structured_path = Path(args.structured_path)
    structured_debug_path = Path(args.structured_debug_path)
    output_root = Path(args.output_root or _default_output_root())
    search_documents_path = output_root / "item-search-documents.jsonl"
    filter_report_path = output_root / "item-filter-report.jsonl"
    summary_path = output_root / "build-summary.json"

    print(f"Loading manifest from {manifest_path} ...")
    manifest_records = [
        ManifestRecord(**row)
        for row in _load_manifest_records(manifest_path)
        if is_supported_item_type(str(row.get("type", "")))
    ]
    print(f"Loading structured data from {structured_path} ...")
    structured_cache = _load_record_cache(structured_path)
    print(f"Loading structured debug from {structured_debug_path} ...")
    debug_cache = _load_record_cache(structured_debug_path)

    summary_payload: dict[str, object] = {}
    if summary_path.exists():
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary_payload = {}
    extraction_model_id = str(
        summary_payload.get("extraction_model_id", DEFAULT_EXTRACTION_MODEL_ID)
    )

    search_document_rows: list[dict[str, object]] = []
    filter_report_rows: list[dict[str, object]] = []
    structured_parse_fail_count = 0

    for record in manifest_records:
        structured_payload = structured_cache.get(record.item_id)
        if structured_payload is None:
            continue
        structured_record = _structured_record_from_payload(structured_payload)
        search_document_rows.append(build_document_record(structured_record).to_dict())
        debug_payload = debug_cache.get(record.item_id)
        if debug_payload is not None:
            debug_record = _structured_debug_from_payload(debug_payload)
            filter_report_rows.extend(_build_filter_report_rows([debug_record]))
        if structured_record.parse_error is not None:
            structured_parse_fail_count += 1

    _write_jsonl(search_documents_path, search_document_rows)
    _write_jsonl(filter_report_path, filter_report_rows)
    refreshed_at = datetime.now(timezone.utc).isoformat()
    _write_json(
        summary_path,
        {
            "extraction_model_id": extraction_model_id,
            "upstash_embedding_model": str(
                summary_payload.get("upstash_embedding_model", DEFAULT_EMBEDDING_MODEL)
            ),
            "item_count": len(search_document_rows),
            "skipped_count": int(summary_payload.get("skipped_count", 0)),
            "missing_icon_count": int(summary_payload.get("missing_icon_count", 0)),
            "missing_overview_count": int(
                summary_payload.get("missing_overview_count", 0)
            ),
            "structured_parse_fail_count": structured_parse_fail_count,
            "build_started_at": str(
                summary_payload.get("build_started_at", refreshed_at)
            ),
            "build_finished_at": refreshed_at,
            "duration_seconds": float(summary_payload.get("duration_seconds", 0)),
            "filter_report_path": str(filter_report_path),
        },
    )

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "structured_path": str(structured_path),
                "structured_debug_path": str(structured_debug_path),
                "search_documents_path": str(search_documents_path),
                "filter_report_path": str(filter_report_path),
                "processed_item_count": len(search_document_rows),
                "structured_parse_fail_count": structured_parse_fail_count,
            },
            indent=2,
        )
    )
    return 0


def run_sync_upstash(args: argparse.Namespace) -> int:
    from image_search.search.upstash import UpstashConfig, create_index, sync_documents

    config = UpstashConfig.from_env(
        rest_url=args.rest_url,
        rest_token=args.rest_token,
        embedding_model=args.embedding_model,
        require_rest=not args.create_index,
    )

    if args.create_index:
        created = create_index(
            name=args.index_name,
            region=args.region,
            dimension_count=args.dimension_count,
            similarity_function=args.metric,
            embedding_model=args.embedding_model,
            sparse_embedding_model=args.sparse_embedding_model,
            index_type=args.index_type,
            config=config,
        )
        print(json.dumps(created, indent=2))
        return 0

    stats = sync_documents(
        documents_path=args.documents_path,
        config=config,
        batch_size=args.batch_size,
    )
    print(json.dumps(stats, indent=2))
    return 0


def run_query_upstash(args: argparse.Namespace) -> int:
    from image_search.models.schemas import QueryRequest
    from image_search.search.upstash import UpstashConfig, query_upstash

    config = UpstashConfig.from_env(
        rest_url=args.rest_url,
        rest_token=args.rest_token,
        embedding_model=args.embedding_model,
    )
    request = QueryRequest(
        q=args.q,
        limit=args.limit,
        item_type=args.item_type or [],
        colors=args.color or [],
    )
    results = query_upstash(request, config)
    print(
        json.dumps(
            [
                {
                    "item_id": result.item_id,
                    "score": result.score,
                    "item_type": result.item_type,
                    "shape": result.shape,
                    "colors": result.colors,
                    "primary_color": result.primary_color,
                    "secondary_color": result.secondary_color,
                    "structured_data": result.structured_data,
                }
                for result in results
            ],
            indent=2,
        )
    )
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    from image_search.search.evaluate import evaluate_queries
    from image_search.search.upstash import UpstashConfig

    config = UpstashConfig.from_env(
        rest_url=args.rest_url,
        rest_token=args.rest_token,
        embedding_model=args.embedding_model,
    )
    report = evaluate_queries(
        queries_path=args.queries,
        metadata_path=args.metadata_path,
        config=config,
        limit=args.limit,
    )
    output_path = Path(args.output)
    _write_json(output_path, report)
    print(json.dumps(report["summary"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infinity Nikki structured image search tooling"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "index", help="Build local structured search artifacts"
    )
    build_parser.add_argument("--item-id", type=int, default=None)
    build_parser.add_argument("--limit", type=int, default=None)
    build_parser.add_argument("--batch-size", type=int, default=10)
    build_parser.add_argument("--extraction-inference-batch-size", type=int, default=2)
    build_parser.add_argument("--device", default="auto")
    build_parser.add_argument(
        "--quantization",
        choices=("none", "8bit", "4bit"),
        default=DEFAULT_MODEL_QUANTIZATION,
    )
    build_parser.add_argument("--extraction-model", default=DEFAULT_EXTRACTION_MODEL_ID)
    build_parser.add_argument("--tracker-root", default=os.getenv("TRACKER_ROOT"))
    build_parser.add_argument(
        "--config-root", default=os.getenv("CONFIG_DECODER_OUTPUT")
    )
    build_parser.add_argument("--sync-report", default=None)
    build_parser.add_argument("--source-version", default=None)
    build_parser.add_argument("--output-root", default=str(_default_output_root()))
    build_parser.add_argument("--manifest-root", default=str(_default_manifest_root()))
    build_parser.add_argument(
        "--regen-manifest",
        action="store_true",
        help="Regenerate manifest even if one already exists",
    )
    build_parser.set_defaults(func=run_build_index)

    refresh_parser = subparsers.add_parser(
        "refresh",
        help="Rebuild search documents from cached manifest and structured data",
    )
    refresh_parser.add_argument(
        "--manifest-path",
        default=str(_default_manifest_root() / "item-manifest.jsonl"),
    )
    refresh_parser.add_argument(
        "--structured-path",
        default=str(_default_output_root() / "item-structured-data.jsonl"),
    )
    refresh_parser.add_argument(
        "--structured-debug-path",
        default=str(_default_output_root() / "item-structured-debug.jsonl"),
    )
    refresh_parser.add_argument("--output-root", default=str(_default_output_root()))
    refresh_parser.set_defaults(func=run_refresh_derived)

    sync_parser = subparsers.add_parser("sync", help="Upsert documents into Upstash")
    sync_parser.add_argument(
        "--documents-path",
        default=str(_default_output_root() / "item-search-documents.jsonl"),
    )
    sync_parser.add_argument("--batch-size", type=int, default=100)
    sync_parser.add_argument("--rest-url", default=None)
    sync_parser.add_argument("--rest-token", default=None)
    sync_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    sync_parser.add_argument(
        "--sparse-embedding-model", default=DEFAULT_SPARSE_EMBEDDING_MODEL
    )
    sync_parser.add_argument("--create-index", action="store_true")
    sync_parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    sync_parser.add_argument("--region", default=DEFAULT_INDEX_REGION)
    sync_parser.add_argument("--index-type", default=DEFAULT_INDEX_TYPE)
    sync_parser.add_argument(
        "--dimension-count", type=int, default=DEFAULT_INDEX_DIMENSION_COUNT
    )
    sync_parser.add_argument("--metric", default=DEFAULT_INDEX_METRIC)
    sync_parser.set_defaults(func=run_sync_upstash)

    query_parser = subparsers.add_parser("query", help="Query Upstash with raw text")
    query_parser.add_argument("--q", required=True)
    query_parser.add_argument("--limit", type=int, default=20)
    query_parser.add_argument(
        "--item-type", "--type", action="append", dest="item_type"
    )
    query_parser.add_argument("--color", action="append")
    query_parser.add_argument("--rest-url", default=None)
    query_parser.add_argument("--rest-token", default=None)
    query_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    query_parser.set_defaults(func=run_query_upstash)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate Upstash search")
    evaluate_parser.add_argument("--queries", required=True)
    evaluate_parser.add_argument(
        "--metadata-path",
        default=str(_default_output_root() / "item-search-documents.jsonl"),
    )
    evaluate_parser.add_argument("--limit", type=int, default=10)
    evaluate_parser.add_argument(
        "--output",
        default=str(_default_output_root() / "evaluation-report.json"),
    )
    evaluate_parser.add_argument("--rest-url", default=None)
    evaluate_parser.add_argument("--rest-token", default=None)
    evaluate_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    evaluate_parser.set_defaults(func=run_evaluate)

    return parser


def main() -> int:
    _load_project_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
