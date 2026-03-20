from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from image_search.constants.settings import (
    DEFAULT_CAPTION_MODEL_ID,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_DIMENSION_COUNT,
    DEFAULT_INDEX_METRIC,
    DEFAULT_INDEX_NAME,
    DEFAULT_INDEX_REGION,
    DEFAULT_INDEX_TYPE,
    DEFAULT_SPARSE_EMBEDDING_MODEL,
    PROJECT_ROOT,
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
    return PROJECT_ROOT / "reports" / "index"


def _cleanup_stale_outputs(output_root: Path) -> None:
    for filename in (
        "item-official-metadata.jsonl",
        "item-documents.jsonl",
        "item-metadata.parquet",
        "item-unmapped-terms.jsonl",
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
    return not required_fields.issubset(payload) or bool(deprecated_fields.intersection(payload))


def _load_caption_cache(captions_path: Path) -> dict[int, dict]:
    cache: dict[int, dict] = {}
    if not captions_path.exists():
        return cache
    with captions_path.open("r", encoding="utf-8") as handle:
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


def _is_complete_cached_caption(caption: dict) -> bool:
    failed_modalities = caption.get("failed_modalities") or []
    visual = str(caption.get("visual", "")).strip()
    item_type = ""

    def _is_type_only(value: object) -> bool:
        if not item_type:
            return False
        text = str(value or "").strip()
        if not text:
            return False
        parts = [part.strip() for part in text.split(",") if part.strip()]
        return bool(parts) and all(part == item_type for part in parts)

    if failed_modalities:
        return False
    if not visual:
        return False
    if _is_type_only(caption.get("icon_caption")) or _is_type_only(caption.get("overview_caption")):
        return False
    return not _is_type_only(visual)


def run_build_index(args: argparse.Namespace) -> int:
    from image_search.models.schemas import (
        BuildSummary,
        CaptionRecord,
        ManifestRecord,
    )
    from image_search.pipeline.captioning import CaptionerConfig, VisionCaptioner
    from image_search.pipeline.color_tags import tag_item_colors
    from image_search.pipeline.documents import build_document_record
    from image_search.pipeline.manifest import build_manifest
    from image_search.pipeline.taxonomy import (
        build_item_input,
        build_metadata_record,
        build_review_records,
        build_structured_candidates,
        build_tag_assignments,
        build_taxonomy_concepts,
        build_unmapped_term_records,
        build_visual_features,
    )

    output_root = Path(args.output_root or _default_output_root())
    manifest_path = output_root / "item-manifest.jsonl"
    captions_path = output_root / "item-captions-debug.jsonl"
    taxonomy_concepts_path = output_root / "taxonomy-concepts.jsonl"
    visual_features_path = output_root / "item-visual-features.jsonl"
    structured_candidates_path = output_root / "item-structured-candidates.jsonl"
    assignments_path = output_root / "item-tag-assignments.jsonl"
    review_path = output_root / "item-review-queue.jsonl"
    unmapped_terms_path = output_root / "item-unmapped-terms.jsonl"
    search_documents_path = output_root / "item-search-documents.jsonl"
    summary_path = output_root / "build-summary.json"
    started_at = datetime.now(timezone.utc)

    if not args.regen_manifest and manifest_path.exists() and not _manifest_needs_regen(manifest_path):
        print(f"Loading existing manifest from {manifest_path} ...")
        manifest_records: list[ManifestRecord] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                manifest_records.append(ManifestRecord(**json.loads(line)))
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
            source_version=args.source_version,
        )
        _write_jsonl(manifest_path, [record.to_dict() for record in manifest_records])

    caption_cache = _load_caption_cache(captions_path)
    already_done = {
        item_id
        for item_id in set(caption_cache)
        if _is_complete_cached_caption(caption_cache[item_id])
    }
    if already_done:
        print(f"Resuming: {len(already_done)} items already captioned, skipping them.")

    pending = [record for record in manifest_records if record.item_id not in already_done]
    print(f"Items to caption: {len(pending)} / {len(manifest_records)}")

    captioner = VisionCaptioner(
        CaptionerConfig(
            model_id=args.caption_model,
            device=args.device,
            device_map=args.device_map,
            dtype=args.dtype,
            batch_size=args.batch_size,
            inference_batch_size=args.caption_inference_batch_size,
        )
    )

    caption_fail_count = 0
    output_root.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_outputs(output_root)

    batch_size = captioner.config.batch_size
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        captions = captioner.caption_records_batch(batch)

        for record, caption in zip(batch, captions):
            if caption.failed_modalities:
                caption_fail_count += 1
            caption_cache[record.item_id] = caption.to_dict()

        done_so_far = len(already_done) + batch_start + len(batch)
        print(f"  captioned {done_so_far}/{len(manifest_records)}", flush=True)

    caption_rows: list[dict[str, object]] = []
    visual_feature_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    unmapped_term_rows: list[dict[str, object]] = []
    search_document_rows: list[dict[str, object]] = []
    concept_rows = [record.to_dict() for record in build_taxonomy_concepts()]
    accepted_tag_count = 0
    review_tag_count = 0
    suppressed_tag_count = 0

    for record in manifest_records:
        caption_payload = caption_cache.get(record.item_id)
        if caption_payload is None:
            continue
        caption = CaptionRecord(**caption_payload)
        color_tags = tag_item_colors(
            record.overview_path,
            record.icon_path,
            record.type,
            caption.visual,
        )
        item_input = build_item_input(record)
        visual_features = build_visual_features(
            item_id=record.item_id,
            item_type=record.type,
            caption_icon=caption.icon_caption,
            caption_overview=caption.overview_caption,
            caption_visual=caption.visual,
            color_tags=color_tags,
        )
        structured_candidates = build_structured_candidates(
            item_input=item_input,
            visual_features=visual_features,
        )
        assignments = build_tag_assignments(
            item_input=item_input,
            visual_features=visual_features,
            model_version=captioner.model_id,
            candidates=structured_candidates,
        )
        reviews = build_review_records(assignments)
        unmapped_terms = build_unmapped_term_records(item_input, visual_features)
        metadata = build_metadata_record(item_input, visual_features, assignments)
        document = build_document_record(metadata, visual_features, assignments)

        caption_rows.append(caption.to_dict())
        visual_feature_rows.append(visual_features.to_dict())
        candidate_rows.extend(candidate.to_dict() for candidate in structured_candidates)
        assignment_rows.extend(assignment.to_dict() for assignment in assignments)
        review_rows.extend(review.to_dict() for review in reviews)
        unmapped_term_rows.extend(record.to_dict() for record in unmapped_terms)
        search_document_rows.append(document.to_dict())

        for assignment in assignments:
            if assignment.status == "accepted":
                accepted_tag_count += 1
            elif assignment.status == "review":
                review_tag_count += 1
            elif assignment.status == "suppressed":
                suppressed_tag_count += 1

    _write_jsonl(captions_path, caption_rows)
    _write_jsonl(taxonomy_concepts_path, concept_rows)
    _write_jsonl(visual_features_path, visual_feature_rows)
    _write_jsonl(structured_candidates_path, candidate_rows)
    _write_jsonl(assignments_path, assignment_rows)
    _write_jsonl(review_path, review_rows)
    _write_jsonl(unmapped_terms_path, unmapped_term_rows)
    _write_jsonl(search_documents_path, search_document_rows)

    finished_at = datetime.now(timezone.utc)
    summary = BuildSummary(
        caption_model_id=captioner.model_id,
        upstash_embedding_model=DEFAULT_EMBEDDING_MODEL,
        item_count=len(search_document_rows),
        skipped_count=int(manifest_stats["skipped_count"]),
        missing_icon_count=int(manifest_stats["missing_icon_count"]),
        missing_overview_count=int(manifest_stats["missing_overview_count"]),
        caption_fail_count=caption_fail_count,
        taxonomy_concept_count=len(concept_rows),
        accepted_tag_count=accepted_tag_count,
        review_tag_count=review_tag_count,
        suppressed_tag_count=suppressed_tag_count,
        unmapped_term_count=len(unmapped_term_rows),
        build_started_at=started_at.isoformat(),
        build_finished_at=finished_at.isoformat(),
        duration_seconds=(finished_at - started_at).total_seconds(),
    )
    _write_json(summary_path, summary.to_dict())

    print(json.dumps({"summary_path": str(summary_path), **summary.to_dict()}, indent=2))
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
        facets=args.facet or [],
    )
    results = query_upstash(request, config)
    print(
        json.dumps(
            [
                {
                    "item_id": result.item_id,
                    "score": result.score,
                    "item_type": result.item_type,
                    "dominant_colors": result.dominant_colors,
                    "accent_colors": result.accent_colors,
                    "accepted_facets": result.accepted_facets,
                    "search_terms": result.search_terms,
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
    parser = argparse.ArgumentParser(description="Infinity Nikki Upstash search tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-index", help="Build local search artifacts")
    build_parser.add_argument("--limit", type=int, default=None)
    build_parser.add_argument("--batch-size", type=int, default=8)
    build_parser.add_argument("--caption-inference-batch-size", type=int, default=2)
    build_parser.add_argument("--device", default="auto")
    build_parser.add_argument("--device-map", choices=("auto", "none"), default="auto")
    build_parser.add_argument("--dtype", default="auto")
    build_parser.add_argument("--caption-model", default=DEFAULT_CAPTION_MODEL_ID)
    build_parser.add_argument("--tracker-root", default=os.getenv("TRACKER_ROOT"))
    build_parser.add_argument("--config-root", default=os.getenv("CONFIG_DECODER_OUTPUT"))
    build_parser.add_argument("--sync-report", default=None)
    build_parser.add_argument("--source-version", default=None)
    build_parser.add_argument("--output-root", default=str(_default_output_root()))
    build_parser.add_argument(
        "--regen-manifest",
        action="store_true",
        help="Regenerate manifest even if one already exists",
    )
    build_parser.set_defaults(func=run_build_index)

    sync_parser = subparsers.add_parser("sync-upstash", help="Upsert documents into Upstash")
    sync_parser.add_argument(
        "--documents-path",
        default=str(_default_output_root() / "item-search-documents.jsonl"),
    )
    sync_parser.add_argument("--batch-size", type=int, default=100)
    sync_parser.add_argument("--rest-url", default=None)
    sync_parser.add_argument("--rest-token", default=None)
    sync_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    sync_parser.add_argument("--sparse-embedding-model", default=DEFAULT_SPARSE_EMBEDDING_MODEL)
    sync_parser.add_argument("--create-index", action="store_true")
    sync_parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    sync_parser.add_argument("--region", default=DEFAULT_INDEX_REGION)
    sync_parser.add_argument("--index-type", default=DEFAULT_INDEX_TYPE)
    sync_parser.add_argument("--dimension-count", type=int, default=DEFAULT_INDEX_DIMENSION_COUNT)
    sync_parser.add_argument("--metric", default=DEFAULT_INDEX_METRIC)
    sync_parser.set_defaults(func=run_sync_upstash)

    query_parser = subparsers.add_parser("query-upstash", help="Query Upstash with raw text")
    query_parser.add_argument("--q", required=True)
    query_parser.add_argument("--limit", type=int, default=20)
    query_parser.add_argument("--item-type", "--type", action="append", dest="item_type")
    query_parser.add_argument("--color", action="append")
    query_parser.add_argument("--facet", action="append")
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
