from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_SPARSE_EMBEDDING_MODEL = "BM25"
DEFAULT_INDEX_TYPE = "HYBRID"


def _load_project_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env", override=False)


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
    return Path(__file__).resolve().parent / "reports" / "index"


def _load_caption_cache(captions_path: Path) -> dict[int, dict]:
    """Load previously completed captions keyed by item_id."""
    cache: dict[int, dict] = {}
    if not captions_path.exists():
        return cache
    with captions_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                cache[int(d["item_id"])] = d
            except Exception:
                pass
    return cache


def _load_document_cache(documents_path: Path) -> dict[int, dict]:
    """Load previously completed documents keyed by item_id."""
    cache: dict[int, dict] = {}
    if not documents_path.exists():
        return cache
    with documents_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                raw_id = d.get("item_id", d.get("id"))
                if raw_id is None:
                    raw_id = (d.get("metadata") or {}).get("item_id")
                if raw_id is None:
                    continue
                cache[int(raw_id)] = d
            except Exception:
                pass
    return cache


def _is_complete_cached_record(caption: dict, document: dict) -> bool:
    failed_modalities = caption.get("failed_modalities") or []
    visual = str(caption.get("visual", "")).strip()
    data = str(document.get("data", "")).strip()
    item_type = str((document.get("metadata") or {}).get("type", "")).strip()

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
    if _is_type_only(caption.get("icon_caption")):
        return False
    if _is_type_only(caption.get("overview_caption")):
        return False
    if _is_type_only(visual):
        return False
    if not visual:
        return False
    if not data:
        return False
    return True


def run_build_index(args: argparse.Namespace) -> int:
    import pandas as pd

    from caption import CaptionerConfig, FlorenceCaptioner
    from color_tags import tag_item_colors
    from documents import build_document_record
    from manifest import build_manifest
    from schemas import BuildSummary, CaptionRecord, ManifestRecord, MetadataRecord

    output_root = Path(args.output_root or _default_output_root())
    manifest_path = output_root / "item-manifest.jsonl"
    captions_path = output_root / "item-captions-debug.jsonl"
    documents_path = output_root / "item-documents.jsonl"
    metadata_path = output_root / "item-metadata.parquet"
    summary_path = output_root / "build-summary.json"
    started_at = datetime.now(timezone.utc)

    if not args.regen_manifest and manifest_path.exists():
        print(f"Loading existing manifest from {manifest_path} …")
        manifest_records: list[ManifestRecord] = []
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                manifest_records.append(ManifestRecord(**d))
                if args.limit is not None and len(manifest_records) >= args.limit:
                    break
        manifest_stats: dict[str, int | str] = {
            "source": "cached",
            "skipped_count": 0,
            "missing_icon_count": 0,
            "missing_overview_count": 0,
        }
    else:
        print("Building manifest …")
        manifest_records, manifest_stats = build_manifest(
            tracker_root=args.tracker_root,
            config_root=args.config_root,
            sync_report_path=args.sync_report,
            limit=args.limit,
            source_version=args.source_version,
        )
        _write_jsonl(manifest_path, [r.to_dict() for r in manifest_records])

    # --- Resume support: load already-completed captions and documents ---
    caption_cache = _load_caption_cache(captions_path)
    document_cache = _load_document_cache(documents_path)
    already_done = {
        item_id
        for item_id in (set(caption_cache.keys()) & set(document_cache.keys()))
        if _is_complete_cached_record(caption_cache[item_id], document_cache[item_id])
    }
    if already_done:
        print(f"Resuming: {len(already_done)} items already captioned, skipping them.")

    pending = [r for r in manifest_records if r.item_id not in already_done]
    print(f"Items to caption: {len(pending)} / {len(manifest_records)}")

    captioner = FlorenceCaptioner(
        CaptionerConfig(
            model_id=args.caption_model,
            fallback_model_id=args.fallback_caption_model,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
        )
    )

    caption_fail_count = 0
    output_root.mkdir(parents=True, exist_ok=True)

    # Open caption and document files in append mode so each batch is durable
    with captions_path.open("a", encoding="utf-8") as captions_fh, \
         documents_path.open("a", encoding="utf-8") as documents_fh:

        batch_size = captioner.config.batch_size
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            captions = captioner.caption_records_batch(batch)

            for record, caption in zip(batch, captions):
                color_tags = tag_item_colors(
                    record.overview_path, record.icon_path, record.type, caption.visual
                )

                if caption.failed_modalities:
                    caption_fail_count += 1

                metadata = MetadataRecord(
                    item_id=record.item_id,
                    name=record.name,
                    type=record.type,
                    quality=record.quality,
                    obtain_type=record.obtain_type,
                    icon_path=record.icon_path,
                    overview_path=record.overview_path,
                    dominant_colors=color_tags.dominant_colors,
                    accent_colors=color_tags.accent_colors,
                    has_icon=record.has_icon,
                    has_overview=record.has_overview,
                    source_version=record.source_version,
                )

                document = build_document_record(metadata, caption)

                # Write immediately so progress survives a crash
                captions_fh.write(json.dumps(caption.to_dict(), ensure_ascii=False) + "\n")
                captions_fh.flush()
                documents_fh.write(json.dumps(document.to_dict(), ensure_ascii=False) + "\n")
                documents_fh.flush()

                caption_cache[record.item_id] = caption.to_dict()
                document_cache[record.item_id] = document.to_dict()

            done_so_far = len(already_done) + batch_start + len(batch)
            print(f"  captioned {done_so_far}/{len(manifest_records)}", flush=True)

    # Rebuild full metadata parquet from all captions (cached + new)
    metadata_records: list[MetadataRecord] = []
    document_records: list[dict[str, object]] = []
    for record in manifest_records:
        cap_dict = caption_cache.get(record.item_id)
        if cap_dict is None:
            continue
        caption = CaptionRecord(**cap_dict)
        color_tags = tag_item_colors(
            record.overview_path, record.icon_path, record.type, caption.visual
        )
        metadata_records.append(MetadataRecord(
            item_id=record.item_id,
            name=record.name,
            type=record.type,
            quality=record.quality,
            obtain_type=record.obtain_type,
            icon_path=record.icon_path,
            overview_path=record.overview_path,
            dominant_colors=color_tags.dominant_colors,
            accent_colors=color_tags.accent_colors,
            has_icon=record.has_icon,
            has_overview=record.has_overview,
            source_version=record.source_version,
        ))
        document_records.append(build_document_record(metadata_records[-1], caption).to_dict())

    pd.DataFrame([r.to_dict() for r in metadata_records]).to_parquet(metadata_path, index=False)
    _write_jsonl(documents_path, document_records)

    finished_at = datetime.now(timezone.utc)
    summary = BuildSummary(
        caption_model_id=captioner.model_id,
        upstash_embedding_model=DEFAULT_EMBEDDING_MODEL,
        item_count=len(metadata_records),
        skipped_count=int(manifest_stats["skipped_count"]),
        missing_icon_count=int(manifest_stats["missing_icon_count"]),
        missing_overview_count=int(manifest_stats["missing_overview_count"]),
        caption_fail_count=caption_fail_count,
        build_started_at=started_at.isoformat(),
        build_finished_at=finished_at.isoformat(),
        duration_seconds=(finished_at - started_at).total_seconds(),
    )
    _write_json(summary_path, summary.to_dict())

    print(json.dumps({"summary_path": str(summary_path), **summary.to_dict()}, indent=2))
    return 0


def run_sync_upstash(args: argparse.Namespace) -> int:
    from upstash_sync import UpstashConfig, create_index, sync_documents

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
    from schemas import QueryRequest
    from upstash_sync import UpstashConfig, query_upstash

    config = UpstashConfig.from_env(
        rest_url=args.rest_url,
        rest_token=args.rest_token,
        embedding_model=args.embedding_model,
    )
    request = QueryRequest(
        q=args.q,
        limit=args.limit,
        type=args.type or [],
        quality=args.quality or [],
        obtain_type=args.obtain_type or [],
        colors=args.color or [],
    )
    results = query_upstash(request, config)
    print(
        json.dumps(
            [
                {
                    "item_id": result.item_id,
                    "score": result.score,
                    "name": result.name,
                    "type": result.type,
                    "quality": result.quality,
                    "obtain_type": result.obtain_type,
                    "dominant_colors": result.dominant_colors,
                    "accent_colors": result.accent_colors,
                }
                for result in results
            ],
            indent=2,
        )
    )
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    from evaluate import evaluate_queries
    from upstash_sync import UpstashConfig

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
    build_parser.add_argument("--device", default="auto")
    build_parser.add_argument("--dtype", default="auto")
    build_parser.add_argument("--caption-model", default="microsoft/Florence-2-large")
    build_parser.add_argument(
        "--fallback-caption-model", default="microsoft/Florence-2-base"
    )
    build_parser.add_argument("--tracker-root", default=os.getenv("TRACKER_ROOT"))
    build_parser.add_argument("--config-root", default=os.getenv("CONFIG_DECODER_OUTPUT"))
    build_parser.add_argument("--sync-report", default=None)
    build_parser.add_argument("--source-version", default=None)
    build_parser.add_argument("--output-root", default=str(_default_output_root()))
    build_parser.add_argument("--regen-manifest", action="store_true", help="Regenerate manifest even if one already exists")
    build_parser.set_defaults(func=run_build_index)

    sync_parser = subparsers.add_parser("sync-upstash", help="Upsert documents into Upstash")
    sync_parser.add_argument(
        "--documents-path",
        default=str(_default_output_root() / f"item-documents.jsonl"),
    )
    sync_parser.add_argument("--batch-size", type=int, default=100)
    sync_parser.add_argument("--rest-url", default=None)
    sync_parser.add_argument("--rest-token", default=None)
    sync_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    sync_parser.add_argument("--sparse-embedding-model", default=DEFAULT_SPARSE_EMBEDDING_MODEL)
    sync_parser.add_argument("--create-index", action="store_true")
    sync_parser.add_argument("--index-name", default="infinity-nikki-items")
    sync_parser.add_argument("--region", default="us-east-1")
    sync_parser.add_argument("--index-type", default=DEFAULT_INDEX_TYPE)
    sync_parser.add_argument("--dimension-count", type=int, default=1024)
    sync_parser.add_argument("--metric", default="COSINE")
    sync_parser.set_defaults(func=run_sync_upstash)

    query_parser = subparsers.add_parser("query-upstash", help="Query Upstash with raw text")
    query_parser.add_argument("--q", required=True)
    query_parser.add_argument("--limit", type=int, default=20)
    query_parser.add_argument("--type", action="append")
    query_parser.add_argument("--quality", type=int, action="append")
    query_parser.add_argument("--obtain-type", type=int, action="append")
    query_parser.add_argument("--color", action="append")
    query_parser.add_argument("--rest-url", default=None)
    query_parser.add_argument("--rest-token", default=None)
    query_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    query_parser.set_defaults(func=run_query_upstash)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate Upstash search")
    evaluate_parser.add_argument("--queries", required=True)
    evaluate_parser.add_argument(
        "--metadata-path",
        default=str(_default_output_root() / f"item-metadata.parquet"),
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
