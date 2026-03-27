from __future__ import annotations

import argparse
import json
from pathlib import Path

from constants.settings import PROJECT_ROOT
from constants.structured import expand_item_type_filters
from pipeline.manifest import build_manifest


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate base-item manifest JSONL for the image-search index."
    )
    parser.add_argument(
        "--tracker-root", default=None, help="Path to nikki-tracker repo root"
    )
    parser.add_argument(
        "--config-root", default=None, help="Path to config-decoder output root"
    )
    parser.add_argument(
        "--sync-report", default=None, help="Path to database-sync-report.json"
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "manifest" / "item-manifest.jsonl"),
        help="Destination JSONL file",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of records"
    )
    parser.add_argument(
        "--type",
        action="append",
        dest="types",
        metavar="TYPE",
        help="Filter by item type (repeatable, supports special values clothing/accessories)",
    )
    parser.add_argument(
        "--clothing",
        action="store_true",
        help="Include hair, tops, bottoms, outerwear, socks, and shoes",
    )
    parser.add_argument(
        "--accessories",
        action="store_true",
        help="Include all supported item types not covered by --clothing",
    )
    parser.add_argument(
        "--indexed-path",
        default=str(PROJECT_ROOT / "manifest" / "item-indexed.jsonl"),
        help="JSONL manifest of already-indexed records to skip if present",
    )
    parser.add_argument(
        "--include-indexed",
        action="store_true",
        help="Include records even if they already appear in the indexed manifest",
    )
    args = parser.parse_args()

    requested_types = list(args.types or [])
    if args.clothing:
        requested_types.append("clothing")
    if args.accessories:
        requested_types.append("accessories")
    try:
        type_filter = expand_item_type_filters(requested_types)
    except ValueError as exc:
        parser.error(str(exc))

    print("Building manifest ...")
    records, stats = build_manifest(
        tracker_root=args.tracker_root,
        config_root=args.config_root,
        sync_report_path=args.sync_report,
        limit=args.limit,
        item_types=type_filter or None,
        indexed_manifest_path=args.indexed_path,
        skip_indexed=not args.include_indexed,
    )

    output_path = Path(args.output)
    _write_jsonl(output_path, [record.to_dict() for record in records])

    summary = {
        "output": str(output_path),
        "record_count": len(records),
        **({"type_filter": sorted(type_filter)} if type_filter else {}),
        **stats,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
