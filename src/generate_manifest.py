from __future__ import annotations

import argparse
import json
from pathlib import Path

from constants.settings import PROJECT_ROOT
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
        "--source-version", default=None, help="Override source_version field"
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
        help="Filter by item type (repeatable, e.g. --type hair --type shoes)",
    )
    args = parser.parse_args()

    print("Building manifest ...")
    records, stats = build_manifest(
        tracker_root=args.tracker_root,
        config_root=args.config_root,
        sync_report_path=args.sync_report,
        limit=args.limit,
        source_version=args.source_version,
    )

    if args.types:
        type_filter = set(args.types)
        records = [record for record in records if record.type in type_filter]

    output_path = Path(args.output)
    _write_jsonl(output_path, [record.to_dict() for record in records])

    summary = {
        "output": str(output_path),
        "record_count": len(records),
        **({"type_filter": args.types} if args.types else {}),
        **stats,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
