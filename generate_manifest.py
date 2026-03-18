"""generate_manifest.py — standalone manifest generator.

Run directly from the project root:

    python generate_manifest.py

Optional flags:
    --tracker-root   PATH   Path to gongeo.us-nikki-tracker checkout
    --config-root    PATH   Path to gongeo.us-config-decoder cfg/config_output
    --sync-report    PATH   Path to database-sync-report.json
    --source-version STR    Override the source_version field in every record
    --output         PATH   Output JSONL path (default: reports/index/item-manifest.jsonl)
    --limit          N      Stop after N records (useful for testing)
    --type           TYPE   Filter by item type (repeatable, e.g. --type hair --type shoes)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script without the package being installed.
# Inserts the *parent* of this file so that `image_search.*` imports resolve.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from manifest import build_manifest  # noqa: E402  (after sys.path patch)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate base-item manifest JSONL for the image-search index."
    )
    parser.add_argument("--tracker-root", default=None, help="Path to nikki-tracker repo root")
    parser.add_argument("--config-root", default=None, help="Path to config-decoder output root")
    parser.add_argument("--sync-report", default=None, help="Path to database-sync-report.json")
    parser.add_argument("--source-version", default=None, help="Override source_version field")
    parser.add_argument(
        "--output",
        default=str(_HERE / "reports" / "index" / "item-manifest.jsonl"),
        help="Destination JSONL file (default: reports/item-manifest.jsonl)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of records (for testing)"
    )
    parser.add_argument(
        "--type", action="append", dest="types", metavar="TYPE",
        help="Filter by item type (repeatable, e.g. --type hair --type shoes)"
    )
    args = parser.parse_args()

    print("Building manifest …")
    records, stats = build_manifest(
        tracker_root=args.tracker_root,
        config_root=args.config_root,
        sync_report_path=args.sync_report,
        limit=args.limit,
        source_version=args.source_version,
    )

    if args.types:
        type_filter = set(args.types)
        records = [r for r in records if r.type in type_filter]

    output_path = Path(args.output)
    _write_jsonl(output_path, [r.to_dict() for r in records])

    summary = {
        "output": str(output_path),
        "record_count": len(records),
        **({"type_filter": args.types} if args.types else {}),
        **{k: v for k, v in stats.items()},
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
