# Image Search

Offline structured extraction for Infinity Nikki items.

## Layout

Source lives under `src/` (added to `sys.path` by the root wrappers):

- `src/constants/` — settings, structured field definitions, and extraction prompts
- `src/models/` — dataclass schemas for manifests, structured records, and search documents
- `src/pipeline/` — manifest ingestion, vision extraction, and search-document generation

Root `cli.py` and `manifest.py` are thin compatibility entry-points that prepend `src/` to `sys.path` before delegating.

## What It Builds

`index` reads:

- `reports/database-sync-report.json` — synced item list and source version
- `gongeo.us-config-decoder/cfg/config_output/item/TbItem.json` — item minor-type mapping
- `gongeo.us-config-decoder/cfg/config_output/clothes/TbClothesMinorTypeInfo.json` — minor-type labels
- `gongeo.us-nikki-tracker/data/item-search/generated/image-search-taxonomy.json` — canonical tracker-owned taxonomy/schema export
- `gongeo.us-nikki-tracker/data/item-search/generated/overrides.json` — curated per-item maintainer overrides
- `manifest/item-attributes.jsonl` — optional canonical item-attributes snapshot used to skip already-published items when rebuilding the manifest
- `gongeo.us-nikki-tracker/public/images/items/` — overview and icon images

It produces:

- `manifest/item-manifest.jsonl`
- `index/item-structured-raw.jsonl`
- `index/item-structured-data.jsonl`
- `index/item-attributes.jsonl`
- `index/item-structured-debug.jsonl`
- `index/item-search-report.jsonl`
- `index/build-summary.json`

## Install

```bash
pip install -r requirements.txt
```

Default extraction uses `Qwen/Qwen3-VL-4B-Instruct` with 4-bit quantization (requires CUDA).

## Environment

Set these in `.env` or your shell:

- `TRACKER_ROOT` — path to `gongeo.us-nikki-tracker` repo
- `CONFIG_DECODER_OUTPUT` — path to `gongeo.us-config-decoder/cfg/config_output`

## Commands

Keep dev runs capped at `--limit 10` unless intentionally wider.

```bash
# Extract structured data and build canonical item attributes
python cli.py index --limit 10
python cli.py index --item-id 1020123456   # single-item debug
python cli.py index --type dresses --type outerwear
python cli.py index --item-ids 1020100001 1020100002

# Rebuild canonical item attributes from cached extraction (no re-extraction)
python cli.py refresh
python cli.py refresh --type dresses
python cli.py refresh --item-id 1020100001

# Regenerate the base-item manifest only
python manifest.py
```

If you want manifest generation to skip items that are already in the current published Supabase dataset, copy the tracker-generated local mirror from `gongeo.us-nikki-tracker/data/item-search/generated/supabase/item-attributes.jsonl` into `manifest/item-attributes.jsonl` before running `python manifest.py` or `python cli.py index --regen-manifest`.

`index --item-id <ID>` prints one JSON bundle: prompt, raw model response, parsed payload, and normalized structured output.

`refresh` rebuilds `index/item-attributes.jsonl`, `index/item-search-report.jsonl`, and `index/build-summary.json` from cached normalized structured data plus tracker overrides — use when override application or downstream normalization changes but re-extraction is not needed.

## Notes

- Imports use bare module names (`from constants.settings import ...`) because `src/` is prepended to `sys.path`.
- The build now materializes three layers per item: raw extracted payload, normalized structured payload, and final canonical `item-attributes` payload after tracker overrides are applied.
- Extraction uses item-type-specific prompts; only fields relevant to the slot are included.
- `item-attributes.jsonl` is the only persisted final artifact.
- `metadata` in `item-attributes.jsonl` contains only actual tag fields and does not repeat `item_id`, `item_type`, `slot`, `category`, or `subcategory`.
- Search documents are derived from canonical item-attributes rows, not stored as a separate final artifact.
- `manifest/item-attributes.jsonl` uses the same row shape as the canonical item-attributes artifact, so the tracker Supabase mirror can be copied there directly.
