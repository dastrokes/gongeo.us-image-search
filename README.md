# Image Search

Offline structured extraction and Upstash-backed semantic item search for Infinity Nikki items.

## Layout

Source lives under `src/` (added to `sys.path` by the root wrappers):

- `src/constants/` — settings, structured field definitions, and extraction prompts
- `src/models/` — dataclass schemas for manifests, structured records, and search documents
- `src/pipeline/` — manifest ingestion, vision extraction, and search-document generation
- `src/search/` — Upstash sync and search evaluation

Root `cli.py` and `manifest.py` are thin compatibility entry-points that prepend `src/` to `sys.path` before delegating.

## What It Builds

`index` reads:

- `reports/database-sync-report.json` — synced item list and source version
- `gongeo.us-config-decoder/cfg/config_output/item/TbItem.json` — item minor-type mapping
- `gongeo.us-config-decoder/cfg/config_output/clothes/TbClothesMinorTypeInfo.json` — minor-type labels
- `gongeo.us-nikki-tracker/public/images/items/` — overview and icon images

It produces:

- `manifest/item-manifest.jsonl`
- `index/item-structured-data.jsonl`
- `index/item-structured-debug.jsonl`
- `index/item-search-documents.jsonl`
- `index/item-filter-report.jsonl`
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
- `UPSTASH_VECTOR_REST_URL`
- `UPSTASH_VECTOR_REST_TOKEN`

Optional for index creation:

- `UPSTASH_EMAIL`
- `UPSTASH_API_KEY`
- `UPSTASH_VECTOR_MANAGEMENT_URL`

## Commands

Keep dev runs capped at `--limit 10` unless intentionally wider.

```bash
# Extract structured data and build search documents
python cli.py index --limit 10
python cli.py index --item-id 1020123456   # single-item debug

# Rebuild search documents from cached extraction (no re-extraction)
python cli.py refresh

# Upload search documents to Upstash Vector
python cli.py sync

# Query Upstash interactively
python cli.py query --q "floral lace dress" --limit 20

# Evaluate search quality against a query set
python cli.py evaluate --queries path/to/queries.jsonl

# Regenerate the base-item manifest only
python manifest.py
```

`index --item-id <ID>` prints one JSON bundle: prompt, raw model response, parsed payload, and normalized structured output.

`refresh` rebuilds `index/item-search-documents.jsonl` and `index/build-summary.json` from the cached manifest and structured data — use when document formatting changes but re-extraction is not needed.

## Notes

- Imports use bare module names (`from constants.settings import ...`) because `src/` is prepended to `sys.path`.
- The canonical output is one normalized JSON object per item keyed by `item_id`.
- Extraction uses item-type-specific prompts; only fields relevant to the slot are included.
- Search documents are derived from normalized structured JSON, not captions or taxonomy assignments.
- Upstash filtering supports item type and color metadata.
