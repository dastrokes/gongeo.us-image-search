# Image Search

Offline structured extraction and Upstash-backed semantic item search for Infinity Nikki items.

## Layout

Core code lives under `image_search/`:

- `image_search/constants/` for settings and structured shape definitions
- `image_search/models/` for manifest, structured output, and search schemas
- `image_search/pipeline/` for manifest ingestion, strict JSON extraction, and search-document generation
- `image_search/search/` for Upstash sync and evaluation

Root `cli.py` and `generate_manifest.py` stay as thin compatibility entrypoints.

## What It Builds

`build-index` reads:

- `reports/database-sync-report.json`
- tracker item names from `gongeo.us-nikki-tracker/app/locales/en/item.json`
- tracker images from `gongeo.us-nikki-tracker/public/images/items`
- config-decoder data from `gongeo.us-config-decoder/cfg/config_output`

It produces:

- `manifest/item-manifest.jsonl`
- `index/item-structured-data.jsonl`
- `index/item-structured-debug.jsonl`
- `index/item-search-documents.jsonl`
- `index/build-summary.json`

## Install

```bash
pip install -r requirements.txt
```

Default extraction uses `Qwen/Qwen3-VL-4B-Instruct`. The pinned `transformers==4.57.6` dependency includes `Qwen3VLForConditionalGeneration`.

## Environment

Set these in `.env` or your shell:

- `TRACKER_ROOT`
- `CONFIG_DECODER_OUTPUT`
- `UPSTASH_VECTOR_REST_URL`
- `UPSTASH_VECTOR_REST_TOKEN`

Optional for index creation:

- `UPSTASH_EMAIL`
- `UPSTASH_API_KEY`
- `UPSTASH_VECTOR_MANAGEMENT_URL`

## Commands

Development should stay capped at `--limit 10` unless intentionally widened.

```bash
python cli.py build-index --limit 10
python cli.py build-index --item-id 123456
python cli.py refresh-derived
python cli.py sync-upstash
python cli.py query-upstash --q "blue floral headwear" --item-type headwear
python cli.py evaluate --queries path/to/queries.jsonl
```

`build-index --item-id <ID>` runs a fast single-item debug path and prints one JSON bundle with the strict prompt, raw response, parsed payload, and normalized structured output.

`refresh-derived` rebuilds:

- `index/item-search-documents.jsonl`
- `index/build-summary.json`

from:

- `manifest/item-manifest.jsonl`
- `index/item-structured-data.jsonl`

Use it when search-document formatting changes and you want to refresh derived outputs without rerunning extraction.

## Notes

- The canonical output contract is one normalized JSON object per item, keyed by `item_id`.
- The extractor uses shape-specific prompts and shape-specific response templates with only relevant fields.
- Model-authored `primary_color` and `secondary_color` are part of the canonical structured payload.
- Search documents are derived directly from normalized structured JSON, not caption terms or taxonomy assignments.
- Upstash filtering currently supports item type and color metadata only.
