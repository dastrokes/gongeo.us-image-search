# Image Search

Offline caption generation and Upstash-backed semantic item search for Infinity Nikki items.

## What it builds

`build-index` reads:

- `reports/database-sync-report.json`
- tracker item names from `gongeo.us-nikki-tracker/app/locales/en/item.json`
- tracker images from `gongeo.us-nikki-tracker/public/images/items`
- config-decoder data from `gongeo.us-config-decoder/cfg/config_output`

It produces:

- `reports/index/item-manifest.jsonl`
- `reports/index/item-metadata.parquet`
- `reports/index/item-documents.jsonl`
- `reports/index/item-captions-debug.jsonl`
- `reports/index/build-summary.json`

## Install

```bash
pip install -r requirements.txt
```

## Environment

Set these in the processor `.env` or shell:

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
python cli.py sync-upstash
python cli.py query-upstash --q "blue floral headwear" --type headwear
python cli.py evaluate --queries path/to/queries.jsonl
```

## Notes

- Florence captions are image-grounding for search text. Upstash handles text embedding and retrieval.
- `style_key`, score tags, and scoring props are excluded from indexed text.
- Explicit filters are passed to Upstash metadata filtering. Natural-language-to-filter parsing is not part of v1.
- Indexed metadata now separates `dominant_colors` from `accent_colors`; `--color` filters only match `dominant_colors`.
