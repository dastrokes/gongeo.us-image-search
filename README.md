# Image Search

Offline caption generation and Upstash-backed semantic item search for Infinity Nikki items.

## Layout

Core code now lives under `image_search/`:

- `image_search/constants/` for prompts, defaults, and shared hardcoded vocabulary
- `image_search/models/` for schemas and item type profiles
- `image_search/pipeline/` for manifest, captioning, document, and color-tag generation
- `image_search/search/` for Upstash sync and evaluation
- `image_search/vision/` for palette extraction

Root `cli.py` and `generate_manifest.py` stay as thin compatibility entrypoints.

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
- Extraction policy treats the overview image as authoritative for silhouette, length, placement, layering, and item identity.
- Extraction policy treats the icon image as authoritative for trim, closures, embroidery, small motifs, ornaments, and tiny accent colors.
- Known type-label drift is handled in profiles where practical: `pendants` may surface bag or garter-like accessories, and `faceDecorations` may include eyewear.
