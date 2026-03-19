# Image Search

Offline structured tagging and Upstash-backed semantic item search for Infinity Nikki items.

## Layout

Core code now lives under `image_search/`:

- `image_search/constants/` for prompts, color maps, and taxonomy registry definitions
- `image_search/models/` for build and search artifact schemas
- `image_search/pipeline/` for manifest ingestion, captioning, structured tagging, and search-document generation
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
- `reports/index/taxonomy-concepts.jsonl`
- `reports/index/item-visual-features.jsonl`
- `reports/index/item-structured-candidates.jsonl`
- `reports/index/item-tag-assignments.jsonl`
- `reports/index/item-review-queue.jsonl`
- `reports/index/item-search-documents.jsonl`
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
python cli.py query-upstash --q "blue floral headwear" --item-type headwear
python cli.py query-upstash --q "mini hat" --facet headwear.subtype:mini_hat
python cli.py evaluate --queries path/to/queries.jsonl
```

## Notes

- Florence captions are now intermediate evidence for structured facet mapping, not the primary search payload.
- The canonical output contract is visual-only and keyed by `item_id`. Official game metadata is not part of the tagging artifacts.
- Search documents are derived from accepted structured tags plus vetted `search_terms`; review tags are emitted to a separate QA artifact.
- The report set is intentionally small: manifest, captions, taxonomy concepts, structured candidates, assignments, review queue, search documents, and build summary.
- Explicit filters are passed to Upstash metadata filtering. Natural-language-to-filter parsing is not part of v1.
- Indexed metadata separates `dominant_colors` from `accent_colors`; `--color` filters match either bucket.
- Structured facet filtering is available via repeated `--facet` flags against `accepted_facets`.
- Extraction policy treats the overview image as authoritative for silhouette, length, placement, layering, and item identity.
- Extraction policy treats the icon image as authoritative for trim, closures, embroidery, small motifs, ornaments, and tiny accent colors.
- Structured visual tags are intentionally conservative. Low-confidence or conflicting facet matches are routed to the review queue or preserved as `search_terms` instead of being promoted into strict filters.
