# Image Search

Offline structured extraction for Infinity Nikki items.

## Layout

Source lives under `src/` (added to `sys.path` by the root wrappers):

- `src/constants/` — settings, structured field definitions, and extraction prompts
- `src/models/` — dataclass schemas for manifests, structured records, and search documents
- `src/pipeline/` — manifest ingestion, vision extraction, and canonical attribute generation

Root `cli.py` and `manifest.py` are thin compatibility entry-points that prepend `src/` to `sys.path` before delegating.

## What It Builds

`index` reads:

- `reports/database-sync-report.json` — synced item list and source version
- `reports/theme-sync-report.json` — processor-side missing-theme item exclusions
- `gongeo.us-config-decoder/cfg/config_output/item/TbItem.json` — item minor-type mapping
- `gongeo.us-config-decoder/cfg/config_output/clothes/TbClothesMinorTypeInfo.json` — minor-type labels
- `gongeo.us-nikki-tracker/data/item-search/generated/image-search-taxonomy.json` — canonical tracker-owned taxonomy/schema export
- `gongeo.us-nikki-tracker/data/item-search/generated/overrides.json` — curated per-item maintainer overrides
- verified D1 `item-attributes.jsonl` — published-ID snapshot used to skip already-published items when rebuilding the manifest
- `gongeo.us-nikki-tracker/public/images/items/` — overview and icon images

It produces:

- `manifest/item-manifest.jsonl`
- `manifest/item-color-manifest.jsonl` when running `colors`
- `index/item-structured-raw.jsonl`
- `index/item-structured-data.jsonl`
- `index/item-attributes.jsonl`
- `index/item-structured-debug.jsonl`
- `index/item-search-report.jsonl`
- `index/build-summary.json`
- `index/item-colors.jsonl` when running `colors`
- `index/item-color-debug.jsonl` when running `colors`
- `index/item-color-report.jsonl` when running `colors`
- `index/color-build-summary.json` when running `colors`

## Install

```bash
uv sync
```

Structured extraction supports two remote providers:

- `google` (default) calls Google AI Studio with `GOOGLE_API_KEY`.
- `vercel` calls Vercel AI Gateway with `AI_GATEWAY_API_KEY`.

Both providers default to Gemini 3.5 Flash Lite. Vercel model IDs include the
provider prefix, such as `google/gemini-3.5-flash-lite`.

## Environment

Set these in `.env` or your shell:

- `GOOGLE_API_KEY` - required for `--provider google`
- `AI_GATEWAY_API_KEY` - required for `--provider vercel`

- `TRACKER_ROOT` — path to `gongeo.us-nikki-tracker` repo
- `CONFIG_DECODER_OUTPUT` — path to `gongeo.us-config-decoder/cfg/config_output`

## Commands

Keep dev runs capped at `--limit 10` unless intentionally wider.

```bash
# Extract structured data and build canonical item attributes
uv run python cli.py index --limit 10
uv run python cli.py index --item-id 1020123456   # single-item debug
uv run python cli.py index --type dresses --type outerwear
uv run python cli.py index --item-ids 1020100001 1020100002
uv run python cli.py index --provider vercel --limit 10
uv run python cli.py index --provider vercel --model google/gemini-3.5-flash-lite --limit 10

# Rebuild canonical item attributes from cached extraction (no re-extraction)
uv run python cli.py refresh
uv run python cli.py refresh --type dresses
uv run python cli.py refresh --item-id 1020100001

# Regenerate the base-item manifest only
uv run python manifest.py

# Build standalone icon-derived color tags for all synced items
uv run python cli.py colors --limit 10
uv run python cli.py colors --type hair --limit 10
uv run python cli.py colors --item-id 1020100001
```

For an incremental game update, follow `$game-data-update`. Pass the
current verified D1-derived `item-attributes.jsonl` with
`--item-attributes-path` to both manifest generation and indexing; the
default local snapshot is only for ad hoc development.

`index --item-id <ID>` prints one JSON bundle: prompt, raw model response, parsed payload, and normalized structured output.

`refresh` rebuilds `index/item-attributes.jsonl`, `index/item-search-report.jsonl`, and `index/build-summary.json` from cached normalized structured data plus tracker overrides — use when override application or downstream normalization changes but re-extraction is not needed.

The in-place TypeScript conversion is tracked in
[docs/typescript-migration.md](docs/typescript-migration.md). The migration
keeps the JSONL contracts and review/publication handoff stable.

## Notes

- Imports use bare module names (`from constants.settings import ...`) because `src/` is prepended to `sys.path`.
- The build now materializes three layers per item: raw extracted payload, normalized structured payload, and final canonical `item-attributes` payload after tracker overrides are applied.
- Extraction uses item-type-specific prompts; only fields relevant to the slot are included.
- `item-attributes.jsonl` is the only persisted final artifact.
- `metadata` in `item-attributes.jsonl` contains only actual tag fields and does not repeat `item_id`, `item_type`, `slot`, `category`, or `subcategory`.
- Search documents are derived from canonical item-attributes rows, not stored as a separate final artifact.
- `manifest/item-attributes.jsonl` remains the default published-ID snapshot;
  pass `--item-attributes-path` to use the current D1 export directly.
- `colors` is intentionally separate from `index` and does not read or write `item-attributes.jsonl`.
- Color tagging is not base-item restricted; it reads all synced items with available transparent icon PNGs.
- Canonical color labels are `black`, `white`, `cream`, `gray`, `red`, `pink`, `orange`, `yellow`, `green`, `olive`, `teal`, `blue`, `navy`, `purple`, `brown`, `beige`, `gold`, `silver`; `multicolor` is derived when enough meaningful color roles are present.
- Color rows expose `primary_colors` and `secondary_colors` arrays for color roles.
- Hair color rows are flagged for review because hair icons can include face, skin, and clothing pixels.
