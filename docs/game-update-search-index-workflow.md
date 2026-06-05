# Game Update Search Index Workflow

This is the routine workflow for turning new game-update item IDs into a published item-search index. It assumes these sibling repos are checked out next to each other:

- `gongeo.us-image-search`: Python extraction worker. Produces canonical `item-attributes.jsonl`.
- `gongeo.us-nikki-tracker`: tracker app, taxonomy owner, localization owner, Supabase/Pinecone publisher.
- `gongeo.us-config-decoder`: decoded game config source.
- `gongeo.us-data-processor`: synced item report source.

Use small dev runs with `--limit 10` unless intentionally running the real batch.

## What Owns What

Tracker-owned source:

- `gongeo.us-nikki-tracker/data/item-search/registry.mjs`
- `gongeo.us-nikki-tracker/data/item-search/terms.json`
- `gongeo.us-nikki-tracker/data/item-search/taxonomy.json`
- `gongeo.us-nikki-tracker/app/locales/<locale>/filter.json`
- `gongeo.us-nikki-tracker/app/locales/<locale>/misc.json`

Tracker-generated operational files:

- `gongeo.us-nikki-tracker/data/item-search/generated/image-search-taxonomy.json`
- `gongeo.us-nikki-tracker/data/item-search/generated/overrides.json`
- `gongeo.us-nikki-tracker/data/item-search/generated/supabase/item-attributes.jsonl`
- `gongeo.us-nikki-tracker/data/item-search/generated/reports/publish/latest.json`

Image-search artifacts:

- `manifest/item-attributes.jsonl`: local skip list copied from the tracker Supabase mirror.
- `manifest/item-manifest.jsonl`: IDs selected for extraction.
- `index/item-structured-raw.jsonl`: raw model JSON.
- `index/item-structured-data.jsonl`: normalized structured data.
- `index/item-structured-debug.jsonl`: prompt, raw response, normalized data, parse errors, filter report.
- `index/item-search-report.jsonl`: normalization and override change report.
- `index/item-attributes.jsonl`: canonical publish input.
- `index/build-summary.json`: build counts and parse status.

## End-To-End Summary

1. Update upstream game data, decoded config, sync reports, and tracker images.
2. Regenerate tracker item-search generated assets if taxonomy, terms, or registry changed.
3. Refresh the tracker Supabase local mirror and copy it into `gongeo.us-image-search/manifest/item-attributes.jsonl`.
4. Generate a new manifest from image-search. This should contain only unpublished, supported base items when the skip list is current.
5. Run a small extraction sanity pass.
6. Run the real extraction for the new IDs.
7. Review parse errors, missing images, taxonomy drift, non-canonical scalar values, and overrides.
8. Sync missing terms/taxonomy back into tracker source when the generated attributes introduce new valid terms.
9. Review and update `en` and `zh` localization labels for new terms.
10. Publish from tracker using the generated `index/item-attributes.jsonl`.
11. Verify the publish report, refreshed local mirror, Supabase detail/facet behavior, and Pinecone-backed search.

## Preflight

From `gongeo.us-image-search`:

```bash
git status --short
uv sync
```

From `gongeo.us-nikki-tracker`:

```bash
git status --short
npm install
```

Environment expectations:

- Image-search `.env` should set `TRACKER_ROOT` and `CONFIG_DECODER_OUTPUT` when the default sibling paths are not correct.
- Tracker `.env` must include `SUPABASE_DATA_URL`, `SUPABASE_DATA_SECRET_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX_HOST` before publish.
- If using Gemini extraction, image-search needs `GOOGLE_API_KEY`.

Upstream inputs to confirm before extraction:

- `gongeo.us-data-processor/reports/database-sync-report.json` contains the new synced item IDs.
- `gongeo.us-data-processor/reports/theme-sync-report.json` is current. Image-search excludes `missingThemes` entries whose `type` is `item`.
- `gongeo.us-config-decoder/cfg/config_output/item/TbItem.json` and `clothes/TbClothesMinorTypeInfo.json` are current.
- `gongeo.us-nikki-tracker/public/images/items/<id>.(png|jpg|jpeg|webp)` exists for overview images.
- `gongeo.us-nikki-tracker/public/images/items/icons/<id>.(png|jpg|jpeg|webp)` exists for icon images.

## Tracker Preparation

Run this when registry, taxonomy, terms, or filters changed:

```bash
cd ../gongeo.us-nikki-tracker
node scripts/generate_filters.mjs
```

This refreshes tracker-side derived assets, including the image-search taxonomy export consumed by Python:

- `data/attribute.json`
- `shared/constants/itemSearchRegistry.ts`
- `shared/constants/itemSearchTaxonomy.ts`
- `app/locales/*/filter.json`
- `data/item-search/generated/image-search-taxonomy.json`

Then refresh the current published item mirror:

```bash
node scripts/refresh-item-search-local-copy.mjs
```

Copy the mirror into image-search so manifest generation can skip already-published rows:

```powershell
Copy-Item ..\gongeo.us-nikki-tracker\data\item-search\generated\supabase\item-attributes.jsonl .\manifest\item-attributes.jsonl
```

Run the copy command from `gongeo.us-image-search`. This file is only a local skip list and uses the same canonical row shape as the final publish artifact.

## Manifest Generation

For a non-destructive sanity manifest:

```bash
uv run python manifest.py --limit 10 --output manifest/item-manifest.sanity.jsonl
```

For the real new-ID manifest:

```bash
uv run python manifest.py
```

Review the printed summary:

- `record_count`: candidate rows that will be extracted.
- `indexed_skipped_count`: rows skipped because they were already present in `manifest/item-attributes.jsonl`.
- `non_base_skipped_count`: glow-up/evolution or other non-base IDs outside the configured base ranges.
- `processor_excluded_count`: item IDs excluded by data processor theme sync.
- `missing_icon_count` and `missing_overview_count`: asset gaps.
- `skipped_count`: unsupported item types plus rows with neither icon nor overview.

If image counts are non-zero, identify the affected IDs before extraction:

```powershell
@'
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / "src"))

from constants.structured import is_supported_item_type
from pipeline.manifest import (
    DEFAULT_ITEM_ATTRIBUTES_MANIFEST,
    _find_image_path,
    _is_base_item,
    _load_existing_item_ids,
    _load_json,
    _load_processor_excluded_item_ids,
    _resolve_item_type,
    resolve_manifest_paths,
)

paths = resolve_manifest_paths()
sync_report = _load_json(paths.sync_report_path)
item_config = _load_json(paths.item_config_path)
minor_type_info = _load_json(paths.minor_type_path)
indexed = _load_existing_item_ids(DEFAULT_ITEM_ATTRIBUTES_MANIFEST)
excluded = _load_processor_excluded_item_ids(paths.sync_report_path)

for raw in sync_report.get("syncedDetails", {}).get("items", []):
    item_id = int(raw["id"])
    if item_id in excluded or not _is_base_item(item_id) or item_id in indexed:
        continue
    item_type = _resolve_item_type(item_config.get(str(item_id)), minor_type_info)
    if not is_supported_item_type(item_type):
        continue
    missing = []
    if _find_image_path(paths.item_icon_root, item_id) is None:
        missing.append("icon")
    if _find_image_path(paths.item_image_root, item_id) is None:
        missing.append("overview")
    if missing:
        print(item_id, item_type, ",".join(missing))
'@ | python -
```

Fix missing assets in tracker when possible. Items with both images missing are skipped by extraction. Items with only one image still extract, but the output is lower confidence.

## Extraction

First debug one representative new item:

```bash
uv run python cli.py index --item-id <item_id>
```

This prints the prompt, raw model response, parsed payload, image paths, and normalized output without writing a full batch.

Run a small batch into temporary roots:

```bash
uv run python cli.py index --limit 10 --regen-manifest --output-root index/sanity --manifest-root manifest/sanity
```

For the real new-ID batch:

```bash
uv run python cli.py index --regen-manifest
```

Useful scoped variants:

```bash
uv run python cli.py index --regen-manifest --item-ids <id1> <id2> <id3>
uv run python cli.py index --regen-manifest --type dresses --type outerwear
uv run python cli.py index --regen-manifest --backend gemini
uv run python cli.py index --regen-manifest --backend local --device cuda --quantization none
```

The index command checkpoints `index/item-structured-data.jsonl` and `index/item-structured-debug.jsonl`. If a run is interrupted, rerun the same command and completed items are skipped.

## Image-Search Sanity Checks

Read the summary:

```bash
Get-Content index/build-summary.json
```

Expected gates before publish:

- `structured_parse_fail_count` is `0`.
- `missing_icon_count` and `missing_overview_count` are understood and either fixed or accepted.
- `item_count` matches the intended publish batch.
- `index/item-attributes.jsonl` has the same intended IDs as `manifest/item-manifest.jsonl`.
- `index/item-search-report.jsonl` contains only expected normalization changes and overrides.
- `filter_report.cross_field_ownership` only shows intentional owner-field moves or drops.

Find parse errors:

```powershell
@'
import json
from pathlib import Path

for path in [Path("index/item-structured-data.jsonl"), Path("index/item-structured-debug.jsonl")]:
    if not path.exists():
        continue
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("parse_error"):
            print(path, row.get("item_id"), row.get("item_type"), row.get("parse_error"))
'@ | python -
```

Spot-check final rows:

```bash
Get-Content index/item-attributes.jsonl -TotalCount 5
Get-Content index/item-search-report.jsonl -TotalCount 20
```

If output is wrong for a few items, prefer tracker overrides for curated corrections:

- Edit `gongeo.us-nikki-tracker/data/item-search/generated/overrides.json`.
- Rebuild derived image-search output without re-extraction:

```bash
uv run python cli.py refresh --item-ids <id1> <id2>
```

If normalization logic changed rather than only overrides, run:

```bash
uv run python cli.py refresh
```

## Terms And Taxonomy Dedupe

Use this step when new valid categories, subcategories, materials, ornaments, patterns, structures, or scalar values appear in `index/item-attributes.jsonl`.

Dry-run the tracker sync helper:

```bash
cd ../gongeo.us-nikki-tracker
node scripts/sync-item-search-terms-from-attributes.mjs --dry-run --item-attributes-path ../gongeo.us-image-search/index/item-attributes.jsonl
```

Review the JSON output:

- `shared`: new shared field tokens to add to `terms.json`.
- `scoped`: new category/subcategory tokens to add under item-type scope.
- `taxonomy`: new subcategory parent mappings to add to `taxonomy.json`.
- `parentConflicts`: must be reviewed manually. The script reports conflicts and does not overwrite existing parent mappings.
- `ungroupedSubcategories`: subcategories without a category. Fix attributes or add a parent before publishing.

Before applying, dedupe conceptually:

- If a new token means the same thing as an existing token, fix it with image-search aliases, a tracker override, or a corrected existing term instead of adding a duplicate.
- If a token belongs to a different canonical owner field, fix it in Python normalization before tracker backfill instead of backfilling the same concept twice.
- Use the repo taxonomy buckets as game-facing categories. For example, `dresses` and `outerwear` are game taxonomy buckets, not strict academic garment definitions.
- Keep category values closed-list and broad.
- Keep subcategory values as direct child refinements of category.
- Keep material, pattern, structure, ornament, length, silhouette, and fit out of subcategory when a dedicated field exists.

The debug report is the fastest per-item review surface:

- `filter_report.non_canonical`: tokens outside the current tracker registry
- `filter_report.cross_field_ownership`: tokens that the normalizer moved to another field or dropped
- `raw_to_normalized_changes`: the concrete per-item normalization diff

Apply after review:

```bash
node scripts/sync-item-search-terms-from-attributes.mjs --item-attributes-path ../gongeo.us-image-search/index/item-attributes.jsonl
```

Inspect the diff:

```bash
git diff -- data/item-search/terms.json data/item-search/taxonomy.json app/locales/en/filter.json shared/constants/itemSearchRegistry.ts shared/constants/itemSearchTaxonomy.ts data/attribute.json
```

If terms changed, regenerate the full tracker filter assets so every locale file gets the new keys:

```bash
node scripts/generate_filters.mjs
```

Then rerun image-search refresh or extraction as needed so Python consumes the updated `image-search-taxonomy.json`.

## Localizations

Normal search publish targets `en` and `zh` namespaces unless `--namespace` is provided.

Tracker Pinecone text is localized from:

- `app/locales/<namespace>/misc.json` for item type labels.
- `app/locales/<namespace>/filter.json` for category, subcategory, and metadata token labels.
- English fallback, then humanized token fallback, when a namespace is missing a label.

After terms or taxonomy change:

```bash
cd ../gongeo.us-nikki-tracker
git diff -- app/locales/en/filter.json app/locales/zh/filter.json
```

Review and edit new `en` and `zh` labels before publish. The generator may seed new labels with title-cased English tokens. That is acceptable for temporary local dev, but reviewed `en` and `zh` labels should be present before production publish.

Localization sanity checks:

- New category/subcategory keys exist under the correct item type.
- Shared field labels exist at `filter.<field>.<token>`.
- Category/subcategory labels exist at `filter.category.<item_type>.<token>` and `filter.subcategory.<item_type>.<token>`.
- Labels that translate to the same visible text are intentional. Pinecone text is deduped after lowercasing localized values, so accidental duplicate labels can reduce recall.

For localization-only republish:

```bash
node scripts/item-search-publish.mjs --scope locales-only --namespace en --namespace zh
```

## Historical Local-Mirror Audit

Use this when the tracker-owned Supabase mirror may contain older rows that no longer match the current normalizer. This is different from reviewing a fresh extraction batch.

From `gongeo.us-nikki-tracker`:

```powershell
@"
import json
import sys
from pathlib import Path

tracker_index = Path(r'C:\Users\dastrokes\Dev\git\gongeo.us-nikki-tracker\data\item-search\generated\supabase\item-attributes.jsonl')
sys.path.insert(0, r'C:\Users\dastrokes\Dev\git\gongeo.us-image-search\src')

from pipeline.extraction import VisionStructuredExtractor

rows = [json.loads(line) for line in tracker_index.read_text(encoding='utf-8').splitlines() if line.strip()]

def compact(payload: dict[str, object]) -> dict[str, object]:
    compacted: dict[str, object] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        compacted[key] = value
    return compacted

for row in rows:
    current = {
        'category': row.get('category'),
        'subcategory': row.get('subcategory'),
        **(row.get('metadata') or {}),
    }
    normalized = compact(
        VisionStructuredExtractor.normalize_payload(row['item_type'], current)
    )
    current = compact(current)
    if normalized != current:
        print(row['item_id'], row['item_type'])
        print(' current   ', json.dumps(current, ensure_ascii=False, sort_keys=True))
        print(' normalized', json.dumps(normalized, ensure_ascii=False, sort_keys=True))
"@ | python -
```

If this audit returns rows, treat them as a targeted live-index cleanup batch. Do not expand tracker terms just to preserve stale historical ownership or synonym choices.

## Publish

Important: `item-search-publish.mjs` publishes every row in the provided `item-attributes.jsonl`. The `--scope item-ids` and `--scope types` values are report/validation scope, not a filter over a larger JSONL file. Make sure the file you pass contains exactly the rows you intend to publish, unless using `--scope full`.

For a full canonical publish:

```bash
cd ../gongeo.us-nikki-tracker
node scripts/item-search-publish.mjs --scope full --item-attributes-path ../gongeo.us-image-search/index/item-attributes.jsonl
```

For a new-ID batch where `index/item-attributes.jsonl` contains only those new IDs:

```bash
node scripts/item-search-publish.mjs --scope item-ids --item-id <id1> --item-id <id2> --item-attributes-path ../gongeo.us-image-search/index/item-attributes.jsonl
```

For a type-scoped file:

```bash
node scripts/item-search-publish.mjs --scope types --type dresses --type outerwear --item-attributes-path ../gongeo.us-image-search/index/item-attributes.jsonl
```

For override-only item fixes that already exist in `data/item-search/generated/overrides.json`:

```bash
node scripts/item-search-publish.mjs --scope item-ids --item-id <id> --overrides-only
```

Publish does this:

1. Normalizes canonical rows.
2. Upserts Supabase `item_attributes` with `--overwrite`.
3. Builds localized Pinecone records for `en` and `zh` by default.
4. Upserts Pinecone with `--overwrite`.
5. Refreshes `data/item-search/generated/supabase/item-attributes.jsonl`.
6. Writes `data/item-search/generated/reports/publish/latest.json` and a timestamped report.

## Post-Publish Verification

Inspect the latest report:

```bash
Get-Content data/item-search/generated/reports/publish/latest.json
```

Check:

- `finalRowCount` matches the JSONL row count you intended.
- `touchedItemIds` contains only the intended IDs for a batch publish.
- `supabase.supabase_written` is plausible.
- `pinecone.pinecone_written.en` and `pinecone.pinecone_written.zh` are plausible.
- `localCopy.exported_count` is current.
- `failedItems` is empty.
- `itemAttributesSource.path` points at the intended file.

Then verify live behavior:

- Item detail endpoint returns the new metadata.
- Item facets include new category/subcategory/metadata tokens.
- Search finds the new items in `en`.
- Search finds the new items in `zh`.
- Existing common searches still return expected results.

Refresh image-search skip list after publish:

```powershell
Copy-Item ..\gongeo.us-nikki-tracker\data\item-search\generated\supabase\item-attributes.jsonl .\manifest\item-attributes.jsonl
```

## Recovery Recipes

Wrong labels only:

1. Fix `app/locales/en/filter.json` and `app/locales/zh/filter.json`.
2. Run `node scripts/item-search-publish.mjs --scope locales-only --namespace en --namespace zh`.

Wrong metadata for a few items:

1. Add or edit tracker overrides.
2. Run `uv run python cli.py refresh --item-ids <id1> <id2>` in image-search.
3. Publish the refreshed small JSONL with `--scope item-ids`.

Bad extraction caused by missing or poor images:

1. Fix `public/images/items` and/or `public/images/items/icons` in tracker.
2. Remove the affected cached rows from the image-search output or run into a clean `--output-root`.
3. Rerun `uv run python cli.py index --item-ids <id>`.
4. Sanity-check and publish.

New valid token missing from filters/search:

1. Run `node scripts/sync-item-search-terms-from-attributes.mjs --dry-run --item-attributes-path ../gongeo.us-image-search/index/item-attributes.jsonl`.
2. Resolve duplicates and parent conflicts.
3. Run the sync script without `--dry-run`.
4. Review `en` and `zh` labels.
5. Run `node scripts/generate_filters.mjs`.
6. Rerun image-search `refresh` or `index`, then publish.

Publish file contains too many rows:

1. Stop before publish when possible.
2. Regenerate image-search output with `--item-ids` or a clean `--output-root`.
3. If already published, republish corrected rows for any affected existing items. Publish upserts do not delete unintended new item IDs, so remove truly unwanted Supabase/Pinecone records manually before considering the index clean.

## Minimal Normal Game-Update Command List

Tracker:

```bash
cd ../gongeo.us-nikki-tracker
node scripts/generate_filters.mjs
node scripts/refresh-item-search-local-copy.mjs
```

Image-search:

```powershell
Copy-Item ..\gongeo.us-nikki-tracker\data\item-search\generated\supabase\item-attributes.jsonl .\manifest\item-attributes.jsonl
uv run python manifest.py
uv run python cli.py index --item-id <representative_new_id>
uv run python cli.py index --limit 10 --regen-manifest --output-root index/sanity --manifest-root manifest/sanity
uv run python cli.py index --regen-manifest
```

Tracker term/localization review:

```bash
cd ../gongeo.us-nikki-tracker
node scripts/sync-item-search-terms-from-attributes.mjs --dry-run --item-attributes-path ../gongeo.us-image-search/index/item-attributes.jsonl
# If dry-run shows valid new terms:
node scripts/sync-item-search-terms-from-attributes.mjs --item-attributes-path ../gongeo.us-image-search/index/item-attributes.jsonl
node scripts/generate_filters.mjs
```

Publish:

```bash
node scripts/item-search-publish.mjs --scope item-ids --item-id <id1> --item-id <id2> --item-attributes-path ../gongeo.us-image-search/index/item-attributes.jsonl
Get-Content data/item-search/generated/reports/publish/latest.json
```
