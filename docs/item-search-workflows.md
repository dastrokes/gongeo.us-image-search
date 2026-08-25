# Item Search Operations: Image Search

`gongeo.us-image-search` owns item manifest generation, visual extraction, prompts, normalization, debug output, and canonical `item-attributes.jsonl` generation. `gongeo.us-nikki-tracker` owns registry/taxonomy, curated overrides, localization, Supabase/Pinecone publishing, and the published mirror.

Use `$review-item-search-batches` for the cross-project review/correction workflow, taxonomy and metadata rules, reusable audits, override preparation, localization gates, safe publishing, and recovery. This document covers image-search operations only.

Keep development runs small with `--limit 10` unless intentionally running the real batch.

## Inputs and artifacts

Expected sibling inputs:

- data-processor database/theme sync reports
- config-decoder item and minor-type config
- tracker overview images and icons
- tracker `data/item-search/generated/image-search-taxonomy.json`
- optional tracker mirror copied to `manifest/item-attributes.jsonl`

Generated artifacts:

- `manifest/item-manifest.jsonl`: selected extraction IDs.
- `index/item-structured-raw.jsonl`: raw model output.
- `index/item-structured-data.jsonl`: normalized structured output.
- `index/item-structured-debug.jsonl`: prompt, response, parse errors, and filter report.
- `index/item-search-report.jsonl`: normalization and override changes.
- `index/item-attributes.jsonl`: canonical tracker publish input.
- `index/build-summary.json`: build counts and parse status.

## Preflight

```powershell
git status --short
uv sync
```

Confirm:

- `.env` points `TRACKER_ROOT` and `CONFIG_DECODER_OUTPUT` at the intended sibling checkouts when defaults do not apply.
- `GOOGLE_API_KEY` exists when using Gemini.
- upstream database and theme reports contain the intended new IDs.
- decoded item/minor-type configs are current.
- tracker overview/icon assets exist or their gaps are understood.
- tracker generated taxonomy is current.

From tracker, regenerate taxonomy assets and refresh the published mirror when needed:

```powershell
node scripts/generate_filters.mjs
node scripts/refresh-item-search-local-copy.mjs
```

Copy the mirror into this project's `manifest/item-attributes.jsonl` to skip already published IDs.

## Manifest generation

Create a disposable small manifest first:

```powershell
uv run python manifest.py --limit 10 --output manifest/item-manifest.sanity.jsonl
```

Then create the real manifest:

```powershell
uv run python manifest.py
```

Review:

- candidate record count
- already-indexed skip count
- non-base and processor-excluded counts
- unsupported item types
- missing icon/overview counts
- final skipped count

Use the `$review-item-search-batches` bundled `missing-images` audit when image counts need item-level inspection. Items missing both views are skipped; one-view extraction is lower confidence.

## Extraction

Debug one representative item without a full batch:

```powershell
uv run python cli.py index --item-id <item_id>
```

Run a small disposable batch:

```powershell
uv run python cli.py index --limit 10 --regen-manifest --output-root index/sanity --manifest-root manifest/sanity
```

Run the real batch only after the small pass is sound:

```powershell
uv run python cli.py index --regen-manifest
```

Scoped variants:

```powershell
uv run python cli.py index --regen-manifest --item-ids <id1> <id2>
uv run python cli.py index --regen-manifest --type dresses --type outerwear
uv run python cli.py index --regen-manifest --backend gemini
uv run python cli.py index --regen-manifest --backend local --device cuda --quantization none
```

The index command checkpoints structured and debug rows. Re-running the same interrupted scope skips completed items.

## Sanity gates

Inspect:

```powershell
Get-Content index/build-summary.json
Get-Content index/item-attributes.jsonl -TotalCount 5
Get-Content index/item-search-report.jsonl -TotalCount 20
```

Require:

- zero unexplained structured parse failures
- understood image gaps
- item count equal to the intended batch
- matching intended IDs in manifest and canonical output
- only expected normalization and override changes
- intentional cross-field ownership moves
- no unresolved parent-child mismatch
- no unresolved unregistered subcategory

Use `$review-item-search-batches` for its bundled parse-error audit and the complete type-by-type review methodology. Overview images are primary evidence; icons only confirm small details. `item_type` is a game-data namespace, not visual evidence.

## Correction ownership

- A few item-specific visual corrections belong in tracker complete-row overrides.
- Repeatable extraction errors belong in this project's prompt, aliases, ownership map, or normalizer.
- Tracker category/subcategory registry and parent mappings remain tracker-owned.
- Do not add narrow prompt rules based on a few IDs.
- Do not introduce fields without explicit approval and tracker support.

After generic normalization changes, refresh the affected scope:

```powershell
uv run python cli.py refresh --item-ids <id1> <id2>
uv run python cli.py refresh --type <item_type>
uv run python cli.py refresh
```

A targeted refresh artifact must not replace the full canonical artifact.

Use the `$review-item-search-batches` normalizer-drift audit when the tracker mirror may contain historical rows that no longer match current normalization.

## Tracker handoff

From tracker, dry-run terms and taxonomy against the reviewed artifact:

```powershell
node scripts/sync-item-search-terms-from-attributes.mjs --dry-run --item-attributes-path ../gongeo.us-image-search/index/item-attributes.jsonl
```

Resolve aliases, field ownership, parent conflicts, and ungrouped subcategories before accepting additions. Tracker then owns term/taxonomy application, generated assets, all locale labels, prepared override validation, publishing, and post-publish verification.

Do not call tracker publish as a capability probe. Publishing requires explicit user approval. For accepted curated corrections, tracker uses `--scope pending-overrides`; for generated artifacts, the supplied file must already contain exactly the intended rows unless it is a verified full-catalog publish.

After publish, copy tracker's refreshed mirror back to `manifest/item-attributes.jsonl` when future manifests should skip those IDs.
