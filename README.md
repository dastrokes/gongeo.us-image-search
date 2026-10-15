# Image Search

TypeScript CLI for Infinity Nikki visual attributes and icon colors. Requires
Node 24.15+ and npm. Extraction uses direct Google or Vercel AI Gateway.

Game-update preparation, review, D1 snapshots, and publication are documented
centrally in the `game-data-update` skill. This project generates local
artifacts; the data processor owns D1/Pinecone publication.

## Setup

```powershell
npm ci
npm run check
```

Set `GOOGLE_API_KEY` for Google or `AI_GATEWAY_API_KEY` for
`--provider vercel` in `.env` or the shell. Shell values take precedence.
Defaults are `gemini-3.5-flash-lite` and `google/gemini-2.5-flash`,
respectively; override with `--model`. Free-tier access is account-dependent.

Sibling checkouts supply decoded configs, processor reports, tracker taxonomy,
overrides, and images. Optional path settings are in [.env.example](.env.example);
CLI flags take precedence over environment settings and sibling defaults.

## Commands

Keep development model runs capped at `--limit 10`. Use `--help` with
any command for flags.

```powershell
npm run manifest -- --item-attributes-path <D1-snapshot.jsonl> --limit 10
npm run index -- --item-id 1020100310 --provider google
npm run index -- --provider vercel --limit 10 --item-attributes-path <D1-snapshot.jsonl> --output-root index/review
npm run refresh -- --structured-path index/review/item-structured-data.jsonl --structured-debug-path index/review/item-structured-debug.jsonl --output-root index/refreshed
npm run colors -- --limit 10 --output-root index/colors-review
npm run audit -- parse-errors --output-root index/review
npm run audit -- normalizer-drift --attributes <D1-snapshot.jsonl>
npm run audit -- missing-images --attributes <D1-snapshot.jsonl>
```

`manifest` selects incremental base items from the processor's synced-item
report, decoded types, and tracker catalog. It excludes published IDs using
the supplied D1 snapshot. `manifest/item-attributes.jsonl` is the development
default; use the verified export explicitly for a game update. Use
`--regen-manifest` on `index` to regenerate before extraction.

`index --item-id` prints a debug bundle without writing batch artifacts.
Batch extraction writes raw, normalized, debug, final attributes, review
reports, and summary files beneath `--output-root`. The processor-facing
artifact is `item-attributes.jsonl`:

```json
{"item_id":1020100310,"item_type":"hair","category":"twin_tails","subcategory":"high_twin_tails","metadata":{"texture":"straight"}}
```

Metadata omits inapplicable values and never repeats identity fields. Tracker
owns taxonomy and curated overrides; extraction preserves reviewable
open-vocabulary values rather than silently treating them as enums.

`refresh` re-normalizes cached structured records and reapplies overrides
without model calls. Its default inputs are in `index/`; pass both structured
input paths for another batch. Smaller scopes refuse to overwrite an output
containing other IDs: choose a separate output directory.

`colors` independently processes eligible local icons, including recolors.
It writes `item-colors.jsonl`, debug/review reports, and a color build summary,
never `item-attributes.jsonl`. Hair icons are flagged for review.
Use `--regen-manifest` after source updates. Limited runs do not replace the
shared color manifest.

Audits are read-only: JSONL findings followed by a count summary. Findings
keep exit code 0; invalid or missing required inputs fail nonzero. Parse-error
counts include records in both structured files, not unique item IDs.

## Resume and recovery

Batch indexing stores `extraction-run.json` and atomic `.checkpoints/<id>.json`
files in its output directory. Repeat the same scope, provider/model,
prompt/schema, and images to resume. Changed inputs require a new directory.
Older JSONL caches remain usable by offline refresh, but are not reused for
new model requests without a verified run identity. Failed responses retain
diagnostics and block final attributes.

Colors use independent `.color-checkpoints/` files; unchanged icons resume
and changed icons are recomputed. Check `run-status.json` or
`color-run-status.json` for `complete` before consuming an artifact set.
Individual files are written atomically; the whole set is not a transaction.

## Development

`npm run check` runs strict typechecking, Biome, and focused offline tests;
`npm run format` applies formatting. No model access or sibling checkout is
needed for tests. [Fixture notes](test/fixtures/README.md) explain the frozen
compatibility baseline.

Source is organized into `src/pipeline/`, `src/providers/`,
`src/constants/`, shared artifact contracts, path resolution, and the CLI.
