import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { z } from "zod";
import { readJson, readJsonl, serializeJsonl } from "../src/artifacts.js";
import { parseCli } from "../src/cli.js";
import { resolvePaths } from "../src/config.js";
import {
  itemAttributesRecordSchema,
  manifestRecordSchema,
  structuredDebugRecordSchema,
  structuredItemRecordSchema,
} from "../src/contracts.js";
import { applyOverride, buildSearchReport } from "../src/pipeline/finalize.js";
import { buildManifest, loadPublishedIds } from "../src/pipeline/manifest.js";
import {
  buildFilterReport,
  normalizePayload,
  parseResponse,
} from "../src/pipeline/normalization.js";
import { buildPrompt } from "../src/pipeline/prompts.js";
import { refreshRecords, runRefresh } from "../src/pipeline/refresh.js";
import { loadTaxonomy, Taxonomy } from "../src/pipeline/taxonomy.js";

const fixture = (name: string) =>
  fileURLToPath(new URL(`fixtures/${name}`, import.meta.url));
const cases = z
  .object({
    extraction: z.array(
      z.object({
        name: z.string(),
        item_id: z.number(),
        item_type: z.string(),
        raw_response: z.string(),
        override: z.record(z.string(), z.unknown()).optional(),
      }),
    ),
    manifest: z.object({
      items: z.array(
        z.object({
          id: z.number(),
          minor_type: z.number(),
          images: z.array(z.string()),
          family_root: z.number().optional(),
          published: z.boolean().optional(),
          excluded: z.boolean().optional(),
        }),
      ),
      scenarios: z.array(
        z.object({
          name: z.string(),
          options: z.object({
            skip_indexed: z.boolean().optional(),
            item_id: z.number().optional(),
            item_types: z.array(z.string()).optional(),
            limit: z.number().optional(),
          }),
        }),
      ),
    }),
  })
  .parse(await readJson(fixture("cases.json")));
const reference = z
  .object({
    extraction: z.array(
      z.object({
        name: z.string(),
        raw: z.unknown(),
        structured: structuredItemRecordSchema,
        debug: structuredDebugRecordSchema,
        attributes: itemAttributesRecordSchema,
        report: z.unknown(),
      }),
    ),
    manifest: z.array(
      z.object({
        name: z.string(),
        rows: z.array(manifestRecordSchema),
        stats: z.record(z.string(), z.number()),
      }),
    ),
    refresh: z.object({
      attributes: z.array(itemAttributesRecordSchema),
      reports: z.array(z.unknown()),
      summary: z.record(z.string(), z.unknown()),
    }),
  })
  .parse(await readJson(fixture("python-reference.json")));
const taxonomy = await loadTaxonomy(fixture("tracker-export.json"));

test("normalization, override removal, diagnostics, and prompts match Python", async (t) => {
  for (const input of cases.extraction)
    await t.test(input.name, () => {
      const expected = reference.extraction.find(
        (row) => row.name === input.name,
      );
      assert.ok(expected);
      const { payload, error } = parseResponse(input.raw_response);
      // Parser wording is runtime-specific; preserve the stable failure category.
      assert.equal(
        error?.split(":")[0] ?? null,
        expected.structured.parse_error?.split(":")[0] ?? null,
      );
      const data = normalizePayload(taxonomy, input.item_type, payload);
      assert.deepEqual(data, expected.structured.data);
      assert.deepEqual(
        buildFilterReport(taxonomy, input.item_type, payload, data),
        expected.debug.filter_report,
      );
      assert.equal(
        buildPrompt(taxonomy, input.item_type),
        expected.debug.prompt,
      );
      const record = {
        item_id: input.item_id,
        item_type: input.item_type,
        data,
        parse_error: error,
      };
      const final = applyOverride(taxonomy, record, input.override);
      assert.deepEqual(final, expected.attributes);
      assert.deepEqual(
        buildSearchReport(taxonomy, record, final, {
          ...expected.debug,
          raw_payload: payload,
        }),
        expected.report,
      );
    });
});

async function setup(root: string) {
  const paths = resolvePaths(
    {
      trackerRoot: join(root, "tracker with spaces"),
      configRoot: join(root, "decoder"),
      syncReport: join(root, "reports/database-sync-report.json"),
      manifestRoot: join(root, "manifest"),
      outputRoot: join(root, "output with spaces"),
    },
    {},
  );
  const json = async (path: string, value: unknown) => {
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, JSON.stringify(value));
  };
  await json(
    paths.taxonomyPath,
    await readJson(fixture("tracker-export.json")),
  );
  await json(paths.overridesPath, { items: {} });
  await json(paths.syncReport, {
    syncedDetails: { items: cases.manifest.items.map(({ id }) => ({ id })) },
  });
  await json(join(dirname(paths.syncReport), "theme-sync-report.json"), {
    missingThemes: cases.manifest.items
      .filter((row) => row.excluded)
      .map(({ id }) => ({ id, type: "item" })),
  });
  await json(
    join(paths.configRoot, "item/TbItem.json"),
    Object.fromEntries(
      cases.manifest.items.map((row) => [
        row.id,
        { minor_type: row.minor_type },
      ]),
    ),
  );
  await json(
    join(paths.configRoot, "clothes/TbClothesMinorTypeInfo.json"),
    Object.fromEntries(
      [1, 2, 10, 17, 32].map((id) => [id, { l10nshow_name: `LGS03_${id}` }]),
    ),
  );
  await json(join(paths.trackerRoot, "public/catalog/index.json"), {
    files: { items: { path: "/catalog/items.json" } },
  });
  await json(
    join(paths.trackerRoot, "public/catalog/items.json"),
    cases.manifest.items.map((row) => [
      row.id,
      null,
      null,
      null,
      null,
      null,
      row.family_root ?? row.id,
    ]),
  );
  await mkdir(paths.manifestRoot, { recursive: true });
  await writeFile(
    paths.attributesPath,
    serializeJsonl(
      cases.manifest.items
        .filter((row) => row.published)
        .map(({ id }) => ({ item_id: id })),
    ),
  );
  for (const row of cases.manifest.items)
    for (const kind of row.images) {
      const path = join(
        kind === "icon" ? paths.iconRoot : paths.imageRoot,
        `${row.id}.png`,
      );
      await mkdir(dirname(path), { recursive: true });
      await writeFile(path, "fixture image presence only");
    }
  return paths;
}

test("manifest selection/order and exclusion counts match Python", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "image-search manifest "));
  t.after(() => rm(root, { recursive: true, force: true }));
  const paths = await setup(root);
  for (const scenario of cases.manifest.scenarios)
    await t.test(scenario.name, async () => {
      const selection: { itemId?: number; types?: string[]; limit?: number } =
        {};
      if (scenario.options.item_id !== undefined)
        selection.itemId = scenario.options.item_id;
      if (scenario.options.item_types !== undefined)
        selection.types = scenario.options.item_types;
      if (scenario.options.limit !== undefined)
        selection.limit = scenario.options.limit;
      const result = await buildManifest(
        paths,
        taxonomy,
        selection,
        scenario.options.skip_indexed === false,
      );
      const expected = reference.manifest.find(
        (row) => row.name === scenario.name,
      );
      assert.ok(expected);
      assert.deepEqual(result, { rows: expected.rows, stats: expected.stats });
    });
  const output = join(root, "cli output with spaces.jsonl");
  const { stdout } = await promisify(execFile)(process.execPath, [
    "--import",
    "tsx",
    fileURLToPath(new URL("../src/cli.ts", import.meta.url)),
    "manifest",
    "--tracker-root",
    paths.trackerRoot,
    "--taxonomy-path",
    paths.taxonomyPath,
    "--config-root",
    paths.configRoot,
    "--sync-report",
    paths.syncReport,
    "--manifest-root",
    paths.manifestRoot,
    "--output",
    output,
    "--limit",
    "2",
  ]);
  assert.equal(JSON.parse(stdout).record_count, 2);
  assert.deepEqual(
    await readJsonl(output, manifestRecordSchema),
    reference.manifest.find((row) => row.name === "limit-after-exclusion")
      ?.rows,
  );
});

test("published snapshots reject empty, malformed, boolean, and duplicate IDs", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "image-search invalid snapshot "));
  t.after(() => rm(root, { recursive: true, force: true }));
  const path = join(root, "published.jsonl");
  for (const content of [
    "",
    "{}\n",
    '{"item_id":true}\n',
    '{"item_id":1}\n{"item_id":1}\n',
    '{"item_id":',
  ]) {
    await writeFile(path, content);
    await assert.rejects(loadPublishedIds(path));
  }
  await assert.rejects(loadPublishedIds(join(root, "missing.jsonl")));
});

test("offline refresh matches Python and refuses scoped artifact loss", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "image-search refresh "));
  t.after(() => rm(root, { recursive: true, force: true }));
  const paths = await setup(root);
  const rows = reference.extraction.slice(0, 3);
  const manifest = rows
    .map(({ structured }) => ({
      item_id: structured.item_id,
      item_type: structured.item_type,
    }))
    .reverse();
  const structured = rows.map((row) => row.structured);
  const debug = rows.map((row) => row.debug);
  const result = refreshRecords(
    taxonomy,
    manifest,
    structured,
    debug,
    new Map(),
  );
  assert.deepEqual(result.attributes, reference.refresh.attributes);
  assert.deepEqual(result.reports, reference.refresh.reports);
  assert.equal(result.failures, 0);
  // Match last-row-wins resume semantics of existing Python checkpoints.
  assert.deepEqual(
    refreshRecords(
      taxonomy,
      manifest,
      [...structured, ...structured],
      debug,
      new Map(),
    ),
    result,
  );
  await mkdir(paths.outputRoot, { recursive: true });
  await writeFile(
    join(paths.manifestRoot, "item-manifest.jsonl"),
    serializeJsonl(manifest),
  );
  await writeFile(
    join(paths.outputRoot, "item-structured-data.jsonl"),
    serializeJsonl(structured),
  );
  await writeFile(
    join(paths.outputRoot, "item-structured-debug.jsonl"),
    serializeJsonl(debug),
  );
  await writeFile(
    join(paths.outputRoot, "build-summary.json"),
    JSON.stringify({
      extraction_model_id: "fixture-model",
      build_started_at: "2000-01-01T00:00:00+00:00",
    }),
  );
  const summary = await runRefresh(paths, taxonomy);
  const expectedSummary = { ...summary };
  delete (expectedSummary as Partial<typeof summary>).summary_path;
  expectedSummary.build_finished_at = "<timestamp>";
  expectedSummary.item_attributes_path = basename(
    expectedSummary.item_attributes_path,
  );
  expectedSummary.search_report_path = basename(
    expectedSummary.search_report_path,
  );
  assert.deepEqual(expectedSummary, reference.refresh.summary);
  const output = join(paths.outputRoot, "item-attributes.jsonl");
  assert.deepEqual(
    await readJsonl(output, itemAttributesRecordSchema),
    reference.refresh.attributes,
  );
  const before = await readFile(output, "utf8");
  await assert.rejects(
    runRefresh(paths, taxonomy, { itemId: rows[0]?.structured.item_id ?? 0 }),
    /separate --output-root/,
  );
  assert.equal(await readFile(output, "utf8"), before);
  const child = await promisify(execFile)(process.execPath, [
    "--import",
    "tsx",
    fileURLToPath(new URL("../src/cli.ts", import.meta.url)),
    "refresh",
    "--tracker-root",
    paths.trackerRoot,
    "--taxonomy-path",
    paths.taxonomyPath,
    "--overrides-path",
    paths.overridesPath,
    "--manifest-root",
    paths.manifestRoot,
    "--output-root",
    paths.outputRoot,
    "--structured-path",
    join(paths.outputRoot, "item-structured-data.jsonl"),
    "--structured-debug-path",
    join(paths.outputRoot, "item-structured-debug.jsonl"),
  ]);
  assert.equal(JSON.parse(child.stdout).item_count, 3);
  const separatePaths = {
    ...paths,
    outputRoot: join(root, "separate refresh"),
  };
  const separate = await runRefresh(separatePaths, taxonomy, {
    structuredPath: join(paths.outputRoot, "item-structured-data.jsonl"),
    structuredDebugPath: join(paths.outputRoot, "item-structured-debug.jsonl"),
  });
  assert.equal(separate.extraction_model_id, "fixture-model");
  assert.equal(separate.build_started_at, "2000-01-01T00:00:00+00:00");
  assert.equal(
    z
      .object({ status: z.string() })
      .parse(await readJson(join(paths.outputRoot, "run-status.json"))).status,
    "complete",
  );
  await writeFile(
    join(paths.outputRoot, "item-structured-data.jsonl"),
    serializeJsonl(
      structured.map((row, index) =>
        index === 0 ? { ...row, parse_error: "empty_response" } : row,
      ),
    ),
  );
  await assert.rejects(
    runRefresh(paths, taxonomy),
    /unresolved parse failures/,
  );
  assert.equal(await readFile(output, "utf8"), before);
});

test("path precedence, type groups, and CLI flags preserve intended scopes", () => {
  const paths = resolvePaths(
    { trackerRoot: "custom tracker" },
    { TRACKER_ROOT: "wrong tracker" },
  );
  assert.ok(paths.taxonomyPath.startsWith(paths.trackerRoot));
  assert.ok(paths.overridesPath.startsWith(paths.trackerRoot));
  assert.notEqual(
    resolvePaths({}, { TRACKER_ITEM_SEARCH_EXPORT: "explicit export" })
      .taxonomyPath,
    paths.taxonomyPath,
  );
  assert.ok(taxonomy.expandTypes(["accessories"]).has("handhelds"));
  assert.ok(!taxonomy.expandTypes(["accessories"]).has("hair"));
  assert.deepEqual(
    [...taxonomy.expandTypes(["abilityHandhelds"])],
    ["handhelds"],
  );
  assert.throws(() => taxonomy.expandTypes(["invalid"]), /Unknown item type/);
  const parsed = parseCli([
    "refresh",
    "--item-ids",
    "1020100310",
    "1021000001",
    "--type",
    "hair",
    "--type",
    "earrings",
    "--output-root",
    "path with spaces",
  ]);
  assert.deepEqual(parsed.selection.itemIds, [1020100310, 1021000001]);
  assert.deepEqual(parsed.selection.types, ["hair", "earrings"]);
  assert.throws(
    () => parseCli(["manifest", "--limit", "0"]),
    /positive integer/,
  );
  assert.throws(() => parseCli(["refresh", "--item-ids"]), /requires/);
  assert.throws(() => parseCli(["refresh", "--output", "x"]), /Manifest/);
});

test("invalid taxonomy and mismatched override identity fail explicitly", async () => {
  const source = await readJson(fixture("tracker-export.json"));
  const modified = z
    .record(z.string(), z.unknown())
    .parse(structuredClone(source));
  modified.fieldKindByName = {};
  assert.throws(() => new Taxonomy(modified), /Unknown schema field/);
  const row = reference.extraction[0]?.structured;
  assert.ok(row);
  assert.throws(
    () =>
      applyOverride(taxonomy, row, {
        item_id: row.item_id + 1,
        item_type: row.item_type,
      }),
    /item_id/,
  );
  assert.throws(
    () =>
      applyOverride(taxonomy, row, {
        item_id: row.item_id,
        item_type: "outerwear",
      }),
    /item_type/,
  );
});
