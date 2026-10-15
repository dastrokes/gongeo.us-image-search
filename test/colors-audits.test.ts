import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import sharp from "sharp";
import { z } from "zod";
import { readJson, readJsonl, serializeJsonl } from "../src/artifacts.js";
import { parseCli } from "../src/cli.js";
import { resolvePaths } from "../src/config.js";
import {
  colorDebugRecordSchema,
  colorManifestRecordSchema,
  itemAttributesRecordSchema,
  itemColorsRecordSchema,
  structuredDebugRecordSchema,
  structuredItemRecordSchema,
} from "../src/contracts.js";
import {
  auditMissingImages,
  auditNormalizerDrift,
  auditParseErrors,
} from "../src/pipeline/audits.js";
import {
  extractColorRecord,
  extractColorsFromRgba,
  roundEven,
} from "../src/pipeline/color-extraction.js";
import {
  buildColorManifest,
  buildColorReport,
  runColors,
} from "../src/pipeline/colors.js";
import { loadTaxonomy } from "../src/pipeline/taxonomy.js";

const fixture = (name: string) =>
  fileURLToPath(new URL(`fixtures/${name}`, import.meta.url));
const cases = z
  .object({
    colors: z.array(
      z.object({
        name: z.string(),
        item_id: z.number(),
        item_type: z.string(),
        width: z.number(),
        height: z.number(),
        pixels: z.array(z.array(z.number())),
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
    }),
  })
  .parse(await readJson(fixture("cases.json")));
const reference = z
  .object({
    colors: z.array(
      z.object({
        name: z.string(),
        record: itemColorsRecordSchema,
        debug: colorDebugRecordSchema,
      }),
    ),
    color_manifest: z.object({
      rows: z.array(colorManifestRecordSchema),
      stats: z.record(z.string(), z.number()),
    }),
    extraction: z.array(
      z.object({
        structured: structuredItemRecordSchema,
        debug: structuredDebugRecordSchema,
        attributes: itemAttributesRecordSchema,
      }),
    ),
  })
  .parse(await readJson(fixture("python-reference.json")));
const taxonomy = await loadTaxonomy(fixture("tracker-export.json"));

async function png(pixels: number[][], width: number, height: number) {
  return sharp(Buffer.from(pixels.flat()), {
    raw: { width, height, channels: 4 },
  })
    .png()
    .toBuffer();
}

async function setup(root: string) {
  const paths = resolvePaths(
    {
      trackerRoot: join(root, "tracker"),
      configRoot: join(root, "config"),
      syncReport: join(root, "reports/database-sync-report.json"),
      manifestRoot: join(root, "manifest"),
      outputRoot: join(root, "color output"),
    },
    {},
  );
  const json = async (path: string, value: unknown) => {
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, JSON.stringify(value));
  };
  const items = cases.manifest.items;
  await json(
    paths.taxonomyPath,
    await readJson(fixture("tracker-export.json")),
  );
  await json(paths.syncReport, {
    syncedDetails: { items: items.map(({ id }) => ({ id })) },
  });
  await json(join(dirname(paths.syncReport), "theme-sync-report.json"), {
    missingThemes: items
      .filter((row) => row.excluded)
      .map(({ id }) => ({ id, type: "item" })),
  });
  await json(
    join(paths.configRoot, "item/TbItem.json"),
    Object.fromEntries(
      items.map((row) => [row.id, { minor_type: row.minor_type }]),
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
    items.map((row) => [
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
      items.filter((row) => row.published).map(({ id }) => ({ item_id: id })),
    ),
  );
  const bytes = await png([[32, 32, 32, 255]], 1, 1);
  for (const row of items)
    for (const kind of row.images) {
      const path = join(
        kind === "icon" ? paths.iconRoot : paths.imageRoot,
        `${row.id}.png`,
      );
      await mkdir(dirname(path), { recursive: true });
      await writeFile(path, bytes);
    }
  return paths;
}

test("RGBA extraction and PNG decoding match Python colors exactly", async (t) => {
  for (const input of cases.colors)
    await t.test(input.name, async () => {
      const expected = reference.colors.find((row) => row.name === input.name);
      assert.ok(expected);
      const manifest = {
        item_id: input.item_id,
        item_type: input.item_type,
        icon_path: `${input.name}.png`,
      };
      const raw = extractColorsFromRgba(
        manifest,
        Buffer.from(input.pixels.flat()),
        input.width,
        input.height,
      );
      assert.deepEqual(raw, { record: expected.record, debug: expected.debug });
      assert.deepEqual(
        await extractColorRecord(
          manifest,
          await png(input.pixels, input.width, input.height),
        ),
        raw,
      );
      const report = buildColorReport(raw.record, raw.debug);
      assert.equal(report !== null, raw.record.needs_review);
      if (report)
        assert.deepEqual(report.debug.color_weights, raw.debug.color_weights);
    });
  assert.equal(roundEven(32.5), 32);
  assert.equal(roundEven(33.5), 34);
  assert.equal(roundEven(2.675, 2), 2.67);
  assert.equal(roundEven(-2.675, 2), -2.67);
  assert.throws(
    () =>
      extractColorsFromRgba(
        { item_id: 1, item_type: "hair", icon_path: "" },
        new Uint8Array(),
        1,
        1,
      ),
    /RGBA/,
  );
});

test("color manifests match Python and include recolors and published/excluded icons", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "image-search colors "));
  t.after(() => rm(root, { recursive: true, force: true }));
  const paths = await setup(root);
  const built = await buildColorManifest(paths, taxonomy);
  assert.deepEqual(
    {
      ...built,
      rows: built.rows.map((row) => ({
        ...row,
        icon_path: `icons/${basename(row.icon_path)}`,
      })),
    },
    reference.color_manifest,
  );
  assert.equal(
    (await buildColorManifest(paths, taxonomy, { limit: 2 })).rows.length,
    2,
  );
  assert.deepEqual(
    (
      await buildColorManifest(paths, taxonomy, { types: ["outerwear"] })
    ).rows.map((row) => row.item_id),
    [1020200003],
  );
});

test("colors resume, invalidate changed images, isolate status, and guard output scope", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "image-search colors resume "));
  t.after(() => rm(root, { recursive: true, force: true }));
  const paths = await setup(root);
  const options = { itemIds: [1020100003, 1021000001] };
  await mkdir(paths.outputRoot, { recursive: true });
  const status = join(paths.outputRoot, "run-status.json");
  await writeFile(status, "extraction-owned");
  await runColors(paths, taxonomy, options);
  const output = join(paths.outputRoot, "item-colors.jsonl");
  const before = await readFile(output, "utf8");
  const checkpoint = join(
    paths.outputRoot,
    ".color-checkpoints/1020100003.json",
  );
  const checkpointBefore = await readFile(checkpoint, "utf8");
  await runColors(paths, taxonomy, options);
  assert.equal(await readFile(output, "utf8"), before);
  assert.equal(await readFile(checkpoint, "utf8"), checkpointBefore);
  assert.equal(await readFile(status, "utf8"), "extraction-owned");
  await assert.rejects(
    runColors(paths, taxonomy, { itemId: 1020100003 }),
    /outside this scope/,
  );
  assert.equal(await readFile(output, "utf8"), before);
  await writeFile(
    join(paths.iconRoot, "1020100003.png"),
    await png([[85, 127, 193, 255]], 1, 1),
  );
  await runColors(paths, taxonomy, options);
  const rows = await readJsonl(output, itemColorsRecordSchema);
  assert.deepEqual(rows[0]?.color_tags, ["blue"]);
  assert.notEqual(await readFile(checkpoint, "utf8"), checkpointBefore);
  await writeFile(join(paths.iconRoot, "1021000001.png"), "invalid image");
  await assert.rejects(
    runColors(paths, taxonomy, options),
    /unsupported image/,
  );
  assert.deepEqual(
    await readJson(join(paths.outputRoot, "color-run-status.json")),
    { status: "failed", expected_count: 2, completed_count: 1 },
  );
  await writeFile(checkpoint, "{}");
  await assert.rejects(runColors(paths, taxonomy, options));
});

test("audits report extraction failures, drift, and missing images without writes", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "image-search audits "));
  t.after(() => rm(root, { recursive: true, force: true }));
  const paths = await setup(root);
  await assert.rejects(auditParseErrors(paths.outputRoot), /No extraction/);
  await mkdir(paths.outputRoot, { recursive: true });
  const structured = reference.extraction.map((row) => row.structured);
  const errorsPath = join(paths.outputRoot, "item-structured-data.jsonl");
  await writeFile(errorsPath, serializeJsonl(structured));
  const audit = await auditParseErrors(paths.outputRoot);
  assert.equal(audit.summary.parseErrorCount, 3);
  await writeFile(
    join(paths.outputRoot, "item-structured-debug.jsonl"),
    serializeJsonl(reference.extraction.map((row) => row.debug)),
  );
  assert.equal(
    (await auditParseErrors(paths.outputRoot)).summary.parseErrorCount,
    6,
  );
  await writeFile(errorsPath, "not JSON\n");
  await assert.rejects(auditParseErrors(paths.outputRoot), /\.jsonl:1/);
  await writeFile(errorsPath, serializeJsonl(structured));
  const snapshot = join(root, "D1 attributes.jsonl");
  const row = reference.extraction[1]?.attributes;
  assert.ok(row);
  await writeFile(snapshot, serializeJsonl([row]));
  assert.equal(
    (await auditNormalizerDrift(taxonomy, snapshot)).summary
      .normalizerDriftCount,
    0,
  );
  await writeFile(
    snapshot,
    serializeJsonl([
      { ...row, metadata: { ...row.metadata, ornament: ["gems"] } },
    ]),
  );
  assert.equal(
    (await auditNormalizerDrift(taxonomy, snapshot)).summary
      .normalizerDriftCount,
    1,
  );
  await writeFile(snapshot, serializeJsonl([row, row]));
  await assert.rejects(auditNormalizerDrift(taxonomy, snapshot), /Duplicate/);
  await writeFile(snapshot, "");
  await assert.rejects(auditNormalizerDrift(taxonomy, snapshot), /empty/);
  const missing = await auditMissingImages(
    paths,
    taxonomy,
    paths.attributesPath,
  );
  assert.deepEqual(
    missing.rows.map((row) => [row.item_id, row.missing]),
    [
      [1020200001, ["icon"]],
      [1021000001, ["overview"]],
      [1020200002, ["icon", "overview"]],
      [1021500001, ["overview"]],
    ],
  );
  assert.equal(missing.summary.missingImageCount, 4);
  const child = await promisify(execFile)(process.execPath, [
    "--import",
    "tsx",
    fileURLToPath(new URL("../src/cli.ts", import.meta.url)),
    "audit",
    "parse-errors",
    "--output-root",
    paths.outputRoot,
    "--taxonomy-path",
    join(root, "not-required.json"),
  ]);
  assert.deepEqual(JSON.parse(child.stdout.trim().split("\n").at(-1) ?? ""), {
    parseErrorCount: 6,
  });
  assert.equal(await readFile(errorsPath, "utf8"), serializeJsonl(structured));
});

test("color and audit CLI flags reject ambiguous or irrelevant inputs", () => {
  assert.equal(
    parseCli(["colors", "--regen-manifest", "--limit", "10"]).selection.limit,
    10,
  );
  assert.equal(
    parseCli(["audit", "normalizer-drift", "--attributes", "D1 snapshot.jsonl"])
      .audit,
    "normalizer-drift",
  );
  for (const args of [
    ["audit"],
    ["audit", "unknown"],
    ["audit", "missing-images"],
    ["audit", "parse-errors", "--limit", "10"],
    ["audit", "parse-errors", "--regen-manifest"],
    ["colors", "--provider", "google"],
    ["colors", "--output", "x"],
    ["colors", "--include-indexed"],
  ])
    assert.throws(() => parseCli(args));
});
