import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { z } from "zod";
import {
  atomicWrite,
  isMissing,
  readJson,
  readJsonl,
  serializeJsonl,
} from "../artifacts.js";
import type { Paths } from "../config.js";
import { palette } from "../constants/colors.js";
import {
  type ColorDebugRecord,
  type ColorManifestRecord,
  colorDebugRecordSchema,
  colorManifestRecordSchema,
  type ItemColorsRecord,
  itemColorsRecordSchema,
} from "../contracts.js";
import { extractColorRecord } from "./color-extraction.js";
import {
  findImage,
  loadManifestSource,
  type Selection,
  selectRecords,
} from "./manifest.js";
import type { Taxonomy } from "./taxonomy.js";

export async function buildColorManifest(
  paths: Paths,
  taxonomy: Taxonomy,
  selection: Selection = {},
) {
  selectRecords(taxonomy, [], selection);
  const ids = new Set(selection.itemIds ?? []);
  if (selection.itemId !== undefined) ids.add(selection.itemId);
  const types = taxonomy.expandTypes(selection.types);
  const source = await loadManifestSource(paths, taxonomy);
  const stats = {
    skipped_count: 0,
    unsupported_type_count: 0,
    missing_icon_count: 0,
  };
  const rows: ColorManifestRecord[] = [];
  for (const row of source) {
    if (ids.size && !ids.has(row.item_id)) continue;
    if (!taxonomy.supported(row.item_type)) {
      stats.unsupported_type_count++;
      stats.skipped_count++;
      continue;
    }
    if (types.size && !types.has(row.item_type)) continue;
    const icon = await findImage(paths.iconRoot, row.item_id);
    if (!icon) {
      stats.missing_icon_count++;
      stats.skipped_count++;
      continue;
    }
    rows.push({ ...row, icon_path: icon });
    if (selection.limit !== undefined && rows.length >= selection.limit) break;
  }
  return { rows, stats };
}

export function buildColorReport(
  record: ItemColorsRecord,
  debug: ColorDebugRecord,
) {
  if (!record.needs_review) return null;
  return {
    item_id: record.item_id,
    item_type: record.item_type,
    review_reasons: record.review_reasons,
    color_tags: record.color_tags,
    swatches: record.swatches,
    debug: {
      visible_pixel_count: debug.visible_pixel_count,
      sampled_pixel_count: debug.sampled_pixel_count,
      ignored_pixel_count: debug.ignored_pixel_count,
      color_weights: debug.color_weights,
      image_path: debug.image_path,
    },
  };
}

const checkpointSchema = z.strictObject({
  version: z.literal(1),
  input_hash: z.string(),
  record: itemColorsRecordSchema,
  debug: colorDebugRecordSchema,
});

export async function runColors(
  paths: Paths,
  taxonomy: Taxonomy,
  options: Selection & { regenManifest?: boolean } = {},
) {
  const started = Date.now();
  const manifestPath = join(paths.manifestRoot, "item-color-manifest.jsonl");
  let rows: ColorManifestRecord[];
  let stats = {
    skipped_count: 0,
    unsupported_type_count: 0,
    missing_icon_count: 0,
  };
  let cached: ColorManifestRecord[] | null = null;
  if (!options.regenManifest)
    try {
      cached = await readJsonl(manifestPath, colorManifestRecordSchema);
    } catch (error) {
      if (!isMissing(error)) throw error;
    }
  if (cached) {
    rows = [];
    for (const row of selectRecords(taxonomy, cached, options)) {
      const path = await findImage(paths.iconRoot, row.item_id);
      if (!path) {
        stats.missing_icon_count++;
        stats.skipped_count++;
        continue;
      }
      rows.push({ ...row, icon_path: path });
    }
  } else {
    const built = await buildColorManifest(paths, taxonomy, options);
    rows = built.rows;
    stats = built.stats;
  }
  if (new Set(rows.map((row) => row.item_id)).size !== rows.length)
    throw new Error("Color manifest contains duplicate IDs");
  if (options.itemId !== undefined && !rows.length)
    throw new Error(`Color item ${options.itemId} not found`);
  const colorsPath = join(paths.outputRoot, "item-colors.jsonl");
  let existing: ItemColorsRecord[] = [];
  try {
    existing = await readJsonl(colorsPath, itemColorsRecordSchema);
  } catch (error) {
    if (!isMissing(error)) throw error;
  }
  const ids = new Set(rows.map((row) => row.item_id));
  if (existing.some((row) => !ids.has(row.item_id)))
    throw new Error(
      "Color output contains items outside this scope. Choose a separate --output-root.",
    );
  const statusPath = join(paths.outputRoot, "color-run-status.json");
  await atomicWrite(
    statusPath,
    `${JSON.stringify({ status: "extracting", expected_count: rows.length })}\n`,
  );
  if (
    !cached &&
    options.itemId === undefined &&
    !options.itemIds?.length &&
    !options.types?.length &&
    options.limit === undefined
  )
    await atomicWrite(manifestPath, serializeJsonl(rows));
  const results: { record: ItemColorsRecord; debug: ColorDebugRecord }[] = [];
  let resumed = 0;
  try {
    for (const row of rows) {
      const bytes = await readFile(row.icon_path);
      const hash = createHash("sha256")
        .update(
          JSON.stringify({ version: 1, palette, item_type: row.item_type }),
        )
        .update(bytes)
        .digest("hex");
      const checkpointPath = join(
        paths.outputRoot,
        ".color-checkpoints",
        `${row.item_id}.json`,
      );
      let checkpoint: z.infer<typeof checkpointSchema> | null = null;
      try {
        checkpoint = checkpointSchema.parse(await readJson(checkpointPath));
      } catch (error) {
        if (!isMissing(error)) throw error;
      }
      if (
        checkpoint &&
        (checkpoint.record.item_id !== row.item_id ||
          checkpoint.debug.item_id !== row.item_id ||
          checkpoint.record.item_type !== row.item_type ||
          checkpoint.debug.item_type !== row.item_type)
      )
        throw new Error(`Color checkpoint identity mismatch: ${row.item_id}`);
      if (checkpoint?.input_hash === hash) {
        results.push({
          record: checkpoint.record,
          debug: { ...checkpoint.debug, image_path: row.icon_path },
        });
        resumed++;
      } else {
        const extracted = await extractColorRecord(row, bytes);
        await atomicWrite(
          checkpointPath,
          `${JSON.stringify({ version: 1, input_hash: hash, ...extracted })}\n`,
        );
        results.push(extracted);
      }
    }
  } catch (error) {
    await atomicWrite(
      statusPath,
      `${JSON.stringify({ status: "failed", expected_count: rows.length, completed_count: results.length })}\n`,
    );
    throw error;
  }
  const colors = results.map((row) => row.record);
  const reports = results.flatMap(({ record, debug }) => {
    const report = buildColorReport(record, debug);
    return report ? [report] : [];
  });
  const reportPath = join(paths.outputRoot, "item-color-report.jsonl");
  const summaryPath = join(paths.outputRoot, "color-build-summary.json");
  const summary = {
    item_count: colors.length,
    ...stats,
    review_required_count: colors.filter((row) => row.needs_review).length,
    build_started_at: new Date(started).toISOString(),
    build_finished_at: new Date().toISOString(),
    duration_seconds: (Date.now() - started) / 1000,
    item_colors_path: colorsPath,
    color_report_path: reportPath,
  };
  await atomicWrite(
    statusPath,
    `${JSON.stringify({ status: "finalizing", expected_count: rows.length })}\n`,
  );
  await atomicWrite(colorsPath, serializeJsonl(colors));
  await atomicWrite(
    join(paths.outputRoot, "item-color-debug.jsonl"),
    serializeJsonl(results.map((row) => row.debug)),
  );
  await atomicWrite(reportPath, serializeJsonl(reports));
  await atomicWrite(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
  await atomicWrite(
    statusPath,
    `${JSON.stringify({ status: "complete", item_count: colors.length })}\n`,
  );
  console.error(
    `Colors: ${colors.length} selected, ${resumed} resumed, ${colors.length - resumed} decoded`,
  );
  return { summary_path: summaryPath, ...summary };
}
