import { dirname, join } from "node:path";
import {
  atomicWrite,
  isMissing,
  readJson,
  readJsonl,
  serializeJsonl,
} from "../artifacts.js";
import type { Paths } from "../config.js";
import {
  buildSummarySchema,
  type ItemAttributesRecord,
  itemAttributesRecordSchema,
  type ManifestRecord,
  manifestRecordSchema,
  type StructuredDebugRecord,
  type StructuredItemRecord,
  structuredDebugRecordSchema,
  structuredItemRecordSchema,
} from "../contracts.js";
import { applyOverride, buildSearchReport, loadOverrides } from "./finalize.js";
import { type Selection, selectRecords } from "./manifest.js";
import { isObject, normalizePayload, type Payload } from "./normalization.js";
import type { Taxonomy } from "./taxonomy.js";

export function refreshRecords(
  taxonomy: Taxonomy,
  manifest: readonly ManifestRecord[],
  structured: readonly StructuredItemRecord[],
  debug: readonly StructuredDebugRecord[],
  overrides: Map<number, Payload>,
  selection: Selection = {},
) {
  // Legacy Python checkpoints append replacement rows; the last row wins.
  const cache = new Map(structured.map((row) => [row.item_id, row]));
  const diagnostics = new Map(debug.map((row) => [row.item_id, row]));
  const overlap = selectRecords(taxonomy, manifest, selection).filter((row) =>
    cache.has(row.item_id),
  );
  const records = overlap.length
    ? overlap
    : selectRecords(taxonomy, [...cache.values()], selection);
  const attributes = [];
  const reports: Payload[] = [];
  let failures = 0;
  for (const selected of records) {
    const cached = cache.get(selected.item_id);
    if (!cached) continue;
    if (cached.item_type !== selected.item_type)
      throw new Error(
        `Manifest/cache item_type mismatch for ${selected.item_id}`,
      );
    if (cached.parse_error !== null) {
      failures++;
      continue;
    }
    const record = {
      ...cached,
      data: normalizePayload(taxonomy, cached.item_type, cached.data),
    };
    const final = applyOverride(
      taxonomy,
      record,
      overrides.get(record.item_id),
    );
    attributes.push(final);
    const diagnostic = diagnostics.get(record.item_id);
    if (diagnostic && diagnostic.item_type !== record.item_type)
      throw new Error(`Debug/cache item_type mismatch for ${record.item_id}`);
    const report = buildSearchReport(taxonomy, record, final, diagnostic);
    if (report) reports.push(report);
  }
  return { attributes, reports, failures };
}

export async function assertOutputScope(
  path: string,
  intendedIds: ReadonlySet<number>,
) {
  let existing: ItemAttributesRecord[];
  try {
    existing = await readJsonl(path, itemAttributesRecordSchema);
  } catch (error) {
    if (isMissing(error)) return;
    throw error;
  }
  const outside = existing.filter((row) => !intendedIds.has(row.item_id));
  if (outside.length)
    throw new Error(
      `Output ${path} contains ${outside.length} items outside this scope. Choose a separate --output-root.`,
    );
}

export interface RefreshOptions extends Selection {
  manifestPath?: string;
  structuredPath?: string;
  structuredDebugPath?: string;
}

export async function runRefresh(
  paths: Paths,
  taxonomy: Taxonomy,
  options: RefreshOptions = {},
) {
  const [manifest, structured, overrides] = await Promise.all([
    readJsonl(
      options.manifestPath ?? join(paths.manifestRoot, "item-manifest.jsonl"),
      manifestRecordSchema,
    ),
    readJsonl(
      options.structuredPath ??
        join(paths.outputRoot, "item-structured-data.jsonl"),
      structuredItemRecordSchema,
    ),
    loadOverrides(paths.overridesPath),
  ]);
  let debug: StructuredDebugRecord[] = [];
  try {
    debug = await readJsonl(
      options.structuredDebugPath ??
        join(paths.outputRoot, "item-structured-debug.jsonl"),
      structuredDebugRecordSchema,
    );
  } catch (error) {
    if (!isMissing(error)) throw error;
  }
  const result = refreshRecords(
    taxonomy,
    manifest,
    structured,
    debug,
    overrides,
    options,
  );
  if (result.failures)
    throw new Error(
      `Refresh has ${result.failures} unresolved parse failures. Repair/re-extract those rows before refreshing.`,
    );
  const attributesPath = join(paths.outputRoot, "item-attributes.jsonl");
  await assertOutputScope(
    attributesPath,
    new Set(result.attributes.map((row) => row.item_id)),
  );
  const summaryPath = join(paths.outputRoot, "build-summary.json");
  let previous: Payload = {};
  const sourceSummaryPath = options.structuredPath
    ? join(dirname(options.structuredPath), "build-summary.json")
    : summaryPath;
  for (const path of new Set([sourceSummaryPath, summaryPath])) {
    try {
      const value = await readJson(path);
      if (!isObject(value)) throw new Error(`Invalid summary: ${path}`);
      previous = value;
      break;
    } catch (error) {
      if (!isMissing(error)) throw error;
    }
  }
  const models = new Set(
    debug.flatMap((row) => (row.request ? [row.request.model] : [])),
  );
  const now = new Date().toISOString();
  const reportPath = join(paths.outputRoot, "item-search-report.jsonl");
  const summary = buildSummarySchema.parse({
    extraction_model_id:
      previous.extraction_model_id ??
      (models.size === 1 ? [...models][0] : "unknown"),
    item_count: result.attributes.length,
    skipped_count: previous.skipped_count ?? 0,
    missing_icon_count: previous.missing_icon_count ?? 0,
    missing_overview_count: previous.missing_overview_count ?? 0,
    structured_parse_fail_count: result.failures,
    build_started_at: previous.build_started_at ?? now,
    build_finished_at: now,
    duration_seconds: previous.duration_seconds ?? 0,
    search_report_path: reportPath,
    item_attributes_path: attributesPath,
  });
  const statusPath = join(paths.outputRoot, "run-status.json");
  const status = { operation: "refresh", item_count: result.attributes.length };
  await atomicWrite(
    statusPath,
    `${JSON.stringify({ ...status, status: "finalizing" })}\n`,
  );
  await atomicWrite(attributesPath, serializeJsonl(result.attributes));
  await atomicWrite(reportPath, serializeJsonl(result.reports));
  await atomicWrite(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
  await atomicWrite(
    statusPath,
    `${JSON.stringify({ ...status, status: "complete" })}\n`,
  );
  return {
    ...summary,
    summary_path: summaryPath,
    search_report_path: reportPath,
    item_attributes_path: attributesPath,
  };
}
