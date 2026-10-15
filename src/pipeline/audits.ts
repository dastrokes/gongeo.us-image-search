import { join } from "node:path";
import { isDeepStrictEqual } from "node:util";
import { isMissing, readJsonl } from "../artifacts.js";
import type { Paths } from "../config.js";
import {
  itemAttributesRecordSchema,
  type StructuredItemRecord,
  structuredDebugRecordSchema,
  structuredItemRecordSchema,
} from "../contracts.js";
import {
  findImage,
  loadCatalogRoots,
  loadExcludedIds,
  loadManifestSource,
  loadPublishedIds,
} from "./manifest.js";
import { normalizePayload, type Payload } from "./normalization.js";
import type { Taxonomy } from "./taxonomy.js";

type ParseRecord = Pick<
  StructuredItemRecord,
  "item_id" | "item_type" | "parse_error"
>;

export async function auditParseErrors(outputRoot: string) {
  const rows: Payload[] = [];
  let files = 0;
  for (const [name, schema] of [
    ["item-structured-data.jsonl", structuredItemRecordSchema],
    ["item-structured-debug.jsonl", structuredDebugRecordSchema],
  ] as const) {
    const path = join(outputRoot, name);
    let records: ParseRecord[];
    try {
      records = await readJsonl<ParseRecord>(path, schema);
      files++;
    } catch (error) {
      if (isMissing(error)) continue;
      throw error;
    }
    for (const row of records)
      if (row.parse_error)
        rows.push({
          path,
          item_id: row.item_id,
          item_type: row.item_type,
          parse_error: row.parse_error,
        });
  }
  if (!files) throw new Error(`No extraction artifacts found in ${outputRoot}`);
  return { rows, summary: { parseErrorCount: rows.length } };
}

function compact(source: Payload) {
  return Object.fromEntries(
    Object.entries(source).filter(
      ([, value]) =>
        value !== null && !(Array.isArray(value) && value.length === 0),
    ),
  );
}

export async function auditNormalizerDrift(
  taxonomy: Taxonomy,
  attributes: string,
) {
  const records = await readJsonl(attributes, itemAttributesRecordSchema);
  if (!records.length)
    throw new Error("Published item attributes snapshot is empty");
  const seen = new Set<number>();
  const rows: Payload[] = [];
  for (const row of records) {
    if (seen.has(row.item_id))
      throw new Error(`Duplicate published item ${row.item_id}`);
    seen.add(row.item_id);
    if (!taxonomy.supported(row.item_type))
      throw new Error(`Unsupported published item type: ${row.item_type}`);
    const current = compact({
      category: row.category,
      subcategory: row.subcategory,
      ...row.metadata,
    });
    const normalized = compact(
      normalizePayload(taxonomy, row.item_type, current),
    );
    if (!isDeepStrictEqual(current, normalized))
      rows.push({
        item_id: row.item_id,
        item_type: row.item_type,
        current,
        normalized,
      });
  }
  return { rows, summary: { normalizerDriftCount: rows.length } };
}

export async function auditMissingImages(
  paths: Paths,
  taxonomy: Taxonomy,
  attributes: string,
) {
  const [source, roots, excluded, published] = await Promise.all([
    loadManifestSource(paths, taxonomy),
    loadCatalogRoots(paths),
    loadExcludedIds(paths),
    loadPublishedIds(attributes),
  ]);
  const rows: Payload[] = [];
  for (const row of source) {
    if (
      excluded.has(row.item_id) ||
      published.has(row.item_id) ||
      roots.get(row.item_id) !== row.item_id ||
      !taxonomy.supported(row.item_type)
    )
      continue;
    const missing: string[] = [];
    if (!(await findImage(paths.iconRoot, row.item_id))) missing.push("icon");
    if (!(await findImage(paths.imageRoot, row.item_id)))
      missing.push("overview");
    if (missing.length) rows.push({ ...row, missing });
  }
  return { rows, summary: { missingImageCount: rows.length } };
}
