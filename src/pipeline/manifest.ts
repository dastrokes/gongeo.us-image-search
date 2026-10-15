import { stat } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { z } from "zod";
import { isMissing, readJson, readJsonl } from "../artifacts.js";
import type { Paths } from "../config.js";
import { rules } from "../constants/rules.js";
import { type ManifestRecord, manifestRecordSchema } from "../contracts.js";
import { isObject } from "./normalization.js";
import type { Taxonomy } from "./taxonomy.js";

export interface Selection {
  itemId?: number;
  itemIds?: readonly number[];
  types?: readonly string[];
  limit?: number;
}

export function selectRecords<T extends ManifestRecord>(
  taxonomy: Taxonomy,
  records: readonly T[],
  selection: Selection,
): T[] {
  const ids = new Set(selection.itemIds ?? []);
  if (selection.itemId !== undefined) ids.add(selection.itemId);
  for (const id of ids)
    if (!Number.isSafeInteger(id) || id <= 0)
      throw new Error("Item IDs must be positive integers");
  if (
    selection.limit !== undefined &&
    (!Number.isSafeInteger(selection.limit) || selection.limit <= 0)
  )
    throw new Error("--limit must be a positive integer");
  const types = taxonomy.expandTypes(selection.types);
  const selected = records.filter(
    (row) =>
      taxonomy.supported(row.item_type) &&
      (!ids.size || ids.has(row.item_id)) &&
      (!types.size || types.has(taxonomy.normalizeType(row.item_type))),
  );
  return selection.limit === undefined
    ? selected
    : selected.slice(0, selection.limit);
}

export async function findImage(
  root: string,
  id: number,
): Promise<string | null> {
  for (const extension of rules.imageExtensions) {
    const path = join(root, `${id}${extension}`);
    try {
      if ((await stat(path)).isFile()) return path;
    } catch (error) {
      if (!isMissing(error)) throw error;
    }
  }
  return null;
}

export async function loadPublishedIds(path: string): Promise<Set<number>> {
  const schema = z.object({ item_id: manifestRecordSchema.shape.item_id });
  const rows = await readJsonl(path, schema);
  const result = new Set<number>();
  for (const row of rows) {
    if (result.has(row.item_id))
      throw new Error(`Duplicate published item ${row.item_id}: ${path}`);
    result.add(row.item_id);
  }
  if (!result.size)
    throw new Error(
      `Published item attributes snapshot is empty: ${path}. Use --include-indexed for an intentional full index.`,
    );
  return result;
}

export async function loadCatalogRoots(
  paths: Paths,
): Promise<Map<number, number>> {
  const index = z
    .object({
      files: z.object({
        items: z.object({ path: z.string().startsWith("/catalog/") }),
      }),
    })
    .parse(
      await readJson(join(paths.trackerRoot, "public/catalog/index.json")),
    );
  const publicRoot = join(paths.trackerRoot, "public");
  const path = resolve(publicRoot, index.files.items.path.slice(1));
  if (relative(join(publicRoot, "catalog"), path).startsWith(".."))
    throw new Error("Catalog path escapes its directory");
  const rows = z.array(z.unknown()).parse(await readJson(path));
  const roots = new Map<number, number>();
  for (const row of rows) {
    if (!Array.isArray(row) || !row.length) continue;
    const ids: unknown[] = Array.isArray(row[0]) ? row[0] : [row[0]];
    for (const id of ids)
      if (typeof id === "number" && Number.isSafeInteger(id))
        roots.set(id, typeof row[6] === "number" ? row[6] : id);
  }
  return roots;
}

export async function loadManifestSource(
  paths: Paths,
  taxonomy: Taxonomy,
): Promise<ManifestRecord[]> {
  const [report, items, minorTypes] = await Promise.all([
    readJson(paths.syncReport),
    readJson(join(paths.configRoot, "item/TbItem.json")),
    readJson(join(paths.configRoot, "clothes/TbClothesMinorTypeInfo.json")),
  ]);
  const synced = z
    .object({
      syncedDetails: z.object({
        items: z.array(z.object({ id: manifestRecordSchema.shape.item_id })),
      }),
    })
    .parse(report).syncedDetails.items;
  if (!isObject(items) || !isObject(minorTypes))
    throw new Error("Invalid decoded item/minor-type configuration");
  const seen = new Set<number>();
  return synced.map(({ id }) => {
    if (seen.has(id)) throw new Error(`Duplicate synced item ID: ${id}`);
    seen.add(id);
    const item = items[String(id)];
    const minor = isObject(item) ? minorTypes[String(item.minor_type)] : null;
    const key =
      isObject(minor) && typeof minor.l10nshow_name === "string"
        ? minor.l10nshow_name
        : "";
    return {
      item_id: id,
      item_type: taxonomy.normalizeType(rules.typeKeyMap[key] ?? "unknown"),
    };
  });
}

export async function loadExcludedIds(paths: Paths): Promise<Set<number>> {
  let source: unknown;
  try {
    source = await readJson(
      join(dirname(paths.syncReport), "theme-sync-report.json"),
    );
  } catch (error) {
    if (isMissing(error)) return new Set();
    throw error;
  }
  const rows = z
    .object({ missingThemes: z.array(z.unknown()).default([]) })
    .parse(source).missingThemes;
  const result = new Set<number>();
  for (const row of rows)
    if (isObject(row) && row.type === "item") {
      const id = Number(row.id);
      if (Number.isSafeInteger(id) && id > 0) result.add(id);
    }
  return result;
}

export async function buildManifest(
  paths: Paths,
  taxonomy: Taxonomy,
  selection: Selection = {},
  includeIndexed = false,
) {
  // Validate before reading large inputs; limits count accepted rows, not candidates.
  selectRecords(taxonomy, [], selection);
  const ids = new Set(selection.itemIds ?? []);
  if (selection.itemId !== undefined) ids.add(selection.itemId);
  const types = taxonomy.expandTypes(selection.types);
  const [source, roots, excluded, published] = await Promise.all([
    loadManifestSource(paths, taxonomy),
    loadCatalogRoots(paths),
    loadExcludedIds(paths),
    includeIndexed || ids.size
      ? Promise.resolve(new Set<number>())
      : loadPublishedIds(paths.attributesPath),
  ]);
  const stats = {
    skipped_count: 0,
    non_base_skipped_count: 0,
    indexed_skipped_count: 0,
    missing_icon_count: 0,
    missing_overview_count: 0,
    processor_excluded_count: 0,
  };
  const rows: ManifestRecord[] = [];
  for (const row of source) {
    if (ids.size && !ids.has(row.item_id)) continue;
    if (excluded.has(row.item_id)) {
      stats.processor_excluded_count++;
      continue;
    }
    if (roots.get(row.item_id) !== row.item_id) {
      stats.non_base_skipped_count++;
      continue;
    }
    if (!taxonomy.supported(row.item_type)) {
      stats.skipped_count++;
      continue;
    }
    if (types.size && !types.has(row.item_type)) continue;
    if (published.has(row.item_id)) {
      stats.indexed_skipped_count++;
      continue;
    }
    const [icon, overview] = await Promise.all([
      findImage(paths.iconRoot, row.item_id),
      findImage(paths.imageRoot, row.item_id),
    ]);
    if (!icon) stats.missing_icon_count++;
    if (!overview) stats.missing_overview_count++;
    if (!icon && !overview) {
      stats.skipped_count++;
      continue;
    }
    rows.push(row);
    if (selection.limit !== undefined && rows.length >= selection.limit) break;
  }
  return { rows, stats };
}
