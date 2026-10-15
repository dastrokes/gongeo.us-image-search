import { isDeepStrictEqual } from "node:util";
import { isMissing, readJson } from "../artifacts.js";
import {
  type ItemAttributesRecord,
  itemAttributesRecordSchema,
  type StructuredDebugRecord,
  type StructuredItemRecord,
} from "../contracts.js";
import {
  buildFilterReport,
  type Data,
  hasValue,
  isObject,
  normalizePayload,
  type Payload,
} from "./normalization.js";
import type { Taxonomy } from "./taxonomy.js";

export function toAttributes(
  taxonomy: Taxonomy,
  record: StructuredItemRecord,
): ItemAttributesRecord {
  const metadata = Object.fromEntries(
    taxonomy
      .fields(record.item_type)
      .filter(
        ({ name }) =>
          name !== "category" &&
          name !== "subcategory" &&
          hasValue(record.data[name]),
      )
      .map(({ name }) => [name, record.data[name]]),
  );
  const scalar = (field: string) =>
    typeof record.data[field] === "string"
      ? record.data[field].trim() || null
      : null;
  return itemAttributesRecordSchema.parse({
    item_id: record.item_id,
    item_type: record.item_type,
    category: scalar("category"),
    subcategory: scalar("subcategory"),
    metadata,
  });
}

export function structuredFromAttributes(
  taxonomy: Taxonomy,
  record: ItemAttributesRecord,
): Data {
  return normalizePayload(taxonomy, record.item_type, {
    category: record.category,
    subcategory: record.subcategory,
    ...record.metadata,
  });
}

export function applyOverride(
  taxonomy: Taxonomy,
  record: StructuredItemRecord,
  override?: Payload,
): ItemAttributesRecord {
  const base = toAttributes(taxonomy, {
    ...record,
    data: normalizePayload(taxonomy, record.item_type, record.data),
  });
  if (!override) return base;
  const metadata = isObject(override.metadata) ? override.metadata : {};
  let payload: Payload;
  if (
    Object.hasOwn(override, "item_id") ||
    Object.hasOwn(override, "item_type")
  ) {
    if (Number(override.item_id) !== record.item_id)
      throw new Error(
        `Curated override item_id does not match ${record.item_id}`,
      );
    if (
      typeof override.item_type !== "string" ||
      taxonomy.normalizeType(override.item_type) !==
        taxonomy.normalizeType(record.item_type)
    )
      throw new Error(
        `Curated override item_type does not match ${record.item_type}`,
      );
    payload = {
      category: override.category ?? null,
      subcategory: override.subcategory ?? null,
      ...metadata,
    };
  } else
    payload = {
      category: base.category,
      subcategory: base.subcategory,
      ...base.metadata,
      ...metadata,
    };
  return toAttributes(taxonomy, {
    ...record,
    data: normalizePayload(taxonomy, record.item_type, payload),
  });
}

export async function loadOverrides(
  path: string,
): Promise<Map<number, Payload>> {
  let source: unknown;
  try {
    source = await readJson(path);
  } catch (error) {
    if (isMissing(error)) return new Map();
    throw error;
  }
  if (!isObject(source) || !isObject(source.items))
    throw new Error(`Invalid override file: ${path}`);
  const result = new Map<number, Payload>();
  for (const [key, value] of Object.entries(source.items)) {
    const id = Number(key);
    if (!Number.isSafeInteger(id) || id <= 0 || !isObject(value))
      throw new Error(`Invalid override entry: ${path} / ${key}`);
    result.set(id, value);
  }
  return result;
}

function changes(before: Payload | null, after: Payload | null) {
  return [
    ...new Set([...Object.keys(before ?? {}), ...Object.keys(after ?? {})]),
  ]
    .sort()
    .flatMap((field) => {
      const previous = before?.[field] ?? null;
      const current = after?.[field] ?? null;
      return isDeepStrictEqual(previous, current) ||
        (!hasValue(previous) && !hasValue(current))
        ? []
        : [{ field, from: previous, to: current }];
    });
}

export function buildSearchReport(
  taxonomy: Taxonomy,
  record: StructuredItemRecord,
  final: ItemAttributesRecord,
  debug?: StructuredDebugRecord,
): Payload | null {
  const report = debug?.raw_payload
    ? buildFilterReport(
        taxonomy,
        record.item_type,
        debug.raw_payload,
        record.data,
      )
    : (debug?.filter_report ?? {});
  const filterReport = Object.fromEntries(
    [
      "cross_field_ownership",
      "non_canonical",
      "parent_child_mismatch",
      "subcategory_not_in_list",
      "unknown_fields",
    ]
      .filter((key) => hasValue(report[key]))
      .map((key) => [key, report[key]]),
  );
  const finalData = structuredFromAttributes(taxonomy, final);
  const rawChanges = changes(debug?.raw_payload ?? null, record.data);
  const overrideChanges = changes(record.data, finalData);
  if (
    !Object.keys(filterReport).length &&
    !rawChanges.length &&
    !overrideChanges.length
  )
    return null;
  return Object.fromEntries(
    Object.entries({
      item_id: record.item_id,
      item_type: record.item_type,
      filter_report: filterReport,
      source_raw_payload: debug?.raw_payload ?? null,
      normalized_data: record.data,
      final_data: finalData,
      raw_to_normalized_changes: rawChanges,
      normalized_to_final_changes: overrideChanges,
    }).filter(([, value]) => hasValue(value)),
  );
}
