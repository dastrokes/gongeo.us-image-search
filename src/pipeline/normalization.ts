import { rules } from "../constants/rules.js";
import type { StructuredItemRecord } from "../contracts.js";
import type { Taxonomy } from "./taxonomy.js";

export type Data = StructuredItemRecord["data"];
export type Payload = Record<string, unknown>;
export const isObject = (value: unknown): value is Payload =>
  value !== null && typeof value === "object" && !Array.isArray(value);
export const hasValue = (value: unknown): boolean =>
  value !== undefined &&
  value !== null &&
  value !== "" &&
  !(Array.isArray(value) && value.length === 0) &&
  !(isObject(value) && Object.keys(value).length === 0);
const detailFields = ["pattern", "material", "structure", "ornament"];

export function normalizeToken(value: unknown): string {
  if (!value) return "";
  return String(value)
    .trim()
    .toLowerCase()
    .replaceAll("&", " and ")
    .replace(/[\s/-]+/g, "_")
    .replace(/[^a-z0-9_]+/g, "")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function collapse(value: unknown, field: string): string {
  const token = normalizeToken(value);
  let normalized =
    rules.fieldAliases[field]?.find(
      ([from]) => normalizeToken(from) === token,
    )?.[1] ?? token;
  normalized = normalizeToken(normalized);
  if (field === "ornament") {
    for (const suffix of rules.ornamentSuffixes) {
      if (normalized.endsWith(`_${suffix}`)) {
        const stripped = normalized
          .slice(0, -suffix.length - 1)
          .replace(/^_+|_+$/g, "");
        if (stripped) return stripped;
      }
    }
  }
  return normalized;
}

function rawValues(value: unknown, field: string): string[] {
  if (value === null || value === undefined || value === "") return [];
  return (Array.isArray(value) ? value : [value])
    .map((entry) => collapse(entry, field))
    .filter(Boolean);
}

export function outputTemplate(taxonomy: Taxonomy, type: string): Data {
  return Object.fromEntries(
    taxonomy
      .fields(type)
      .map(({ name, kind }) => [name, kind === "array" ? [] : null]),
  );
}

export function normalizePayload(
  taxonomy: Taxonomy,
  type: string,
  payload: Payload | null,
): Data {
  type = taxonomy.normalizeType(type);
  const normalized = outputTemplate(taxonomy, type);
  if (!payload) return normalized;
  const definitions = new Map(
    taxonomy.fields(type).map((field) => [field.name, field.kind]),
  );
  for (const [field, value] of Object.entries(payload)) {
    if (!definitions.has(field)) continue;
    if (definitions.get(field) === "array")
      normalized[field] = [...new Set(rawValues(value, field))];
    else {
      const candidate = Array.isArray(value)
        ? value.find((entry) => entry !== null && entry !== "")
        : value;
      normalized[field] = collapse(candidate, field) || null;
    }
  }
  const moves = new Map<string, string[]>();
  for (const field of detailFields) {
    const values = normalized[field];
    if (!Array.isArray(values)) continue;
    normalized[field] = values.filter((value) => {
      const target = taxonomy.detailOwners.get(value);
      if (!target || target === field || definitions.get(target) !== "array")
        return true;
      moves.set(target, [...(moves.get(target) ?? []), value]);
      return false;
    });
  }
  for (const [target, values] of moves)
    normalized[target] = [
      ...new Set([
        ...(Array.isArray(normalized[target]) ? normalized[target] : []),
        ...values,
      ]),
    ];

  for (const [source, ownership] of Object.entries(rules.crossFieldOwnership)) {
    const values = normalized[source];
    if (!Array.isArray(values)) continue;
    const kept: string[] = [];
    for (const value of values) {
      if (source === "material" && value === "braided_cord") {
        kept.push("cord");
        const structure = normalized.structure;
        if (Array.isArray(structure) && !structure.includes("braided"))
          structure.push("braided");
        continue;
      }
      const rule = ownership[value];
      if (!rule) {
        kept.push(value);
        continue;
      }
      const [target, token] = rule;
      if (!target) continue;
      if (definitions.get(target) !== "array") {
        kept.push(value);
        continue;
      }
      const destination = Array.isArray(normalized[target])
        ? normalized[target]
        : [];
      const normalizedToken = collapse(token || value, target);
      if (normalizedToken && !destination.includes(normalizedToken))
        destination.push(normalizedToken);
      normalized[target] = destination;
    }
    normalized[source] = kept;
  }
  if (normalized.neckline === "strapless") {
    if (!normalized.shoulder_style) normalized.shoulder_style = "strapless";
    normalized.neckline = null;
  }
  if (
    normalized.neckline === "off_shoulder" &&
    normalized.shoulder_style === "off_shoulder"
  )
    normalized.neckline = null;

  let category =
    typeof normalized.category === "string" ? normalized.category : "";
  let subcategory =
    typeof normalized.subcategory === "string" ? normalized.subcategory : "";
  const hierarchy = taxonomy.hierarchy(type);
  const inferredParent = hierarchy[subcategory];
  if (!category && inferredParent) {
    category = inferredParent;
    normalized.category = category;
  }
  const promotedParent = hierarchy[category];
  if (
    category &&
    !taxonomy.scoped("category", type).includes(category) &&
    promotedParent
  ) {
    const promoted = category;
    category = promotedParent;
    normalized.category = category;
    if (!subcategory || subcategory === category) {
      subcategory = promoted;
      normalized.subcategory = subcategory;
    }
  }
  const finalParent = hierarchy[subcategory];
  if (
    finalParent &&
    (!category || !taxonomy.ancestors(type, subcategory).includes(category))
  ) {
    normalized.category = finalParent;
    category = finalParent;
  }
  if (subcategory && subcategory === category) normalized.subcategory = null;
  for (const field of detailFields) {
    const values = normalized[field];
    if (Array.isArray(values))
      normalized[field] = values.filter(
        (value) =>
          value !== normalized.category && value !== normalized.subcategory,
      );
  }
  if (type === "shoes" && normalized.category !== "barefoot") {
    if (!normalized.heel_height) normalized.heel_height = "flat";
    if (!normalized.sole_height) normalized.sole_height = "flat";
  }
  const length = normalized.bottom_length;
  if (category && typeof length === "string")
    normalized.bottom_length =
      rules.bottomLengthAliases[category]?.[length] ?? length;
  return normalized;
}

export function parseResponse(text: string): {
  payload: Payload | null;
  error: string | null;
} {
  if (!text.trim()) return { payload: null, error: "empty_response" };
  try {
    const payload: unknown = JSON.parse(text);
    return isObject(payload)
      ? { payload, error: null }
      : { payload: null, error: "json_root_not_object" };
  } catch {
    return { payload: null, error: "json_decode_error:invalid_json" };
  }
}

export function buildFilterReport(
  taxonomy: Taxonomy,
  type: string,
  raw: Payload | null,
  data: Data,
): Payload {
  type = taxonomy.normalizeType(type);
  raw ??= {};
  const fields = new Map(
    taxonomy.fields(type).map((field) => [field.name, field.kind]),
  );
  const nonCanonical: Payload[] = [];
  const crossFieldOwnership: Payload[] = [];
  const seen = new Set<string>();
  for (const [field, value] of Object.entries(raw)) {
    if (!fields.has(field)) continue;
    const canonical =
      field === "category"
        ? taxonomy.scoped("category", type)
        : fields.get(field) === "scalar" && field !== "subcategory"
          ? taxonomy.shared(field)
          : [];
    for (const token of rawValues(value, field)) {
      if (canonical.length && !canonical.includes(token))
        nonCanonical.push({ field, token });
      const ownership = rules.crossFieldOwnership[field]?.[token];
      if (ownership) {
        const fingerprint = JSON.stringify([field, token, ...ownership]);
        if (!seen.has(fingerprint)) {
          seen.add(fingerprint);
          crossFieldOwnership.push({
            field,
            token,
            target_field: ownership[0],
            target_token: ownership[1],
          });
        }
      }
    }
  }
  const unknown = Object.keys(raw)
    .filter((field) => !fields.has(field))
    .sort();
  const category = typeof data.category === "string" ? data.category : "";
  const subcategory =
    typeof data.subcategory === "string" ? data.subcategory : "";
  const parent = taxonomy.hierarchy(type)[subcategory];
  const mismatch =
    category &&
    subcategory &&
    parent &&
    !taxonomy.ancestors(type, subcategory).includes(category)
      ? { item_type: type, category, subcategory, expected_parent: parent }
      : null;
  const unlisted =
    subcategory && !taxonomy.scoped("subcategory", type).includes(subcategory)
      ? { item_type: type, category, subcategory }
      : null;
  return {
    cross_field_ownership: crossFieldOwnership,
    cross_field_ownership_count: crossFieldOwnership.length,
    non_canonical: nonCanonical,
    non_canonical_count: nonCanonical.length,
    unknown_fields: unknown,
    unknown_field_count: unknown.length,
    parent_child_mismatch: mismatch,
    parent_child_mismatch_count: mismatch ? 1 : 0,
    subcategory_not_in_list: unlisted,
    subcategory_not_in_list_count: unlisted ? 1 : 0,
  };
}
