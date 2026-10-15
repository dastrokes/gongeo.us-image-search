import { z } from "zod";
import { readJson } from "../artifacts.js";

const strings = z.record(z.string(), z.string());
const lists = z.record(z.string(), z.array(z.string()));
const taxonomySchema = z.object({
  supportedItemTypes: z.array(z.string()).min(1),
  itemTypeAliasToSupported: strings,
  fieldKindByName: z.record(z.string(), z.enum(["scalar", "array"])),
  schemaKeyByItemType: strings,
  fieldNamesByItemType: lists,
  terms: z.object({
    sharedFieldValues: lists,
    scopedFieldValues: z.record(z.string(), lists),
  }),
  taxonomy: z.object({
    subcategoryParentByType: z.record(z.string(), strings),
  }),
});

export class Taxonomy {
  readonly source: z.infer<typeof taxonomySchema>;
  readonly detailOwners = new Map<string, string>();

  constructor(input: unknown) {
    this.source = taxonomySchema.parse(input);
    const ambiguous = new Set<string>();
    for (const field of ["pattern", "material", "structure", "ornament"]) {
      for (const token of this.shared(field)) {
        const owner = this.detailOwners.get(token);
        if (owner && owner !== field) ambiguous.add(token);
        else this.detailOwners.set(token, field);
      }
    }
    for (const token of ambiguous) this.detailOwners.delete(token);
    this.validate();
  }

  normalizeType(value: string | undefined | null): string {
    if (!value) return "unknown";
    return (
      (this.source.itemTypeAliasToSupported[value] ?? value).trim() || "unknown"
    );
  }

  supported(value: string): boolean {
    return this.source.supportedItemTypes.includes(this.normalizeType(value));
  }

  fields(type: string) {
    const normalized = this.normalizeType(type);
    const schemaKey =
      this.source.schemaKeyByItemType[normalized] ?? "accessory";
    const fallback = this.source.supportedItemTypes.find(
      (name) => this.source.schemaKeyByItemType[name] === schemaKey,
    );
    const names =
      this.source.fieldNamesByItemType[normalized] ??
      this.source.fieldNamesByItemType[fallback ?? ""] ??
      [];
    return names.map((name) => {
      const kind = this.source.fieldKindByName[name];
      if (!kind) throw new Error(`Unknown schema field: ${name}`);
      return { name, kind };
    });
  }

  shared(field: string): string[] {
    return this.source.terms.sharedFieldValues[field] ?? [];
  }
  scoped(field: string, type: string): string[] {
    return (
      this.source.terms.scopedFieldValues[field]?.[this.normalizeType(type)] ??
      []
    );
  }
  hierarchy(type: string): Record<string, string> {
    return (
      this.source.taxonomy.subcategoryParentByType[this.normalizeType(type)] ??
      {}
    );
  }
  ancestors(type: string, child: string): string[] {
    const hierarchy = this.hierarchy(type);
    const ancestors: string[] = [];
    const visited = new Set([child]);
    let current = hierarchy[child];
    while (current) {
      if (visited.has(current))
        throw new Error(`Taxonomy cycle for ${type}: ${child}`);
      visited.add(current);
      ancestors.push(current);
      current = hierarchy[current];
    }
    return ancestors;
  }

  expandTypes(types: readonly string[] = []): Set<string> {
    const clothing = [
      "hair",
      "dresses",
      "tops",
      "bottoms",
      "outerwear",
      "socks",
      "shoes",
    ].filter((type) => this.supported(type));
    const result = new Set<string>();
    for (const raw of types) {
      const type = this.normalizeType(raw);
      const values =
        type === "clothing"
          ? clothing
          : type === "accessories"
            ? this.source.supportedItemTypes.filter(
                (value) => !clothing.includes(value),
              )
            : [type];
      for (const value of values) {
        if (!this.supported(value))
          throw new Error(`Unknown item type filter: ${value}`);
        result.add(value);
      }
    }
    return result;
  }

  private validate() {
    const supported = new Set(this.source.supportedItemTypes);
    for (const field of ["category", "subcategory"]) {
      const actual = Object.keys(
        this.source.terms.scopedFieldValues[field] ?? {},
      ).map((type) => this.normalizeType(type));
      if (
        actual.length !== supported.size ||
        actual.some((type) => !supported.has(type))
      )
        throw new Error(`Tracker taxonomy mismatch in ${field}`);
    }
    for (const type of supported) {
      const fields = this.fields(type);
      if (!fields.some((field) => field.name === "category"))
        throw new Error(`Missing category field for ${type}`);
      const children = new Set(this.scoped("subcategory", type));
      const hierarchy = this.hierarchy(type);
      if (Object.keys(hierarchy).length !== children.size)
        throw new Error(`Tracker taxonomy parent mismatch: ${type}`);
      for (const child of children) {
        const parent = hierarchy[child];
        if (!parent || !this.scoped("category", type).includes(parent))
          throw new Error(`Invalid taxonomy parent for ${type}/${child}`);
        this.ancestors(type, child);
      }
    }
  }
}

export async function loadTaxonomy(path: string): Promise<Taxonomy> {
  return new Taxonomy(await readJson(path));
}
