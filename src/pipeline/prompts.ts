import { rules } from "../constants/rules.js";
import { outputTemplate } from "./normalization.js";
import type { Taxonomy } from "./taxonomy.js";

export function buildPromptParts(taxonomy: Taxonomy, type: string) {
  type = taxonomy.normalizeType(type);
  const fields = taxonomy.fields(type);
  const closed: string[] = [];
  const categories = taxonomy.scoped("category", type);
  if (categories.length) closed.push(`  category: ${categories.join(", ")}`);
  for (const { name, kind } of fields) {
    if (kind !== "scalar" || name === "category" || name === "subcategory")
      continue;
    const tokens = taxonomy.shared(name);
    if (tokens.length) closed.push(`  ${name}: ${tokens.join(", ")}`);
  }
  const children = fields.some((field) => field.name === "subcategory")
    ? categories.map((category) => {
        const names = Object.entries(taxonomy.hierarchy(type))
          .filter(([, parent]) => parent === category)
          .map(([child]) => child)
          .sort();
        return `  ${category}: ${names.length ? names.join(", ") : "null"}`;
      })
    : [];
  const guidance = [
    rules.itemTypeGuidance[type],
    ...fields.map(({ name }) => {
      const text = rules.fieldGuidance[name];
      const examples = rules.fieldExamples[name];
      return text && examples
        ? `${text} Examples: ${examples.join(", ")}.`
        : text;
    }),
  ].filter(Boolean);
  const parts = [
    `ITEM TYPE\n- slot = ${type}`,
    `SCHEMA TEMPLATE\n${JSON.stringify(outputTemplate(taxonomy, type), null, 2)}`,
    rules.evidencePrompt,
  ];
  if (guidance.length) parts.push(guidance.join("\n"));
  if (closed.length) parts.push(`CLOSED LISTS\n${closed.join("\n")}`);
  if (children.length)
    parts.push(`PREFERRED SUBCATEGORY EXAMPLES\n${children.join("\n")}`);
  parts.push(
    "OUTPUT\n- Return exactly one JSON object matching the schema template.\n- Preserve the field order shown in SCHEMA TEMPLATE.\n- Use null or [] when unsupported, invisible, or uncertain.",
  );
  return { system: rules.systemPrompt, user: parts.join("\n\n") };
}

export function buildPrompt(taxonomy: Taxonomy, type: string): string {
  const { system, user } = buildPromptParts(taxonomy, type);
  return `${system}\n${user}`;
}

export function buildResponseSchema(taxonomy: Taxonomy, type: string) {
  return {
    type: "object",
    properties: Object.fromEntries(
      taxonomy
        .fields(type)
        .map(({ name, kind }) => [
          name,
          kind === "array"
            ? { type: "array", items: { type: "string" }, uniqueItems: true }
            : { type: ["string", "null"] },
        ]),
    ),
    required: taxonomy.fields(type).map((field) => field.name),
    additionalProperties: false,
  };
}
