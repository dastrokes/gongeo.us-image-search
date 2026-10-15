import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { z } from "zod";
import { parseJsonl, readJsonl, serializeJsonl } from "../src/artifacts.js";
import {
  buildSummarySchema,
  colorDebugRecordSchema,
  colorManifestRecordSchema,
  itemAttributesRecordSchema,
  itemColorsRecordSchema,
  manifestRecordSchema,
  structuredDebugRecordSchema,
  structuredItemRecordSchema,
} from "../src/contracts.js";

const referenceSchema = z.object({
  format_version: z.literal(1),
  extraction: z.array(
    z.object({
      name: z.string(),
      structured: structuredItemRecordSchema,
      debug: structuredDebugRecordSchema,
      attributes: itemAttributesRecordSchema,
    }),
  ),
  manifest: z.array(z.object({ rows: z.array(manifestRecordSchema) })),
  color_manifest: z.object({ rows: z.array(colorManifestRecordSchema) }),
  colors: z.array(
    z.object({
      record: itemColorsRecordSchema,
      debug: colorDebugRecordSchema,
    }),
  ),
  refresh: z.object({
    attributes: z.array(itemAttributesRecordSchema),
    summary: buildSummarySchema,
  }),
});

const reference = referenceSchema.parse(
  JSON.parse(
    await readFile(
      new URL("fixtures/python-reference.json", import.meta.url),
      "utf8",
    ),
  ),
);

test("Python artifacts satisfy the TypeScript contracts without losing fields", () => {
  for (const row of reference.extraction) {
    assert.deepEqual(
      structuredItemRecordSchema.parse(row.structured),
      row.structured,
    );
    assert.deepEqual(structuredDebugRecordSchema.parse(row.debug), row.debug);
    assert.deepEqual(
      itemAttributesRecordSchema.parse(row.attributes),
      row.attributes,
    );
  }
  for (const row of reference.colors) {
    assert.deepEqual(itemColorsRecordSchema.parse(row.record), row.record);
    assert.deepEqual(colorDebugRecordSchema.parse(row.debug), row.debug);
  }
});

test("final attributes omit inapplicable metadata and reject repeated identity", () => {
  const row = reference.refresh.attributes[0];
  assert.ok(row);
  for (const invalid of [null, "", [], { nested: "value" }]) {
    assert.equal(
      itemAttributesRecordSchema.safeParse({
        ...row,
        metadata: { ornament: invalid },
      }).success,
      false,
    );
  }
  for (const key of [
    "item_id",
    "item_type",
    "slot",
    "category",
    "subcategory",
  ]) {
    assert.equal(
      itemAttributesRecordSchema.safeParse({
        ...row,
        metadata: { [key]: "unexpected" },
      }).success,
      false,
    );
  }
  assert.equal(
    itemAttributesRecordSchema.safeParse({
      ...row,
      metadata: { ornament: ["new_reviewable_token"] },
    }).success,
    true,
  );
});

test("artifact identity rejects invalid IDs and unexpected top-level fields", () => {
  for (const id of [0, -1, 1.5, true, "123", Number.MAX_SAFE_INTEGER + 1]) {
    assert.equal(
      manifestRecordSchema.safeParse({ item_id: id, item_type: "hair" })
        .success,
      false,
    );
  }
  assert.equal(
    manifestRecordSchema.safeParse({
      item_id: 1,
      item_type: "hair",
      extra: true,
    }).success,
    false,
  );
});

test("JSONL retains reference row/array order and UTF-8 text across CRLF and BOM", () => {
  const rows = reference.refresh.attributes;
  assert.deepEqual(
    parseJsonl(serializeJsonl(rows), itemAttributesRecordSchema),
    rows,
  );
  const localizedSchema = z.strictObject({
    text: z.string(),
    tags: z.array(z.string()),
  });
  const localized = [{ text: "外套 披肩/斗篷 蝴蝶结", tags: ["bow", "bell"] }];
  const text = `\uFEFF\r\n${serializeJsonl(localized).replace(/\n/g, "\r\n")}\r\n`;
  assert.deepEqual(parseJsonl(text, localizedSchema), localized);
  assert.equal(serializeJsonl([]), "");
});

test("JSONL reports the source and physical line for syntax and schema errors", () => {
  const valid = '{"item_id":1,"item_type":"hair"}';
  for (const invalid of [
    '{"item_id":',
    '{"item_id":false,"item_type":"hair"}',
  ]) {
    assert.throws(
      () =>
        parseJsonl(
          `${valid}\n\n${invalid}`,
          manifestRecordSchema,
          "fixture with spaces.jsonl",
        ),
      (error: unknown) => {
        assert.ok(error instanceof Error);
        assert.equal(
          error.message,
          "Invalid JSONL record at fixture with spaces.jsonl:3",
        );
        assert.ok(error.cause instanceof Error);
        return true;
      },
    );
  }
});

test("JSONL reads files with spaces and propagates missing-file errors", async () => {
  const directory = await mkdtemp(join(tmpdir(), "image-search contracts "));
  try {
    const path = join(directory, "reviewed attributes.jsonl");
    await writeFile(path, serializeJsonl(reference.refresh.attributes), "utf8");
    assert.deepEqual(
      await readJsonl(path, itemAttributesRecordSchema),
      reference.refresh.attributes,
    );
    await assert.rejects(
      readJsonl(join(directory, "missing.jsonl"), itemAttributesRecordSchema),
      { code: "ENOENT" },
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
