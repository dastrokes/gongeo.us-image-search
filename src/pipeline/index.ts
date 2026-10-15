import { createHash } from "node:crypto";
import { readdir, readFile, stat } from "node:fs/promises";
import { extname, join } from "node:path";
import { isDeepStrictEqual } from "node:util";
import { z } from "zod";
import {
  atomicWrite,
  isMissing,
  readJson,
  readJsonl,
  serializeJsonl,
} from "../artifacts.js";
import { defaultModels, type Paths } from "../config.js";
import { rules } from "../constants/rules.js";
import {
  buildSummarySchema,
  type ManifestRecord,
  manifestRecordSchema,
  type StructuredDebugRecord,
  type StructuredItemRecord,
  structuredDebugRecordSchema,
  structuredItemRecordSchema,
} from "../contracts.js";
import {
  createAdapter,
  type ExtractionAdapter,
  type ImageInput,
  type ProviderName,
  requestSettings,
  responseShape,
} from "../providers/provider.js";
import { applyOverride, buildSearchReport, loadOverrides } from "./finalize.js";
import {
  buildManifest,
  findImage,
  loadPublishedIds,
  type Selection,
  selectRecords,
} from "./manifest.js";
import {
  buildFilterReport,
  normalizePayload,
  parseResponse,
} from "./normalization.js";
import { buildPrompt, buildPromptParts } from "./prompts.js";
import { assertOutputScope } from "./refresh.js";
import type { Taxonomy } from "./taxonomy.js";

const descriptorSchema = z.strictObject({
  version: z.literal(1),
  provider: z.enum(["google", "vercel"]),
  model: z.string(),
  config_hash: z.string(),
  items: z.array(manifestRecordSchema.extend({ input_hash: z.string() })),
});
const checkpointSchema = z.strictObject({
  version: z.literal(1),
  input_hash: z.string(),
  structured: structuredItemRecordSchema,
  debug: structuredDebugRecordSchema,
});
type Checkpoint = z.infer<typeof checkpointSchema>;
type Descriptor = z.infer<typeof descriptorSchema>;
const digest = (content: string | Uint8Array) =>
  createHash("sha256").update(content).digest("hex");

export interface IndexOptions extends Selection {
  provider?: ProviderName;
  model?: string;
  regenManifest?: boolean;
  includeIndexed?: boolean;
  maxItems?: number;
  batchSize?: number;
}
interface PreparedItem {
  record: ManifestRecord;
  input_hash: string;
  image_paths: { overview: string | null; icon: string | null };
}

async function imagesFor(
  paths: Paths,
  record: ManifestRecord,
): Promise<ImageInput[]> {
  const images: ImageInput[] = [];
  for (const [label, root] of [
    ["overview", paths.imageRoot],
    ["icon", paths.iconRoot],
  ] as const) {
    const path = await findImage(root, record.item_id);
    if (!path) continue;
    const mediaType =
      {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
      }[extname(path)] ?? "image/jpeg";
    images.push({ label, path, mediaType, bytes: await readFile(path) });
  }
  if (!images.length) throw new Error(`No images for item ${record.item_id}`);
  return images;
}

function imageIdentity(images: ImageInput[], record: ManifestRecord): string {
  return digest(
    JSON.stringify({
      item_type: record.item_type,
      images: images.map(({ label, mediaType, bytes }) => ({
        label,
        mediaType,
        hash: digest(bytes),
      })),
    }),
  );
}

async function existing(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch (error) {
    if (isMissing(error)) return false;
    throw error;
  }
}

async function ensureDescriptor(root: string, descriptor: Descriptor) {
  const path = join(root, "extraction-run.json");
  if (await existing(path)) {
    const previous = descriptorSchema.parse(await readJson(path));
    if (!isDeepStrictEqual(previous, descriptor))
      throw new Error(
        "Extraction inputs/provider/model differ from this run. Choose a separate --output-root.",
      );
    return;
  }
  const artifacts = [
    "item-structured-data.jsonl",
    "item-structured-debug.jsonl",
    "item-structured-raw.jsonl",
    "item-attributes.jsonl",
    "build-summary.json",
  ];
  for (const name of artifacts)
    if (await existing(join(root, name)))
      throw new Error(
        "Existing extraction artifacts have no verified run identity. Use offline refresh, or choose a separate --output-root for new extraction.",
      );
  let checkpoints: string[] = [];
  try {
    checkpoints = await readdir(join(root, ".checkpoints"));
  } catch (error) {
    if (!isMissing(error)) throw error;
  }
  if (checkpoints.some((name) => name.endsWith(".json")))
    throw new Error(
      "Checkpoints have no run descriptor. Choose a separate --output-root.",
    );
  await atomicWrite(path, `${JSON.stringify(descriptor, null, 2)}\n`);
}

async function loadCheckpoint(
  root: string,
  item: PreparedItem,
  descriptor: Descriptor,
): Promise<Checkpoint | null> {
  const path = join(root, ".checkpoints", `${item.record.item_id}.json`);
  let source: unknown;
  try {
    source = await readJson(path);
  } catch (error) {
    if (isMissing(error)) return null;
    throw error;
  }
  const value = checkpointSchema.parse(source);
  if (
    value.input_hash !== item.input_hash ||
    value.structured.item_id !== item.record.item_id ||
    value.debug.item_id !== item.record.item_id ||
    value.structured.item_type !== item.record.item_type ||
    value.debug.item_type !== item.record.item_type ||
    value.debug.request?.provider !== descriptor.provider ||
    value.debug.request.model !== descriptor.model ||
    value.structured.parse_error !== value.debug.parse_error
  )
    throw new Error(`Checkpoint identity mismatch: ${path}`);
  return value;
}

async function extractItem(
  paths: Paths,
  taxonomy: Taxonomy,
  item: PreparedItem,
  provider: ProviderName,
  model: string,
  adapter: ExtractionAdapter,
): Promise<Checkpoint> {
  const images = await imagesFor(paths, item.record);
  if (imageIdentity(images, item.record) !== item.input_hash)
    throw new Error(
      `Images changed during extraction for ${item.record.item_id}; use a separate output root after reviewing the source.`,
    );
  const prompts = buildPromptParts(taxonomy, item.record.item_type);
  const result = await adapter.extract({
    ...prompts,
    schema: responseShape(taxonomy, item.record.item_type),
    images,
  });
  const { payload, error } = parseResponse(result.rawText);
  const shape = responseShape(taxonomy, item.record.item_type).safeParse(
    payload,
  );
  const parseError =
    result.error ??
    error ??
    (shape.success ? null : "invalid_structured_response");
  const data = normalizePayload(taxonomy, item.record.item_type, payload);
  const structured = structuredItemRecordSchema.parse({
    ...item.record,
    data,
    parse_error: parseError,
  });
  const debug = structuredDebugRecordSchema.parse({
    ...item.record,
    image_paths: item.image_paths,
    prompt: buildPrompt(taxonomy, item.record.item_type),
    raw_response: result.rawText,
    raw_payload: payload,
    normalized_data: data,
    filter_report: buildFilterReport(
      taxonomy,
      item.record.item_type,
      payload,
      data,
    ),
    parse_error: parseError,
    request: {
      provider,
      model,
      finish_reason: result.finishReason,
      usage: result.usage,
    },
  });
  return { version: 1, input_hash: item.input_hash, structured, debug };
}

export async function runIndex(
  paths: Paths,
  taxonomy: Taxonomy,
  options: IndexOptions = {},
  adapterFactory: (
    provider: ProviderName,
    model: string,
  ) => ExtractionAdapter = createAdapter,
) {
  const started = Date.now();
  const provider = options.provider ?? "google";
  const model = options.model ?? defaultModels[provider];
  const maxItems = options.maxItems ?? 500;
  const batchSize = options.batchSize ?? 10;
  if (
    !Number.isSafeInteger(maxItems) ||
    maxItems <= 0 ||
    !Number.isSafeInteger(batchSize) ||
    batchSize <= 0
  )
    throw new Error("--max-items and --batch-size must be positive integers");
  if (options.includeIndexed && !options.regenManifest)
    throw new Error("--include-indexed requires --regen-manifest");
  selectRecords(taxonomy, [], options);
  const singleDebug =
    options.itemId !== undefined &&
    !options.itemIds?.length &&
    !options.types?.length;
  const manifestPath = join(paths.manifestRoot, "item-manifest.jsonl");
  let rows: ManifestRecord[];
  let stats = {
    skipped_count: 0,
    missing_icon_count: 0,
    missing_overview_count: 0,
  };
  const regenerate = options.regenManifest || !(await existing(manifestPath));
  if (regenerate) {
    const generated = await buildManifest(
      paths,
      taxonomy,
      options,
      options.includeIndexed,
    );
    rows = generated.rows;
    stats = {
      skipped_count: generated.stats.skipped_count,
      missing_icon_count: generated.stats.missing_icon_count,
      missing_overview_count: generated.stats.missing_overview_count,
    };
  } else {
    rows = selectRecords(
      taxonomy,
      await readJsonl(manifestPath, manifestRecordSchema),
      options,
    );
    if (
      !options.itemId &&
      !options.itemIds?.length &&
      !options.includeIndexed
    ) {
      const published = await loadPublishedIds(paths.attributesPath);
      if (rows.some((row) => published.has(row.item_id)))
        throw new Error(
          "Cached manifest contains published IDs; rerun --regen-manifest with the verified D1 snapshot.",
        );
    }
  }
  if (new Set(rows.map((row) => row.item_id)).size !== rows.length)
    throw new Error("Manifest contains duplicate item IDs");
  if (rows.length > maxItems)
    throw new Error(
      `Manifest has ${rows.length} items, above --max-items ${maxItems}. Inspect the snapshot and scope before raising the ceiling.`,
    );
  if (singleDebug && !rows.length)
    throw new Error(`Item ${options.itemId} not found`);

  const prepared: PreparedItem[] = [];
  for (const record of rows) {
    const images = await imagesFor(paths, record);
    prepared.push({
      record,
      input_hash: imageIdentity(images, record),
      image_paths: {
        overview:
          images.find((image) => image.label === "overview")?.path ?? null,
        icon: images.find((image) => image.label === "icon")?.path ?? null,
      },
    });
  }
  if (!regenerate)
    stats = {
      ...stats,
      missing_icon_count: prepared.filter((item) => !item.image_paths.icon)
        .length,
      missing_overview_count: prepared.filter(
        (item) => !item.image_paths.overview,
      ).length,
    };
  if (singleDebug) {
    const item = prepared[0];
    if (!item) throw new Error("Missing debug item");
    const result = await extractItem(
      paths,
      taxonomy,
      item,
      provider,
      model,
      adapterFactory(provider, model),
    );
    return {
      kind: "debug" as const,
      bundle: {
        ...item.record,
        image_paths: item.image_paths,
        structured_debug: result.debug,
      },
    };
  }

  await assertOutputScope(
    join(paths.outputRoot, "item-attributes.jsonl"),
    new Set(rows.map((row) => row.item_id)),
  );
  const descriptor: Descriptor = {
    version: 1,
    provider,
    model,
    config_hash: digest(
      JSON.stringify({
        adapter_version: 1,
        settings: requestSettings(model),
        taxonomy: taxonomy.source,
        rules,
      }),
    ),
    items: prepared.map((item) => ({
      ...item.record,
      input_hash: item.input_hash,
    })),
  };
  await ensureDescriptor(paths.outputRoot, descriptor);
  const cache = new Map<number, Checkpoint>();
  // Preflight every checkpoint before making another paid request.
  for (const item of prepared) {
    const checkpoint = await loadCheckpoint(paths.outputRoot, item, descriptor);
    if (checkpoint) cache.set(item.record.item_id, checkpoint);
  }
  const overrides = await loadOverrides(paths.overridesPath);
  const statusPath = join(paths.outputRoot, "run-status.json");
  await atomicWrite(
    statusPath,
    `${JSON.stringify({ operation: "index", status: "extracting", expected_count: rows.length })}\n`,
  );
  if (
    regenerate &&
    !options.itemIds?.length &&
    !options.types?.length &&
    options.itemId === undefined
  )
    await atomicWrite(manifestPath, serializeJsonl(rows));
  const pending = prepared.filter(
    (item) => cache.get(item.record.item_id)?.structured.parse_error !== null,
  );
  console.error(
    `Extraction: ${rows.length} selected, ${rows.length - pending.length} resumed, ${pending.length} pending (${provider}/${model})`,
  );
  let fatal: unknown;
  try {
    const adapter = pending.length ? adapterFactory(provider, model) : null;
    for (const [index, item] of pending.entries()) {
      if (!adapter) throw new Error("Missing extraction adapter");
      const checkpoint = await extractItem(
        paths,
        taxonomy,
        item,
        provider,
        model,
        adapter,
      );
      await atomicWrite(
        join(paths.outputRoot, ".checkpoints", `${item.record.item_id}.json`),
        `${JSON.stringify(checkpoint)}\n`,
      );
      cache.set(item.record.item_id, checkpoint);
      if ((index + 1) % batchSize === 0 || index + 1 === pending.length)
        console.error(
          `Checkpointed ${rows.length - pending.length + index + 1}/${rows.length}`,
        );
    }
  } catch (error) {
    fatal = error;
  }

  const structured: StructuredItemRecord[] = [];
  const diagnostics: StructuredDebugRecord[] = [];
  const attributes = [];
  const reports = [];
  for (const item of prepared) {
    const value = cache.get(item.record.item_id);
    if (!value) continue;
    const record = {
      ...value.structured,
      data: normalizePayload(
        taxonomy,
        item.record.item_type,
        value.debug.raw_payload,
      ),
    };
    const debug = {
      ...value.debug,
      image_paths: item.image_paths,
      normalized_data: record.data,
    };
    structured.push(record);
    diagnostics.push(debug);
    if (record.parse_error === null) {
      const final = applyOverride(
        taxonomy,
        record,
        overrides.get(record.item_id),
      );
      attributes.push(final);
      const report = buildSearchReport(taxonomy, record, final, debug);
      if (report) reports.push(report);
    }
  }
  const failures = structured.filter((row) => row.parse_error !== null).length;
  const complete =
    fatal === undefined && failures === 0 && structured.length === rows.length;
  const summary = buildSummarySchema.parse({
    extraction_model_id: model,
    item_count: attributes.length,
    ...stats,
    structured_parse_fail_count: failures,
    build_started_at: new Date(started).toISOString(),
    build_finished_at: new Date().toISOString(),
    duration_seconds: (Date.now() - started) / 1000,
    item_attributes_path: join(paths.outputRoot, "item-attributes.jsonl"),
    search_report_path: join(paths.outputRoot, "item-search-report.jsonl"),
  });
  await atomicWrite(
    statusPath,
    `${JSON.stringify({ operation: "index", status: "finalizing", expected_count: rows.length })}\n`,
  );
  await atomicWrite(
    join(paths.outputRoot, "item-structured-data.jsonl"),
    serializeJsonl(structured),
  );
  await atomicWrite(
    join(paths.outputRoot, "item-structured-debug.jsonl"),
    serializeJsonl(diagnostics),
  );
  await atomicWrite(
    join(paths.outputRoot, "item-structured-raw.jsonl"),
    serializeJsonl(
      diagnostics.map((row) => ({
        item_id: row.item_id,
        item_type: row.item_type,
        data: row.raw_payload ?? {},
        parse_error: row.parse_error,
      })),
    ),
  );
  if (complete) {
    await atomicWrite(
      join(paths.outputRoot, "item-attributes.jsonl"),
      serializeJsonl(attributes),
    );
    await atomicWrite(
      join(paths.outputRoot, "item-search-report.jsonl"),
      serializeJsonl(reports),
    );
  }
  const summaryPath = join(paths.outputRoot, "build-summary.json");
  await atomicWrite(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
  await atomicWrite(
    statusPath,
    `${JSON.stringify({ operation: "index", status: complete ? "complete" : "failed", expected_count: rows.length, completed_count: attributes.length, parse_failure_count: failures })}\n`,
  );
  if (fatal !== undefined) throw fatal;
  if (!complete)
    throw new Error(
      `Extraction has ${failures} invalid responses; diagnostics and checkpoints retained. Rerun the same command to retry failed items.`,
    );
  return {
    kind: "batch" as const,
    summary: { summary_path: summaryPath, ...summary },
  };
}
