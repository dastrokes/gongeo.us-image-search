import assert from "node:assert/strict";
import {
  mkdir,
  mkdtemp,
  readdir,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { APICallError } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import { readJson, readJsonl, serializeJsonl } from "../src/artifacts.js";
import { resolvePaths } from "../src/config.js";
import {
  itemAttributesRecordSchema,
  structuredDebugRecordSchema,
} from "../src/contracts.js";
import { runIndex } from "../src/pipeline/index.js";
import { outputTemplate } from "../src/pipeline/normalization.js";
import { loadTaxonomy } from "../src/pipeline/taxonomy.js";
import {
  createAdapter,
  type ExtractionAdapter,
  type ExtractionRequest,
  type ExtractionResult,
  ProviderRequestError,
  responseShape,
  SdkAdapter,
} from "../src/providers/provider.js";

const taxonomy = await loadTaxonomy(
  fileURLToPath(new URL("fixtures/tracker-export.json", import.meta.url)),
);
const rawText = ` { "bangs": null, "category": "loose", "hair_length": "long", "haircut": null, "subcategory": null, "texture": "straight" } `;
const request: ExtractionRequest = {
  system: "extract",
  user: "hair",
  schema: responseShape(taxonomy, "hair"),
  images: [
    {
      label: "overview",
      path: "fixture.png",
      mediaType: "image/png",
      bytes: new Uint8Array([1, 2]),
    },
  ],
};
const generation = (text = rawText) => ({
  content: [{ type: "text" as const, text }],
  finishReason: { unified: "stop" as const, raw: "stop" },
  usage: {
    inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 4, text: 4, reasoning: 0 },
  },
  warnings: [],
});
const validResult = (): ExtractionResult => ({
  rawText: JSON.stringify({
    ...outputTemplate(taxonomy, "hair"),
    category: "loose",
  }),
  error: null,
  finishReason: "stop",
  usage: { input_tokens: 10, output_tokens: 4 },
});

test("SDK keeps the original response text, schema validation, and image labels", async () => {
  const model = new MockLanguageModelV4({ doGenerate: generation() });
  const adapter = new SdkAdapter("vercel", "google/gemini-2.5-flash", model);
  const result = await adapter.extract(request);
  assert.equal(result.rawText, rawText);
  assert.equal(result.error, null);
  assert.equal(result.usage.total_tokens, 14);
  assert.equal(model.doGenerateCalls.length, 1);
  const call = model.doGenerateCalls[0];
  assert.equal(call?.temperature, 0);
  assert.equal(call?.responseFormat?.type, "json");
  assert.match(JSON.stringify(call?.prompt), /IMAGE: overview/);
});

test("SDK failures retain malformed or schema-invalid raw responses without retrying", async () => {
  for (const text of ['{"category":', '{"category":"loose"}']) {
    const model = new MockLanguageModelV4({ doGenerate: generation(text) });
    const adapter = new SdkAdapter("vercel", "fixture", model);
    const result = await adapter.extract(request);
    assert.equal(result.error, "invalid_structured_response");
    assert.equal(result.rawText, text);
    assert.equal(model.doGenerateCalls.length, 1);
  }
});

test("one retry layer honors Retry-After, bounded attempts, and Google spacing", async () => {
  let time = 0;
  const waits: number[] = [];
  let attempts = 0;
  const model = new MockLanguageModelV4({
    doGenerate: async () => {
      if (attempts++ === 0)
        throw new APICallError({
          message: "rate limited",
          url: "https://fixture.invalid",
          requestBodyValues: {},
          statusCode: 429,
          responseHeaders: { "Retry-After": "1.5" },
        });
      return generation();
    },
  });
  const adapter = new SdkAdapter("google", "gemini-3.5-flash-lite", model, {
    now: () => time,
    sleep: async (delay) => {
      waits.push(delay);
      time += delay;
    },
  });
  await adapter.extract(request);
  await adapter.extract(request);
  assert.equal(attempts, 3);
  assert.deepEqual(waits, [1500, 2500, 4000]);

  let failures = 0;
  const failing = new MockLanguageModelV4({
    doGenerate: async () => {
      failures++;
      throw new APICallError({
        message: "server",
        url: "https://fixture.invalid",
        requestBodyValues: {},
        statusCode: 503,
      });
    },
  });
  await assert.rejects(
    new SdkAdapter("vercel", "fixture", failing, {
      maxRetries: 2,
      sleep: async () => {},
    }).extract(request),
    /HTTP 503/,
  );
  assert.equal(failures, 3);
});

test("authorization errors fail once and request timeouts abort the model call", async () => {
  const model = new MockLanguageModelV4({
    doGenerate: async () => {
      throw new APICallError({
        message: "sensitive provider error",
        url: "https://fixture.invalid",
        requestBodyValues: { secret: "do-not-log" },
        statusCode: 403,
      });
    },
  });
  await assert.rejects(
    new SdkAdapter("vercel", "fixture", model).extract(request),
    (error: unknown) => {
      assert.ok(error instanceof ProviderRequestError);
      assert.equal(error.status, 403);
      assert.doesNotMatch(error.message, /sensitive|do-not-log/);
      return true;
    },
  );
  assert.equal(model.doGenerateCalls.length, 1);
  const slow = new MockLanguageModelV4({
    doGenerate: async ({ abortSignal }) => {
      await sleep(1000, undefined, { signal: abortSignal });
      return generation();
    },
  });
  await assert.rejects(
    new SdkAdapter("vercel", "fixture", slow, {
      timeoutMs: 5,
      maxRetries: 0,
    }).extract(request),
    /network\/timeout/,
  );
});

test("explicit provider factories use the correct host and credential without fallback", async () => {
  assert.throws(() => createAdapter("google", "fixture", {}), /GOOGLE_API_KEY/);
  assert.throws(
    () => createAdapter("vercel", "fixture", {}),
    /AI_GATEWAY_API_KEY/,
  );
  const googleUrls: string[] = [];
  const direct = createAdapter(
    "google",
    "gemini-3.5-flash-lite",
    { GOOGLE_API_KEY: "fixture-key" },
    async (url) => {
      googleUrls.push(String(url));
      return Response.json({
        candidates: [
          {
            content: { role: "model", parts: [{ text: rawText }] },
            finishReason: "STOP",
            index: 0,
          },
        ],
        usageMetadata: {
          promptTokenCount: 10,
          candidatesTokenCount: 4,
          totalTokenCount: 14,
        },
      });
    },
  );
  assert.equal((await direct.extract(request)).error, null);
  assert.equal(googleUrls.length, 1);
  assert.ok(
    googleUrls[0]?.startsWith("https://generativelanguage.googleapis.com/"),
  );
  const gatewayUrls: string[] = [];
  const gateway = createAdapter(
    "vercel",
    "google/gemini-2.5-flash",
    { AI_GATEWAY_API_KEY: "fixture-key" },
    async (url) => {
      gatewayUrls.push(String(url));
      return Response.json({ error: { message: "denied" } }, { status: 403 });
    },
  );
  await assert.rejects(gateway.extract(request), /HTTP 403/);
  assert.equal(gatewayUrls.length, 1);
  assert.ok(gatewayUrls[0]?.startsWith("https://ai-gateway.vercel.sh/"));
});

async function workspace(root: string) {
  const paths = resolvePaths(
    {
      trackerRoot: join(root, "tracker"),
      configRoot: join(root, "decoder"),
      syncReport: join(root, "reports/database-sync-report.json"),
      manifestRoot: join(root, "manifest"),
      outputRoot: join(root, "output"),
    },
    {},
  );
  await mkdir(paths.manifestRoot, { recursive: true });
  await mkdir(paths.iconRoot, { recursive: true });
  await writeFile(
    join(paths.manifestRoot, "item-manifest.jsonl"),
    serializeJsonl([
      { item_id: 11, item_type: "hair" },
      { item_id: 12, item_type: "hair" },
    ]),
  );
  for (const id of [11, 12])
    await writeFile(join(paths.imageRoot, `${id}.png`), "fixture-image");
  return paths;
}

const options = {
  provider: "vercel" as const,
  model: "google/gemini-2.5-flash",
  itemIds: [11, 12],
};
const good: ExtractionAdapter = { extract: async () => validResult() };

test("interrupted extraction resumes only pending items and rebuilds stable artifacts", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "image-search checkpoint "));
  t.after(() => rm(root, { recursive: true, force: true }));
  const paths = await workspace(root);
  let calls = 0;
  await assert.rejects(
    runIndex(paths, taxonomy, options, () => ({
      extract: async () => {
        if (++calls === 2) throw new ProviderRequestError("vercel", 503, true);
        return validResult();
      },
    })),
    /HTTP 503/,
  );
  assert.equal(calls, 2);
  assert.deepEqual(await readJson(join(paths.outputRoot, "run-status.json")), {
    operation: "index",
    status: "failed",
    expected_count: 2,
    completed_count: 1,
    parse_failure_count: 0,
  });
  assert.deepEqual(
    (await readdir(join(paths.outputRoot, ".checkpoints"))).filter((name) =>
      name.endsWith(".json"),
    ),
    ["11.json"],
  );
  await assert.rejects(
    readFile(join(paths.outputRoot, "item-attributes.jsonl")),
    { code: "ENOENT" },
  );
  let resumed = 0;
  const result = await runIndex(paths, taxonomy, options, () => ({
    extract: async () => {
      resumed++;
      return validResult();
    },
  }));
  assert.equal(resumed, 1);
  assert.equal(result.kind, "batch");
  assert.deepEqual(
    (
      await readJsonl(
        join(paths.outputRoot, "item-attributes.jsonl"),
        itemAttributesRecordSchema,
      )
    ).map((row) => row.item_id),
    [11, 12],
  );
  await writeFile(
    join(paths.outputRoot, ".checkpoints/11.json.unfinished.tmp"),
    "partial write",
  );
  await runIndex(paths, taxonomy, options, () => {
    throw new Error(
      "Completed resume must not need a credential or model call",
    );
  });
  assert.equal(
    (
      await readJsonl(
        join(paths.outputRoot, "item-structured-debug.jsonl"),
        structuredDebugRecordSchema,
      )
    ).length,
    2,
  );
});

test("changed model, schema, images, or malformed checkpoints reject before any request", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "image-search identity "));
  t.after(() => rm(root, { recursive: true, force: true }));
  const paths = await workspace(root);
  await runIndex(paths, taxonomy, options, () => good);
  const noCall = () => {
    throw new Error("unexpected provider construction");
  };
  await assert.rejects(
    runIndex(paths, taxonomy, { ...options, model: "other" }, noCall),
    /differ/,
  );
  await assert.rejects(
    runIndex(paths, taxonomy, { ...options, provider: "google" }, noCall),
    /differ/,
  );
  const changed = structuredClone(taxonomy.source);
  changed.terms.sharedFieldValues.material?.push("test_new_material");
  const { Taxonomy } = await import("../src/pipeline/taxonomy.js");
  await assert.rejects(
    runIndex(paths, new Taxonomy(changed), options, noCall),
    /differ/,
  );
  await writeFile(join(paths.imageRoot, "11.png"), "changed-image");
  await assert.rejects(runIndex(paths, taxonomy, options, noCall), /differ/);
  await writeFile(join(paths.imageRoot, "11.png"), "fixture-image");
  await writeFile(
    join(paths.outputRoot, ".checkpoints/12.json"),
    '{"unfinished":',
  );
  await assert.rejects(
    runIndex(paths, taxonomy, options, noCall),
    /Cannot read JSON/,
  );
});

test("invalid responses retain raw diagnostics and retry only failed items", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "image-search failed response "));
  t.after(() => rm(root, { recursive: true, force: true }));
  const paths = await workspace(root);
  let calls = 0;
  await assert.rejects(
    runIndex(paths, taxonomy, options, () => ({
      extract: async () =>
        ++calls === 1
          ? {
              ...validResult(),
              rawText: '{"category":',
              error: "invalid_structured_response",
            }
          : validResult(),
    })),
    /invalid responses/,
  );
  const debug = await readJsonl(
    join(paths.outputRoot, "item-structured-debug.jsonl"),
    structuredDebugRecordSchema,
  );
  assert.equal(debug[0]?.raw_response, '{"category":');
  assert.equal(debug[0]?.parse_error, "invalid_structured_response");
  await assert.rejects(
    readFile(join(paths.outputRoot, "item-attributes.jsonl")),
    { code: "ENOENT" },
  );
  calls = 0;
  await runIndex(paths, taxonomy, options, () => ({
    extract: async () => {
      calls++;
      return validResult();
    },
  }));
  assert.equal(calls, 1);
});

test("ceilings, legacy caches, and single-item debug preserve output boundaries", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "image-search guards "));
  t.after(() => rm(root, { recursive: true, force: true }));
  const paths = await workspace(root);
  const noCall = () => {
    throw new Error("unexpected model call");
  };
  await assert.rejects(
    runIndex(paths, taxonomy, { ...options, maxItems: 1 }, noCall),
    /above --max-items/,
  );
  await mkdir(paths.outputRoot, { recursive: true });
  await writeFile(join(paths.outputRoot, "item-structured-data.jsonl"), "");
  await assert.rejects(
    runIndex(paths, taxonomy, options, noCall),
    /no verified run identity/,
  );
  const before = await readdir(paths.outputRoot);
  const result = await runIndex(
    paths,
    taxonomy,
    { provider: "vercel", itemId: 11 },
    () => good,
  );
  assert.equal(result.kind, "debug");
  assert.deepEqual(await readdir(paths.outputRoot), before);
});

test("regenerated manifests finalize a schema-valid batch summary", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "image-search regenerate "));
  t.after(() => rm(root, { recursive: true, force: true }));
  const paths = await workspace(root);
  const json = async (path: string, value: unknown) => {
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, JSON.stringify(value));
  };
  await json(paths.syncReport, {
    syncedDetails: { items: [{ id: 11 }, { id: 12 }] },
  });
  await json(join(paths.configRoot, "item/TbItem.json"), {
    "11": { minor_type: 1 },
    "12": { minor_type: 1 },
  });
  await json(join(paths.configRoot, "clothes/TbClothesMinorTypeInfo.json"), {
    "1": { l10nshow_name: "LGS03_1" },
  });
  await json(join(paths.trackerRoot, "public/catalog/index.json"), {
    files: { items: { path: "/catalog/items.json" } },
  });
  await json(join(paths.trackerRoot, "public/catalog/items.json"), [
    [11],
    [12],
  ]);
  const result = await runIndex(
    paths,
    taxonomy,
    { ...options, regenManifest: true },
    () => good,
  );
  assert.equal(result.kind, "batch");
  if (result.kind === "batch") {
    assert.equal(result.summary.item_count, 2);
    assert.equal(result.summary.missing_icon_count, 2);
    assert.equal(result.summary.structured_parse_fail_count, 0);
  }
});
