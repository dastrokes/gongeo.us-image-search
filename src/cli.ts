import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { atomicWrite, serializeJsonl } from "./artifacts.js";
import { loadProjectEnv, projectRoot, resolvePaths } from "./config.js";
import {
  auditMissingImages,
  auditNormalizerDrift,
  auditParseErrors,
} from "./pipeline/audits.js";
import { runColors } from "./pipeline/colors.js";
import { type IndexOptions, runIndex } from "./pipeline/index.js";
import { buildManifest, type Selection } from "./pipeline/manifest.js";
import { runRefresh } from "./pipeline/refresh.js";
import { loadTaxonomy } from "./pipeline/taxonomy.js";

const help = `Image-search commands:
  npm run manifest -- --item-attributes-path <D1 snapshot> [--limit 10]
  npm run refresh -- [--output-root <directory>] [--item-id <id>]
  npm run index -- --provider google --limit 10 --output-root index/review
  npm run colors -- --limit 10 --output-root index/colors-review
  npm run audit -- parse-errors --output-root index/review
  npm run audit -- normalizer-drift --attributes <D1 snapshot>
  npm run audit -- missing-images --attributes <D1 snapshot>

Shared options: --tracker-root, --config-root, --sync-report,
  --taxonomy-path, --overrides-path, --manifest-root, --output-root,
  --item-id, --item-ids <id> <id>, --type <type> (repeatable), --limit.
Manifest: --output <file>, --include-indexed, --clothing, --accessories.
Refresh: --manifest-path, --structured-path, --structured-debug-path.
Index: --provider google|vercel, --model, --regen-manifest,
  --include-indexed (with --regen-manifest), --max-items (500), --batch-size (10).
Colors: --regen-manifest; local icons only, no model calls.
Audits: JSONL findings followed by a count summary; no writes.
  normalizer-drift and missing-images require --attributes <D1 snapshot>.`;

export function parseCli(args: string[]) {
  const expanded: string[] = [];
  for (let index = 0; index < args.length; index++) {
    const value = args[index];
    if (value === undefined) continue;
    if (value === "--item-ids") {
      let count = 0;
      while (
        args[index + 1] !== undefined &&
        !args[index + 1]?.startsWith("--")
      ) {
        expanded.push("--item-ids", args[++index] ?? "");
        count++;
      }
      if (!count) throw new Error("--item-ids requires at least one ID");
    } else expanded.push(value);
  }
  const parsed = parseArgs({
    args: expanded,
    allowPositionals: true,
    strict: true,
    options: {
      help: { type: "boolean", short: "h" },
      "tracker-root": { type: "string" },
      "config-root": { type: "string" },
      "sync-report": { type: "string" },
      "taxonomy-path": { type: "string" },
      "overrides-path": { type: "string" },
      "item-attributes-path": { type: "string" },
      attributes: { type: "string" },
      "manifest-root": { type: "string" },
      "output-root": { type: "string" },
      output: { type: "string" },
      "manifest-path": { type: "string" },
      "structured-path": { type: "string" },
      "structured-debug-path": { type: "string" },
      "item-id": { type: "string" },
      "item-ids": { type: "string", multiple: true },
      type: { type: "string", multiple: true },
      limit: { type: "string" },
      "include-indexed": { type: "boolean" },
      clothing: { type: "boolean" },
      accessories: { type: "boolean" },
      provider: { type: "string" },
      model: { type: "string" },
      "max-items": { type: "string" },
      "batch-size": { type: "string" },
      "regen-manifest": { type: "boolean" },
    },
  });
  const [command, ...extra] = parsed.positionals;
  if (extra.length && !(command === "audit" && extra.length === 1))
    throw new Error(`Unexpected positional arguments: ${extra.join(" ")}`);
  if (
    command &&
    !["manifest", "refresh", "index", "colors", "audit"].includes(command)
  )
    throw new Error(`Unknown command: ${command}`);
  const v = parsed.values;
  const audit = command === "audit" ? extra[0] : undefined;
  if (
    command === "audit" &&
    !v.help &&
    !["parse-errors", "normalizer-drift", "missing-images"].includes(
      audit ?? "",
    )
  )
    throw new Error(
      "Audit requires parse-errors, normalizer-drift, or missing-images",
    );
  if (v.attributes !== undefined && command !== "audit")
    throw new Error("--attributes requires the audit command");
  if (command === "audit" && !v.help) {
    if (v["item-attributes-path"] !== undefined)
      throw new Error("Use --attributes for the audit snapshot");
    if (audit !== "parse-errors" && !v.attributes)
      throw new Error(
        "This audit requires --attributes pointing to a D1-exported JSONL snapshot",
      );
    if (
      [
        v["item-id"],
        v["item-ids"],
        v.type,
        v.limit,
        v.clothing,
        v.accessories,
        v["include-indexed"],
        v.output,
        v["manifest-path"],
        v["structured-path"],
        v["structured-debug-path"],
      ].some((value) => value !== undefined)
    )
      throw new Error(
        "Audits scan their complete input; selection and extraction input flags are not supported",
      );
    if (audit === "parse-errors" && v.attributes !== undefined)
      throw new Error("parse-errors reads --output-root, not --attributes");
  }
  const pathOptions = Object.fromEntries(
    Object.entries({
      trackerRoot: v["tracker-root"],
      configRoot: v["config-root"],
      syncReport: v["sync-report"],
      taxonomyPath: v["taxonomy-path"],
      overridesPath: v["overrides-path"],
      attributesPath: v["item-attributes-path"],
      outputRoot: v["output-root"],
      manifestRoot: v["manifest-root"],
    }).filter(([, value]) => value !== undefined),
  );
  const number = (value: string) => {
    if (
      !/^\d+$/.test(value) ||
      !Number.isSafeInteger(Number(value)) ||
      Number(value) <= 0
    )
      throw new Error(`Expected positive integer: ${value}`);
    return Number(value);
  };
  const selection: Selection = {
    types: [
      ...(v.type ?? []),
      ...(v.clothing ? ["clothing"] : []),
      ...(v.accessories ? ["accessories"] : []),
    ],
  };
  if (v["item-id"] !== undefined) selection.itemId = number(v["item-id"]);
  if (v["item-ids"] !== undefined)
    selection.itemIds = v["item-ids"].map(number);
  if (v.limit !== undefined) selection.limit = number(v.limit);
  const indexOptions: IndexOptions = { ...selection };
  if (v.provider !== undefined) {
    if (v.provider !== "google" && v.provider !== "vercel")
      throw new Error("--provider must be google or vercel");
    indexOptions.provider = v.provider;
  }
  if (v.model !== undefined) indexOptions.model = v.model;
  if (v["max-items"] !== undefined)
    indexOptions.maxItems = number(v["max-items"]);
  if (v["batch-size"] !== undefined)
    indexOptions.batchSize = number(v["batch-size"]);
  if (v["regen-manifest"] !== undefined)
    indexOptions.regenManifest = v["regen-manifest"];
  if (v["include-indexed"] !== undefined)
    indexOptions.includeIndexed = v["include-indexed"];
  if (
    command !== "index" &&
    [v.provider, v.model, v["max-items"], v["batch-size"]].some(
      (value) => value !== undefined,
    )
  )
    throw new Error("Extraction flags require the index command");
  if (
    v["regen-manifest"] !== undefined &&
    command !== "index" &&
    command !== "colors"
  )
    throw new Error("--regen-manifest requires index or colors");
  if (
    (command === "index" || command === "colors") &&
    [
      v.output,
      v["manifest-path"],
      v["structured-path"],
      v["structured-debug-path"],
    ].some((value) => value !== undefined)
  )
    throw new Error(
      "Output/input override flags are not supported with index/colors; use --output-root and --manifest-root",
    );
  if (
    command === "manifest" &&
    [v["manifest-path"], v["structured-path"], v["structured-debug-path"]].some(
      (value) => value !== undefined,
    )
  )
    throw new Error("Refresh input flags cannot be used with manifest");
  if (
    command === "refresh" &&
    (v.output !== undefined || v["include-indexed"] !== undefined)
  )
    throw new Error("Manifest output/index flags cannot be used with refresh");
  if (command === "colors" && v["include-indexed"] !== undefined)
    throw new Error(
      "Colors include all eligible icons; --include-indexed does not apply",
    );
  return {
    command,
    audit,
    help: v.help ?? !command,
    selection,
    pathOptions,
    indexOptions,
    values: v,
  };
}

export async function main(args = process.argv.slice(2)) {
  loadProjectEnv();
  const parsed = parseCli(args);
  if (parsed.help) {
    console.log(help);
    return;
  }
  const paths = resolvePaths(parsed.pathOptions);
  if (parsed.command === "audit" && parsed.audit === "parse-errors") {
    const result = await auditParseErrors(paths.outputRoot);
    for (const row of result.rows) console.log(JSON.stringify(row));
    console.log(JSON.stringify(result.summary));
    return;
  }
  const taxonomy = await loadTaxonomy(paths.taxonomyPath);
  if (parsed.command === "audit") {
    const attributes = resolve(parsed.values.attributes ?? "");
    const result =
      parsed.audit === "normalizer-drift"
        ? await auditNormalizerDrift(taxonomy, attributes)
        : await auditMissingImages(paths, taxonomy, attributes);
    for (const row of result.rows) console.log(JSON.stringify(row));
    console.log(JSON.stringify(result.summary));
  } else if (parsed.command === "colors") {
    const options: Selection & { regenManifest?: boolean } = {
      ...parsed.selection,
    };
    if (parsed.values["regen-manifest"] !== undefined)
      options.regenManifest = parsed.values["regen-manifest"];
    console.log(
      JSON.stringify(await runColors(paths, taxonomy, options), null, 2),
    );
  } else if (parsed.command === "manifest") {
    const { rows, stats } = await buildManifest(
      paths,
      taxonomy,
      parsed.selection,
      parsed.values["include-indexed"] ?? false,
    );
    const output = resolve(
      parsed.values.output ??
        resolve(paths.manifestRoot, "item-manifest.jsonl"),
    );
    await atomicWrite(output, serializeJsonl(rows));
    console.log(
      JSON.stringify({ output, record_count: rows.length, ...stats }, null, 2),
    );
  } else if (parsed.command === "index") {
    const result = await runIndex(paths, taxonomy, parsed.indexOptions);
    console.log(
      JSON.stringify(
        result.kind === "debug" ? result.bundle : result.summary,
        null,
        2,
      ),
    );
    if (
      result.kind === "debug" &&
      result.bundle.structured_debug.parse_error !== null
    )
      process.exitCode = 1;
  } else {
    const options = { ...parsed.selection };
    const inputOptions = Object.fromEntries(
      Object.entries({
        manifestPath: parsed.values["manifest-path"],
        structuredPath:
          parsed.values["structured-path"] ??
          resolve(projectRoot, "index/item-structured-data.jsonl"),
        structuredDebugPath:
          parsed.values["structured-debug-path"] ??
          resolve(projectRoot, "index/item-structured-debug.jsonl"),
      }).filter(([, value]) => value !== undefined),
    );
    console.log(
      JSON.stringify(
        await runRefresh(paths, taxonomy, { ...options, ...inputOptions }),
        null,
        2,
      ),
    );
  }
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
