import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const projectRoot = fileURLToPath(new URL("../", import.meta.url));
export const defaultModels = {
  google: "gemini-3.5-flash-lite",
  vercel: "google/gemini-2.5-flash",
} as const;

export interface PathOptions {
  trackerRoot?: string;
  configRoot?: string;
  syncReport?: string;
  taxonomyPath?: string;
  overridesPath?: string;
  attributesPath?: string;
  outputRoot?: string;
  manifestRoot?: string;
}

export function resolvePaths(options: PathOptions = {}, env = process.env) {
  const siblings = dirname(projectRoot);
  const trackerRoot = resolve(
    options.trackerRoot ||
      env.TRACKER_ROOT ||
      join(siblings, "gongeo.us-nikki-tracker"),
  );
  const configRoot = resolve(
    options.configRoot ||
      env.CONFIG_DECODER_OUTPUT ||
      join(siblings, "gongeo.us-config-decoder/cfg/config_output"),
  );
  const manifestRoot = resolve(
    options.manifestRoot || join(projectRoot, "manifest"),
  );
  return {
    trackerRoot,
    configRoot,
    manifestRoot,
    outputRoot: resolve(options.outputRoot || join(projectRoot, "index")),
    syncReport: resolve(
      options.syncReport ||
        join(
          siblings,
          "gongeo.us-data-processor/reports/database-sync-report.json",
        ),
    ),
    taxonomyPath: resolve(
      options.taxonomyPath ||
        env.TRACKER_ITEM_SEARCH_EXPORT ||
        join(
          trackerRoot,
          "data/item-search/generated/image-search-taxonomy.json",
        ),
    ),
    overridesPath: resolve(
      options.overridesPath ||
        env.TRACKER_ITEM_SEARCH_OVERRIDES ||
        join(trackerRoot, "data/item-search/generated/overrides.json"),
    ),
    attributesPath: resolve(
      options.attributesPath || join(manifestRoot, "item-attributes.jsonl"),
    ),
    imageRoot: join(trackerRoot, "public/images/items"),
    iconRoot: join(trackerRoot, "public/images/items/icons"),
  };
}

export type Paths = ReturnType<typeof resolvePaths>;

export function loadProjectEnv() {
  try {
    process.loadEnvFile(join(projectRoot, ".env"));
  } catch (error) {
    if (!(error instanceof Error && "code" in error && error.code === "ENOENT"))
      throw error;
  }
}
