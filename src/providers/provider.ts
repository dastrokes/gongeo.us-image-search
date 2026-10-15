import { setTimeout as sleep } from "node:timers/promises";
import { createGateway } from "@ai-sdk/gateway";
import { createGoogle } from "@ai-sdk/google";
import {
  APICallError,
  generateText,
  type LanguageModel,
  type LanguageModelUsage,
  NoObjectGeneratedError,
  Output,
  type UserContent,
} from "ai";
import { z } from "zod";
import type { Data } from "../pipeline/normalization.js";
import type { Taxonomy } from "../pipeline/taxonomy.js";

export type ProviderName = "google" | "vercel";
export interface ImageInput {
  label: "overview" | "icon";
  path: string;
  mediaType: string;
  bytes: Uint8Array;
}
export interface ExtractionRequest {
  system: string;
  user: string;
  schema: z.ZodType<Data>;
  images: ImageInput[];
}
export interface ExtractionResult {
  rawText: string;
  error: string | null;
  finishReason: string | null;
  usage: Record<string, number>;
}
export interface ExtractionAdapter {
  extract(request: ExtractionRequest): Promise<ExtractionResult>;
}

export function responseShape(
  taxonomy: Taxonomy,
  type: string,
): z.ZodType<Data> {
  return z.strictObject(
    Object.fromEntries(
      taxonomy
        .fields(type)
        .map(({ name, kind }) => [
          name,
          kind === "array" ? z.array(z.string()) : z.string().nullable(),
        ]),
    ),
  );
}

export function requestSettings(model: string) {
  // Older OpenAI reasoning models do not accept temperature. Other overrides
  // retain their provider default; explicitly pin the settings we have checked.
  if (model.startsWith("openai/gpt-6"))
    return { reasoning: "minimal" as const };
  if (model.includes("gemini-") || model.startsWith("openai/gpt-4"))
    return { temperature: 0 };
  return {};
}

function usageCounts(
  usage: LanguageModelUsage | undefined,
): Record<string, number> {
  return Object.fromEntries(
    Object.entries({
      input_tokens: usage?.inputTokens,
      output_tokens: usage?.outputTokens,
      total_tokens: usage?.totalTokens,
      reasoning_tokens: usage?.outputTokenDetails.reasoningTokens,
    }).filter(
      (entry): entry is [string, number] =>
        typeof entry[1] === "number" &&
        Number.isFinite(entry[1]) &&
        entry[1] >= 0,
    ),
  );
}

export class ProviderRequestError extends Error {
  constructor(
    readonly provider: ProviderName,
    readonly status: number | null,
    readonly retryable: boolean,
    readonly retryAfterMs: number | null = null,
  ) {
    const hint =
      status === 401 || status === 403
        ? " Check the key permissions and model eligibility."
        : status === 400 || status === 404
          ? " Check the model ID and response schema."
          : "";
    super(
      `${provider} request failed${status === null ? " (network/timeout)" : ` (HTTP ${status})`}.${hint}`,
    );
  }
}

function retryAfter(
  headers: Record<string, string> | undefined,
): number | null {
  const value = Object.entries(headers ?? {}).find(
    ([key]) => key.toLowerCase() === "retry-after",
  )?.[1];
  if (!value) return null;
  const numeric = Number(value);
  if (Number.isFinite(numeric) && numeric >= 0) return numeric * 1000;
  const date = Date.parse(value);
  return Number.isFinite(date) ? Math.max(0, date - Date.now()) : null;
}

function requestError(
  provider: ProviderName,
  error: unknown,
): ProviderRequestError {
  let current = error;
  for (
    let depth = 0;
    depth < 8 && current instanceof Error;
    depth++, current = current.cause
  ) {
    if (APICallError.isInstance(current)) {
      const status = current.statusCode ?? null;
      return new ProviderRequestError(
        provider,
        status,
        status === null
          ? current.isRetryable
          : status === 408 || status === 429 || status >= 500,
        retryAfter(current.responseHeaders),
      );
    }
    if ("statusCode" in current && typeof current.statusCode === "number") {
      const status = current.statusCode;
      return new ProviderRequestError(
        provider,
        status,
        status === 408 || status === 429 || status >= 500,
      );
    }
    const code = "code" in current ? current.code : null;
    if (
      [
        "ECONNRESET",
        "ECONNREFUSED",
        "ETIMEDOUT",
        "EAI_AGAIN",
        "ENOTFOUND",
        "UND_ERR_CONNECT_TIMEOUT",
      ].includes(String(code)) ||
      ["AbortError", "TimeoutError"].includes(current.name) ||
      (current instanceof TypeError && current.message === "fetch failed")
    )
      return new ProviderRequestError(provider, null, true);
  }
  return new ProviderRequestError(provider, null, false);
}

export interface AdapterOptions {
  maxRetries?: number;
  timeoutMs?: number;
  minIntervalMs?: number;
  sleep?: (ms: number) => Promise<void>;
  now?: () => number;
}

export class SdkAdapter implements ExtractionAdapter {
  private lastCall = -Infinity;
  constructor(
    readonly provider: ProviderName,
    readonly modelId: string,
    private readonly model: LanguageModel,
    private readonly options: AdapterOptions = {},
  ) {}

  async extract(request: ExtractionRequest): Promise<ExtractionResult> {
    const content: UserContent = [{ type: "text", text: request.user }];
    for (const image of request.images)
      content.push(
        { type: "text", text: `IMAGE: ${image.label}` },
        { type: "file", data: image.bytes, mediaType: image.mediaType },
      );
    const wait =
      this.options.sleep ?? ((ms: number) => sleep(ms).then(() => {}));
    const now = this.options.now ?? (() => performance.now());
    const maxRetries = this.options.maxRetries ?? 3;
    for (let attempt = 0; ; attempt++) {
      const spacing =
        (this.options.minIntervalMs ??
          (this.provider === "google" ? 4000 : 0)) -
        (now() - this.lastCall);
      if (spacing > 0) await wait(spacing);
      this.lastCall = now();
      try {
        const result = await generateText({
          model: this.model,
          system: request.system,
          messages: [{ role: "user", content }],
          output: Output.object({
            schema: request.schema,
            name: "item_attributes",
          }),
          ...requestSettings(this.modelId),
          maxRetries: 0,
          abortSignal: AbortSignal.timeout(this.options.timeoutMs ?? 180_000),
        });
        // Reading output also validates the schema; retain original assistant text.
        void result.output;
        return {
          rawText: result.text,
          error:
            result.finishReason === "stop"
              ? null
              : `incomplete_response:${result.finishReason}`,
          finishReason: result.finishReason,
          usage: usageCounts(result.usage),
        };
      } catch (error) {
        if (NoObjectGeneratedError.isInstance(error))
          return {
            rawText: error.text ?? "",
            error: "invalid_structured_response",
            finishReason: error.finishReason ?? null,
            usage: usageCounts(error.usage),
          };
        const failure = requestError(this.provider, error);
        if (!failure.retryable || attempt >= maxRetries) throw failure;
        const delay = failure.retryAfterMs ?? 2000 * 2 ** attempt;
        console.error(
          `${failure.message} Retry ${attempt + 1}/${maxRetries} in ${(delay / 1000).toFixed(1)}s`,
        );
        await wait(delay);
      }
    }
  }
}

export function createAdapter(
  provider: ProviderName,
  model: string,
  env = process.env,
  fetchFunction?: typeof fetch,
): ExtractionAdapter {
  const keyName =
    provider === "google" ? "GOOGLE_API_KEY" : "AI_GATEWAY_API_KEY";
  const apiKey = env[keyName];
  if (!apiKey?.trim()) throw new Error(`${keyName} is not set`);
  if (provider === "google")
    return new SdkAdapter(
      provider,
      model,
      createGoogle({
        apiKey,
        ...(fetchFunction ? { fetch: fetchFunction } : {}),
      }).languageModel(model),
    );
  if (!model.includes("/"))
    throw new Error("Gateway model IDs must include the provider prefix");
  return new SdkAdapter(
    provider,
    model,
    createGateway({
      apiKey,
      ...(fetchFunction ? { fetch: fetchFunction } : {}),
    }).languageModel(model),
  );
}
