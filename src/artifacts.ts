import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { z } from "zod";

export function parseJsonl<T>(
  text: string,
  schema: z.ZodType<T>,
  source = "<jsonl>",
): T[] {
  const rows: T[] = [];
  const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/);
  for (const [index, line] of lines.entries()) {
    if (!line.trim()) continue;
    try {
      rows.push(schema.parse(JSON.parse(line)));
    } catch (cause) {
      throw new Error(`Invalid JSONL record at ${source}:${index + 1}`, {
        cause,
      });
    }
  }
  return rows;
}

export async function readJsonl<T>(
  path: string | URL,
  schema: z.ZodType<T>,
): Promise<T[]> {
  return parseJsonl(await readFile(path, "utf8"), schema, String(path));
}

export function serializeJsonl<T>(rows: readonly T[]): string {
  return rows.map((row) => `${JSON.stringify(row)}\n`).join("");
}

export async function readJson(path: string): Promise<unknown> {
  try {
    return JSON.parse((await readFile(path, "utf8")).replace(/^\uFEFF/, ""));
  } catch (cause) {
    throw new Error(`Cannot read JSON: ${path}`, { cause });
  }
}

export function isMissing(error: unknown): boolean {
  return (
    error instanceof Error &&
    (("code" in error && error.code === "ENOENT") ||
      (error.cause !== undefined && isMissing(error.cause)))
  );
}

export async function atomicWrite(
  path: string,
  content: string,
): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${randomUUID()}.tmp`;
  await writeFile(temporary, content, "utf8");
  await rename(temporary, path);
}
