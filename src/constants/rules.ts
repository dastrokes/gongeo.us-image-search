import { z } from "zod";
import source from "./extraction-rules.json" with { type: "json" };

const strings = z.record(z.string(), z.string());
export const rules = z
  .object({
    fieldAliases: z.record(
      z.string(),
      z.array(z.tuple([z.string(), z.string()])),
    ),
    bottomLengthAliases: z.record(z.string(), strings),
    crossFieldOwnership: z.record(
      z.string(),
      z.record(
        z.string(),
        z.tuple([z.string().nullable(), z.string().nullable()]),
      ),
    ),
    ornamentSuffixes: z.array(z.string()),
    systemPrompt: z.string(),
    evidencePrompt: z.string(),
    fieldGuidance: strings,
    fieldExamples: z.record(z.string(), z.array(z.string())),
    itemTypeGuidance: strings,
    typeKeyMap: strings,
    imageExtensions: z.array(z.string()),
  })
  .parse(source);
