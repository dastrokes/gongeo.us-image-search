import { z } from "zod";

const itemId = z.number().int().positive().max(Number.MAX_SAFE_INTEGER);
const nonempty = z.string().min(1);
const count = z.number().int().nonnegative();
const fieldValue = z.union([z.string(), z.array(z.string()), z.null()]);
const structuredData = z.record(z.string(), fieldValue);

export const manifestRecordSchema = z.strictObject({
  item_id: itemId,
  item_type: nonempty,
});

export const structuredItemRecordSchema = manifestRecordSchema.extend({
  data: structuredData,
  parse_error: z.string().nullable(),
});

export const structuredDebugRecordSchema = manifestRecordSchema.extend({
  image_paths: z.strictObject({
    overview: z.string().nullable(),
    icon: z.string().nullable(),
  }),
  prompt: z.string(),
  raw_response: z.string(),
  raw_payload: z.record(z.string(), z.unknown()).nullable(),
  normalized_data: structuredData,
  filter_report: z.record(z.string(), z.unknown()),
  parse_error: z.string().nullable(),
  request: z
    .strictObject({
      provider: z.enum(["google", "vercel"]),
      model: nonempty,
      finish_reason: z.string().nullable(),
      usage: z.record(z.string(), z.number().nonnegative()),
    })
    .optional(),
});

const metadata = z
  .record(z.string(), z.union([nonempty, z.array(nonempty).min(1)]))
  .superRefine((values, context) => {
    for (const key of [
      "item_id",
      "item_type",
      "slot",
      "category",
      "subcategory",
    ]) {
      if (Object.hasOwn(values, key)) {
        context.addIssue({
          code: "custom",
          path: [key],
          message: "Identity fields must not appear in metadata",
        });
      }
    }
  });

export const itemAttributesRecordSchema = manifestRecordSchema.extend({
  category: nonempty.nullable(),
  subcategory: nonempty.nullable(),
  metadata,
});

export const colorManifestRecordSchema = manifestRecordSchema.extend({
  icon_path: nonempty,
});

export const colorSwatchSchema = z.strictObject({
  label: nonempty,
  hex: z.string().regex(/^#[0-9a-f]{6}$/),
  weight: z.number().min(0).max(1),
});

const colorResult = {
  swatches: z.array(colorSwatchSchema),
  needs_review: z.boolean(),
  review_reasons: z.array(nonempty),
};

export const itemColorsRecordSchema = manifestRecordSchema.extend({
  primary_colors: z.array(nonempty),
  secondary_colors: z.array(nonempty),
  color_tags: z.array(nonempty),
  ...colorResult,
});

export const colorDebugRecordSchema = manifestRecordSchema.extend({
  image_path: nonempty,
  visible_pixel_count: count,
  sampled_pixel_count: count,
  ignored_pixel_count: count,
  color_weights: z.record(z.string(), z.number().min(0).max(1)),
  ...colorResult,
});

export const buildSummarySchema = z.strictObject({
  extraction_model_id: nonempty,
  item_count: count,
  skipped_count: count,
  missing_icon_count: count,
  missing_overview_count: count,
  structured_parse_fail_count: count,
  build_started_at: nonempty,
  build_finished_at: nonempty,
  duration_seconds: z.number().nonnegative(),
  search_report_path: z.string().nullable(),
  item_attributes_path: z.string().nullable(),
});

export type ManifestRecord = z.infer<typeof manifestRecordSchema>;
export type StructuredItemRecord = z.infer<typeof structuredItemRecordSchema>;
export type StructuredDebugRecord = z.infer<typeof structuredDebugRecordSchema>;
export type ItemAttributesRecord = z.infer<typeof itemAttributesRecordSchema>;
export type ColorManifestRecord = z.infer<typeof colorManifestRecordSchema>;
export type ColorSwatch = z.infer<typeof colorSwatchSchema>;
export type ItemColorsRecord = z.infer<typeof itemColorsRecordSchema>;
export type ColorDebugRecord = z.infer<typeof colorDebugRecordSchema>;
export type BuildSummary = z.infer<typeof buildSummarySchema>;
