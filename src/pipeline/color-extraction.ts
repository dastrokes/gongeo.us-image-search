import sharp from "sharp";
import { palette } from "../constants/colors.js";
import {
  type ColorManifestRecord,
  type ColorSwatch,
  colorDebugRecordSchema,
  itemColorsRecordSchema,
} from "../contracts.js";

// Match Python's rounding of the exact IEEE-754 input, including decimal ties.
export function roundEven(value: number, digits = 0): number {
  if (
    !Number.isFinite(value) ||
    !Number.isInteger(digits) ||
    digits < 0 ||
    digits > 4
  )
    throw new Error("Invalid color rounding input");
  if (value === 0) return value;
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, Math.abs(value));
  const bits = view.getBigUint64(0);
  const exponent = Number((bits >> 52n) & 0x7ffn);
  const significand =
    (bits & ((1n << 52n) - 1n)) + (exponent === 0 ? 0n : 1n << 52n);
  const power = (exponent === 0 ? -1022 : exponent - 1023) - 52;
  let numerator = significand * 10n ** BigInt(digits);
  const denominator = power >= 0 ? 1n : 1n << BigInt(-power);
  if (power > 0) numerator <<= BigInt(power);
  let rounded = numerator / denominator;
  const remainder = numerator % denominator;
  if (
    remainder * 2n > denominator ||
    (remainder * 2n === denominator && rounded % 2n !== 0n)
  )
    rounded++;
  return (Math.sign(value) * Number(rounded)) / 10 ** digits;
}

type RGB = readonly [number, number, number];
const luma = ([r, g, b]: RGB) => r * 0.3 + g * 0.59 + b * 0.11;
const highlight = (rgb: RGB) =>
  Math.min(...rgb) >= 238 && Math.max(...rgb) - Math.min(...rgb) <= 28;
const hex = (rgb: RGB) =>
  `#${rgb.map((value) => roundEven(value).toString(16).padStart(2, "0")).join("")}`;
const fromHex = (value: string): RGB => [
  Number.parseInt(value.slice(1, 3), 16),
  Number.parseInt(value.slice(3, 5), 16),
  Number.parseInt(value.slice(5, 7), 16),
];

function searchHue(rgb: RGB): string | null {
  const high = Math.max(...rgb);
  const low = Math.min(...rgb);
  const chroma = high - low;
  if (chroma < 12) return null;
  const [r, g, b] = rgb.map((value) => value / 255) as [number, number, number];
  const maximum = Math.max(r, g, b);
  const range = maximum - Math.min(r, g, b);
  const saturation = range / maximum;
  if (saturation < 0.045) return null;
  const rc = (maximum - r) / range;
  const gc = (maximum - g) / range;
  const bc = (maximum - b) / range;
  let hue =
    (r === maximum ? bc - gc : g === maximum ? 2 + rc - bc : 4 + gc - rc) / 6;
  hue %= 1;
  if (hue < 0) hue += 1;
  hue *= 360;
  const light = luma(rgb);
  if (hue >= 330 || hue < 10)
    return light >= 150 && saturation <= 0.4 ? "pink" : null;
  if (hue >= 300 && hue < 330 && light >= 140) return "pink";
  if (hue >= 250 && hue < 300) return "purple";
  if (hue >= 195 && hue < 250)
    return light < 90 && saturation >= 0.18 ? "navy" : "blue";
  if (hue >= 165 && hue < 195) return "teal";
  if (hue >= 45 && hue < 95 && light < 175 && saturation >= 0.18)
    return "olive";
  if (hue >= 80 && hue < 165 && chroma >= 16) return "green";
  if (hue >= 15 && hue < 80 && light >= 200 && saturation <= 0.22)
    return "cream";
  if (hue >= 45 && hue < 80 && light >= 185 && saturation <= 0.34)
    return "yellow";
  if (hue >= 15 && hue < 45 && light >= 185 && saturation <= 0.34)
    return light >= 210 && saturation <= 0.1 ? "cream" : "beige";
  return null;
}

function classify(rgb: RGB): string {
  const hue = searchHue(rgb);
  if (hue) return hue;
  if (Math.max(...rgb) - Math.min(...rgb) <= 24) {
    const light = luma(rgb);
    return light < 85
      ? "black"
      : light < 175
        ? "gray"
        : light < 220
          ? "silver"
          : "white";
  }
  if (highlight(rgb)) return "white";
  let nearest = "black";
  let distance = Infinity;
  for (const [label, [r, g, b]] of palette) {
    const current =
      ((rgb[0] - r) * 0.3) ** 2 +
      ((rgb[1] - g) * 0.59) ** 2 +
      ((rgb[2] - b) * 0.45) ** 2;
    if (current < distance) {
      nearest = label;
      distance = current;
    }
  }
  return nearest;
}

const noise = new Set(["white", "cream", "beige", "olive"]);
const taupe = new Set(["gray", "brown", "olive", "cream", "beige"]);
const lightTaupe = new Set(["cream", "beige"]);
const shadowTaupe = new Set(["gray", "brown", "olive"]);
const sortSwatches = (rows: ColorSwatch[]) =>
  rows
    .sort(
      (a, b) =>
        b.weight - a.weight ||
        (a.label < b.label ? -1 : a.label > b.label ? 1 : 0),
    )
    .slice(0, 5);

function normalizeHair(swatches: ColorSwatch[]): ColorSwatch[] {
  if (!swatches.length) return swatches;
  const selected = swatches.filter((row) => taupe.has(row.label));
  const weight = selected.reduce((total, row) => total + row.weight, 0);
  const light = selected
    .filter((row) => lightTaupe.has(row.label))
    .reduce((total, row) => total + row.weight, 0);
  const shadow = selected
    .filter((row) => shadowTaupe.has(row.label))
    .reduce((total, row) => total + row.weight, 0);
  const chromatic = swatches.some(
    (row) => !taupe.has(row.label) && row.weight >= 0.18,
  );
  if (
    weight < 0.65 ||
    light < 0.18 ||
    shadow < 0.35 ||
    swatches[0]?.label === "brown" ||
    chromatic
  )
    return swatches;
  const rgb: [number, number, number] = [0, 0, 0];
  for (const row of selected) {
    const color = fromHex(row.hex);
    for (const channel of [0, 1, 2] as const)
      rgb[channel] += color[channel] * row.weight;
  }
  return sortSwatches([
    {
      label: "beige",
      hex: hex(rgb.map((value) => value / weight) as [number, number, number]),
      weight: roundEven(weight, 4),
    },
    ...swatches.filter((row) => !taupe.has(row.label)),
  ]);
}

class Sum {
  value = 0;
  private correction = 0;
  add(value: number) {
    const total = this.value + value;
    this.correction +=
      Math.abs(this.value) >= Math.abs(value)
        ? this.value - total + value
        : value - total + this.value;
    this.value = total;
  }
  total() {
    return this.value + this.correction;
  }
}

export function extractColorsFromRgba(
  record: ColorManifestRecord,
  bytes: Uint8Array,
  width: number,
  height: number,
) {
  if (
    !Number.isSafeInteger(width) ||
    width <= 0 ||
    !Number.isSafeInteger(height) ||
    height <= 0 ||
    bytes.length !== width * height * 4
  )
    throw new Error("Invalid RGBA pixel buffer");
  const groups = new Map<
    string,
    { weight: number; sum: Sum; red: number; green: number; blue: number }
  >();
  const totalWeight = new Sum();
  let visible = 0;
  let ignored = 0;
  let sampled = 0;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const offset = (y * width + x) * 4;
      const rgb: RGB = [
        bytes[offset] ?? 0,
        bytes[offset + 1] ?? 0,
        bytes[offset + 2] ?? 0,
      ];
      const alpha = bytes[offset + 3] ?? 0;
      if (alpha < 32) {
        ignored++;
        continue;
      }
      visible++;
      if (highlight(rgb) && alpha < 192) {
        ignored++;
        continue;
      }
      let multiplier = 1;
      if (record.item_type === "hair") {
        const center = Math.abs(x / Math.max(1, width - 1) - 0.5);
        const vertical = y / Math.max(1, height - 1);
        const skin =
          rgb[0] > rgb[1] * 1.03 && rgb[1] > rgb[2] * 1.05 && rgb[0] > 135;
        const face =
          Math.min(...rgb) >= 220 && Math.max(...rgb) - Math.min(...rgb) <= 48;
        if (
          center < 0.32 &&
          vertical > 0.14 &&
          vertical < 0.82 &&
          (skin || face)
        )
          multiplier = 1 / 12;
      }
      const weight = (alpha / 255) * multiplier;
      const label = classify(rgb);
      let group = groups.get(label);
      if (!group) {
        group = { weight: 0, sum: new Sum(), red: 0, green: 0, blue: 0 };
        groups.set(label, group);
      }
      group.weight += weight;
      group.sum.add(weight);
      totalWeight.add(weight);
      group.red += rgb[0] * weight;
      group.green += rgb[1] * weight;
      group.blue += rgb[2] * weight;
      sampled++;
    }
  let swatches = sortSwatches(
    [...groups].map(([label, group]) => ({
      label,
      hex: hex([
        group.red / group.weight,
        group.green / group.weight,
        group.blue / group.weight,
      ]),
      weight: roundEven(group.sum.total() / totalWeight.total(), 4),
    })),
  );
  const hair = record.item_type === "hair";
  if (hair) swatches = normalizeHair(swatches);
  let primary = swatches
    .filter(
      (row) =>
        row.weight >= 0.18 &&
        !(hair && noise.has(row.label) && row.label !== swatches[0]?.label),
    )
    .slice(0, 3);
  if (!primary.length) primary = swatches.slice(0, 1);
  const labels = new Set(primary.map((row) => row.label));
  const secondary = swatches
    .filter(
      (row) =>
        !labels.has(row.label) &&
        row.weight >= 0.1 &&
        !(hair && noise.has(row.label)),
    )
    .slice(0, 3);
  const colorTags = [...primary, ...secondary].map((row) => row.label);
  if (colorTags.length >= 3) colorTags.push("multicolor");
  const reasons: string[] = [];
  if (hair) reasons.push("hair_icon_composite");
  if (!colorTags.length) reasons.push("no_color_tags");
  const identity = { item_id: record.item_id, item_type: record.item_type };
  const review = { needs_review: reasons.length > 0, review_reasons: reasons };
  return {
    record: itemColorsRecordSchema.parse({
      ...identity,
      primary_colors: primary.map((row) => row.label),
      secondary_colors: secondary.map((row) => row.label),
      color_tags: colorTags,
      swatches: [...primary, ...secondary],
      ...review,
    }),
    debug: colorDebugRecordSchema.parse({
      ...identity,
      image_path: record.icon_path,
      visible_pixel_count: visible,
      sampled_pixel_count: sampled,
      ignored_pixel_count: ignored,
      color_weights: Object.fromEntries(
        swatches.map((row) => [row.label, row.weight]),
      ),
      swatches,
      ...review,
    }),
  };
}

export async function extractColorRecord(
  record: ColorManifestRecord,
  input?: Uint8Array,
) {
  // Pillow convert(RGBA) does not apply embedded profiles or EXIF rotation.
  const { data, info } = await sharp(input ?? record.icon_path, {
    ignoreIcc: true,
  })
    .toColourspace("srgb")
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return extractColorsFromRgba(record, data, info.width, info.height);
}
