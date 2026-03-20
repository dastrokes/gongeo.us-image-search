from __future__ import annotations

import unittest

from image_search.models.schemas import ManifestRecord
from image_search.pipeline.captioning import VisionCaptioner


class CaptioningTests(unittest.TestCase):
    def test_prompt_construction_covers_representative_paths(self) -> None:
        hair_record = ManifestRecord(
            item_id=1,
            type="hair",
            icon_path="a",
            overview_path="b",
            has_icon=True,
            has_overview=True,
            source_version="1",
        )

        with self.subTest("plain prompt"):
            prompt = VisionCaptioner._build_plain_prompt("tops", "overview")
            self.assertIn(
                "Return only a comma-separated list of short lowercase phrases.",
                prompt,
            )
            self.assertIn("Subcategory examples for `tops`:", prompt)

        with self.subTest("hair record prompt"):
            prompt = VisionCaptioner._build_record_prompt(
                hair_record,
                has_icon=True,
                has_overview=True,
            )
            self.assertIn(
                "Describe only the hairstyle or hair-attached decorations.",
                prompt,
            )
            self.assertIn(
                "Use the icon array for small details and the overview array for whole-item structure.",
                prompt,
            )

        with self.subTest("overview prompt"):
            prompt = VisionCaptioner._build_prompt("dresses", "overview")
            self.assertIn(
                "subcategory, length, silhouette, neckline, sleeve length, sleeve shape, straps",
                prompt,
            )
            self.assertIn(
                "The overview should prioritize whole-item structure first",
                prompt,
            )

        with self.subTest("accessory prompt"):
            prompt = VisionCaptioner._build_prompt("headwear", "overview")
            self.assertIn(
                "subcategory, shape, size, placement, attachment style",
                prompt,
            )
            self.assertIn("whole-item shape, placement, scale", prompt)

    def test_normalize_caption_filters_noise_and_cross_item_leaks(self) -> None:
        cases = [
            (
                "long blonde hair, ribbon bow, gold top, gold necklace, white dress",
                "hair",
                "overview",
                "long blonde hair, ribbon bow",
            ),
            (
                "black hat, gold top, white blouse, long dress",
                "headwear",
                "overview",
                "black hat",
            ),
            (
                "pink hair, white turtleneck, long sleeves",
                "tops",
                "overview",
                "white turtleneck, long sleeves",
            ),
            (
                "tight-fitting, halter neck, zipper detail",
                "tops",
                "overview",
                "halter neck, zipper detail",
            ),
        ]

        for raw_caption, item_type, modality, expected in cases:
            with self.subTest(raw_caption=raw_caption, item_type=item_type):
                normalized = VisionCaptioner._normalize_caption(
                    raw_caption,
                    item_type,
                    modality,
                )
                self.assertEqual(normalized, expected)

    def test_parse_qwen_response_handles_structured_json_and_fallback_text(self) -> None:
        cases = [
            (
                """
                {
                  "colors": ["blue", "white trim"],
                  "shape": ["wide brim hat"],
                  "patterns": ["floral"],
                  "motifs": ["bow"],
                  "construction": ["ribbon tie"],
                  "details": ["gold charm"]
                }
                """,
                "blue, white trim, wide brim hat, floral, bow, ribbon tie, gold charm",
            ),
            (
                """```json
                {
                  "colors": ["red"],
                  "shape": ["long coat"],
                  "details": ["double-breasted front"]
                }
                ```""",
                "red, long coat, double-breasted front",
            ),
            (
                """
                colors: blue, white trim
                shape: cropped jacket
                details: pearl buttons
                """,
                "blue, white trim, cropped jacket, pearl buttons",
            ),
        ]

        for raw_text, expected in cases:
            with self.subTest(raw_text=raw_text.strip().splitlines()[0]):
                self.assertEqual(VisionCaptioner._parse_qwen_response(raw_text), expected)


if __name__ == "__main__":
    unittest.main()
