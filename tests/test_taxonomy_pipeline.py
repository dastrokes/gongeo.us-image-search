from __future__ import annotations

import unittest

from image_search.models.schemas import (
    ItemInputRecord,
    StructuredCandidateRecord,
)
from image_search.pipeline.color_tags import ColorTaggingResult
from image_search.pipeline.documents import build_document_record
from image_search.pipeline.taxonomy import (
    build_metadata_record,
    build_structured_candidates,
    build_tag_assignments,
    build_taxonomy_concepts,
    build_unmapped_term_records,
    build_visual_features,
)


class TaxonomyPipelineTests(unittest.TestCase):
    def test_build_taxonomy_concepts_includes_core_keys(self) -> None:
        concepts = build_taxonomy_concepts()
        keys = [concept.key for concept in concepts]

        self.assertEqual(len(keys), len(set(keys)))

        concept_keys = {concept.key for concept in concepts}
        for concept_key in {
            "subcategory:jacket",
            "subcategory:cardigan",
            "subcategory:drop_earrings",
            "length:maxi",
            "hair.length:long",
            "theme:fairycore",
            "occasion_setting:bridal",
            "character_archetype:princess",
            "subcategory:sunglasses",
            "subcategory:horns",
            "motif:moon",
            "subcategory:garter",
        }:
            with self.subTest(concept_key=concept_key):
                self.assertIn(concept_key, concept_keys)

    def test_build_structured_candidates_extracts_representative_terms(self) -> None:
        cases = [
            (
                ItemInputRecord(item_id=1, item_type="hair", icon_path="", overview_path=""),
                build_visual_features(
                    item_id=1,
                    item_type="hair",
                    caption_icon="blue hair, side ponytail, full bangs",
                    caption_overview="falls over shoulders, wavy hair, side ponytail, full bangs",
                    caption_visual="blue hair, side ponytail, falls over shoulders, wavy hair, full bangs",
                    color_tags=ColorTaggingResult(
                        dominant_colors=["blue", "silver"],
                        accent_colors=["pink"],
                    ),
                ),
                {
                    "color.primary:blue",
                    "hair.length:long",
                    "hair.arrangement:side_ponytail",
                    "hair.texture:wavy",
                    "hair.bangs:full_bangs",
                },
            ),
            (
                ItemInputRecord(item_id=2, item_type="tops", icon_path="", overview_path=""),
                build_visual_features(
                    item_id=2,
                    item_type="tops",
                    caption_icon="white jacket, button front, puff sleeves",
                    caption_overview="cardigan, white jacket, v-neckline, long sleeves",
                    caption_visual="white jacket, cardigan, v-neckline, button front, long puff sleeves",
                    color_tags=ColorTaggingResult(
                        dominant_colors=["white"],
                        accent_colors=["black"],
                    ),
                ),
                {
                    "subcategory:jacket",
                    "subcategory:cardigan",
                    "neckline:v_neck",
                    "closure:button_front",
                    "sleeve_length:long",
                    "sleeve_shape:puff",
                },
            ),
            (
                ItemInputRecord(item_id=9, item_type="tops", icon_path="", overview_path=""),
                build_visual_features(
                    item_id=9,
                    item_type="tops",
                    caption_icon="sports bra, white trim, heart pattern",
                    caption_overview="sports bra, square neckline, no sleeves, crop top",
                    caption_visual="sports bra, white trim, heart pattern, square neckline, no sleeves, crop top",
                    color_tags=ColorTaggingResult(
                        dominant_colors=["red"],
                        accent_colors=["white"],
                    ),
                ),
                {
                    "subcategory:sports_bra",
                    "neckline:square",
                    "sleeve_length:sleeveless",
                    "length:cropped",
                },
            ),
            (
                ItemInputRecord(item_id=6, item_type="dresses", icon_path="", overview_path=""),
                build_visual_features(
                    item_id=6,
                    item_type="dresses",
                    caption_icon="princess dress, celestial details, wedding styling",
                    caption_overview="fairycore gown, ethereal bridal look, formal ballroom silhouette",
                    caption_visual="princess fairycore gown, celestial details, ethereal bridal ballroom look",
                    color_tags=ColorTaggingResult(
                        dominant_colors=["white", "blue"],
                        accent_colors=["gold"],
                    ),
                ),
                {
                    "character_archetype:princess",
                    "theme:fairycore",
                    "theme:celestial",
                    "theme:ethereal",
                    "occasion_setting:bridal",
                    "occasion_setting:formal",
                },
            ),
            (
                ItemInputRecord(item_id=7, item_type="faceDecorations", icon_path="", overview_path=""),
                build_visual_features(
                    item_id=7,
                    item_type="faceDecorations",
                    caption_icon="sunglasses, monocle chain, star sticker",
                    caption_overview="face accessory with sunglasses and monocle",
                    caption_visual="sunglasses, monocle chain, star sticker",
                    color_tags=ColorTaggingResult(
                        dominant_colors=["black"],
                        accent_colors=["gold"],
                    ),
                ),
                {
                    "subcategory:sunglasses",
                    "subcategory:monocle",
                    "subcategory:sticker",
                    "ornament:chain",
                    "motif:star",
                },
            ),
            (
                ItemInputRecord(item_id=8, item_type="headwear", icon_path="", overview_path=""),
                build_visual_features(
                    item_id=8,
                    item_type="headwear",
                    caption_icon="horns, antlers, feather trim, crescent moon motif",
                    caption_overview="winged crown with halo and antlers",
                    caption_visual="horns, antlers, winged crown, halo, feather trim, crescent moon motif",
                    color_tags=ColorTaggingResult(
                        dominant_colors=["white"],
                        accent_colors=["silver"],
                    ),
                ),
                {
                    "subcategory:horns",
                    "subcategory:antlers",
                    "subcategory:halo",
                    "subcategory:winged_headpiece",
                    "trim:feather_trim",
                    "motif:moon",
                },
            ),
        ]

        for item_input, visual, expected_keys in cases:
            with self.subTest(item_id=item_input.item_id, item_type=item_input.item_type):
                candidate_keys = {
                    candidate.concept_key
                    for candidate in build_structured_candidates(item_input, visual)
                }
                for concept_key in expected_keys:
                    self.assertIn(concept_key, candidate_keys)

    def test_build_tag_assignments_cover_conflicts_implications_and_documents(self) -> None:
        with self.subTest("exclusive conflict"):
            candidates = [
                StructuredCandidateRecord(
                    item_id=3,
                    facet_key="length",
                    value_key="midi",
                    concept_key="length:midi",
                    source="test",
                    source_modalities=["overview"],
                    pre_validation_score=0.91,
                ),
                StructuredCandidateRecord(
                    item_id=3,
                    facet_key="length",
                    value_key="maxi",
                    concept_key="length:maxi",
                    source="test",
                    source_modalities=["overview"],
                    pre_validation_score=0.84,
                ),
            ]

            item_input = ItemInputRecord(
                item_id=3,
                item_type="dresses",
                icon_path="",
                overview_path="",
            )
            visual = build_visual_features(
                item_id=3,
                item_type="dresses",
                caption_icon="",
                caption_overview="",
                caption_visual="",
                color_tags=ColorTaggingResult(dominant_colors=[], accent_colors=[]),
            )
            assignments = build_tag_assignments(
                item_input,
                visual,
                "test-model",
                candidates=candidates,
            )
            statuses = {assignment.concept_key: assignment.status for assignment in assignments}

            self.assertEqual(statuses["length:midi"], "accepted")
            self.assertEqual(statuses["length:maxi"], "suppressed")

        with self.subTest("implied parent and multi-value facets"):
            item_input = ItemInputRecord(item_id=4, item_type="shoes", icon_path="", overview_path="")
            visual = build_visual_features(
                item_id=4,
                item_type="shoes",
                caption_icon="ankle boots, pointed toe",
                caption_overview="ankle boots with pointed toe and platform sole",
                caption_visual="ankle boots, pointed toe, platform sole",
                color_tags=ColorTaggingResult(dominant_colors=["black"], accent_colors=["gold"]),
            )
            assignments = build_tag_assignments(item_input, visual, "test-model")
            accepted = {
                assignment.concept_key
                for assignment in assignments
                if assignment.status == "accepted"
            }

            self.assertIn("subcategory:ankle_boots", accepted)
            self.assertIn("subcategory:boots", accepted)
            self.assertIn("shoes.shaft_height:ankle", accepted)
            self.assertIn("shoes.toe_shape:pointed_toe", accepted)
            self.assertIn("shoes.platform:platform", accepted)

        with self.subTest("garter remains distinct from garter stockings"):
            item_input = ItemInputRecord(item_id=10, item_type="socks", icon_path="", overview_path="")
            visual = build_visual_features(
                item_id=10,
                item_type="socks",
                caption_icon="garter belt, sheer lace",
                caption_overview="garter stockings with sheer finish",
                caption_visual="garter belt, garter stockings, sheer lace",
                color_tags=ColorTaggingResult(dominant_colors=["black"], accent_colors=[]),
            )
            assignments = build_tag_assignments(item_input, visual, "test-model")
            statuses = {assignment.concept_key: assignment.status for assignment in assignments}

            self.assertEqual(statuses["subcategory:garter_stockings"], "accepted")
            self.assertEqual(statuses["subcategory:garter"], "suppressed")

        with self.subTest("sunglasses imply glasses"):
            item_input = ItemInputRecord(item_id=11, item_type="faceDecorations", icon_path="", overview_path="")
            visual = build_visual_features(
                item_id=11,
                item_type="faceDecorations",
                caption_icon="sunglasses with dark frames",
                caption_overview="fashion eyewear",
                caption_visual="sunglasses with dark frames",
                color_tags=ColorTaggingResult(dominant_colors=["black"], accent_colors=[]),
            )
            assignments = build_tag_assignments(item_input, visual, "test-model")
            accepted = {
                assignment.concept_key
                for assignment in assignments
                if assignment.status == "accepted"
            }

            self.assertIn("subcategory:sunglasses", accepted)
            self.assertIn("subcategory:glasses", accepted)

        with self.subTest("button closure variants resolve to button front"):
            item_input = ItemInputRecord(item_id=12, item_type="tops", icon_path="", overview_path="")
            visual = build_visual_features(
                item_id=12,
                item_type="tops",
                caption_icon="button closure, ornamented buttons",
                caption_overview="front closure with buttons, square neckline",
                caption_visual="button closure, ornamented buttons, front closure with buttons, square neckline",
                color_tags=ColorTaggingResult(dominant_colors=["white"], accent_colors=["gold"]),
            )
            assignments = build_tag_assignments(item_input, visual, "test-model")
            accepted = {
                assignment.concept_key
                for assignment in assignments
                if assignment.status == "accepted"
            }

            self.assertIn("closure:button_front", accepted)
            self.assertIn("neckline:square", accepted)

        with self.subTest("metadata and document"):
            item_input = ItemInputRecord(item_id=5, item_type="handhelds", icon_path="", overview_path="")
            visual = build_visual_features(
                item_id=5,
                item_type="handhelds",
                caption_icon="parasol, ribbon ornament",
                caption_overview="umbrella with decorative ribbon",
                caption_visual="parasol, umbrella, decorative ribbon",
                color_tags=ColorTaggingResult(dominant_colors=["blue"], accent_colors=["white"]),
            )
            assignments = build_tag_assignments(item_input, visual, "test-model")
            metadata = build_metadata_record(item_input, visual, assignments)
            document = build_document_record(metadata, visual, assignments)

            self.assertIn("subcategory:umbrella", metadata.accepted_facets)
            self.assertIn("subcategory", metadata.facet_values)
            self.assertIn("umbrella", metadata.search_terms)
            self.assertIn("parasol", metadata.search_terms)
            self.assertIn("facet_values", document.metadata)
            self.assertIn("search terms", document.data)

        with self.subTest("unmapped taxonomy gap report"):
            item_input = ItemInputRecord(item_id=9, item_type="headwear", icon_path="", overview_path="")
            visual = build_visual_features(
                item_id=9,
                item_type="headwear",
                caption_icon="hinnin ornament, halo, antlers",
                caption_overview="ornate headwear with hinnin ornament",
                caption_visual="hinnin ornament, halo, antlers",
                color_tags=ColorTaggingResult(dominant_colors=["gold"], accent_colors=[]),
            )
            unmapped = build_unmapped_term_records(item_input, visual)

            self.assertEqual([record.term for record in unmapped], ["hinnin ornament"])


if __name__ == "__main__":
    unittest.main()
