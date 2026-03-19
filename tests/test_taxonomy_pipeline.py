from __future__ import annotations

import unittest

from image_search.models.schemas import ItemInputRecord
from image_search.pipeline.color_tags import ColorTaggingResult
from image_search.pipeline.documents import build_document_record
from image_search.pipeline.taxonomy import (
    build_metadata_record,
    build_structured_candidates,
    build_tag_assignments,
    build_taxonomy_concepts,
    build_visual_features,
)


class TaxonomyPipelineTests(unittest.TestCase):
    def test_taxonomy_concepts_are_unique_and_cover_tops_subtypes(self) -> None:
        concepts = build_taxonomy_concepts()
        keys = [concept.key for concept in concepts]
        self.assertEqual(len(keys), len(set(keys)))

        concept_keys = {concept.key for concept in concepts}
        for concept_key in {
            "tops.subtype:jacket",
            "tops.subtype:cardigan",
            "tops.subtype:vest",
            "tops.subtype:blazer",
            "tops.subtype:shirt",
            "tops.subtype:blouse",
            "tops.subtype:sweater",
            "tops.subtype:camisole",
            "tops.subtype:halter",
            "tops.subtype:corset",
        }:
            self.assertIn(concept_key, concept_keys)

    def test_hair_assignments_capture_length_structure_and_colors(self) -> None:
        item_input = ItemInputRecord(item_id=1, item_type="hair", icon_path="", overview_path="")
        visual = build_visual_features(
            item_id=1,
            item_type="hair",
            caption_icon="blue hair, ponytail, braid",
            caption_overview="falls over shoulders, twin tails, half-down look",
            caption_visual="blue hair, ponytail, braid, falls over shoulders, twin tails, half-down look",
            color_tags=ColorTaggingResult(
                dominant_colors=["blue", "silver"],
                accent_colors=[],
            ),
        )

        assignments = build_tag_assignments(item_input, visual, "test-model")
        accepted = {
            assignment.concept_key
            for assignment in assignments
            if assignment.status == "accepted"
        }
        review = {
            assignment.concept_key
            for assignment in assignments
            if assignment.status == "review"
        }
        suppressed = {
            assignment.concept_key
            for assignment in assignments
            if assignment.status == "suppressed"
        }

        self.assertIn("color.primary:blue", accepted)
        self.assertIn("color.secondary:silver", accepted)
        self.assertIn("hair.length:long", accepted)
        self.assertIn("hair.structure:twin_tails", accepted)
        self.assertIn("hair.structure:half_up", accepted)
        self.assertIn("hair.structure:ponytail", review)
        self.assertIn("hair.structure:braid", suppressed)

    def test_tops_candidates_capture_subtypes_and_search_only_fields(self) -> None:
        item_input = ItemInputRecord(item_id=2, item_type="tops", icon_path="", overview_path="")
        visual = build_visual_features(
            item_id=2,
            item_type="tops",
            caption_icon="white jacket, buttons, bow",
            caption_overview="cardigan, white jacket, v-neckline",
            caption_visual="white jacket, cardigan, v-neckline, button-down front, bow",
            color_tags=ColorTaggingResult(
                dominant_colors=["white"],
                accent_colors=["black"],
            ),
        )

        candidates = build_structured_candidates(item_input, visual)
        candidate_keys = {candidate.concept_key for candidate in candidates}

        self.assertIn("tops.subtype:jacket", candidate_keys)
        self.assertIn("tops.subtype:cardigan", candidate_keys)
        self.assertIn("tops.neckline:v_neck", candidate_keys)
        self.assertIn("tops.closure:button_down", candidate_keys)
        self.assertNotIn("dresses.length:maxi", candidate_keys)

    def test_term_specific_rules_expand_phrase_level_evidence(self) -> None:
        item_input = ItemInputRecord(item_id=22, item_type="tops", icon_path="", overview_path="")
        visual = build_visual_features(
            item_id=22,
            item_type="tops",
            caption_icon="collar is tied in a bow, checkered halter top",
            caption_overview="long sleeves with brown stripes on the sleeves",
            caption_visual="collar is tied in a bow, checkered halter top, long sleeves with brown stripes on the sleeves",
            color_tags=ColorTaggingResult(
                dominant_colors=["black"],
                accent_colors=["white"],
            ),
        )

        candidates = build_structured_candidates(item_input, visual)
        sources_by_key: dict[str, set[str]] = {}
        for candidate in candidates:
            sources_by_key.setdefault(candidate.concept_key, set()).add(candidate.source)

        self.assertIn("term_rule", sources_by_key.get("tops.subtype:halter", set()))
        self.assertIn("term_rule", sources_by_key.get("pattern:checkered", set()))
        self.assertIn("term_rule", sources_by_key.get("tops.sleeve_style:long_sleeve", set()))
        self.assertIn("term_rule", sources_by_key.get("motif:bow", set()))
        self.assertIn("tops.neckline:collared", sources_by_key)

    def test_ambiguous_top_subtypes_stay_out_of_accepted_facets_but_land_in_search_terms(self) -> None:
        item_input = ItemInputRecord(item_id=3, item_type="tops", icon_path="", overview_path="")
        visual = build_visual_features(
            item_id=3,
            item_type="tops",
            caption_icon="buttons, bow",
            caption_overview="cardigan, white jacket, v-neckline",
            caption_visual="white jacket, cardigan, v-neckline, button-down front, bow",
            color_tags=ColorTaggingResult(
                dominant_colors=["white"],
                accent_colors=["black"],
            ),
        )

        candidates = build_structured_candidates(item_input, visual)
        assignments = build_tag_assignments(item_input, visual, "test-model", candidates=candidates)
        metadata = build_metadata_record(item_input, visual, assignments)

        self.assertNotIn("tops.subtype:jacket", metadata.accepted_facets)
        self.assertNotIn("tops.subtype:cardigan", metadata.accepted_facets)
        self.assertIn("tops.subtype:cardigan", metadata.review_facets)
        self.assertIn("white jacket", metadata.search_terms)
        self.assertIn("cardigan", metadata.search_terms)
        self.assertIn("v-neckline", metadata.search_terms)
        self.assertIn("button-down front", metadata.search_terms)

    def test_document_keeps_search_terms_without_leaking_review_facets_into_filters(self) -> None:
        item_input = ItemInputRecord(item_id=4, item_type="tops", icon_path="", overview_path="")
        visual = build_visual_features(
            item_id=4,
            item_type="tops",
            caption_icon="white jacket, buttons, bow",
            caption_overview="cardigan, white jacket, v-neckline",
            caption_visual="white jacket, cardigan, v-neckline, button-down front, bow",
            color_tags=ColorTaggingResult(
                dominant_colors=["white"],
                accent_colors=["black"],
            ),
        )

        candidates = build_structured_candidates(item_input, visual)
        assignments = build_tag_assignments(item_input, visual, "test-model", candidates=candidates)
        metadata = build_metadata_record(item_input, visual, assignments)
        document = build_document_record(metadata, visual, assignments)

        self.assertIn("motif:bow", metadata.accepted_facets)
        self.assertNotIn("tops.neckline:v_neck", metadata.accepted_facets)
        self.assertIn("v-neckline", document.data)
        self.assertIn("white jacket", document.data)
        self.assertIn("bow", document.data)
        self.assertEqual(document.metadata["item_type"], "tops")
        self.assertIn("search_terms", document.metadata)


if __name__ == "__main__":
    unittest.main()
