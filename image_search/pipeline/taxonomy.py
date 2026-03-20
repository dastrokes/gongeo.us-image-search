from __future__ import annotations

import re

from image_search.constants.taxonomy import (
    CONCEPT_BY_KEY,
    EXCLUSIVE_FACETS,
    IMPLICATION_RULES,
    NEAR_EXCLUSIVE_FACETS,
    PARENT_CHILD_RELATIONSHIPS,
    TAXONOMY_CONCEPTS,
    TERM_EXTRACTION_RULES_BY_TYPE,
    VISUAL_CONCEPTS_BY_TYPE,
    ConceptDefinition,
    ExtractionRule,
)
from image_search.models.schemas import (
    ItemInputRecord,
    ManifestRecord,
    MetadataRecord,
    ReviewRecord,
    StructuredCandidateRecord,
    TagAssignmentRecord,
    TaxonomyConceptRecord,
    UnmappedTermRecord,
    VisualFeatureRecord,
)
from image_search.models.type_profiles import get_type_profile
from image_search.pipeline.color_tags import ColorTaggingResult


ACCEPTED_THRESHOLD = 0.88
REVIEW_THRESHOLD = 0.70
SEARCH_TERM_THRESHOLD = 0.66
SUBCATEGORY_CONFLICT_MARGIN = 0.08
DEFAULT_CONFLICT_MARGIN = 0.06
SUBCATEGORY_ACCEPTED_THRESHOLD = 0.82
SUBCATEGORY_REVIEW_THRESHOLD = 0.68


def build_taxonomy_concepts() -> list[TaxonomyConceptRecord]:
    return [
        TaxonomyConceptRecord(
            key=definition.key,
            layer=definition.layer,
            facet_key=definition.facet_key,
            value_key=definition.value_key,
            display=definition.display,
            allowed_item_types=list(definition.allowed_item_types),
            multi_value=definition.multi_value,
            preferred_evidence=definition.preferred_evidence,
            aliases=list(definition.aliases),
            search_only=definition.search_only,
            is_filterable=definition.is_filterable,
        )
        for definition in TAXONOMY_CONCEPTS
    ]


def build_item_input(record: ManifestRecord) -> ItemInputRecord:
    return ItemInputRecord(
        item_id=record.item_id,
        item_type=record.type,
        icon_path=record.icon_path,
        overview_path=record.overview_path,
        has_icon=record.has_icon,
        has_overview=record.has_overview,
    )


def _split_caption_terms(value: str) -> list[str]:
    seen: set[str] = set()
    parts: list[str] = []
    for chunk in value.split(","):
        normalized = re.sub(r"\s+", " ", chunk.lower()).strip(" ,")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        parts.append(normalized)
    return parts


def _status_for_score(score: float, facet_key: str = "") -> str:
    accepted_threshold = ACCEPTED_THRESHOLD
    review_threshold = REVIEW_THRESHOLD
    if facet_key == "subcategory":
        accepted_threshold = SUBCATEGORY_ACCEPTED_THRESHOLD
        review_threshold = SUBCATEGORY_REVIEW_THRESHOLD

    if score >= accepted_threshold:
        return "accepted"
    if score >= review_threshold:
        return "review"
    return "suppressed"


def _facet_conflict_margin(facet_key: str) -> float:
    if facet_key == "subcategory":
        return SUBCATEGORY_CONFLICT_MARGIN
    if facet_key in NEAR_EXCLUSIVE_FACETS:
        return DEFAULT_CONFLICT_MARGIN + 0.01
    return DEFAULT_CONFLICT_MARGIN


def _display_label(concept_key: str) -> str:
    definition = CONCEPT_BY_KEY.get(concept_key)
    if definition is None:
        _, _, value_key = concept_key.partition(":")
        return value_key.replace("_", " ")
    return definition.display


def _matches_definition(definition: ConceptDefinition, text: str) -> bool:
    return any(re.search(pattern, text) for pattern in definition.patterns)


def _candidate_search_phrases(term: str) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()

    def add_variant(value: str) -> None:
        normalized = re.sub(r"\s+", " ", value.strip().lower()).strip(" ,")
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        variants.append(normalized)

    add_variant(term)

    stripped = re.sub(
        r"^(?:the )?(?:item|top|shirt|jacket|coat|dress|garment)\s+(?:has|have|is)\s+",
        "",
        term.lower(),
    )
    add_variant(stripped)

    for separator in (" with ", " on ", " at "):
        if separator in stripped:
            prefix, _, _ = stripped.partition(separator)
            add_variant(prefix)

    return variants


def _type_relevant_caption_terms(item_type: str, terms: list[str]) -> list[str]:
    concepts = VISUAL_CONCEPTS_BY_TYPE.get(item_type, ())
    profile_keywords = set(get_type_profile(item_type).keywords)
    kept: list[str] = []
    seen: set[str] = set()

    for term in terms:
        relevant_variants: list[str] = []
        for variant in _candidate_search_phrases(term):
            if variant == item_type:
                continue
            if any(_matches_definition(definition, variant) for definition in concepts):
                relevant_variants.append(variant)
                continue

            if any(keyword in variant for keyword in profile_keywords):
                relevant_variants.append(variant)

        if not relevant_variants:
            continue

        best_variant = min(
            set(relevant_variants),
            key=lambda value: (len(value.split()), len(value)),
        )
        if best_variant not in seen:
            seen.add(best_variant)
            kept.append(best_variant)

    return kept


def build_visual_features(
    item_id: int,
    item_type: str,
    caption_icon: str,
    caption_overview: str,
    caption_visual: str,
    color_tags: ColorTaggingResult,
) -> VisualFeatureRecord:
    palette: list[str] = []
    for value in [*color_tags.dominant_colors, *color_tags.accent_colors]:
        if value and value not in palette:
            palette.append(value)

    primary_color = color_tags.dominant_colors[0] if color_tags.dominant_colors else None
    secondary_color = None
    if len(color_tags.dominant_colors) > 1:
        secondary_color = color_tags.dominant_colors[1]
    elif color_tags.accent_colors:
        secondary_color = color_tags.accent_colors[0]

    icon_terms = _split_caption_terms(caption_icon)
    overview_terms = _split_caption_terms(caption_overview)
    raw_terms = _split_caption_terms(caption_visual)

    return VisualFeatureRecord(
        item_id=item_id,
        item_type=item_type,
        dominant_colors=list(color_tags.dominant_colors),
        accent_colors=list(color_tags.accent_colors),
        primary_color=primary_color,
        secondary_color=secondary_color,
        palette=palette,
        icon_caption=caption_icon,
        overview_caption=caption_overview,
        caption_visual=caption_visual,
        icon_terms=icon_terms,
        overview_terms=overview_terms,
        raw_terms=raw_terms,
        search_terms=_type_relevant_caption_terms(item_type, raw_terms),
        signals={
            "color_source": "deterministic_cv",
            "caption_modalities": {
                "icon": bool(caption_icon),
                "overview": bool(caption_overview),
            },
        },
    )


def _score_candidate_base(
    base_confidence: float,
    preferred_evidence: str,
    matched_modalities: set[str],
    match_count: int,
) -> float:
    score = base_confidence

    if preferred_evidence in {"icon", "overview"}:
        if preferred_evidence in matched_modalities:
            score += 0.08
        elif matched_modalities:
            score += 0.02
    elif preferred_evidence == "both":
        if "icon" in matched_modalities:
            score += 0.04
        if "overview" in matched_modalities:
            score += 0.04
    elif preferred_evidence == "cv_only":
        score += 0.08

    if "icon" in matched_modalities and "overview" in matched_modalities:
        score += 0.08
    elif matched_modalities:
        score += 0.03

    if match_count > 1:
        score += min(0.05, 0.02 * (match_count - 1))

    return min(score, 0.98)


def _score_candidate(
    definition: ConceptDefinition,
    matched_modalities: set[str],
    match_count: int,
) -> float:
    return _score_candidate_base(
        definition.base_confidence,
        definition.preferred_evidence,
        matched_modalities,
        match_count,
    )


def _color_candidates(
    item_input: ItemInputRecord,
    visual_features: VisualFeatureRecord,
) -> list[StructuredCandidateRecord]:
    candidates: list[StructuredCandidateRecord] = []

    if visual_features.primary_color:
        concept_key = f"color.primary:{visual_features.primary_color}"
        if concept_key in CONCEPT_BY_KEY:
            candidates.append(
                StructuredCandidateRecord(
                    item_id=item_input.item_id,
                    facet_key="color.primary",
                    value_key=visual_features.primary_color,
                    concept_key=concept_key,
                    source="cv",
                    source_modalities=["cv"],
                    pre_validation_score=0.97,
                    evidence={"signal": "dominant_colors"},
                )
            )

    if visual_features.secondary_color:
        concept_key = f"color.secondary:{visual_features.secondary_color}"
        if concept_key in CONCEPT_BY_KEY:
            candidates.append(
                StructuredCandidateRecord(
                    item_id=item_input.item_id,
                    facet_key="color.secondary",
                    value_key=visual_features.secondary_color,
                    concept_key=concept_key,
                    source="cv",
                    source_modalities=["cv"],
                    pre_validation_score=0.9,
                    evidence={"signal": "secondary_colors"},
                )
            )

    accent_value = None
    for color_value in visual_features.accent_colors:
        if color_value and color_value not in {
            visual_features.primary_color,
            visual_features.secondary_color,
        }:
            accent_value = color_value
            break

    if accent_value:
        concept_key = f"color.accent:{accent_value}"
        if concept_key in CONCEPT_BY_KEY:
            candidates.append(
                StructuredCandidateRecord(
                    item_id=item_input.item_id,
                    facet_key="color.accent",
                    value_key=accent_value,
                    concept_key=concept_key,
                    source="cv",
                    source_modalities=["cv"],
                    pre_validation_score=0.88,
                    evidence={"signal": "accent_colors"},
                )
            )

    return candidates


def _term_rule_candidates(
    item_input: ItemInputRecord,
    visual_features: VisualFeatureRecord,
) -> list[StructuredCandidateRecord]:
    candidates: list[StructuredCandidateRecord] = []
    modality_terms = {
        "icon": visual_features.icon_terms,
        "overview": visual_features.overview_terms,
        "combined": visual_features.raw_terms,
    }

    for rule in TERM_EXTRACTION_RULES_BY_TYPE.get(item_input.item_type, ()):
        matched_modalities: set[str] = set()
        matched_terms: list[str] = []
        matched_patterns: list[str] = []

        for modality, terms in modality_terms.items():
            for term in terms:
                for pattern in rule.patterns:
                    if re.search(pattern, term):
                        matched_modalities.add(modality)
                        matched_terms.append(term)
                        matched_patterns.append(pattern)

        if not matched_terms:
            continue

        for concept_key in rule.concept_keys:
            definition = CONCEPT_BY_KEY.get(concept_key)
            if definition is None:
                continue
            score = _score_candidate_base(
                min(definition.base_confidence + rule.score_bonus, 0.92),
                rule.preferred_evidence,
                matched_modalities,
                len(set(matched_patterns)),
            )
            candidates.append(
                StructuredCandidateRecord(
                    item_id=item_input.item_id,
                    facet_key=definition.facet_key,
                    value_key=definition.value_key,
                    concept_key=definition.key,
                    source="term_rule",
                    source_modalities=sorted(matched_modalities),
                    pre_validation_score=score,
                    search_only=definition.search_only,
                    evidence={
                        "rule": rule.name,
                        "matched_patterns": sorted(set(matched_patterns)),
                        "matched_terms": sorted(set(matched_terms)),
                        "preferred_evidence": rule.preferred_evidence,
                    },
                )
            )

    return candidates


def build_structured_candidates(
    item_input: ItemInputRecord,
    visual_features: VisualFeatureRecord,
) -> list[StructuredCandidateRecord]:
    candidates = [
        *_color_candidates(item_input, visual_features),
        *_term_rule_candidates(item_input, visual_features),
    ]
    icon_text = visual_features.icon_caption.lower()
    overview_text = visual_features.overview_caption.lower()
    combined_text = visual_features.caption_visual.lower()

    for definition in VISUAL_CONCEPTS_BY_TYPE.get(item_input.item_type, ()):
        if definition.preferred_evidence == "cv_only":
            continue

        matched_modalities: set[str] = set()
        matched_patterns: list[str] = []
        for pattern in definition.patterns:
            if icon_text and re.search(pattern, icon_text):
                matched_modalities.add("icon")
                matched_patterns.append(pattern)
            if overview_text and re.search(pattern, overview_text):
                matched_modalities.add("overview")
                matched_patterns.append(pattern)
            if not matched_modalities and combined_text and re.search(pattern, combined_text):
                matched_modalities.add("combined")
                matched_patterns.append(pattern)

        if not matched_patterns:
            continue

        score = _score_candidate(
            definition,
            matched_modalities,
            len(set(matched_patterns)),
        )
        candidates.append(
            StructuredCandidateRecord(
                item_id=item_input.item_id,
                facet_key=definition.facet_key,
                value_key=definition.value_key,
                concept_key=definition.key,
                source="caption_parse",
                source_modalities=sorted(matched_modalities),
                pre_validation_score=score,
                search_only=definition.search_only,
                evidence={
                    "matched_patterns": sorted(set(matched_patterns)),
                    "preferred_evidence": definition.preferred_evidence,
                },
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.facet_key,
            -candidate.pre_validation_score,
            candidate.value_key,
        ),
    )


def _dedupe_candidates(
    candidates: list[StructuredCandidateRecord],
) -> list[StructuredCandidateRecord]:
    deduped: dict[str, StructuredCandidateRecord] = {}
    for candidate in candidates:
        current = deduped.get(candidate.concept_key)
        if current is None or candidate.pre_validation_score > current.pre_validation_score:
            deduped[candidate.concept_key] = candidate
    return list(deduped.values())


def _prune_parent_candidates(
    candidates: list[StructuredCandidateRecord],
) -> list[StructuredCandidateRecord]:
    by_key = {candidate.concept_key: candidate for candidate in candidates}
    suppressed_keys: set[str] = set()

    for child_key, parent_keys in PARENT_CHILD_RELATIONSHIPS.items():
        child_candidate = by_key.get(child_key)
        if child_candidate is None:
            continue
        for parent_key in parent_keys:
            parent_candidate = by_key.get(parent_key)
            if parent_candidate is None:
                continue
            if parent_candidate.pre_validation_score <= child_candidate.pre_validation_score:
                suppressed_keys.add(parent_key)

    return [
        candidate
        for candidate in candidates
        if candidate.concept_key not in suppressed_keys
    ]


def _assignment_from_candidate(
    candidate: StructuredCandidateRecord,
    *,
    model_version: str,
    status: str,
    confidence: float | None = None,
    extra_evidence: dict[str, object] | None = None,
) -> TagAssignmentRecord:
    definition = CONCEPT_BY_KEY[candidate.concept_key]
    evidence = dict(candidate.evidence)
    evidence["source_modalities"] = list(candidate.source_modalities)
    if extra_evidence:
        evidence.update(extra_evidence)
    return TagAssignmentRecord(
        item_id=candidate.item_id,
        concept_key=candidate.concept_key,
        facet_key=candidate.facet_key,
        value_key=candidate.value_key,
        layer=definition.layer,
        source=candidate.source,
        confidence=round(
            candidate.pre_validation_score if confidence is None else confidence,
            4,
        ),
        status=status,
        model_version=model_version,
        search_only=definition.search_only,
        evidence=evidence,
    )


def _implied_assignments(
    assignments: list[TagAssignmentRecord],
    *,
    item_id: int,
    item_type: str,
    model_version: str,
) -> list[TagAssignmentRecord]:
    implied: list[TagAssignmentRecord] = []
    existing_keys = {assignment.concept_key for assignment in assignments}

    for assignment in assignments:
        if assignment.status != "accepted":
            continue
        for implied_key in IMPLICATION_RULES.get(assignment.concept_key, ()):
            if implied_key in existing_keys:
                continue
            definition = CONCEPT_BY_KEY.get(implied_key)
            if definition is None or item_type not in definition.allowed_item_types:
                continue
            existing_keys.add(implied_key)
            implied.append(
                TagAssignmentRecord(
                    item_id=item_id,
                    concept_key=definition.key,
                    facet_key=definition.facet_key,
                    value_key=definition.value_key,
                    layer=definition.layer,
                    source="implied",
                    confidence=round(max(0.72, assignment.confidence - 0.04), 4),
                    status="accepted",
                    model_version=model_version,
                    search_only=definition.search_only,
                    evidence={
                        "validation": "implied_parent",
                        "implied_by": assignment.concept_key,
                    },
                )
            )

    return implied


def _validate_candidates(
    item_type: str,
    candidates: list[StructuredCandidateRecord],
    *,
    model_version: str,
) -> list[TagAssignmentRecord]:
    grouped: dict[str, list[StructuredCandidateRecord]] = {}
    for candidate in _dedupe_candidates(candidates):
        definition = CONCEPT_BY_KEY.get(candidate.concept_key)
        if definition is None:
            continue
        grouped.setdefault(candidate.facet_key, []).append(candidate)

    assignments: list[TagAssignmentRecord] = []
    for facet_key, facet_candidates in grouped.items():
        facet_candidates = _prune_parent_candidates(facet_candidates)
        facet_candidates.sort(
            key=lambda candidate: (
                candidate.pre_validation_score,
                len(candidate.source_modalities),
            ),
            reverse=True,
        )

        first_definition = CONCEPT_BY_KEY[facet_candidates[0].concept_key]
        facet_is_exclusive = facet_key in EXCLUSIVE_FACETS
        facet_is_multi = first_definition.multi_value and not facet_is_exclusive

        if facet_is_multi:
            for index, candidate in enumerate(facet_candidates):
                definition = CONCEPT_BY_KEY[candidate.concept_key]
                status = _status_for_score(candidate.pre_validation_score, candidate.facet_key)
                if index >= definition.max_values:
                    if index == definition.max_values and status == "accepted":
                        status = "review"
                        extra = {"validation": "facet_limit_review"}
                    elif status == "review" and index == definition.max_values:
                        extra = {"validation": "facet_limit_review"}
                    else:
                        status = "suppressed"
                        extra = {"validation": "facet_limit"}
                else:
                    extra = None
                assignments.append(
                    _assignment_from_candidate(
                        candidate,
                        model_version=model_version,
                        status=status,
                        extra_evidence=extra,
                    )
                )
            continue

        top_candidate = facet_candidates[0]
        top_definition = CONCEPT_BY_KEY[top_candidate.concept_key]
        top_status = _status_for_score(top_candidate.pre_validation_score, top_candidate.facet_key)
        top_extra: dict[str, object] | None = None

        if len(facet_candidates) > 1:
            second_candidate = facet_candidates[1]
            margin = _facet_conflict_margin(facet_key)
            if (top_candidate.pre_validation_score - second_candidate.pre_validation_score) < margin:
                if top_status == "accepted":
                    top_extra = {
                        "validation": "accepted_with_conflict",
                        "competing_values": [
                            candidate.value_key
                            for candidate in facet_candidates[: min(3, len(facet_candidates))]
                        ],
                    }
                elif top_candidate.pre_validation_score >= REVIEW_THRESHOLD:
                    top_status = "review"
                    top_extra = {
                        "validation": "ambiguous_conflict",
                        "competing_values": [
                            candidate.value_key
                            for candidate in facet_candidates[: min(3, len(facet_candidates))]
                        ],
                    }
                else:
                    top_status = "suppressed"
                    top_extra = {
                        "validation": "ambiguous_conflict",
                        "competing_values": [
                            candidate.value_key
                            for candidate in facet_candidates[: min(3, len(facet_candidates))]
                        ],
                    }

        assignments.append(
            _assignment_from_candidate(
                top_candidate,
                model_version=model_version,
                status=top_status,
                extra_evidence=top_extra,
            )
        )

        for candidate in facet_candidates[1:]:
            assignments.append(
                _assignment_from_candidate(
                    candidate,
                    model_version=model_version,
                    status="suppressed",
                    confidence=min(candidate.pre_validation_score, top_candidate.pre_validation_score),
                    extra_evidence={
                        "validation": "facet_conflict",
                        "selected_value": top_definition.value_key,
                    },
                )
            )

    return sorted(
        assignments,
        key=lambda assignment: (
            assignment.facet_key,
            -assignment.confidence,
            assignment.value_key,
        ),
    )


def build_tag_assignments(
    item_input: ItemInputRecord,
    visual_features: VisualFeatureRecord,
    model_version: str,
    candidates: list[StructuredCandidateRecord] | None = None,
) -> list[TagAssignmentRecord]:
    resolved_candidates = candidates or build_structured_candidates(item_input, visual_features)
    assignments = _validate_candidates(
        item_input.item_type,
        resolved_candidates,
        model_version=model_version,
    )
    assignments.extend(
        _implied_assignments(
            assignments,
            item_id=item_input.item_id,
            item_type=item_input.item_type,
            model_version=model_version,
        )
    )
    return sorted(
        assignments,
        key=lambda assignment: (
            assignment.facet_key,
            -assignment.confidence,
            assignment.value_key,
        ),
    )


def build_review_records(assignments: list[TagAssignmentRecord]) -> list[ReviewRecord]:
    return [
        ReviewRecord(
            item_id=assignment.item_id,
            concept_key=assignment.concept_key,
            facet_key=assignment.facet_key,
            value_key=assignment.value_key,
            confidence=assignment.confidence,
            status=assignment.status,
            evidence=assignment.evidence,
        )
        for assignment in assignments
        if assignment.status == "review"
    ]


def _term_source_modalities(
    term: str,
    visual_features: VisualFeatureRecord,
) -> list[str]:
    modalities: list[str] = []
    if term in visual_features.icon_terms:
        modalities.append("icon")
    if term in visual_features.overview_terms:
        modalities.append("overview")
    if term in visual_features.raw_terms and not modalities:
        modalities.append("combined")
    return modalities


def _term_matches_taxonomy(term: str, item_type: str) -> bool:
    for variant in _candidate_search_phrases(term):
        if any(
            _matches_definition(definition, variant)
            for definition in VISUAL_CONCEPTS_BY_TYPE.get(item_type, ())
        ):
            return True
        for rule in TERM_EXTRACTION_RULES_BY_TYPE.get(item_type, ()):
            if any(re.search(pattern, variant) for pattern in rule.patterns):
                return True
    return False


def build_unmapped_term_records(
    item_input: ItemInputRecord,
    visual_features: VisualFeatureRecord,
) -> list[UnmappedTermRecord]:
    records: list[UnmappedTermRecord] = []
    seen: set[str] = set()
    report_terms: list[str] = list(visual_features.search_terms)

    for term in visual_features.raw_terms:
        if re.search(
            r"\b(?:motif|ornament|trim|pattern|embroidery|embroidered|material|headpiece|accessory|horn|antler|halo|monocle|glasses|sunglasses|wing)\b",
            term,
        ):
            report_terms.append(term)

    for term in report_terms:
        if term in seen or _term_matches_taxonomy(term, item_input.item_type):
            continue
        seen.add(term)
        records.append(
            UnmappedTermRecord(
                item_id=item_input.item_id,
                item_type=item_input.item_type,
                term=term,
                source_modalities=_term_source_modalities(term, visual_features),
                evidence={
                    "icon_caption": visual_features.icon_caption,
                    "overview_caption": visual_features.overview_caption,
                },
            )
        )

    return records


def _collect_search_terms(
    item_input: ItemInputRecord,
    visual_features: VisualFeatureRecord,
    assignments: list[TagAssignmentRecord],
) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def add_term(value: str) -> None:
        normalized = re.sub(r"\s+", " ", value.strip().lower())
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        terms.append(normalized)

    for value in visual_features.search_terms:
        add_term(value)

    for assignment in assignments:
        definition = CONCEPT_BY_KEY.get(assignment.concept_key)
        label = _display_label(assignment.concept_key).lower()
        if assignment.search_only and assignment.status in {"accepted", "review"}:
            add_term(label)
        elif assignment.facet_key == "subcategory" and assignment.status != "accepted":
            add_term(label)
        elif assignment.status == "review" and assignment.facet_key in {
            "pattern",
            "motif",
            "ornament",
            "trim",
            "material",
        }:
            add_term(label)
        elif assignment.status == "accepted" and assignment.facet_key in {"subcategory", "pattern"}:
            add_term(label)

        if definition is None:
            continue
        if assignment.status in {"accepted", "review"}:
            for alias in definition.aliases:
                add_term(alias)
            for parent_key in PARENT_CHILD_RELATIONSHIPS.get(assignment.concept_key, ()):
                add_term(_display_label(parent_key))

    return terms


def build_metadata_record(
    item_input: ItemInputRecord,
    visual_features: VisualFeatureRecord,
    assignments: list[TagAssignmentRecord],
) -> MetadataRecord:
    accepted = [assignment for assignment in assignments if assignment.status == "accepted"]
    review = [assignment for assignment in assignments if assignment.status == "review"]
    accepted_filterable = [assignment for assignment in accepted if not assignment.search_only]
    facet_values: dict[str, list[str]] = {}
    for assignment in accepted_filterable:
        facet_values.setdefault(assignment.facet_key, []).append(assignment.value_key)

    return MetadataRecord(
        item_id=item_input.item_id,
        item_type=item_input.item_type,
        dominant_colors=list(visual_features.dominant_colors),
        accent_colors=list(visual_features.accent_colors),
        primary_color=visual_features.primary_color,
        secondary_color=visual_features.secondary_color,
        facet_values=facet_values,
        accepted_facets=[assignment.concept_key for assignment in accepted_filterable],
        review_facets=[assignment.concept_key for assignment in review],
        search_terms=_collect_search_terms(item_input, visual_features, assignments),
    )
