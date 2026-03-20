from __future__ import annotations

import re

from image_search.constants.colors import strip_leading_color_or_material_phrase
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

ACCEPTED_THRESHOLD = 0.8
REVIEW_THRESHOLD = 0.6
SEARCH_TERM_THRESHOLD = 0.6
SUBCATEGORY_CONFLICT_MARGIN = 0.08
DEFAULT_CONFLICT_MARGIN = 0.06
SUBCATEGORY_ACCEPTED_THRESHOLD = 0.8
SUBCATEGORY_REVIEW_THRESHOLD = 0.6
PREFERRED_MODALITY_MATCH_BONUS = 0.05
NON_PREFERRED_MODALITY_MATCH_BONUS = 0.03
BOTH_MODALITY_MATCH_BONUS = 0.03
COMBINED_MODALITY_MATCH_BONUS = 0.02
CV_ONLY_MATCH_BONUS = 0.08
DUAL_MODALITY_COVERAGE_BONUS = 0.06
SINGLE_MODALITY_COVERAGE_BONUS = 0.02
EXTRA_PATTERN_MATCH_BONUS = 0.02
MAX_PATTERN_MATCH_BONUS = 0.05
MAX_CANDIDATE_SCORE = 0.98

_OBVIOUS_UNMAPPED_TERMS = frozenset(
    {
        "ankle",
        "decorative trim",
        "hem",
        "material",
        "motif",
        "ornate trim",
        "ornament",
        "pattern",
        "patterned trim",
        "shoe",
        "shoes",
        "sparkling trim",
        "toe",
        "trim",
        "waist",
    }
)

_NEGATIVE_UNMAPPED_TERM_PATTERN = re.compile(
    r"^no(?: visible)? (?:pattern|ornament|motif|trim|material)$"
)


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
    for chunk in re.split(r",|\s+with\s+", value):
        normalized = re.sub(r"\s+", " ", chunk.lower()).strip(" ,")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        parts.append(normalized)
    return parts


def _strip_leading_color_or_material(term: str) -> str | None:
    stripped = strip_leading_color_or_material_phrase(term)
    if stripped is None or len(stripped.split()) < 1:
        return None
    return stripped


def _expand_compound_caption_terms(item_type: str, terms: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()

    def add_term(value: str) -> None:
        normalized = _normalize_term(value)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        expanded.append(normalized)

    for term in terms:
        add_term(term)
        stripped = _strip_leading_color_or_material(term)
        if stripped:
            add_term(stripped)
        words = term.split()
        if len(words) < 2:
            continue

        for start in range(len(words)):
            for end in range(len(words), start, -1):
                if end - start >= len(words):
                    continue
                candidate = " ".join(words[start:end])
                if _term_matches_taxonomy(candidate, item_type):
                    add_term(candidate)

    return expanded


def _normalize_term(value: str) -> str:
    normalized = re.sub(r"\s+", " ", str(value).replace("_", " ").strip().lower())
    return normalized.strip(" ,")


def _is_negative_term(value: str) -> bool:
    normalized = _normalize_term(value)
    return bool(normalized) and bool(re.match(r"^(?:no|without)\b", normalized))


def _unique_terms(values: list[str]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_term(value)
        if not normalized or normalized in seen or _is_negative_term(normalized):
            continue
        seen.add(normalized)
        terms.append(normalized)
    return terms


def _candidate_modalities_for_term(
    term: str,
    icon_terms: list[str],
    overview_terms: list[str],
    raw_terms: list[str],
) -> list[str]:
    modalities: list[str] = []
    if term in icon_terms:
        modalities.append("icon")
    if term in overview_terms:
        modalities.append("overview")
    if term in raw_terms and not modalities:
        modalities.append("combined")
    return modalities


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


_TYPE_ECHO_OVERRIDES = {"tops": {"top"}, "shoes": {"shoe"}, "socks": {"sock"}}


def _definition_requires_exact_term_match(definition: ConceptDefinition) -> bool:
    return definition.facet_key == "material" or definition.facet_key.startswith(
        "color."
    )


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
    color_tags: ColorTaggingResult,
) -> VisualFeatureRecord:
    palette: list[str] = []
    for value in [*color_tags.dominant_colors, *color_tags.accent_colors]:
        if value and value not in palette:
            palette.append(value)

    primary_color = (
        color_tags.dominant_colors[0] if color_tags.dominant_colors else None
    )
    secondary_color = None
    if len(color_tags.dominant_colors) > 1:
        secondary_color = color_tags.dominant_colors[1]
    elif color_tags.accent_colors:
        secondary_color = color_tags.accent_colors[0]

    icon_terms = _expand_compound_caption_terms(
        item_type, _split_caption_terms(caption_icon)
    )
    overview_terms = _expand_compound_caption_terms(
        item_type, _split_caption_terms(caption_overview)
    )
    raw_terms = _unique_terms([*overview_terms, *icon_terms])
    derived_search_terms = _unique_terms(raw_terms)
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
        icon_terms=icon_terms,
        overview_terms=overview_terms,
        raw_terms=raw_terms,
        search_terms=_type_relevant_caption_terms(item_type, derived_search_terms),
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
            score += PREFERRED_MODALITY_MATCH_BONUS
        elif "combined" in matched_modalities:
            score += COMBINED_MODALITY_MATCH_BONUS
        elif matched_modalities:
            score += NON_PREFERRED_MODALITY_MATCH_BONUS
    elif preferred_evidence == "both":
        if "icon" in matched_modalities:
            score += BOTH_MODALITY_MATCH_BONUS
        if "overview" in matched_modalities:
            score += BOTH_MODALITY_MATCH_BONUS
        if matched_modalities == {"combined"}:
            score += COMBINED_MODALITY_MATCH_BONUS
    elif preferred_evidence == "cv_only":
        score += CV_ONLY_MATCH_BONUS

    if "icon" in matched_modalities and "overview" in matched_modalities:
        score += DUAL_MODALITY_COVERAGE_BONUS
    elif matched_modalities:
        score += SINGLE_MODALITY_COVERAGE_BONUS

    if match_count > 1:
        score += min(
            MAX_PATTERN_MATCH_BONUS, EXTRA_PATTERN_MATCH_BONUS * (match_count - 1)
        )

    return min(score, MAX_CANDIDATE_SCORE)


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
    combined_text = " ".join(
        text for text in (icon_text, overview_text) if text
    ).strip()

    for definition in VISUAL_CONCEPTS_BY_TYPE.get(item_input.item_type, ()):
        if definition.preferred_evidence == "cv_only":
            continue

        if _definition_requires_exact_term_match(definition):
            matched_modalities: set[str] = set()
            matched_patterns: list[str] = []
            modality_terms = {
                "icon": visual_features.icon_terms,
                "overview": visual_features.overview_terms,
                "combined": visual_features.raw_terms,
            }
            for modality, terms in modality_terms.items():
                for term in terms:
                    if any(re.fullmatch(pattern, term) for pattern in definition.patterns):
                        matched_modalities.add(modality)
                        matched_patterns.extend(definition.patterns)
            if not matched_modalities:
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
            if (
                not matched_modalities
                and combined_text
                and re.search(pattern, combined_text)
            ):
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

    deduped = _dedupe_candidates(candidates)
    return sorted(
        deduped,
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
        if (
            current is None
            or candidate.pre_validation_score > current.pre_validation_score
        ):
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
        child_status = _status_for_score(
            child_candidate.pre_validation_score,
            child_candidate.facet_key,
        )
        if child_status != "accepted":
            continue
        for parent_key in parent_keys:
            parent_candidate = by_key.get(parent_key)
            if parent_candidate is None:
                continue
            if (
                parent_candidate.pre_validation_score
                <= child_candidate.pre_validation_score
            ):
                suppressed_keys.add(parent_key)

    return [
        candidate
        for candidate in candidates
        if candidate.concept_key not in suppressed_keys
    ]


def _prefer_specific_subcategory_candidate(
    candidates: list[StructuredCandidateRecord],
) -> list[StructuredCandidateRecord]:
    if not candidates:
        return candidates

    candidate_by_key = {candidate.concept_key: candidate for candidate in candidates}
    preferred_child_keys = {
        candidate.concept_key
        for candidate in candidates
        if _status_for_score(candidate.pre_validation_score, candidate.facet_key)
        == "accepted"
        and any(
            parent_key in candidate_by_key
            for parent_key in PARENT_CHILD_RELATIONSHIPS.get(
                candidate.concept_key,
                (),
            )
        )
    }
    if not preferred_child_keys:
        return candidates

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.concept_key in preferred_child_keys,
            candidate.pre_validation_score,
            len(candidate.source_modalities),
        ),
        reverse=True,
    )


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
        if facet_key == "subcategory":
            facet_candidates = _prefer_specific_subcategory_candidate(facet_candidates)

        first_definition = CONCEPT_BY_KEY[facet_candidates[0].concept_key]
        facet_is_exclusive = facet_key in EXCLUSIVE_FACETS
        facet_is_multi = first_definition.multi_value and not facet_is_exclusive

        if facet_is_multi:
            for index, candidate in enumerate(facet_candidates):
                definition = CONCEPT_BY_KEY[candidate.concept_key]
                status = _status_for_score(
                    candidate.pre_validation_score, candidate.facet_key
                )
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

        if not facet_is_exclusive:
            for candidate in facet_candidates:
                assignments.append(
                    _assignment_from_candidate(
                        candidate,
                        model_version=model_version,
                        status=_status_for_score(
                            candidate.pre_validation_score, candidate.facet_key
                        ),
                    )
                )
            continue

        top_candidate = facet_candidates[0]
        top_definition = CONCEPT_BY_KEY[top_candidate.concept_key]
        top_status = _status_for_score(
            top_candidate.pre_validation_score, top_candidate.facet_key
        )
        top_extra: dict[str, object] | None = None

        if len(facet_candidates) > 1:
            second_candidate = facet_candidates[1]
            margin = _facet_conflict_margin(facet_key)
            if (
                top_candidate.pre_validation_score
                - second_candidate.pre_validation_score
            ) < margin:
                if top_status == "accepted":
                    top_extra = {
                        "validation": "accepted_with_conflict",
                        "competing_values": [
                            candidate.value_key
                            for candidate in facet_candidates[
                                : min(3, len(facet_candidates))
                            ]
                        ],
                    }
                elif top_candidate.pre_validation_score >= REVIEW_THRESHOLD:
                    top_status = "review"
                    top_extra = {
                        "validation": "ambiguous_conflict",
                        "competing_values": [
                            candidate.value_key
                            for candidate in facet_candidates[
                                : min(3, len(facet_candidates))
                            ]
                        ],
                    }
                else:
                    top_status = "suppressed"
                    top_extra = {
                        "validation": "ambiguous_conflict",
                        "competing_values": [
                            candidate.value_key
                            for candidate in facet_candidates[
                                : min(3, len(facet_candidates))
                            ]
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
                    confidence=min(
                        candidate.pre_validation_score,
                        top_candidate.pre_validation_score,
                    ),
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
    resolved_candidates = candidates or build_structured_candidates(
        item_input, visual_features
    )
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
            (
                any(re.fullmatch(pattern, variant) for pattern in definition.patterns)
                if _definition_requires_exact_term_match(definition)
                else _matches_definition(definition, variant)
            )
            for definition in VISUAL_CONCEPTS_BY_TYPE.get(item_type, ())
        ):
            return True
        for rule in TERM_EXTRACTION_RULES_BY_TYPE.get(item_type, ()):
            if any(re.search(pattern, variant) for pattern in rule.patterns):
                return True
    return False


def _observed_term_variants(
    assignment: TagAssignmentRecord,
    observed_terms: set[str] | None = None,
) -> list[str]:
    definition = CONCEPT_BY_KEY.get(assignment.concept_key)
    candidate_values: list[str] = [_display_label(assignment.concept_key).lower()]

    matched_terms = assignment.evidence.get("matched_terms")
    if isinstance(matched_terms, list):
        candidate_values.extend(
            str(value).lower().strip() for value in matched_terms if str(value).strip()
        )

    if definition is not None:
        candidate_values.extend(alias.lower() for alias in definition.aliases)

    variants: list[str] = []
    seen_variants: set[str] = set()
    for value in candidate_values:
        normalized = re.sub(r"\s+", " ", value.strip().lower())
        if not normalized or normalized in seen_variants:
            continue
        seen_variants.add(normalized)
        if observed_terms is not None and not any(
            observed == normalized
            or observed.startswith(f"{normalized} ")
            or normalized in observed
            for observed in observed_terms
        ):
            continue
        variants.append(normalized)

    return variants


def build_unmapped_term_records(
    item_input: ItemInputRecord,
    visual_features: VisualFeatureRecord,
    assignments: list[TagAssignmentRecord] | None = None,
) -> list[UnmappedTermRecord]:
    records: list[UnmappedTermRecord] = []
    seen: set[str] = set()
    report_terms: list[str] = list(visual_features.search_terms)
    observed_terms = {
        re.sub(r"\s+", " ", value.strip().lower())
        for value in (
            *visual_features.search_terms,
            *visual_features.raw_terms,
            *visual_features.icon_terms,
            *visual_features.overview_terms,
        )
        if value and value.strip()
    }
    covered_variants: set[str] = set()
    for assignment in assignments or []:
        if assignment.status not in {"accepted", "review"}:
            continue
        covered_variants.update(_observed_term_variants(assignment, observed_terms))

    for term in visual_features.raw_terms:
        if re.search(
            r"\b(?:motif|ornament|trim|pattern|embroidery|embroidered|material|headpiece|accessory|horn|antler|halo|monocle|glasses|sunglasses|wing)\b",
            term,
        ):
            report_terms.append(term)

    for term in report_terms:
        if (
            term in seen
            or _is_obvious_noise_term(term)
            or _is_type_echo_term(term, item_input.item_type)
            or _term_is_covered_by_assignments(term, covered_variants)
            or _term_matches_taxonomy(term, item_input.item_type)
        ):
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


def _term_is_covered_by_assignments(
    term: str,
    covered_variants: set[str],
) -> bool:
    normalized = _normalize_term(term)
    if not normalized or not covered_variants:
        return False
    for variant in _candidate_search_phrases(normalized):
        if any(
            covered == variant or covered in variant or variant in covered
            for covered in covered_variants
        ):
            return True
    return False


def _is_obvious_noise_term(term: str) -> bool:
    normalized = re.sub(r"\s+", " ", term.strip().lower())
    if not normalized:
        return True
    if normalized in _OBVIOUS_UNMAPPED_TERMS:
        return True
    return bool(_NEGATIVE_UNMAPPED_TERM_PATTERN.fullmatch(normalized))


def _is_type_echo_term(term: str, item_type: str) -> bool:
    normalized_term = _normalize_term(term)
    normalized_type = _normalize_term(item_type)
    if not normalized_term or not normalized_type:
        return False
    if normalized_term == normalized_type:
        return True
    if normalized_term in _TYPE_ECHO_OVERRIDES.get(normalized_type, set()):
        return True
    singular_type = normalized_type[:-1] if normalized_type.endswith("s") else ""
    if singular_type and normalized_term == singular_type:
        return True
    return False


def _collect_search_terms(
    item_input: ItemInputRecord,
    visual_features: VisualFeatureRecord,
    assignments: list[TagAssignmentRecord],
) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    observed_terms = {
        re.sub(r"\s+", " ", value.strip().lower())
        for value in (
            *visual_features.search_terms,
            *visual_features.raw_terms,
            *visual_features.icon_terms,
            *visual_features.overview_terms,
        )
        if value and value.strip()
    }

    def add_term(value: str) -> None:
        normalized = re.sub(r"\s+", " ", value.strip().lower())
        if not normalized or normalized in seen or _is_negative_term(normalized):
            return
        seen.add(normalized)
        terms.append(normalized)

    def was_observed(value: str) -> bool:
        normalized = re.sub(r"\s+", " ", value.strip().lower())
        if not normalized:
            return False
        return any(
            observed == normalized
            or observed.startswith(f"{normalized} ")
            or normalized in observed
            for observed in observed_terms
        )

    for value in visual_features.search_terms:
        add_term(value)

    for assignment in assignments:
        definition = CONCEPT_BY_KEY.get(assignment.concept_key)
        label = _display_label(assignment.concept_key).lower()
        if assignment.search_only and assignment.status in {"accepted", "review"}:
            for variant in _observed_term_variants(assignment, observed_terms):
                add_term(variant)
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
        elif assignment.status == "accepted" and assignment.facet_key in {
            "subcategory",
            "pattern",
        }:
            add_term(label)

        if definition is None:
            continue
        if assignment.status in {"accepted", "review"}:
            for alias in definition.aliases:
                if was_observed(alias):
                    add_term(alias)
            for parent_key in PARENT_CHILD_RELATIONSHIPS.get(
                assignment.concept_key, ()
            ):
                add_term(_display_label(parent_key))

    return terms


def build_metadata_record(
    item_input: ItemInputRecord,
    visual_features: VisualFeatureRecord,
    assignments: list[TagAssignmentRecord],
) -> MetadataRecord:
    accepted = [
        assignment for assignment in assignments if assignment.status == "accepted"
    ]
    review = [assignment for assignment in assignments if assignment.status == "review"]
    accepted_filterable = [
        assignment for assignment in accepted if not assignment.search_only
    ]
    facet_values: dict[str, list[str]] = {}
    for assignment in accepted_filterable:
        facet_values.setdefault(assignment.facet_key, []).append(assignment.value_key)
    subtype_values = facet_values.get("subcategory", [])

    return MetadataRecord(
        item_id=item_input.item_id,
        item_type=item_input.item_type,
        subtype=subtype_values[0] if subtype_values else None,
        dominant_colors=list(visual_features.dominant_colors),
        accent_colors=list(visual_features.accent_colors),
        primary_color=visual_features.primary_color,
        secondary_color=visual_features.secondary_color,
        facet_values=facet_values,
        accepted_facets=[assignment.concept_key for assignment in accepted_filterable],
        review_facets=[assignment.concept_key for assignment in review],
        search_terms=_collect_search_terms(item_input, visual_features, assignments),
    )
