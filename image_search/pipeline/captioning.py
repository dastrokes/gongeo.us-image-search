from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from image_search.constants.colors import (
    COLOR_DETAIL_NOUNS,
    COLOR_ONLY_PATTERN,
    COLOR_PATTERN,
    normalize_color_label,
)
from image_search.constants.settings import DEFAULT_CAPTION_MODEL_ID
from image_search.constants.text import (
    BOTTOM_GARMENT_PATTERNS,
    CAPTION_NOISE_TERMS,
    DRESS_GARMENT_PATTERNS,
    LOW_SIGNAL_VISUAL_PATTERN,
    OUTERWEAR_GARMENT_PATTERNS,
    QWEN_STRUCTURED_DETAIL_PROMPT,
    STOP_VISUAL_PATTERN,
    TOP_GARMENT_PATTERNS,
)
from image_search.constants.taxonomy import subcategory_labels_for_item_type
from image_search.models.schemas import CaptionRecord, ManifestRecord
from image_search.models.type_profiles import (
    ACCESSORY_TYPES,
    APPAREL_TYPES,
    FACE_DETAIL_TYPES,
    get_type_profile,
    is_visual_term_relevant,
)


@dataclass(slots=True)
class CaptionerConfig:
    model_id: str = DEFAULT_CAPTION_MODEL_ID
    device: str = "auto"
    device_map: str = "auto"
    dtype: str = "auto"
    batch_size: int = 8
    inference_batch_size: int = 2


class VisionCaptioner:
    _APPAREL_LEAK_PATTERNS: tuple[str, ...] = (
        *TOP_GARMENT_PATTERNS,
        *BOTTOM_GARMENT_PATTERNS,
        *OUTERWEAR_GARMENT_PATTERNS,
        *DRESS_GARMENT_PATTERNS,
        r"\bcorset\b",
        r"\bvest\b",
        r"\bapron\b",
    )
    _JEWELRY_LEAK_PATTERNS: tuple[str, ...] = (
        r"\bnecklace\b",
        r"\bchoker\b",
        r"\bpendant\b",
        r"\bearrings?\b",
        r"\bbracelets?\b",
        r"\bring\b",
        r"\bbrooch\b",
    )
    _BODY_LEAK_PATTERNS: tuple[str, ...] = (
        r"\bface\b",
        r"\bskin\b",
        r"\bhand\b",
        r"\barm\b",
        r"\btorso\b",
        r"\bbody\b",
    )
    _HAIR_LEAK_PATTERNS: tuple[str, ...] = (
        r"\bhair\b",
        r"\bponytails?\b",
        r"\bpigtails?\b",
        r"\btwin tails?\b",
        r"\bbangs?\b",
        r"\bbun\b",
        r"\bbraid(?:ed)?\b",
        r"\bcurls?\b",
        r"\bwaves?\b",
    )

    @staticmethod
    def _extract_style_descriptors(source_text: str, item_type: str) -> list[str]:
        profile = get_type_profile(item_type)
        descriptors: list[str] = []
        seen: set[str] = set()
        for pattern, descriptor in profile.style_patterns:
            if descriptor in seen:
                continue
            if re.search(pattern, source_text):
                seen.add(descriptor)
                descriptors.append(descriptor)
        return descriptors

    @staticmethod
    def _is_default_showcase_descriptor(
        candidate: str, item_type: str, modality: str
    ) -> bool:
        normalized = candidate.strip().lower()
        if not normalized:
            return False
        profile = get_type_profile(item_type)

        if item_type in {"outerwear", "tops"} and any(
            re.search(pattern, normalized) for pattern in BOTTOM_GARMENT_PATTERNS
        ):
            return True
        if item_type in {"outerwear", "tops", "bottoms", "socks", "shoes"} and any(
            re.search(pattern, normalized) for pattern in DRESS_GARMENT_PATTERNS
        ):
            return True

        if item_type != "hair" and COLOR_ONLY_PATTERN.fullmatch(normalized):
            return True

        blocked_patterns: tuple[str, ...] = ()
        if modality == "overview":
            if profile.blocked_patterns_overview:
                blocked_patterns = profile.blocked_patterns_overview
            elif item_type == "outerwear":
                blocked_patterns = TOP_GARMENT_PATTERNS + BOTTOM_GARMENT_PATTERNS
            elif item_type == "tops":
                blocked_patterns = BOTTOM_GARMENT_PATTERNS + DRESS_GARMENT_PATTERNS
            elif item_type in {"bottoms", "socks", "shoes"}:
                blocked_patterns = (
                    TOP_GARMENT_PATTERNS
                    + OUTERWEAR_GARMENT_PATTERNS
                    + DRESS_GARMENT_PATTERNS
                )
            elif item_type not in APPAREL_TYPES and item_type != "hair":
                blocked_patterns = (
                    TOP_GARMENT_PATTERNS
                    + BOTTOM_GARMENT_PATTERNS
                    + DRESS_GARMENT_PATTERNS
                )
        elif profile.blocked_patterns_icon:
            blocked_patterns = profile.blocked_patterns_icon

        return any(re.search(pattern, normalized) for pattern in blocked_patterns)

    @staticmethod
    def _candidate_parts(part: str) -> list[str]:
        part = part.lower().strip(" ,")
        if not part:
            return []

        candidates: list[str] = []
        seen: set[str] = set()

        def add_candidate(value: str) -> None:
            normalized = value.replace("grey", "gray")
            normalized = re.sub(r"\s+", " ", normalized).strip(" ,")
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            candidates.append(normalized)

        simplified = re.sub(
            r"^(?:there (?:is|are)|there's|this is|it is|image is(?: of)?|image shows?|pictured is)\s+",
            "",
            part,
        )
        simplified = simplified.replace("'s", "")
        if not re.search(r"\b(?:is|are|around|near|inside|covering)\b", simplified):
            add_candidate(simplified)

        color_object_match = re.search(
            rf"\b(?P<color>{COLOR_PATTERN})\s+(?P<object>{'|'.join(COLOR_DETAIL_NOUNS)})\b",
            simplified,
        )
        if color_object_match:
            add_candidate(
                f"{color_object_match.group('color')} {color_object_match.group('object')}"
            )

        subject_color_match = re.search(
            rf"\b(?P<subject>[a-z]+(?: [a-z]+){{0,2}})\s+(?:is|are)\s+(?P<color>{COLOR_PATTERN})\b",
            simplified,
        )
        if subject_color_match:
            subject = re.sub(
                r"^(?:a|an|the)\s+",
                "",
                subject_color_match.group("subject"),
            ).strip()
            if subject:
                add_candidate(f"{subject_color_match.group('color')} {subject}")

        stripped_clause = re.sub(
            r"\b(?:around|on|at|near|inside|covering)\b.*$",
            "",
            simplified,
        ).strip(" ,")
        if stripped_clause and stripped_clause != simplified:
            add_candidate(stripped_clause)

        return candidates

    @staticmethod
    def _build_prompt(item_type: str, modality: str) -> str:
        focus = "the item's visible structure, materials, and distinguishing details"
        if modality == "overview":
            focus = "overall silhouette, length, layering, placement, and major materials"
        elif modality == "icon":
            focus = "small motifs, trims, closures, embroidery, ornaments, and accent details"

        prompt = QWEN_STRUCTURED_DETAIL_PROMPT.format(focus=focus)
        if item_type:
            prompt += f"\nTreat the main item as a `{item_type}`."
            prompt += VisionCaptioner._subcategory_prompt_rule(item_type)
            prompt += VisionCaptioner._attribute_prompt_rule(item_type, modality)
            prompt += VisionCaptioner._coverage_prompt_rule(item_type, modality)
        return prompt

    @staticmethod
    def _build_plain_prompt(item_type: str, modality: str) -> str:
        focus = "the item's visible structure, materials, and distinguishing details"
        if modality == "overview":
            focus = "overall silhouette, length, layering, placement, and major materials"
        elif modality == "icon":
            focus = "small motifs, trims, closures, embroidery, ornaments, and accent details"

        prompt = (
            "Describe only the main wearable item in this image.\n"
            f"Focus on {focus}.\n"
            "Start with the main garment/accessory subcategory if it is visible.\n"
            "Be exhaustive about visible attributes, but omit anything not clearly visible.\n"
            "Prefer concrete visual facts such as length, silhouette, shape, placement, material, pattern, motif, trim, closure, and ornament.\n"
            "Return only a comma-separated list of short lowercase phrases.\n"
            "Do not return JSON.\n"
            "Do not mention the character, pose, background, or nearby items."
        )
        if item_type:
            prompt += f"\nTreat the main item as a `{item_type}`."
            prompt += VisionCaptioner._subcategory_prompt_rule(item_type)
            prompt += VisionCaptioner._attribute_prompt_rule(item_type, modality)
            prompt += VisionCaptioner._coverage_prompt_rule(item_type, modality)
        return prompt

    @staticmethod
    def _subcategory_prompt_rule(item_type: str) -> str:
        subcategory_labels = subcategory_labels_for_item_type(item_type)
        if not subcategory_labels:
            return ""

        examples = ", ".join(subcategory_labels[:6])
        return (
            "\nIf visible, include the item's subcategory as one of the first phrases."
            f"\nSubcategory examples for `{item_type}`: {examples}."
        )

    @staticmethod
    def _attribute_prompt_rule(item_type: str, modality: str) -> str:
        if item_type == "hair":
            return (
                "\nCover visible hair attributes: length, arrangement, bangs, texture, parting, and attached ornaments."
            )
        if item_type == "dresses":
            return (
                "\nCover visible dress attributes: subcategory, length, silhouette, neckline, sleeve length, sleeve shape, straps, waist, hem, layering, materials, pattern, motif, trim, ornament, and closures."
            )
        if item_type in {"tops", "outerwear"}:
            return (
                "\nCover visible apparel attributes: subcategory, neckline, collar, sleeve length, sleeve shape, garment length, hem, layering, front opening or closure, materials, pattern, motif, trim, and ornament."
            )
        if item_type == "bottoms":
            return (
                "\nCover visible bottom attributes: subcategory, rise, length, silhouette, pleats or layering, hem, materials, pattern, motif, trim, and ornament."
            )
        if item_type == "socks":
            return (
                "\nCover visible legwear attributes: subcategory, height, opacity, trim, pattern, motif, and ornament."
            )
        if item_type == "shoes":
            return (
                "\nCover visible footwear attributes: subcategory, heel height, shaft height, toe shape, platform, straps, buckles or closures, materials, pattern, trim, and ornament."
            )
        if item_type in ACCESSORY_TYPES:
            return (
                "\nCover visible accessory attributes: subcategory, shape, size, placement, attachment style, materials, pattern, motif, trim, ornament, gems, bows, ribbons, and closures."
            )
        if item_type in FACE_DETAIL_TYPES:
            return (
                "\nCover visible face-detail attributes: placement, shape, finish, intensity, color, pattern, motif, and decorative accents."
            )
        if item_type == "bodyPaint":
            return (
                "\nCover visible body-paint attributes: placement, coverage, shape, pattern, motif, finish, and color accents."
            )
        if item_type == "skinTones":
            return "\nDescribe only visible skin tone or complexion cues of the target cosmetic item."
        return (
            "\nCover visible attributes such as subcategory, shape, placement, length, materials, pattern, motif, trim, ornament, and closures when applicable."
        )

    @staticmethod
    def _coverage_prompt_rule(item_type: str, modality: str) -> str:
        if modality == "overview":
            if item_type in APPAREL_TYPES or item_type == "hair":
                return (
                    "\nThe overview should prioritize whole-item structure first: subcategory, length or height, silhouette, placement, and large construction details."
                )
            return (
                "\nThe overview should prioritize whole-item shape, placement, scale, and how details are distributed across the item."
            )
        if modality == "icon":
            return (
                "\nThe icon should prioritize small details second: trim, closures, motifs, ornaments, textures, and accent materials."
            )
        return ""

    @staticmethod
    def _collect_terms(value: Any, terms: list[str], seen: set[str]) -> None:
        if isinstance(value, str):
            for candidate in re.split(r"[,;\n]+", value):
                normalized = candidate.replace("_", " ").strip().lower()
                normalized = re.sub(r"\s+", " ", normalized)
                normalized = normalized.strip(" .,:;\"'`()[]{}")
                if normalized in {"", "n/a", "none", "unknown"}:
                    continue
                if normalized not in seen:
                    seen.add(normalized)
                    terms.append(normalized)
            return

        if isinstance(value, list):
            for item in value:
                VisionCaptioner._collect_terms(item, terms, seen)
            return

        if isinstance(value, dict):
            for key, nested in value.items():
                if key.lower() == "confidence":
                    continue
                VisionCaptioner._collect_terms(nested, terms, seen)

    @staticmethod
    def _strip_code_fences(raw_text: str) -> str:
        normalized = raw_text.strip()
        if normalized.startswith("```"):
            normalized = re.sub(r"^```(?:json)?\s*", "", normalized, flags=re.IGNORECASE)
            normalized = re.sub(r"\s*```$", "", normalized)
        return normalized

    @staticmethod
    def _clean_candidate_text(value: str) -> str:
        normalized = value.replace("_", " ").strip().lower()
        normalized = normalized.replace('\\"', '"')
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.strip(" .,:;\"'`()[]{}")
        normalized = re.sub(r"^(?:a|an|the)\s+", "", normalized)
        if normalized.startswith("json "):
            normalized = normalized[5:].strip()
        normalized = re.sub(r"^(icon|overview|merged)\s*:\s*", "", normalized)
        if any(token in normalized for token in ('{"', '"icon"', '"overview"', '"merged"')):
            return ""
        if normalized in {"", "n/a", "none", "unknown"}:
            return ""
        return normalized

    @staticmethod
    def _extract_bracket_content(text: str, start_index: int) -> tuple[str, int] | None:
        depth = 0
        collected: list[str] = []
        for index in range(start_index, len(text)):
            char = text[index]
            if char == "[":
                depth += 1
                if depth == 1:
                    continue
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return "".join(collected), index + 1
            if depth >= 1:
                collected.append(char)
        return None

    @staticmethod
    def _salvage_qwen_payload(raw_text: str) -> dict[str, list[str]] | None:
        normalized = VisionCaptioner._strip_code_fences(raw_text)
        payload: dict[str, list[str]] = {}
        for key in ("icon", "overview", "merged", "visual"):
            match = re.search(rf'"{key}"\s*:\s*\[', normalized, flags=re.IGNORECASE)
            if not match:
                continue
            extracted = VisionCaptioner._extract_bracket_content(normalized, match.end() - 1)
            if extracted is None:
                continue
            content, _ = extracted
            terms: list[str] = []
            seen: set[str] = set()
            for quoted in re.findall(r'"([^"]+)"', content):
                cleaned = VisionCaptioner._clean_candidate_text(quoted)
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    terms.append(cleaned)
            if not terms:
                for piece in content.split(","):
                    cleaned = VisionCaptioner._clean_candidate_text(piece)
                    if cleaned and cleaned not in seen:
                        seen.add(cleaned)
                        terms.append(cleaned)
            if terms:
                payload[key] = terms
        return payload or None

    @staticmethod
    def _extract_qwen_payload(raw_text: str) -> dict[str, Any] | None:
        normalized = VisionCaptioner._strip_code_fences(raw_text)
        if not normalized:
            return None

        start = normalized.find("{")
        end = normalized.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return VisionCaptioner._salvage_qwen_payload(normalized)

        try:
            payload = json.loads(normalized[start : end + 1])
        except json.JSONDecodeError:
            return VisionCaptioner._salvage_qwen_payload(normalized)
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _parse_qwen_response(raw_text: str) -> str:
        payload = VisionCaptioner._extract_qwen_payload(raw_text)
        terms: list[str] = []
        seen: set[str] = set()

        if payload is not None:
            ordered_keys = (
                "colors",
                "shape",
                "materials",
                "patterns",
                "motifs",
                "construction",
                "details",
            )
            for key in ordered_keys:
                VisionCaptioner._collect_terms(payload.get(key), terms, seen)
            for key, value in payload.items():
                if key in ordered_keys:
                    continue
                VisionCaptioner._collect_terms(value, terms, seen)
            if terms:
                return ", ".join(terms)
            return ""

        for line in raw_text.splitlines():
            if ":" not in line:
                continue
            _, value = line.split(":", 1)
            VisionCaptioner._collect_terms(value, terms, seen)

        if terms:
            return ", ".join(terms)
        return ""

    def __init__(self, config: CaptionerConfig) -> None:
        self.config = config
        self.model = None
        self.processor = None
        self.model_id = config.model_id
        self.device = None
        self.torch_dtype = None

    def _resolve_torch(self):
        import torch

        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        if self.config.dtype == "auto":
            if self.device == "cuda":
                self.torch_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = getattr(torch, self.config.dtype)
        return torch

    def _normalize_inputs(self, inputs: Any, torch: Any) -> dict[str, Any]:
        normalized_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                normalized_inputs[key] = value
                continue
            if torch.is_floating_point(value):
                normalized_inputs[key] = value.to(self.device, dtype=self.torch_dtype)
            else:
                normalized_inputs[key] = value.to(self.device)
        return normalized_inputs

    @staticmethod
    def _record_image_paths(record: ManifestRecord) -> list[tuple[str, str]]:
        paths: list[tuple[str, str]] = []
        if record.icon_path:
            paths.append(("icon", record.icon_path))
        if record.overview_path:
            paths.append(("overview", record.overview_path))
        return paths

    @staticmethod
    def _failed_modalities_for_record(
        *,
        has_icon: bool,
        has_overview: bool,
        icon_caption: str,
        overview_caption: str,
        visual: str,
    ) -> list[str]:
        failed: list[str] = []
        if has_icon and not icon_caption:
            failed.append("icon")
        if has_overview and not overview_caption:
            failed.append("overview")
        if not visual:
            failed.append("visual")
        return failed

    @staticmethod
    def _load_rgb_image(image_path: str | Path) -> Any:
        with Image.open(image_path) as image:
            return image.convert("RGB")

    @staticmethod
    def _merge_unique_parts(*values: str) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        for value in values:
            for part in [chunk.strip() for chunk in value.split(",") if chunk.strip()]:
                if part in seen:
                    continue
                seen.add(part)
                parts.append(part)
        return ", ".join(parts)

    @staticmethod
    def _compact_descriptors(values: list[str]) -> list[str]:
        compacted: list[str] = []
        for value in values:
            if any(
                value != other
                and re.search(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", other)
                for other in values
            ):
                continue
            compacted.append(value)
        return compacted

    @staticmethod
    def _chunk_records(
        records: list[ManifestRecord], batch_size: int
    ) -> list[list[ManifestRecord]]:
        if batch_size <= 0:
            batch_size = 1
        return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]

    def _retry_missing_modalities(
        self,
        record: ManifestRecord,
        modalities: list[str],
    ) -> dict[str, str]:
        if not modalities:
            return {}

        path_by_modality = {
            label: path for label, path in self._record_image_paths(record)
        }
        retry_modalities = [label for label in modalities if path_by_modality.get(label)]
        if not retry_modalities:
            return {}

        retried = self._caption_batch_qwen(
            [path_by_modality[label] for label in retry_modalities],
            [record.type] * len(retry_modalities),
            retry_modalities,
        )
        normalized = {
            label: self._normalize_caption(raw_caption, record.type, label)
            for label, raw_caption in zip(retry_modalities, retried)
        }
        still_missing = [
            label for label in retry_modalities if not normalized.get(label)
        ]
        if still_missing:
            raw_fallback = self._caption_batch_qwen(
                [path_by_modality[label] for label in still_missing],
                [record.type] * len(still_missing),
                still_missing,
                prompt_builder=self._build_plain_prompt,
                parse_structured=False,
            )
            for label, raw_caption in zip(still_missing, raw_fallback):
                normalized[label] = self._normalize_caption(raw_caption, record.type, label)
        return normalized

    @staticmethod
    def _flatten_qwen_field(value: Any) -> str:
        terms: list[str] = []
        seen: set[str] = set()
        VisionCaptioner._collect_terms(value, terms, seen)
        return ", ".join(terms)

    @staticmethod
    def _build_record_prompt(
        record: ManifestRecord,
        has_icon: bool,
        has_overview: bool,
    ) -> str:
        if has_icon and has_overview:
            focus = (
                "icon details in the first image and silhouette, placement, and scale "
                "in the second image"
            )
        elif has_icon:
            focus = "small motifs, trims, closures, embroidery, ornaments, and accent details"
        else:
            focus = "overall silhouette, length, layering, placement, and major materials"

        prompt = QWEN_STRUCTURED_DETAIL_PROMPT.format(focus=focus)
        if record.type:
            prompt += f"\nTreat the main item as a `{record.type}`."
            prompt += VisionCaptioner._subcategory_prompt_rule(record.type)
            prompt += VisionCaptioner._attribute_prompt_rule(record.type, "overview" if has_overview and not has_icon else "icon" if has_icon and not has_overview else "record")
        prompt += VisionCaptioner._type_specific_prompt_rules(record.type)
        if has_icon and has_overview:
            prompt += (
                "\nThe first image is the icon view and the second image is the "
                "overview view."
                "\nUse the icon array for small details and the overview array for whole-item structure."
                "\nEnsure the merged array preserves the strongest subcategory and the most important structural and decorative attributes from both views."
            )
        elif has_icon:
            prompt += "\nOnly the icon view is available."
        elif has_overview:
            prompt += "\nOnly the overview view is available."
        return prompt

    @staticmethod
    def _type_specific_prompt_rules(item_type: str) -> str:
        if item_type == "hair":
            return (
                "\nDescribe only the hairstyle or hair-attached decorations."
                "\nIgnore clothing, dresses, tops, jewelry, skin, face, and anything below the neck."
                "\nMention bows, ribbons, clips, or headbands only when they are attached to the hair."
            )
        if item_type in ACCESSORY_TYPES or item_type in FACE_DETAIL_TYPES:
            return (
                "\nDescribe only the target accessory or face detail."
                "\nIgnore surrounding clothing, adjacent jewelry, hairstyle, body parts, and nearby items unless they are part of the target item."
            )
        return ""

    @staticmethod
    def _parse_record_payload(
        raw_text: str,
        has_icon: bool,
        has_overview: bool,
    ) -> tuple[str, str, str]:
        payload = VisionCaptioner._extract_qwen_payload(raw_text)
        if payload is None:
            merged = VisionCaptioner._parse_qwen_response(raw_text)
            if has_icon and has_overview:
                return "", "", merged
            if has_icon:
                return merged, "", merged
            if has_overview:
                return "", merged, merged
            return "", "", merged

        icon = VisionCaptioner._flatten_qwen_field(payload.get("icon"))
        overview = VisionCaptioner._flatten_qwen_field(payload.get("overview"))
        merged = VisionCaptioner._flatten_qwen_field(payload.get("merged"))

        if not merged:
            merged = VisionCaptioner._flatten_qwen_field(payload.get("visual"))
        if not merged:
            merged = VisionCaptioner._merge_unique_parts(icon, overview)
        if not merged:
            merged = VisionCaptioner._parse_qwen_response(raw_text)

        if not icon and has_icon:
            icon = merged
        if not overview and has_overview:
            overview = merged
        return icon, overview, merged

    @staticmethod
    def _sanitized_generation_config(model: Any):
        import copy

        generation_config = copy.deepcopy(model.generation_config)
        generation_config.do_sample = False
        for key in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, key):
                setattr(generation_config, key, None)
        return generation_config

    def _caption_batch_qwen(
        self,
        image_paths: list[str | Path],
        item_types: list[str],
        modalities: list[str],
        *,
        prompt_builder: Any | None = None,
        parse_structured: bool = True,
    ) -> list[str]:
        import torch

        images = [Path(path) for path in image_paths]
        loaded_images = [Image.open(path).convert("RGB") for path in images]
        texts: list[str] = []
        resolved_prompt_builder = prompt_builder or self._build_prompt
        for item_type, modality in zip(item_types, modalities):
            prompt = resolved_prompt_builder(item_type, modality)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        inputs = self.processor(
            text=texts,
            images=loaded_images,
            padding=True,
            return_tensors="pt",
        )
        inputs = self._normalize_inputs(inputs, torch)
        generation_config = self._sanitized_generation_config(self.model)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                generation_config=generation_config,
                use_model_defaults=False,
                max_new_tokens=96,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        decoded_list = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not parse_structured:
            return decoded_list
        return [self._parse_qwen_response(decoded) for decoded in decoded_list]

    def _resolve_model_class(self):
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration

    def ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        self._resolve_torch()
        from transformers import AutoProcessor

        model_class = self._resolve_model_class()
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"
        model_kwargs: dict[str, Any] = {"dtype": self.torch_dtype}
        if self.device == "cuda" and self.config.device_map != "none":
            model_kwargs["device_map"] = "auto"
        self.model = model_class.from_pretrained(self.config.model_id, **model_kwargs)
        if self.device != "cuda" or self.config.device_map == "none":
            self.model.to(self.device)
        self.model.eval()
        self.model_id = self.config.model_id

    def _caption_records_batch_qwen(
        self, records: list[ManifestRecord]
    ) -> list[CaptionRecord]:
        import torch

        if not records:
            return []

        results: list[CaptionRecord] = []
        batch_size = max(1, int(self.config.inference_batch_size))
        generation_config = self._sanitized_generation_config(self.model)

        for batch in self._chunk_records(records, batch_size):
            texts: list[str] = []
            images: list[list[Any]] = []
            metadata: list[tuple[bool, bool, int, str]] = []
            records_by_id = {record.item_id: record for record in batch}

            for record in batch:
                image_specs = self._record_image_paths(record)
                has_icon = any(label == "icon" for label, _ in image_specs)
                has_overview = any(label == "overview" for label, _ in image_specs)
                prompt = self._build_record_prompt(record, has_icon, has_overview)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            *[
                                {
                                    "type": "image",
                                    "image": str(Path(path).resolve()),
                                }
                                for _label, path in image_specs
                            ],
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                texts.append(
                    self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                images.append([self._load_rgb_image(path) for _label, path in image_specs])
                metadata.append((has_icon, has_overview, record.item_id, record.type))

            inputs = self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            inputs = self._normalize_inputs(inputs, torch)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    use_model_defaults=False,
                    max_new_tokens=96,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            decoded_list = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for decoded, (has_icon, has_overview, item_id, item_type) in zip(
                decoded_list,
                metadata,
            ):
                record = records_by_id[item_id]
                icon, overview, merged = self._parse_record_payload(
                    decoded,
                    has_icon,
                    has_overview,
                )
                icon_caption = self._normalize_caption(icon, item_type, "icon")
                overview_caption = self._normalize_caption(
                    overview,
                    item_type,
                    "overview",
                )
                visual = self._normalize_caption(merged, item_type, "overview")

                if has_icon and not icon_caption:
                    icon_caption = visual
                if has_overview and not overview_caption:
                    overview_caption = visual

                missing_modalities = []
                if has_icon and not icon_caption:
                    missing_modalities.append("icon")
                if has_overview and not overview_caption:
                    missing_modalities.append("overview")

                if missing_modalities:
                    retried = self._retry_missing_modalities(record, missing_modalities)
                    if not icon_caption:
                        icon_caption = retried.get("icon", "")
                    if not overview_caption:
                        overview_caption = retried.get("overview", "")

                if not visual:
                    visual = self._normalize_caption(
                        self._merge_unique_parts(icon_caption, overview_caption),
                        item_type,
                        "overview",
                    )

                failed_modalities = self._failed_modalities_for_record(
                    has_icon=has_icon,
                    has_overview=has_overview,
                    icon_caption=icon_caption,
                    overview_caption=overview_caption,
                    visual=visual,
                )

                results.append(
                    CaptionRecord(
                        item_id=item_id,
                        icon_caption=icon_caption,
                        overview_caption=overview_caption,
                        visual=visual,
                        failed_modalities=failed_modalities,
                    )
                )

        return results

    def caption_image(
        self,
        image_path: str | Path,
        item_type: str = "",
        modality: str = "",
    ) -> str:
        self.ensure_loaded()
        return self.caption_images_batch([image_path], [item_type], [modality])[0]

    def caption_images_batch(
        self,
        image_paths: list[str | Path],
        item_types: list[str] | None = None,
        modalities: list[str] | None = None,
    ) -> list[str]:
        self.ensure_loaded()
        normalized_item_types = item_types or [""] * len(image_paths)
        normalized_modalities = modalities or [""] * len(image_paths)
        if not image_paths:
            return []

        results: list[str] = []
        batch_size = max(1, int(self.config.inference_batch_size))
        for start in range(0, len(image_paths), batch_size):
            end = start + batch_size
            batch_paths = image_paths[start:end]
            batch_types = normalized_item_types[start:end]
            batch_modalities = normalized_modalities[start:end]
            results.extend(
                self._caption_batch_qwen(batch_paths, batch_types, batch_modalities)
            )
        return results

    @staticmethod
    def _normalize_caption(
        raw_caption: str,
        item_type: str,
        modality: str = "",
    ) -> str:
        caption = raw_caption.lower()
        caption = re.sub(r"</?s>|<.*?>", " ", caption)
        caption = caption.replace(".", ",")
        caption = caption.replace(";", ",")
        caption = caption.replace(" and ", ", ")
        caption = re.sub(r"\s+", " ", caption).strip(" ,")
        parts = [part.strip(" ,") for part in caption.split(",") if part.strip(" ,")]

        grounded: list[str] = []
        fallback: list[str] = []
        seen: set[str] = set()
        profile = get_type_profile(item_type)

        def add_descriptor(target: list[str], value: str) -> None:
            normalized = value.strip(" ,")
            if len(normalized) < 2 or normalized in seen:
                return
            seen.add(normalized)
            target.append(normalized)

        if profile.extract_hair_color:
            for match in re.finditer(rf"(?P<color>{COLOR_PATTERN})\s+hair", caption):
                add_descriptor(
                    grounded,
                    f"{normalize_color_label(match.group('color'))} hair",
                )

        for part in parts:
            for candidate in VisionCaptioner._candidate_parts(part):
                candidate = re.sub(r"^(a|an|the)\s+", "", candidate)
                if any(noise in candidate for noise in CAPTION_NOISE_TERMS):
                    continue
                if STOP_VISUAL_PATTERN.search(candidate):
                    continue
                if LOW_SIGNAL_VISUAL_PATTERN.search(candidate):
                    continue
                candidate = re.sub(r"\b(she|her|hers)\b", "", candidate)
                candidate = re.sub(r"^(has|have|with)\s+", "", candidate)
                candidate = re.sub(r"\s+", " ", candidate).strip(" ,")
                if len(candidate) < 3 or candidate in seen:
                    continue
                if any(token in candidate for token in {"wearing", "face", "head tilted"}):
                    continue
                if VisionCaptioner._is_default_showcase_descriptor(
                    candidate,
                    item_type,
                    modality,
                ):
                    continue
                if VisionCaptioner._is_cross_item_leak(candidate, item_type):
                    continue
                if not is_visual_term_relevant(candidate, item_type):
                    continue
                add_descriptor(grounded, candidate)

        style_source = ", ".join(grounded) if grounded else caption
        for descriptor in VisionCaptioner._extract_style_descriptors(
            style_source,
            item_type,
        ):
            add_descriptor(fallback, descriptor)

        descriptors = VisionCaptioner._compact_descriptors(grounded + fallback)
        return ", ".join(descriptors[:10])

    @classmethod
    def _is_cross_item_leak(cls, candidate: str, item_type: str) -> bool:
        normalized = candidate.strip().lower()
        if not normalized:
            return False

        profile = get_type_profile(item_type)
        has_profile_keyword = any(keyword in normalized for keyword in profile.keywords)
        matches_apparel = any(
            re.search(pattern, normalized) for pattern in cls._APPAREL_LEAK_PATTERNS
        )
        matches_jewelry = any(
            re.search(pattern, normalized) for pattern in cls._JEWELRY_LEAK_PATTERNS
        )
        matches_body = any(
            re.search(pattern, normalized) for pattern in cls._BODY_LEAK_PATTERNS
        )
        matches_hair = any(
            re.search(pattern, normalized) for pattern in cls._HAIR_LEAK_PATTERNS
        )

        if item_type == "hair":
            return matches_apparel or matches_jewelry or matches_body

        if item_type in APPAREL_TYPES:
            if has_profile_keyword:
                return False
            return matches_hair or matches_body or matches_jewelry

        if item_type in ACCESSORY_TYPES or item_type in FACE_DETAIL_TYPES:
            if has_profile_keyword:
                return False
            return matches_apparel or matches_jewelry or matches_body or matches_hair

        return False

    @staticmethod
    def _caption_parts(normalized_caption: str) -> list[str]:
        return [
            chunk.strip()
            for chunk in normalized_caption.split(",")
            if chunk.strip()
        ]

    @classmethod
    def _is_low_signal_caption(cls, normalized_caption: str, item_type: str) -> bool:
        parts = cls._caption_parts(normalized_caption)
        if not parts:
            return True
        return all(part == item_type for part in parts)

    @staticmethod
    def _preferred_modalities(record: ManifestRecord) -> list[tuple[str, str]]:
        profile = get_type_profile(record.type)
        if profile.prefer_icon or record.type in ACCESSORY_TYPES or record.type in FACE_DETAIL_TYPES:
            ordered = (("icon", record.icon_path), ("overview", record.overview_path))
        else:
            ordered = (("overview", record.overview_path), ("icon", record.icon_path))

        modalities: list[tuple[str, str]] = []
        for label, path in ordered:
            if path:
                modalities.append((label, path))
        return modalities

    def caption_record(self, record: ManifestRecord) -> CaptionRecord:
        return self.caption_records_batch([record])[0]

    def caption_records_batch(self, records: list[ManifestRecord]) -> list[CaptionRecord]:
        if not records:
            return []
        self.ensure_loaded()
        return self._caption_records_batch_qwen(records)
