from __future__ import annotations

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
    ACCESSORY_ATTRIBUTE_PROMPT_RULE,
    ACCESSORY_TYPE_SPECIFIC_PROMPT_RULES,
    APPAREL_LEAK_PATTERNS,
    APPAREL_ATTRIBUTE_PROMPT_RULE,
    BODY_PAINT_ATTRIBUTE_PROMPT_RULE,
    BOTTOM_GARMENT_PATTERNS,
    BOTTOM_ATTRIBUTE_PROMPT_RULE,
    BODY_LEAK_PATTERNS,
    CAPTION_NOISE_PATTERN,
    DEFAULT_PROMPT_FOCUS,
    DRESS_GARMENT_PATTERNS,
    DRESS_ATTRIBUTE_PROMPT_RULE,
    FACE_DETAIL_ATTRIBUTE_PROMPT_RULE,
    GENERIC_ATTRIBUTE_PROMPT_RULE,
    HAIR_ATTRIBUTE_PROMPT_RULE,
    HAIR_TYPE_SPECIFIC_PROMPT_RULES,
    HAIR_LEAK_PATTERNS,
    JEWELRY_LEAK_PATTERNS,
    JOINT_PLAIN_DETAIL_PROMPT_LINES,
    LOW_SIGNAL_VISUAL_PATTERN,
    OUTERWEAR_GARMENT_PATTERNS,
    PLAIN_DETAIL_PROMPT_LINES,
    PROMPT_FOCUS_BY_MODALITY,
    JOINT_DETAIL_PROMPT,
    STRUCTURED_DETAIL_PROMPT,
    SHOES_ATTRIBUTE_PROMPT_RULE,
    SKIN_TONE_ATTRIBUTE_PROMPT_RULE,
    SOCKS_ATTRIBUTE_PROMPT_RULE,
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
    quantization: str = "none"
    batch_size: int = 8
    inference_batch_size: int = 2


class VisionCaptioner:
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
    def _prompt_focus(modality: str) -> str:
        return PROMPT_FOCUS_BY_MODALITY.get(modality, DEFAULT_PROMPT_FOCUS)

    @classmethod
    def _item_prompt_rules(cls, item_type: str, modality: str) -> str:
        if not item_type:
            return ""
        lines = [
            f"Treat the main item as a `{item_type}`.",
            cls._type_specific_prompt_rules(item_type),
            cls._subcategory_prompt_rule(item_type),
            cls._attribute_prompt_rule(item_type, modality),
            cls._coverage_prompt_rule(item_type, modality),
        ]
        return "\n" + "\n".join(line for line in lines if line)

    @classmethod
    def _build_prompt(cls, item_type: str, modality: str) -> str:
        prompt = STRUCTURED_DETAIL_PROMPT.format(focus=cls._prompt_focus(modality))
        return prompt + cls._item_prompt_rules(item_type, modality)

    @classmethod
    def _build_plain_prompt(cls, item_type: str, modality: str) -> str:
        focus = cls._prompt_focus(modality)
        lines = [line.format(focus=focus) for line in PLAIN_DETAIL_PROMPT_LINES]
        return "\n".join(lines) + cls._item_prompt_rules(item_type, modality)

    @classmethod
    def _build_joint_prompt(
        cls,
        item_type: str,
        *,
        has_overview: bool,
        has_icon: bool,
    ) -> str:
        prompt = JOINT_DETAIL_PROMPT
        prompt += f"\nTreat the main item as a `{item_type}`."
        prompt += cls._type_specific_prompt_rules(item_type)
        prompt += cls._subcategory_prompt_rule(item_type)
        if has_overview:
            prompt += cls._attribute_prompt_rule(item_type, "overview")
            prompt += cls._coverage_prompt_rule(item_type, "overview")
        if has_icon:
            prompt += cls._attribute_prompt_rule(item_type, "icon")
            prompt += cls._coverage_prompt_rule(item_type, "icon")
        return prompt

    @classmethod
    def _build_joint_plain_prompt(
        cls,
        item_type: str,
        *,
        has_overview: bool,
        has_icon: bool,
    ) -> str:
        lines = list(JOINT_PLAIN_DETAIL_PROMPT_LINES)
        lines.append(f"Treat the main item as a `{item_type}`.")
        extra = cls._type_specific_prompt_rules(item_type).strip()
        if extra:
            lines.append(extra)
        subcategory = cls._subcategory_prompt_rule(item_type).strip()
        if subcategory:
            lines.append(subcategory)
        if has_overview:
            lines.append(cls._attribute_prompt_rule(item_type, "overview").strip())
            lines.append(cls._coverage_prompt_rule(item_type, "overview").strip())
        if has_icon:
            lines.append(cls._attribute_prompt_rule(item_type, "icon").strip())
            lines.append(cls._coverage_prompt_rule(item_type, "icon").strip())
        return "\n".join(line for line in lines if line)

    @staticmethod
    def _subcategory_prompt_rule(item_type: str) -> str:
        subcategory_labels = subcategory_labels_for_item_type(item_type)
        if not subcategory_labels:
            return ""

        examples = ", ".join(subcategory_labels[:6])
        return (
            "If visible, include the item's subcategory as one of the first phrases.\n"
            f"Subcategory examples for `{item_type}`: {examples}."
        )

    @staticmethod
    def _attribute_prompt_rule(item_type: str, modality: str) -> str:
        if item_type == "hair":
            return HAIR_ATTRIBUTE_PROMPT_RULE
        if item_type == "dresses":
            return DRESS_ATTRIBUTE_PROMPT_RULE
        if item_type in {"tops", "outerwear"}:
            return APPAREL_ATTRIBUTE_PROMPT_RULE
        if item_type == "bottoms":
            return BOTTOM_ATTRIBUTE_PROMPT_RULE
        if item_type == "socks":
            return SOCKS_ATTRIBUTE_PROMPT_RULE
        if item_type == "shoes":
            return SHOES_ATTRIBUTE_PROMPT_RULE
        if item_type in ACCESSORY_TYPES:
            return ACCESSORY_ATTRIBUTE_PROMPT_RULE
        if item_type in FACE_DETAIL_TYPES:
            return FACE_DETAIL_ATTRIBUTE_PROMPT_RULE
        if item_type == "bodyPaint":
            return BODY_PAINT_ATTRIBUTE_PROMPT_RULE
        if item_type == "skinTones":
            return SKIN_TONE_ATTRIBUTE_PROMPT_RULE
        return GENERIC_ATTRIBUTE_PROMPT_RULE

    @staticmethod
    def _coverage_prompt_rule(item_type: str, modality: str) -> str:
        if modality == "overview":
            if item_type in APPAREL_TYPES or item_type == "hair":
                return "The overview should prioritize whole-item structure first: subcategory, length or height, silhouette, placement, and large construction details."
            return "The overview should prioritize whole-item shape, placement, scale, and how details are distributed across the item."
        if modality == "icon":
            return "The icon should prioritize small details second: trim, closures, motifs, ornaments, textures, and accent materials."
        return ""

    @staticmethod
    def _parse_qwen_response(raw_text: str) -> str:
        return raw_text.strip()

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

        if self.device == "cuda":
            self.torch_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            self.torch_dtype = torch.float32
        return torch

    def _single_device_map(self) -> dict[str, int | str]:
        if self.device.startswith("cuda"):
            suffix = self.device.partition(":")[2]
            if suffix.isdigit():
                return {"": int(suffix)}
            return {"": 0}
        return {"": self.device}

    def _input_device(self, torch: Any) -> str:
        if self.model is not None:
            hf_device_map = getattr(self.model, "hf_device_map", None)
            if isinstance(hf_device_map, dict):
                root_device = hf_device_map.get("")
                if isinstance(root_device, int):
                    return f"cuda:{root_device}"
                if isinstance(root_device, str) and root_device:
                    return root_device

            try:
                parameter = next(self.model.parameters())
            except (AttributeError, StopIteration, TypeError):
                parameter = None
            if parameter is not None:
                return str(parameter.device)

        return str(torch.device(self.device))

    def _normalize_inputs(self, inputs: Any, torch: Any) -> dict[str, Any]:
        target_device = self._input_device(torch)
        normalized_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                normalized_inputs[key] = value
                continue
            if torch.is_floating_point(value):
                normalized_inputs[key] = value.to(target_device, dtype=self.torch_dtype)
            else:
                normalized_inputs[key] = value.to(target_device)
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
    ) -> list[str]:
        failed: list[str] = []
        if has_icon and not icon_caption:
            failed.append("icon")
        if has_overview and not overview_caption:
            failed.append("overview")
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
            if VisionCaptioner._is_cross_image_filler(value):
                continue
            if VisionCaptioner._is_negative_descriptor(value):
                continue
            if any(
                value != other
                and re.search(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", other)
                for other in values
            ):
                continue
            compacted.append(value)
        return compacted

    @staticmethod
    def _is_cross_image_filler(value: str) -> bool:
        return bool(
            re.search(
                r"\b(?:consistent|visible)\s+across?\s+both\s+images\b",
                value,
            )
        )

    @staticmethod
    def _is_negative_descriptor(value: str) -> bool:
        return bool(re.match(r"^(?:no|without)\b", value.lower().strip()))

    @staticmethod
    def _chunk_records(
        records: list[ManifestRecord], batch_size: int
    ) -> list[list[ManifestRecord]]:
        if batch_size <= 0:
            batch_size = 1
        return [
            records[index : index + batch_size]
            for index in range(0, len(records), batch_size)
        ]

    @staticmethod
    def _type_specific_prompt_rules(item_type: str) -> str:
        if item_type == "hair":
            return HAIR_TYPE_SPECIFIC_PROMPT_RULES
        if item_type in ACCESSORY_TYPES or item_type in FACE_DETAIL_TYPES:
            return ACCESSORY_TYPE_SPECIFIC_PROMPT_RULES
        return ""

    @staticmethod
    def _sanitized_generation_config(model: Any):
        import copy

        generation_config = copy.deepcopy(model.generation_config)
        generation_config.do_sample = False
        for key in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, key):
                setattr(generation_config, key, None)
        return generation_config

    def _decode_batch_qwen(
        self,
        image_paths: list[str | Path],
        item_types: list[str],
        modalities: list[str],
        *,
        prompt_builder: Any | None = None,
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
        return decoded_list

    def _caption_batch_qwen(
        self,
        image_paths: list[str | Path],
        item_types: list[str],
        modalities: list[str],
        *,
        prompt_builder: Any | None = None,
        parse_structured: bool = True,
    ) -> list[str]:
        decoded_list = self._decode_batch_qwen(
            image_paths,
            item_types,
            modalities,
            prompt_builder=prompt_builder,
        )
        if not parse_structured:
            return decoded_list
        return [self._parse_qwen_response(decoded) for decoded in decoded_list]

    def _decode_joint_qwen(
        self,
        image_specs: list[tuple[str, str | Path]],
        item_type: str,
        *,
        plain: bool = False,
    ) -> str:
        import torch

        ordered_specs = [
            spec
            for label in ("overview", "icon")
            for spec in image_specs
            if spec[0] == label
        ]
        if not ordered_specs:
            return ""

        prompt_builder = (
            self._build_joint_plain_prompt if plain else self._build_joint_prompt
        )
        prompt = prompt_builder(
            item_type,
            has_overview=any(label == "overview" for label, _ in ordered_specs),
            has_icon=any(label == "icon" for label, _ in ordered_specs),
        )
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image"} for _label, _path in ordered_specs),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        loaded_images = [
            self._load_rgb_image(Path(path)) for _label, path in ordered_specs
        ]
        inputs = self.processor(
            text=[text],
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
                max_new_tokens=160,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        decoded = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""

    @staticmethod
    def _parse_joint_response(raw_text: str) -> dict[str, str]:
        parsed = {"overview": "", "icon": ""}
        current_key = ""

        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(r"^(overview|icon)\s*:\s*(.*)$", line, re.I)
            if match:
                current_key = match.group(1).lower()
                parsed[current_key] = match.group(2).strip()
                continue
            if current_key:
                separator = ", " if parsed[current_key] else ""
                parsed[current_key] = f"{parsed[current_key]}{separator}{line}"

        if any(parsed.values()):
            return parsed

        fallback = raw_text.strip()
        if fallback:
            parsed["overview"] = fallback
        return parsed

    def _resolve_model_class(self):
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration

    def ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        self._resolve_torch()
        from transformers import AutoProcessor, BitsAndBytesConfig

        model_class = self._resolve_model_class()
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        if (
            hasattr(self.processor, "tokenizer")
            and self.processor.tokenizer is not None
        ):
            self.processor.tokenizer.padding_side = "left"
        model_kwargs: dict[str, Any] = {}
        use_quantization = self.config.quantization == "8bit"
        if use_quantization:
            if not self.device.startswith("cuda"):
                raise RuntimeError("8-bit quantization requires a CUDA device.")
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "8-bit quantization requires bitsandbytes in the active environment. "
                    "bitsandbytes is not available for this Python install; use Python 3.12 or 3.13."
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = self._single_device_map()
        else:
            model_kwargs["dtype"] = self.torch_dtype
        self.model = model_class.from_pretrained(self.config.model_id, **model_kwargs)
        if not use_quantization:
            self.model.to(self.device)
        self.model.eval()
        self.model_id = self.config.model_id

    def _caption_records_batch_qwen(
        self, records: list[ManifestRecord]
    ) -> list[CaptionRecord]:
        if not records:
            return []

        results: list[CaptionRecord] = []
        for record in records:
            image_specs = self._record_image_paths(record)
            ordered_specs = [
                spec
                for label in ("overview", "icon")
                for spec in image_specs
                if spec[0] == label
            ]
            has_icon = any(label == "icon" for label, _ in ordered_specs)
            has_overview = any(label == "overview" for label, _ in ordered_specs)
            item_type = record.type

            raw_joint = self._decode_joint_qwen(ordered_specs, item_type)
            parsed_joint = self._parse_joint_response(raw_joint)
            missing_expected_line = (
                (has_overview and not parsed_joint["overview"])
                or (has_icon and not parsed_joint["icon"])
            )
            if missing_expected_line:
                retry_joint = self._parse_joint_response(
                    self._decode_joint_qwen(ordered_specs, item_type, plain=True)
                )
                for key, value in retry_joint.items():
                    if value:
                        parsed_joint[key] = value

            icon_caption = self._normalize_caption(
                parsed_joint["icon"],
                item_type,
                "icon",
            )
            overview_caption = self._normalize_caption(
                parsed_joint["overview"],
                item_type,
                "overview",
            )

            if self._is_low_signal_caption(icon_caption, item_type):
                icon_caption = ""
            if self._is_low_signal_caption(overview_caption, item_type):
                overview_caption = ""

            merged_caption = self._normalize_caption(
                self._merge_unique_parts(overview_caption, icon_caption),
                item_type,
                "overview",
            )
            if has_icon and not icon_caption:
                icon_caption = merged_caption
            if has_overview and not overview_caption:
                overview_caption = merged_caption

            failed_modalities = self._failed_modalities_for_record(
                has_icon=has_icon,
                has_overview=has_overview,
                icon_caption=icon_caption,
                overview_caption=overview_caption,
            )

            results.append(
                CaptionRecord(
                    item_id=record.item_id,
                    icon_caption=icon_caption,
                    overview_caption=overview_caption,
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
                if CAPTION_NOISE_PATTERN.search(candidate):
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
                if any(
                    token in candidate for token in {"wearing", "face", "head tilted"}
                ):
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
            re.search(pattern, normalized) for pattern in APPAREL_LEAK_PATTERNS
        )
        matches_jewelry = any(
            re.search(pattern, normalized) for pattern in JEWELRY_LEAK_PATTERNS
        )
        matches_body = any(
            re.search(pattern, normalized) for pattern in BODY_LEAK_PATTERNS
        )
        matches_hair = any(
            re.search(pattern, normalized) for pattern in HAIR_LEAK_PATTERNS
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
            chunk.strip() for chunk in normalized_caption.split(",") if chunk.strip()
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
        if (
            profile.prefer_icon
            or record.type in ACCESSORY_TYPES
            or record.type in FACE_DETAIL_TYPES
        ):
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

    def caption_records_batch(
        self, records: list[ManifestRecord]
    ) -> list[CaptionRecord]:
        if not records:
            return []
        self.ensure_loaded()
        return self._caption_records_batch_qwen(records)
