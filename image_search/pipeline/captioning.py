from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from image_search.constants.colors import (
    COLOR_DETAIL_NOUNS,
    COLOR_ONLY_PATTERN,
    COLOR_PATTERN,
    normalize_color_label,
)
from image_search.constants.settings import (
    DEFAULT_CAPTION_MODEL_ID,
    DEFAULT_FALLBACK_CAPTION_MODEL_ID,
)
from image_search.constants.text import (
    BOTTOM_GARMENT_PATTERNS,
    CAPTION_NOISE_TERMS,
    DRESS_GARMENT_PATTERNS,
    FLORENCE_DETAIL_PROMPT,
    LOW_SIGNAL_VISUAL_PATTERN,
    OUTERWEAR_GARMENT_PATTERNS,
    STOP_VISUAL_PATTERN,
    TOP_GARMENT_PATTERNS,
)
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
    fallback_model_id: str = DEFAULT_FALLBACK_CAPTION_MODEL_ID
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 8


class FlorenceCaptioner:
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

    def _load_model(self, model_id: str) -> None:
        self._resolve_torch()
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            device_map="cuda" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        self.model_id = model_id

    def ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        try:
            self._load_model(self.config.model_id)
        except Exception:
            if self.config.fallback_model_id == self.config.model_id:
                raise
            self._load_model(self.config.fallback_model_id)

    def caption_image(self, image_path: str | Path) -> str:
        self.ensure_loaded()
        import torch

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=FLORENCE_DETAIL_PROMPT,
            images=image,
            return_tensors="pt",
        )
        normalized_inputs = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                normalized_inputs[key] = value
                continue
            if torch.is_floating_point(value):
                normalized_inputs[key] = value.to(self.device, dtype=self.torch_dtype)
            else:
                normalized_inputs[key] = value.to(self.device)
        inputs = normalized_inputs

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=3,
                do_sample=False,
            )

        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        try:
            parsed = self.processor.post_process_generation(
                decoded,
                task=FLORENCE_DETAIL_PROMPT,
                image_size=image.size,
            )
            if isinstance(parsed, dict):
                parsed_value = parsed.get(FLORENCE_DETAIL_PROMPT)
                if isinstance(parsed_value, str):
                    return parsed_value
                if parsed_value is not None:
                    return str(parsed_value)
        except Exception:
            pass
        return decoded

    def caption_images_batch(self, image_paths: list[str | Path]) -> list[str]:
        self.ensure_loaded()
        import torch

        images = [Image.open(path).convert("RGB") for path in image_paths]
        prompts = [FLORENCE_DETAIL_PROMPT] * len(images)
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        normalized_inputs = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                normalized_inputs[key] = value
                continue
            if torch.is_floating_point(value):
                normalized_inputs[key] = value.to(self.device, dtype=self.torch_dtype)
            else:
                normalized_inputs[key] = value.to(self.device)
        inputs = normalized_inputs

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=3,
                do_sample=False,
            )

        decoded_list = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )

        results: list[str] = []
        for decoded, image in zip(decoded_list, images):
            try:
                parsed = self.processor.post_process_generation(
                    decoded,
                    task=FLORENCE_DETAIL_PROMPT,
                    image_size=image.size,
                )
                if isinstance(parsed, dict):
                    parsed_value = parsed.get(FLORENCE_DETAIL_PROMPT)
                    if isinstance(parsed_value, str):
                        results.append(parsed_value)
                        continue
                    if parsed_value is not None:
                        results.append(str(parsed_value))
                        continue
            except Exception:
                pass
            results.append(decoded)
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
            for candidate in FlorenceCaptioner._candidate_parts(part):
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
                if FlorenceCaptioner._is_default_showcase_descriptor(
                    candidate,
                    item_type,
                    modality,
                ):
                    continue
                if not is_visual_term_relevant(candidate, item_type):
                    continue
                add_descriptor(grounded, candidate)

        style_source = ", ".join(grounded) if grounded else caption
        for descriptor in FlorenceCaptioner._extract_style_descriptors(
            style_source,
            item_type,
        ):
            add_descriptor(fallback, descriptor)

        return ", ".join((grounded + fallback)[:10])

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
        icon_caption = ""
        overview_caption = ""
        merged_parts: list[str] = []
        seen: set[str] = set()
        failed: list[str] = []

        for modality, path in self._preferred_modalities(record):
            try:
                normalized = self._normalize_caption(
                    self.caption_image(path),
                    record.type,
                    modality,
                )
            except Exception as exc:
                failed.append(f"{modality}:{exc}")
                continue

            if self._is_low_signal_caption(normalized, record.type):
                continue

            if modality == "icon":
                icon_caption = normalized
            else:
                overview_caption = normalized

            for part in self._caption_parts(normalized):
                if part not in seen:
                    seen.add(part)
                    merged_parts.append(part)

        return CaptionRecord(
            item_id=record.item_id,
            icon_caption=icon_caption,
            overview_caption=overview_caption,
            visual=", ".join(merged_parts),
            failed_modalities=failed,
        )

    def caption_records_batch(self, records: list[ManifestRecord]) -> list[CaptionRecord]:
        self.ensure_loaded()

        tasks: list[tuple[int, str, str]] = []
        for index, record in enumerate(records):
            for modality, path in self._preferred_modalities(record):
                tasks.append((index, modality, path))

        if not tasks:
            return [
                CaptionRecord(
                    item_id=record.item_id,
                    icon_caption="",
                    overview_caption="",
                    visual="",
                    failed_modalities=[],
                )
                for record in records
            ]

        paths = [task[2] for task in tasks]
        try:
            raw_captions = self.caption_images_batch(paths)
        except Exception:
            raw_captions = []
            for path in paths:
                try:
                    raw_captions.append(self.caption_image(path))
                except Exception as exc:
                    raw_captions.append(f"__error__:{exc}")

        icon_captions: dict[int, str] = {}
        overview_captions: dict[int, str] = {}
        merged: dict[int, list[str]] = {index: [] for index in range(len(records))}
        seen_parts: dict[int, set[str]] = {index: set() for index in range(len(records))}
        failed: dict[int, list[str]] = {index: [] for index in range(len(records))}

        for (index, modality, _path), raw in zip(tasks, raw_captions):
            if raw.startswith("__error__:"):
                failed[index].append(f"{modality}:{raw[len('__error__:'):]}")
                continue
            normalized = self._normalize_caption(raw, records[index].type, modality)
            if self._is_low_signal_caption(normalized, records[index].type):
                continue
            if modality == "icon":
                icon_captions[index] = normalized
            else:
                overview_captions[index] = normalized
            for part in self._caption_parts(normalized):
                if part not in seen_parts[index]:
                    seen_parts[index].add(part)
                    merged[index].append(part)

        return [
            CaptionRecord(
                item_id=records[index].item_id,
                icon_caption=icon_captions.get(index, ""),
                overview_caption=overview_captions.get(index, ""),
                visual=", ".join(merged[index]),
                failed_modalities=failed[index],
            )
            for index in range(len(records))
        ]
