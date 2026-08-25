from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from constants.prompts import (
    CANONICAL_ATTRIBUTE_TOKENS,
    CANONICAL_CATEGORY_TOKENS,
    CANONICAL_SUBCATEGORY_TOKENS,
    DETAIL_FIELD_OWNER_BY_TOKEN,
    FILTERED_CANONICAL_ATTRIBUTE_FIELDS,
    STRUCTURED_EXTRACTION_SYSTEM_PROMPT,
    SUBCATEGORY_HIERARCHY,
    build_extraction_user_message,
    get_subcategory_ancestors,
)
from constants.settings import (
    DEFAULT_EXTRACTION_MODEL_ID,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MODEL_QUANTIZATION,
)
from constants.structured import (
    StructuredFieldDefinition,
    StructuredSchemaDefinition,
    schema_definition_for_item_type,
)
from constants.tracker_export import normalize_supported_item_type
from models.schemas import (
    ManifestRecord,
    StructuredDebugRecord,
    StructuredItemRecord,
)
from pipeline.manifest import _find_image_path, resolve_manifest_paths

_ORNAMENT_SUFFIXES_TO_STRIP: tuple[str, ...] = (
    "detail",
    "motif",
    "charm",
    "emblem",
    "applique",
)

_CATEGORY_BOTTOM_LENGTH_TOKEN_ALIASES: dict[str, dict[str, str]] = {
    "dress": {
        "micro": "mini",
        "upper_thigh": "mini",
        "mid_thigh": "mini",
        "mid_calf": "midi",
        "ankle_length": "maxi",
    },
    "jumpsuit": {
        "micro": "upper_thigh",
        "mini": "upper_thigh",
        "midi": "ankle_length",
        "maxi": "ankle_length",
        "floor_length": "ankle_length",
    },
    "skirt": {
        "micro": "mini",
        "upper_thigh": "mini",
        "mid_thigh": "mini",
        "mid_calf": "midi",
        "ankle_length": "maxi",
    },
    "shorts": {
        "micro": "upper_thigh",
        "mini": "upper_thigh",
        "midi": "knee_length",
        "maxi": "knee_length",
        "ankle_length": "knee_length",
        "floor_length": "knee_length",
    },
    "skort": {
        "micro": "upper_thigh",
        "mini": "upper_thigh",
        "mid_calf": "midi",
        "midi": "knee_length",
        "maxi": "knee_length",
        "ankle_length": "knee_length",
        "floor_length": "knee_length",
    },
    "pants": {
        "micro": "knee_length",
        "mini": "knee_length",
        "upper_thigh": "knee_length",
        "mid_thigh": "knee_length",
        "midi": "ankle_length",
        "maxi": "ankle_length",
        "floor_length": "ankle_length",
    },
    "leggings": {
        "micro": "knee_length",
        "mini": "knee_length",
        "upper_thigh": "knee_length",
        "mid_thigh": "knee_length",
        "midi": "ankle_length",
        "maxi": "ankle_length",
        "floor_length": "ankle_length",
    },
    "overalls": {
        "micro": "upper_thigh",
        "mini": "upper_thigh",
        "midi": "ankle_length",
        "maxi": "ankle_length",
        "floor_length": "ankle_length",
    },
}


_CROSS_FIELD_TOKEN_OWNERSHIP: dict[
    str,
    dict[str, tuple[str | None, str | None]],
] = {
    "material": {},
    "ornament": {
        "metal_ornament": ("material", "metal"),
        "paper": ("material", "paper"),
        "water_splash": ("structure", "splash"),
    },
    "pattern": {
        "filigree": ("ornament", "filigree"),
        "musical_notation": ("ornament", "musical_note"),
        "snowflake": ("ornament", "snowflake"),
    },
    "structure": {
        "buckle": ("ornament", "buckle"),
        "cage": ("ornament", "cage"),
        "cloud_shaped": ("ornament", "cloud"),
        "cross": ("pattern", "cross"),
        "cuff": ("ornament", "cuff"),
        "engraved": ("ornament", "engraving"),
        "filigree": ("ornament", "filigree"),
        "mesh": ("material", "mesh"),
        "pinecone": ("ornament", "pinecone"),
        "spiked": ("ornament", "spike"),
    },
}


@dataclass(slots=True)
class StructuredExtractorConfig:
    model_id: str = DEFAULT_EXTRACTION_MODEL_ID
    device: str = "auto"
    quantization: str = DEFAULT_MODEL_QUANTIZATION
    inference_batch_size: int = 2
    max_new_tokens: int = 256
    tracker_root: str | None = None


class VisionStructuredExtractor:
    def __init__(self, config: StructuredExtractorConfig) -> None:
        self.config = config
        self.model = None
        self.processor = None
        self.model_id = config.model_id
        self.device = None
        self.dtype = None

    def _resolve_model_class(self):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText

    def _resolve_torch(self):
        import torch

        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        if self.device == "cuda":
            self.dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            self.dtype = torch.float32
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
                normalized_inputs[key] = value.to(target_device, dtype=self.dtype)
            else:
                normalized_inputs[key] = value.to(target_device)
        return normalized_inputs

    @staticmethod
    def _sanitized_generation_config(
        model: Any,
        *,
        max_new_tokens: int,
        pad_token_id: int | None = None,
    ):
        from transformers import GenerationConfig

        generation_config = GenerationConfig.from_model_config(model.config)
        generation_config.do_sample = False
        generation_config.max_new_tokens = max_new_tokens
        if pad_token_id is not None:
            generation_config.pad_token_id = pad_token_id
        return generation_config

    @staticmethod
    def _normalize_model_generation_config(model: Any) -> None:
        generation_config = getattr(model, "generation_config", None)
        if generation_config is None:
            return
        generation_config.do_sample = False
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = 50

    def ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        torch = self._resolve_torch()
        from transformers import AutoProcessor, BitsAndBytesConfig

        model_class = self._resolve_model_class()
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )
        if (
            hasattr(self.processor, "tokenizer")
            and self.processor.tokenizer is not None
        ):
            self.processor.tokenizer.padding_side = "left"
        model_kwargs: dict[str, Any] = {}
        quantization = self.config.quantization
        use_quantization = quantization in {"4bit", "8bit"}
        if use_quantization:
            if not self.device.startswith("cuda"):
                raise RuntimeError(
                    f"{quantization} quantization requires a CUDA device."
                )
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    f"{quantization} quantization requires bitsandbytes in the active environment. "
                    "bitsandbytes is not available for this Python install; use Python 3.12 or 3.13."
                ) from exc
            if quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            model_kwargs["device_map"] = self._single_device_map()
            model_kwargs["dtype"] = torch.float16
        else:
            model_kwargs["dtype"] = self.dtype
        self.model = model_class.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            **model_kwargs,
        )
        self._normalize_model_generation_config(self.model)
        if not use_quantization:
            self.model.to(self.device)
        self.model.eval()
        self.model_id = self.config.model_id

    @staticmethod
    def _load_rgb_image(image_path: str | Path) -> Any:
        with Image.open(image_path) as image:
            return image.convert("RGB")

    @classmethod
    def _load_rgb_images(cls, image_paths: list[str | Path]) -> list[Any]:
        if not image_paths:
            return []
        if len(image_paths) == 1:
            return [cls._load_rgb_image(image_paths[0])]
        max_workers = min(8, len(image_paths))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(cls._load_rgb_image, image_paths))

    @staticmethod
    def _image_paths_for_item(
        item_id: int,
        tracker_root: str | None = None,
    ) -> dict[str, str | None]:
        manifest_paths = resolve_manifest_paths(tracker_root=tracker_root)
        overview_path = _find_image_path(manifest_paths.item_image_root, item_id)
        icon_path = _find_image_path(manifest_paths.item_icon_root, item_id)
        return {
            "overview": str(overview_path) if overview_path is not None else None,
            "icon": str(icon_path) if icon_path is not None else None,
        }

    @classmethod
    def _record_image_paths(
        cls,
        record: ManifestRecord,
        tracker_root: str | None = None,
    ) -> list[tuple[str, str]]:
        paths: list[tuple[str, str]] = []
        image_paths = cls._image_paths_for_item(record.item_id, tracker_root)
        if image_paths["overview"]:
            paths.append(("overview", str(image_paths["overview"])))
        if image_paths["icon"]:
            paths.append(("icon", str(image_paths["icon"])))
        return paths

    @staticmethod
    def _normalize_token(value: object) -> str:
        normalized = str(value or "").strip().lower()
        normalized = normalized.replace("&", " and ")
        normalized = re.sub(r"[\s/\-]+", "_", normalized)
        normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized

    @classmethod
    def _collapse_alias(
        cls,
        value: str,
        field_definition: StructuredFieldDefinition,
    ) -> str:
        if not value:
            return value
        alias_map = {
            cls._normalize_token(source): cls._normalize_token(target)
            for source, target in field_definition.alias_pairs
        }
        normalized = alias_map.get(value, value)
        return cls._post_process_token(normalized, field_definition)

    @staticmethod
    def _post_process_token(
        value: str,
        field_definition: StructuredFieldDefinition,
    ) -> str:
        if field_definition.name != "ornament":
            return value
        for suffix in _ORNAMENT_SUFFIXES_TO_STRIP:
            suffix_token = f"_{suffix}"
            if not value.endswith(suffix_token):
                continue
            stripped = value[: -len(suffix_token)].strip("_")
            if stripped:
                return stripped
        return value

    @classmethod
    def _normalize_scalar(
        cls,
        value: object,
        field_definition: StructuredFieldDefinition,
    ) -> str | None:
        candidate = value
        if isinstance(candidate, list):
            candidate = next(
                (entry for entry in candidate if entry not in (None, "")),
                None,
            )
        if candidate in (None, ""):
            return None
        normalized = cls._collapse_alias(
            cls._normalize_token(candidate),
            field_definition,
        )
        return normalized or None

    @classmethod
    def _normalize_array(
        cls,
        value: object,
        field_definition: StructuredFieldDefinition,
    ) -> list[str]:
        if value in (None, ""):
            return []
        candidates = value if isinstance(value, list) else [value]

        normalized_values: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = cls._collapse_alias(
                cls._normalize_token(candidate),
                field_definition,
            )
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_values.append(normalized)
        return normalized_values

    @classmethod
    def _normalize_cross_field_concepts(
        cls,
        normalized: dict[str, object],
        field_definitions: dict[str, StructuredFieldDefinition],
    ) -> dict[str, object]:
        for source_field, ownership_map in _CROSS_FIELD_TOKEN_OWNERSHIP.items():
            source_definition = field_definitions.get(source_field)
            if source_definition is None or source_definition.kind != "array":
                continue

            source_values = normalized.get(source_field)
            if not isinstance(source_values, list) or not source_values:
                continue

            kept_values: list[str] = []
            for value in source_values:
                if source_field == "material" and value == "braided_cord":
                    kept_values.append("cord")
                    structure = normalized.get("structure")
                    if isinstance(structure, list) and "braided" not in structure:
                        structure.append("braided")
                    continue

                ownership = ownership_map.get(value)
                if ownership is None:
                    kept_values.append(value)
                    continue

                target_field, target_value = ownership
                if target_field is None:
                    continue

                target_definition = field_definitions.get(target_field)
                if target_definition is None or target_definition.kind != "array":
                    kept_values.append(value)
                    continue

                destination = normalized.get(target_field)
                if not isinstance(destination, list):
                    destination = []
                    normalized[target_field] = destination

                normalized_target = cls._collapse_alias(
                    target_value or value,
                    target_definition,
                )
                if normalized_target and normalized_target not in destination:
                    destination.append(normalized_target)

            normalized[source_field] = kept_values
        return normalized

    @staticmethod
    def _normalize_detail_field_ownership(
        normalized: dict[str, object],
        field_definitions: dict[str, StructuredFieldDefinition],
    ) -> dict[str, object]:
        moves: dict[str, list[str]] = {}
        for source_field in ("pattern", "material", "structure", "ornament"):
            source_definition = field_definitions.get(source_field)
            source_values = normalized.get(source_field)
            if (
                source_definition is None
                or source_definition.kind != "array"
                or not isinstance(source_values, list)
            ):
                continue

            kept_values: list[str] = []
            for value in source_values:
                target_field = DETAIL_FIELD_OWNER_BY_TOKEN.get(value)
                target_definition = field_definitions.get(target_field or "")
                if (
                    target_field is None
                    or target_field == source_field
                    or target_definition is None
                    or target_definition.kind != "array"
                ):
                    kept_values.append(value)
                    continue
                moves.setdefault(target_field, []).append(value)
            normalized[source_field] = kept_values

        for target_field, values in moves.items():
            destination = normalized.get(target_field)
            if not isinstance(destination, list):
                destination = []
                normalized[target_field] = destination
            for value in values:
                if value not in destination:
                    destination.append(value)
        return normalized

    @staticmethod
    def _output_template(
        schema_definition: StructuredSchemaDefinition,
    ) -> dict[str, object]:
        template: dict[str, object] = {}
        for field_definition in schema_definition.fields:
            template[field_definition.name] = (
                [] if field_definition.kind == "array" else None
            )
        return template

    @classmethod
    def build_prompt(cls, item_type: str) -> str:
        user_message = build_extraction_user_message(
            item_type,
            field_names=tuple(
                field_definition.name
                for field_definition in schema_definition_for_item_type(
                    item_type
                ).fields
            ),
        )
        return f"{STRUCTURED_EXTRACTION_SYSTEM_PROMPT}\n{user_message}"

    def _decode_prompted_joint_qwen(
        self,
        image_specs: list[tuple[str, str | Path]],
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        import torch

        if not image_specs:
            return ""
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image"} for _label, _path in image_specs),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        loaded_images = self._load_rgb_images(
            [Path(path) for _label, path in image_specs]
        )
        inputs = self.processor(
            text=[text],
            images=loaded_images,
            padding=True,
            return_tensors="pt",
        )
        inputs = self._normalize_inputs(inputs, torch)
        pad_token_id = None
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, "eos_token_id", None)
        generation_config = self._sanitized_generation_config(
            self.model,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids, strict=False)
        ]
        decoded = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""

    @staticmethod
    def _response_looks_truncated(
        raw_text: str,
        parse_error: str | None,
    ) -> bool:
        if not raw_text.strip():
            return False
        if not parse_error or not parse_error.startswith("json_decode_error:"):
            return False
        truncated_markers = (
            "Unterminated string",
            "Expecting value",
            "Expecting ',' delimiter",
            "Expecting property name enclosed in double quotes",
        )
        if any(marker in parse_error for marker in truncated_markers):
            return True
        stripped = raw_text.rstrip()
        return stripped.startswith("{") and not stripped.endswith("}")

    @staticmethod
    def _parse_json_object(
        raw_text: str,
    ) -> tuple[dict[str, object] | None, str | None]:
        stripped = raw_text.strip()
        if not stripped:
            return None, "empty_response"
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            return None, f"json_decode_error:{exc.msg}"
        if not isinstance(payload, dict):
            return None, "json_root_not_object"
        return payload, None

    @classmethod
    def _normalize_taxonomy_fields(
        cls,
        item_type: str,
        normalized: dict[str, object],
    ) -> dict[str, object]:
        category = str(normalized.get("category") or "").strip()
        subcategory = str(normalized.get("subcategory") or "").strip()
        canonical_categories = set(CANONICAL_CATEGORY_TOKENS.get(item_type, ()))

        hierarchy = SUBCATEGORY_HIERARCHY.get(item_type, {})

        if not category and subcategory in hierarchy:
            normalized["category"] = hierarchy[subcategory]
            category = str(normalized.get("category") or "").strip()

        promoted_subcategory: str | None = None
        if category and category not in canonical_categories:
            promoted_parent = hierarchy.get(category)
            if promoted_parent is not None:
                promoted_subcategory = category
                normalized["category"] = promoted_parent
                category = promoted_parent

        if promoted_subcategory and (not subcategory or subcategory == category):
            normalized["subcategory"] = promoted_subcategory
            subcategory = promoted_subcategory

        if subcategory in hierarchy:
            ancestors = get_subcategory_ancestors(item_type, subcategory)
            if not category or category not in ancestors:
                normalized["category"] = hierarchy[subcategory]
                category = str(normalized.get("category") or "").strip()

        if subcategory and subcategory == category:
            normalized["subcategory"] = None

        identity_values = {normalized.get("category"), normalized.get("subcategory")}
        for field_name in ("pattern", "material", "structure", "ornament"):
            values = normalized.get(field_name)
            if isinstance(values, list):
                normalized[field_name] = [
                    value for value in values if value not in identity_values
                ]
        return normalized

    @staticmethod
    def _normalize_scalar_ownership(
        normalized: dict[str, object],
    ) -> dict[str, object]:
        if normalized.get("neckline") == "strapless":
            if not normalized.get("shoulder_style"):
                normalized["shoulder_style"] = "strapless"
            normalized["neckline"] = None
        if (
            normalized.get("neckline") == "off_shoulder"
            and normalized.get("shoulder_style") == "off_shoulder"
        ):
            normalized["neckline"] = None
        return normalized

    @staticmethod
    def _normalize_shoe_fields(
        item_type: str,
        normalized: dict[str, object],
    ) -> dict[str, object]:
        if item_type != "shoes":
            return normalized

        if normalized.get("category") == "barefoot":
            return normalized

        if not normalized.get("heel_height"):
            normalized["heel_height"] = "flat"
        if not normalized.get("sole_height"):
            normalized["sole_height"] = "flat"
        return normalized

    @staticmethod
    def _normalize_bottom_length_for_category(
        normalized: dict[str, object],
    ) -> dict[str, object]:
        category = str(normalized.get("category") or "").strip()
        bottom_length = str(normalized.get("bottom_length") or "").strip()
        if not category or not bottom_length:
            return normalized

        category_aliases = _CATEGORY_BOTTOM_LENGTH_TOKEN_ALIASES.get(category)
        if not category_aliases:
            return normalized

        normalized["bottom_length"] = category_aliases.get(
            bottom_length,
            bottom_length,
        )
        return normalized

    @classmethod
    def normalize_payload(
        cls,
        item_type: str,
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        item_type = normalize_supported_item_type(item_type)
        schema_definition = schema_definition_for_item_type(item_type)
        normalized = cls._output_template(schema_definition)
        if not isinstance(payload, dict):
            return normalized

        field_definitions = {
            field_definition.name: field_definition
            for field_definition in schema_definition.fields
        }
        for field_name, raw_value in payload.items():
            field_definition = field_definitions.get(field_name)
            if field_definition is None:
                continue
            if field_definition.kind == "array":
                normalized[field_name] = cls._normalize_array(
                    raw_value, field_definition
                )
            else:
                normalized[field_name] = cls._normalize_scalar(
                    raw_value, field_definition
                )
        normalized = cls._normalize_detail_field_ownership(
            normalized,
            field_definitions,
        )
        normalized = cls._normalize_cross_field_concepts(
            normalized,
            field_definitions,
        )
        normalized = cls._normalize_scalar_ownership(normalized)
        normalized = cls._normalize_taxonomy_fields(item_type, normalized)
        normalized = cls._normalize_shoe_fields(item_type, normalized)
        return cls._normalize_bottom_length_for_category(normalized)

    @classmethod
    def _normalized_raw_values(
        cls,
        value: object,
        field_definition: StructuredFieldDefinition,
    ) -> list[str]:
        if value in (None, ""):
            return []
        candidates = value if isinstance(value, list) else [value]
        normalized_values: list[str] = []
        for candidate in candidates:
            normalized = cls._collapse_alias(
                cls._normalize_token(candidate),
                field_definition,
            )
            if normalized:
                normalized_values.append(normalized)
        return normalized_values

    @classmethod
    def _cross_field_ownership_rows(
        cls,
        raw_payload: dict[str, object],
        field_definitions: dict[str, StructuredFieldDefinition],
    ) -> list[dict[str, str | None]]:
        rows: list[dict[str, str | None]] = []
        seen: set[tuple[str, str, str | None, str | None]] = set()

        for field_name, raw_value in raw_payload.items():
            field_definition = field_definitions.get(field_name)
            if field_definition is None:
                continue
            ownership_map = _CROSS_FIELD_TOKEN_OWNERSHIP.get(field_name)
            if not ownership_map:
                continue

            for token in cls._normalized_raw_values(raw_value, field_definition):
                ownership = ownership_map.get(token)
                if ownership is None:
                    continue
                target_field, target_token = ownership
                fingerprint = (field_name, token, target_field, target_token)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                rows.append(
                    {
                        "field": field_name,
                        "token": token,
                        "target_field": target_field,
                        "target_token": target_token,
                    }
                )
        return rows

    @staticmethod
    def _category_mismatch(
        item_type: str,
        normalized_payload: dict[str, object],
    ) -> dict[str, str] | None:
        category = str(normalized_payload.get("category") or "").strip()
        subcategory = str(normalized_payload.get("subcategory") or "").strip()
        if not category or not subcategory:
            return None
        hierarchy = SUBCATEGORY_HIERARCHY.get(item_type, {})
        parent = hierarchy.get(subcategory)
        if parent is None:
            return None
        ancestors = get_subcategory_ancestors(item_type, subcategory)
        if category in ancestors:
            return None
        return {
            "item_type": item_type,
            "category": category,
            "subcategory": subcategory,
            "expected_parent": parent,
        }

    @staticmethod
    def _subcategory_not_in_list(
        item_type: str,
        normalized_payload: dict[str, object],
    ) -> dict[str, object] | None:
        subcategory = str(normalized_payload.get("subcategory") or "").strip()
        if not subcategory:
            return None
        if subcategory in CANONICAL_SUBCATEGORY_TOKENS.get(item_type, ()):
            return None
        return {
            "item_type": item_type,
            "category": str(normalized_payload.get("category") or "").strip(),
            "subcategory": subcategory,
        }

    @classmethod
    def build_filter_report(
        cls,
        item_type: str,
        raw_payload: dict[str, object] | None,
        normalized_payload: dict[str, object],
    ) -> dict[str, object]:
        item_type = normalize_supported_item_type(item_type)
        if not isinstance(raw_payload, dict):
            raw_payload = {}

        schema_definition = schema_definition_for_item_type(item_type)
        field_definitions = {
            field_definition.name: field_definition
            for field_definition in schema_definition.fields
        }

        non_canonical: list[dict[str, str]] = []
        for field_name, raw_value in raw_payload.items():
            field_definition = field_definitions.get(field_name)
            if field_definition is None:
                continue
            normalized_values = cls._normalized_raw_values(raw_value, field_definition)
            if field_name == "category":
                canonical_tokens = set(CANONICAL_CATEGORY_TOKENS.get(item_type, ()))
            elif field_name in FILTERED_CANONICAL_ATTRIBUTE_FIELDS:
                canonical_tokens = set(CANONICAL_ATTRIBUTE_TOKENS.get(field_name, ()))
            else:
                canonical_tokens = set()
            if not canonical_tokens:
                continue
            for token in normalized_values:
                if token not in canonical_tokens:
                    non_canonical.append(
                        {
                            "field": field_name,
                            "token": token,
                        }
                    )

        unknown_fields = sorted(
            field_name
            for field_name in raw_payload
            if field_name not in field_definitions
        )
        cross_field_ownership = cls._cross_field_ownership_rows(
            raw_payload,
            field_definitions,
        )
        category_mismatch = cls._category_mismatch(item_type, normalized_payload)
        subcategory_not_in_list = cls._subcategory_not_in_list(
            item_type, normalized_payload
        )
        return {
            "cross_field_ownership": cross_field_ownership,
            "cross_field_ownership_count": len(cross_field_ownership),
            "non_canonical": non_canonical,
            "non_canonical_count": len(non_canonical),
            "unknown_fields": unknown_fields,
            "unknown_field_count": len(unknown_fields),
            "parent_child_mismatch": category_mismatch,
            "parent_child_mismatch_count": 1 if category_mismatch else 0,
            "subcategory_not_in_list": subcategory_not_in_list,
            "subcategory_not_in_list_count": 1 if subcategory_not_in_list else 0,
        }

    def extract_record(
        self,
        record: ManifestRecord,
    ) -> tuple[StructuredItemRecord, StructuredDebugRecord]:
        self.ensure_loaded()

        image_specs = self._record_image_paths(record, self.config.tracker_root)
        image_paths = self._image_paths_for_item(
            record.item_id,
            self.config.tracker_root,
        )
        prompt = self.build_prompt(record.item_type)
        raw_response = self._decode_prompted_joint_qwen(image_specs, prompt)
        raw_payload, parse_error = self._parse_json_object(raw_response)
        if self._response_looks_truncated(raw_response, parse_error):
            raw_response = self._decode_prompted_joint_qwen(
                image_specs,
                prompt,
                max_new_tokens=512,
            )
            raw_payload, parse_error = self._parse_json_object(raw_response)
        normalized_payload = self.normalize_payload(record.item_type, raw_payload)
        filter_report = self.build_filter_report(
            record.item_type,
            raw_payload,
            normalized_payload,
        )

        structured_record = StructuredItemRecord(
            item_id=record.item_id,
            item_type=record.item_type,
            data=normalized_payload,
            parse_error=parse_error,
        )
        debug_record = StructuredDebugRecord(
            item_id=record.item_id,
            item_type=record.item_type,
            image_paths=image_paths,
            prompt=prompt,
            raw_response=raw_response,
            raw_payload=raw_payload,
            normalized_data=normalized_payload,
            filter_report=filter_report,
            parse_error=parse_error,
        )
        return structured_record, debug_record

    def extract_records_batch(
        self,
        records: list[ManifestRecord],
    ) -> list[tuple[StructuredItemRecord, StructuredDebugRecord]]:
        if not records:
            return []
        results: list[tuple[StructuredItemRecord, StructuredDebugRecord]] = []
        batch_size = max(1, int(self.config.inference_batch_size))
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            results.extend(self.extract_record(record) for record in batch)
        return results


# ---------------------------------------------------------------------------
# Google AI Studio (Gemini) backend
# ---------------------------------------------------------------------------

_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _get_mime_type(image_path: str | Path) -> str:
    suffix = Path(image_path).suffix.lower()
    return _MIME_TYPES.get(suffix, "image/jpeg")


@dataclass(slots=True)
class GeminiExtractorConfig:
    model_id: str = DEFAULT_GEMINI_MODEL
    api_key: str | None = None
    rpm_limit: int = 15
    tracker_root: str | None = None


class GeminiStructuredExtractor:
    """Vision extractor backed by Google AI Studio (Gemini) instead of a local model.

    Requires ``google-genai`` (``pip install google-genai``) and an API key
    available either via the *api_key* config field or the ``GOOGLE_API_KEY``
    environment variable.
    """

    def __init__(self, config: GeminiExtractorConfig) -> None:
        self.config = config
        self.model_id = config.model_id
        self._client: Any = None
        self._min_interval: float = 60.0 / max(1, config.rpm_limit)
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import os

        from google import genai

        api_key = self.config.api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. "
                "Either pass --gemini-api-key or export GOOGLE_API_KEY."
            )
        self._client = genai.Client(api_key=api_key)
        return self._client

    def _call_api(
        self,
        image_specs: list[tuple[str, str | Path]],
        prompt: str,
        max_retries: int = 5,
    ) -> str:
        from google.genai import types
        from google.genai.errors import ClientError

        client = self._get_client()

        # Build content parts: one image part per image, then the text prompt.
        parts: list[Any] = []
        for _label, image_path in image_specs:
            image_bytes = Path(image_path).read_bytes()
            parts.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=_get_mime_type(image_path),
                )
            )
        parts.append(prompt)

        cfg: dict[str, Any] = {"response_mime_type": "application/json"}

        for attempt in range(max_retries + 1):
            # Enforce RPM limit
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request_time = time.monotonic()

            try:
                response = client.models.generate_content(
                    model=self.config.model_id,
                    contents=parts,
                    config=cfg,
                )
                return response.text or ""

            except ClientError as exc:
                if exc.code != 429 or attempt >= max_retries:
                    raise

                # Default backoff
                retry_delay = 10.0 * (2**attempt)

                try:
                    # Attempt to extract delay from API payload details
                    # Example payload: [... {'@type': '...RetryInfo', 'retryDelay': '20s'}]
                    if hasattr(exc, "response_json") and exc.response_json:
                        details = exc.response_json.get("error", {}).get("details", [])
                        for detail in details:
                            if "retryDelay" in detail:
                                delay_str = detail["retryDelay"].rstrip("s")
                                if delay_str.replace(".", "", 1).isdigit():
                                    retry_delay = float(delay_str) + 1.0  # +1s buffer
                                    break
                except Exception:
                    pass

                print(
                    f"  [Rate limited (429)] Retrying in {retry_delay:.2f}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(retry_delay)

        return ""

    # ------------------------------------------------------------------
    # Public interface — mirrors VisionStructuredExtractor
    # ------------------------------------------------------------------

    @property
    def model(self) -> None:
        """Kept for compatibility with callers that check ``extractor.model``."""
        return None

    def ensure_loaded(self) -> None:
        """No-op: Gemini client is initialised lazily on first call."""

    def extract_record(
        self,
        record: ManifestRecord,
    ) -> tuple[StructuredItemRecord, StructuredDebugRecord]:
        image_specs = VisionStructuredExtractor._record_image_paths(
            record,
            self.config.tracker_root,
        )
        image_paths = VisionStructuredExtractor._image_paths_for_item(
            record.item_id,
            self.config.tracker_root,
        )
        prompt = VisionStructuredExtractor.build_prompt(record.item_type)
        raw_response = self._call_api(image_specs, prompt)
        raw_payload, parse_error = VisionStructuredExtractor._parse_json_object(
            raw_response
        )
        normalized_payload = VisionStructuredExtractor.normalize_payload(
            record.item_type, raw_payload
        )
        filter_report = VisionStructuredExtractor.build_filter_report(
            record.item_type,
            raw_payload,
            normalized_payload,
        )

        structured_record = StructuredItemRecord(
            item_id=record.item_id,
            item_type=record.item_type,
            data=normalized_payload,
            parse_error=parse_error,
        )
        debug_record = StructuredDebugRecord(
            item_id=record.item_id,
            item_type=record.item_type,
            image_paths=image_paths,
            prompt=prompt,
            raw_response=raw_response,
            raw_payload=raw_payload,
            normalized_data=normalized_payload,
            filter_report=filter_report,
            parse_error=parse_error,
        )
        return structured_record, debug_record

    def extract_records_batch(
        self,
        records: list[ManifestRecord],
    ) -> list[tuple[StructuredItemRecord, StructuredDebugRecord]]:
        if not records:
            return []
        return [self.extract_record(record) for record in records]
