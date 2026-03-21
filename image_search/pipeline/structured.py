from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from image_search.constants.settings import DEFAULT_EXTRACTION_MODEL_ID
from image_search.constants.structured import (
    StructuredFieldDefinition,
    StructuredShapeDefinition,
    shape_definition_for_item_type,
)
from image_search.models.schemas import (
    ManifestRecord,
    StructuredDebugRecord,
    StructuredItemRecord,
)


@dataclass(slots=True)
class StructuredExtractorConfig:
    model_id: str = DEFAULT_EXTRACTION_MODEL_ID
    device: str = "auto"
    quantization: str = "none"
    inference_batch_size: int = 2


class VisionStructuredExtractor:
    _SUBTYPE_GUIDANCE: dict[str, str] = {
        "outerwear": (
            "Choose the closest visible outerwear class, such as jacket, coat, cape, "
            "cardigan, shawl, bolero, or shrug."
        ),
        "tops": (
            "Choose the closest visible top class, such as blouse, shirt, t_shirt, "
            "sweater, vest, camisole, corset, or hoodie."
        ),
        "bottoms": (
            "Choose the closest visible bottom class, such as skirt, pleated_skirt, "
            "pants, shorts, leggings, jeans, or overalls."
        ),
        "dresses": (
            "Choose the closest visible dress class, such as dress, gown, slip_dress, "
            "pinafore, cheongsam, or sundress."
        ),
        "shoes": (
            "Choose the closest visible shoe class, such as boots, heels, sandals, "
            "flats, loafers, sneakers, pumps, or mary_janes."
        ),
        "socks": (
            "Choose the closest visible legwear class, such as socks, stockings, "
            "tights, thigh_highs, ankle_socks, or leg_warmers."
        ),
        "hairAccessories": (
            "Choose the closest visible accessory class, such as bow, ribbon, flower, "
            "clip, headband, veil, or fascinator."
        ),
        "headwear": (
            "Choose the closest visible headwear class, such as hat, bonnet, beret, "
            "hood, crown, tiara, or headpiece."
        ),
        "earrings": (
            "Choose the closest visible earring class, such as studs, hoops, drops, "
            "dangling_earrings, or cuffs."
        ),
        "neckwear": (
            "Choose the closest visible neckwear class, such as necklace, scarf, tie, "
            "cravat, or pendant_necklace."
        ),
        "bracelets": (
            "Choose the closest visible bracelet class, such as bracelet, bangle, cuff, "
            "beaded_bracelet, or charm_bracelet."
        ),
        "chokers": (
            "Choose the closest visible choker class, such as choker, ribbon_choker, "
            "lace_choker, collar_choker, or pendant_choker."
        ),
        "gloves": (
            "Choose the closest visible glove class, such as gloves, mittens, "
            "fingerless_gloves, opera_gloves, or arm_warmers."
        ),
        "handhelds": (
            "Choose the closest visible handheld class, such as bag, basket, parasol, "
            "umbrella, fan, lantern, or book."
        ),
        "chestAccessories": (
            "Choose the closest visible chest accessory class, such as brooch, corsage, "
            "badge, sash, or chest_pin."
        ),
        "pendants": (
            "Choose the closest visible pendant class, such as pendant, locket, charm, "
            "medallion, or tassel."
        ),
        "backpieces": (
            "Choose the closest visible backpiece class, such as wings, capelet, backpack, "
            "back_bow, or back_ornament."
        ),
        "rings": (
            "Choose the closest visible ring class, such as ring, signet_ring, gemstone_ring, "
            "band, or stacked_rings."
        ),
        "armDecorations": (
            "Choose the closest visible arm decoration class, such as armlet, arm_band, "
            "sleeve_garter, or upper_arm_cuff."
        ),
        "abilityHandhelds": (
            "Choose the closest visible handheld class, such as wand, staff, lantern, fan, "
            "parasol, or magical_tool."
        ),
        "baseMakeup": (
            "Choose the closest visible cosmetic class, such as foundation, blush, contour, "
            "highlight, or face_base."
        ),
        "eyebrows": (
            "Choose the closest visible eyebrow class, such as straight_brows, arched_brows, "
            "soft_brows, or bold_brows."
        ),
        "eyelashes": (
            "Choose the closest visible eyelash class, such as natural_lashes, dramatic_lashes, "
            "cat_eye_lashes, or lower_lashes."
        ),
        "contactLenses": (
            "Choose the closest visible lens class, such as natural_lenses, circle_lenses, "
            "gradient_lenses, or fantasy_lenses."
        ),
        "lips": (
            "Choose the closest visible lip class, such as lipstick, gloss, tint, gradient_lips, "
            "or matte_lips."
        ),
        "skinTones": (
            "Choose the closest visible skin-tone effect class, such as natural_skin, rosy_skin, "
            "tan_skin, or cool_tone_skin."
        ),
        "faceDecorations": (
            "Choose the closest visible face decoration class, such as face_sticker, cheek_mark, "
            "freckles, beauty_mark, or gem_decor."
        ),
        "fullMakeup": (
            "Choose the closest visible makeup set class, such as natural_makeup, glam_makeup, "
            "fantasy_makeup, or themed_makeup."
        ),
    }

    def __init__(self, config: StructuredExtractorConfig) -> None:
        self.config = config
        self.model = None
        self.processor = None
        self.model_id = config.model_id
        self.device = None
        self.torch_dtype = None

    def _resolve_model_class(self):
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration

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
    def _sanitized_generation_config(model: Any):
        import copy

        generation_config = copy.deepcopy(model.generation_config)
        generation_config.do_sample = False
        for key in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, key):
                setattr(generation_config, key, None)
        return generation_config

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
    def _record_image_paths(record: ManifestRecord) -> list[tuple[str, str]]:
        paths: list[tuple[str, str]] = []
        if record.overview_path:
            paths.append(("overview", record.overview_path))
        if record.icon_path:
            paths.append(("icon", record.icon_path))
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
        return alias_map.get(value, value)

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

    @staticmethod
    def _output_template(shape_definition: StructuredShapeDefinition) -> dict[str, object]:
        template: dict[str, object] = {}
        for field_definition in shape_definition.fields:
            template[field_definition.name] = [] if field_definition.kind == "array" else None
        return template

    @classmethod
    def shape_schema(cls, item_type: str) -> dict[str, dict[str, object]]:
        shape_definition = shape_definition_for_item_type(item_type)
        return {
            field_definition.name: {
                "type": field_definition.kind,
                "description": field_definition.description,
            }
            for field_definition in shape_definition.fields
        }

    @classmethod
    def build_prompt(cls, item_type: str) -> str:
        shape_definition = shape_definition_for_item_type(item_type)
        subtype_guidance = cls._SUBTYPE_GUIDANCE.get(item_type)
        subtype_guidance_line = (
            f"Subtype guidance: {subtype_guidance}\n" if subtype_guidance else ""
        )
        field_lines = [
            f'- "{field_definition.name}": '
            f'{"array" if field_definition.kind == "array" else "string_or_null"}'
            f" ({field_definition.description})"
            for field_definition in shape_definition.fields
        ]
        output_template = cls._output_template(shape_definition)
        return (
            "You are a deterministic structured extraction engine for Infinity Nikki items.\n"
            "Return exactly one JSON object and nothing else.\n"
            "Do not include markdown.\n"
            "Do not include comments.\n"
            "Do not include explanations.\n"
            "Do not include keys not listed in the schema.\n"
            "Describe only the target item shown in the provided image set.\n"
            "Use only directly visible details.\n"
            "Do not infer hidden details.\n"
            "Use short lowercase underscore tokens.\n"
            "If a single-value field is unclear, use null.\n"
            "If a multi-value field is unclear, use [].\n"
            "For subtype, make a best-effort visible category decision instead of using null.\n"
            "Use null for subtype only when the item class is genuinely not visually distinguishable.\n"
            "If the exact fashion term is uncertain, choose the closest generic visible class.\n"
            f"Item type: {item_type}\n"
            f"Shape: {shape_definition.name}\n"
            "Image order: overview first, icon second when both are present.\n"
            f"{subtype_guidance_line}"
            "Schema:\n"
            f"{chr(10).join(field_lines)}\n"
            "Output template:\n"
            f"{json.dumps(output_template, indent=2, ensure_ascii=False)}"
        )

    def _decode_prompted_joint_qwen(
        self,
        image_specs: list[tuple[str, str | Path]],
        prompt: str,
        *,
        max_new_tokens: int = 320,
    ) -> str:
        import torch

        if not image_specs:
            return ""

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

        loaded_images = self._load_rgb_images([Path(path) for _label, path in image_specs])
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
                max_new_tokens=max_new_tokens,
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
    def _parse_json_object(raw_text: str) -> tuple[dict[str, object] | None, str | None]:
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
    def normalize_payload(
        cls,
        item_type: str,
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        shape_definition = shape_definition_for_item_type(item_type)
        normalized = cls._output_template(shape_definition)
        if not isinstance(payload, dict):
            return normalized

        field_definitions = {
            field_definition.name: field_definition
            for field_definition in shape_definition.fields
        }
        for field_name, raw_value in payload.items():
            field_definition = field_definitions.get(field_name)
            if field_definition is None:
                continue
            if field_definition.kind == "array":
                normalized[field_name] = cls._normalize_array(raw_value, field_definition)
            else:
                normalized[field_name] = cls._normalize_scalar(raw_value, field_definition)
        return normalized

    def extract_record(
        self,
        record: ManifestRecord,
    ) -> tuple[StructuredItemRecord, StructuredDebugRecord]:
        self.ensure_loaded()

        image_specs = self._record_image_paths(record)
        shape_definition = shape_definition_for_item_type(record.type)
        prompt = self.build_prompt(record.type)
        raw_response = self._decode_prompted_joint_qwen(image_specs, prompt)
        raw_payload, parse_error = self._parse_json_object(raw_response)
        normalized_payload = self.normalize_payload(record.type, raw_payload)

        structured_record = StructuredItemRecord(
            item_id=record.item_id,
            item_type=record.type,
            shape=shape_definition.name,
            source_version=record.source_version,
            data=normalized_payload,
            parse_error=parse_error,
        )
        debug_record = StructuredDebugRecord(
            item_id=record.item_id,
            item_type=record.type,
            shape=shape_definition.name,
            image_paths={
                "overview": record.overview_path or None,
                "icon": record.icon_path or None,
            },
            prompt=prompt,
            raw_response=raw_response,
            raw_payload=raw_payload,
            normalized_data=normalized_payload,
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
