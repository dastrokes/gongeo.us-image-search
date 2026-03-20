from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent

DEFAULT_CAPTION_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_EMBEDDING_MODEL = "BGE_M3"
DEFAULT_SPARSE_EMBEDDING_MODEL = "BGE_M3"
DEFAULT_INDEX_TYPE = "HYBRID"
DEFAULT_INDEX_NAME = "infinity-nikki-items"
DEFAULT_INDEX_REGION = "us-east-1"
DEFAULT_INDEX_DIMENSION_COUNT = 1024
DEFAULT_INDEX_METRIC = "COSINE"
DEFAULT_MANAGEMENT_URL = "https://api.upstash.com/v2/vector"
