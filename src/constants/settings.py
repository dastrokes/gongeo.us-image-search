from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent

DEFAULT_EXTRACTION_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_MODEL_QUANTIZATION = "none"  # "none", "4bit", "8bit"
DEFAULT_EXTRACTION_BACKEND = "gemini"  # "gemini" or "local"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"  # default model for Gemini backend
