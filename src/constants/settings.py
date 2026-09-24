from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent

DEFAULT_EXTRACTION_PROVIDER = "google"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_VERCEL_MODEL = "google/gemini-3.5-flash-lite"
