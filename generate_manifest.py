import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import importlib

manifest = importlib.import_module("image_search.manifest")
main = manifest.main

if __name__ == "__main__":
    raise SystemExit(main())
