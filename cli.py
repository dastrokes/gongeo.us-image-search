import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import importlib

cli = importlib.import_module("cli")
main = cli.main

if __name__ == "__main__":
    raise SystemExit(main())
