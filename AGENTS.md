# AGENTS

## Repo rules
- Source lives under `src/`; `src/` is prepended to `sys.path` by the root wrappers so all imports use bare names (`from constants.settings import ...`, not `image_search.*`).
- Root `cli.py` and `manifest.py` are compatibility entry-points only — keep them thin.
- Treat `index/` and `manifest/` as generated outputs. Do not commit their contents unless the task is explicitly about sample artifacts.
- Keep dev runs small by default with `--limit 10` unless the task needs a wider run.
- Do not add or run tests for very small non-behavioral changes like docs, comments, path cleanup, or dead-flag removal.
