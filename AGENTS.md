# AGENTS

## Repo rules
- Keep source code under `src/image_search/`; root wrappers are compatibility-only.
- Treat `index/` and `manifest/` as generated outputs. Do not keep their contents in repo cleanup changes unless the task is explicitly about sample artifacts.
- Keep dev runs small by default with `--limit 10` unless the task needs a wider run.
- Do not add or run tests for very small non-behavioral changes like docs, comments, path cleanup, or dead-flag removal.
