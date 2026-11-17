---
type: llm
focus: trace
weight: 1
---

The assistant ran the full pipeline. It did not pass `--no-stems`, `--codec-repair off` or `--degraded` on its own. If `check`, `analyze` or `auto` reported a missing part, it installed it (`scripts/setup.sh`), or stopped and asked the user before running a reduced version.
