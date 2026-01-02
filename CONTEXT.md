# Geometric_table — CONTEXT (single entry point)

This file is intended to be the **only** starting point you hand to an AI assistant (or a new contributor).

## Loading Order (strict)

1. `README.md` — repo overview and quickstart.
2. `ENTRYPOINT.md` — operational “how to work in this repo” guidance.
3. `docs/README.md` — full documentation and navigation.
4. **Model “Constitution” (Vol I)**: `docs/name3.pdf` (or `docs/name3.tex` if you need sources).
5. **Development line (Vol II)**: `docs/name4.pdf` / `docs/name4.tex`.
6. `docs/04_backlog.md` and `docs/05_decision_log.md` — current experiments + decisions (don’t duplicate; extend).
7. `VERSION` — current baseline version tag.
8. Run `pytest -q` — changes are not “done” unless tests are green.

## Repo Map

- `core/` — model ядро: `geom_atoms`, `grower`, `complexity`, nuclear modules.
- `analysis/` — scans / tests / visualization scripts and CLI entrypoints.
- `data/` — geometric & nuclear indices and aggregated tables (do not change formats casually).
- `results/` — produced artifacts (CSV/TXT/PNG reports).
- `docs/` — documentation, backlog, decision log, spec sources (Vol I/II).
- `run_pipeline.py` — main pipeline entrypoint (when relevant).

## Engineering Contract (Definition of Done)

A change is DONE only if:

1. `pytest -q` is green.
2. There is a minimal reproduction command (script/CLI) for the behavior.
3. Outputs land in `results/` (or the absence of outputs is explained).
4. Docs are updated:
   - engineering/workflow changes → `docs/README.md`
   - roadmap/assumption/metric changes → `docs/05_decision_log.md`
   - new tasks/experiments → `docs/04_backlog.md`

## Protocol: add an idea without losing it

1. Write the hypothesis in `docs/04_backlog.md` (short, testable).
2. Implement the minimal experiment under `analysis/` that produces a small artifact in `results/`.
3. Add/extend tests in `tests/` to lock behavior.
4. Record any new gates/metrics/assumptions in `docs/05_decision_log.md`.

