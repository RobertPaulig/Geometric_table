# Geometric_table - CONTEXT (single entry point)

This file is intended to be the only starting point you hand to an AI assistant (or a new contributor).

## Loading Order (strict)

1. `README.md` - repo overview and quickstart.
2. `docs/README.md` - full documentation and navigation.
3. Vol I (Constitution/spec): `docs/name3.pdf` (or `docs/name3.tex`, `docs/name3.md`).
4. Vol II (development line): `docs/name4.pdf` / `docs/name4.tex`.
5. `docs/04_backlog.md` and `docs/05_decision_log.md` - experiments + decisions.
6. `docs/90_lineage.md` - lineage of decisions/experiments/tests.
7. `docs/99_index.md` - index of literature/refs.
8. `VERSION` - current baseline version tag.
9. Run `pytest -q` - changes are not done unless tests are green.

## Repo Map

- `core/` - model core (`geom_atoms`, `grower`, `complexity`, nuclear modules).
- `analysis/` - scans/tests/visualizations and CLI entrypoints.
- `data/` - indices and aggregated tables (do not change formats casually).
- `results/` - produced artifacts (CSV/TXT/PNG reports).
- `docs/` - documentation, backlog, decision log, spec sources (Vol I/II).
- `run_pipeline.py` - main pipeline entrypoint (when relevant).

## Definition of Done

A change is DONE only if:

1. `pytest -q` is green.
2. There is a minimal reproduction command (script/CLI).
3. Outputs land in `results/` (or the absence is explained).
4. Docs are updated:
   - workflow/engineering changes -> `docs/README.md`
   - assumptions/metrics/gates -> `docs/05_decision_log.md`
   - new tasks/experiments -> `docs/04_backlog.md`
   - proven evolution -> `docs/90_lineage.md`

## Protocol: add an idea without losing it

1. Put the hypothesis into `docs/04_backlog.md` (short, testable).
2. Implement the minimal experiment under `analysis/` producing an artifact in `results/`.
3. Add/extend tests under `tests/`.
4. Record decisions/gates in `docs/05_decision_log.md`.
5. Record the completed story in `docs/90_lineage.md`.
