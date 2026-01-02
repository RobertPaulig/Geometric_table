# Geometric_table (geom-spec v4.0 Visualization Suite)

[![pytest](https://github.com/RobertPaulig/Geometric_table/actions/workflows/pytest.yml/badge.svg)](https://github.com/RobertPaulig/Geometric_table/actions/workflows/pytest.yml)

Старт: `CONTEXT.md` (единая точка входа).

CI обязателен: merge/PR считаются DONE только при зелёном GitHub Actions `pytest`.

This repository contains Python code and analysis scripts for the geometric–spectral model (geom-spec v4.0). The core model lives in `core/` (`geom_atoms`, `grower`, `complexity`, nuclear modules), while scans, tests and visualizations are in `analysis/`.

The original documentation lives in `docs/README.md`. For full usage details (requirements, scripts, workflow, troubleshooting) see:

- `docs/README.md` – geom-spec v4.0 visualization suite overview
- `REPORT.md` / `REPORT_baseline.md` / `REPORT_wsZ1.md` – project reports and baselines

## Quick start

- Install runtime dependencies: `pip install -r requirements.txt`
- (Optional) install dev tools: `pip install -r requirements-dev.txt`
- Run tests:

```bash
pytest -q
```

## Key directories

- `core/` – model (geom_atoms, grower, complexity, thermo, nuclear)
- `analysis/` – scans, R&D experiments, CLI entrypoints
- `data/` – geometric and nuclear indices, aggregated tables
- `results/` – CSV/TXT/PNG artefacts from scans
- `docs/` – documentation, backlog, decision log

For detailed script-level documentation and examples, open `docs/README.md`.

