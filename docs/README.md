# geom-spec v4.0 Visualization Suite

> Start from `CONTEXT.md` (single entry point). Then continue with this `docs/README.md`.

## Overview

This directory contains Python scripts to visualize and validate the geometric-spectral model (geom-spec v4.0). The scripts perform numerical experiments to verify theoretical laws about atomic properties, molecular complexity, and virtual element stability.

## Requirements

- **Python**: 3.8 or higher
- **Dependencies**: `numpy`, `pandas`, `matplotlib`, `scipy`
- **Local modules**: `geom_atoms.py`, `grower.py`, `complexity.py`, `analyze_geom_table.py`

For development and tests:

- Install runtime deps: `pip install -r requirements.txt`
- Install dev tools (pytest, etc.): `pip install -r requirements-dev.txt`

Install dependencies:
```bash
pip install numpy pandas matplotlib scipy
```
or, for full dev environment:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

To run tests locally:
```bash
pytest -q
```

## HETERO-1A audit (стабильный JSON-отчёт)

Публичный entrypoint: `python -m analysis.chem.audit`.

Пример запуска:
```bash
python -m analysis.chem.audit --input tests/data/hetero_audit_min.json --seed 0 --out audit.json
```

Выход: JSON со стабильными ключами:
- `schema_version`
- `version`, `dataset_id`
- `n_pos`, `n_neg`
- `auc_tie_aware`
- `neg_controls`: `{null_q, perm_q, rand_q, neg_auc_max, null_q_method, method, reps_used, gate, margin, slack, verdict}`
- `warnings`
- `run`: `{seed, timestamp, cmd}`

Примечание: `null_q` сейчас считается по `(n_pos, n_neg)` без учёта весов. Если веса не все равны 1.0, в `warnings`
появляется `weights_used_in_auc_but_null_q_is_unweighted`. Это сигнал к будущему расширению (weighted null).

## HETERO-1A decoys (tree, degree-preserving)

Публичный entrypoint: `python -m analysis.chem.decoys`.

Пример запуска:
```bash
python -m analysis.chem.decoys --input tests/data/hetero_tree_min.json --out decoys.json
```

Дополнительные флаги:
- `--min_dist_to_original` (float, default 0.0)
- `--min_pair_dist` (float, default 0.0)
- `--max_attempts` (int, default: `k*200`)

Вход: дерево с `node_types`, `edges`, `k`, `seed`, `timestamp`, `max_valence`.

Выход (стабильная схема):
- `schema_version`, `mol_id`, `n`
- `k_requested`, `k_generated`
- `constraints`: `{preserve_degree, preserve_types}`
- `metrics`: `{dist_to_original, pairwise_dist, edge_overlap_to_original_mean}`
- `filter`: `{min_dist_to_original, min_pair_dist, attempts, rejected_too_close_to_original, rejected_too_close_to_existing, rejected_duplicate}`
- `decoys`: список `{edges, hash}`
- `warnings`
- `run`: `{seed, timestamp, cmd}`

Гарантии v1:
- декои — деревья с тем же degree-sequence, что и исходный граф
- типы узлов сохраняются (индексы вершин не меняются)
- не нарушаются `max_valence`

## HETERO-1A pipeline (decoys -> selection -> audit)

Публичный entrypoint: `python -m analysis.chem.pipeline`.

Пример запуска:
```bash
python -m analysis.chem.pipeline --tree_input tests/data/hetero_tree_min.json --k 50 --seed 0 --timestamp 2026-01-02T00:00:00+00:00 --select_k 20 --selection maxmin --out pipeline.json
```

Что делает:
- генерирует decoys (см. `analysis.chem.decoys`)
- делает coverage selection (`firstk` или `maxmin`)
- собирает toy audit-датасет и запускает audit (см. `analysis.chem.audit`)

Примечание: `score_mode=toy_edge_dist` — это демонстрационный скоринг (score = 1 - dist_to_original).
На следующих итерациях будет поддержан `score_mode=external_scores`.

Warnings:
- верхнеуровневый `warnings` — это объединение `decoys.warnings`, `audit.warnings` и предупреждений селекции.

Метод selection:
- `firstk`: первые `k` по стабильному порядку (по `hash`)
- `maxmin`: жадный max-min по `pairwise_dist` (tie-break: `hash`)

## Scripts

### 1. `scan_virtual_island.py`

**Purpose**: Map the "island of stability" for virtual atoms `X(p, eps)` by scanning parameter space.

**Usage**:
```bash
# Default scan (p=1..4, eps=-6..-0.1)
python scan_virtual_island.py

# Extended scan (p=0..8, eps=-10..0)
python scan_virtual_island.py --p-min 0 --p-max 8 --eps-min -10.0 --eps-max 0.0 --n-eps 30
```

**Parameters**:
- `--p-min`: Minimum number of valence ports (default: 1)
- `--p-max`: Maximum number of valence ports (default: 4)
- `--eps-min`: Minimum epsilon (spectral depth) (default: -6.0)
- `--eps-max`: Maximum epsilon (default: -0.1)
- `--n-eps`: Number of epsilon grid points (default: 20)

**Output**:
- `virtual_island_scan.png`: 2-panel heatmap (Chi_X and Max |F|)
- Console summary of "island of stability"

**Expected runtime**: ~30 seconds for default, ~3 minutes for extended scan

---

### 2. `scan_living_sectors.py`

**Purpose**: Measure topological complexity of molecular trees grown from each element in the periodic table.

**Usage**:
```bash
python scan_living_sectors.py
```

**How it works**:
- For each element Z=1..36 (excluding inerts):
  - Grow N=20 molecular trees using `grower.py`
  - Compute graph complexity via `complexity.py`
  - Collect statistics: Avg_C, Max_C, Avg_Size

**Output**:
- `living_sectors_scan.png`: Scatter plot on D/A plane, colored by complexity
- Console table: Top 10 elements by Max_Complexity

**Expected result**: Elements on the p-acceptor plateau (C, N, O, Si, P, S, Ge, As, Se) show C_max ~ 10-15, while donors (Li, Na, K) show C_max ~ 0.

**Expected runtime**: ~2 minutes

---

### 3. `scan_living_sectors_segregated.py`

**Purpose**: Compare molecular complexity in three "chemical worlds":
1. **Donor-only**: Li, Na, K, Be, Mg, Ca
2. **Acceptor-only**: C, N, O, F, Si, P, S, Cl
3. **Mixed**: Full periodic table

**Usage**:
```bash
python scan_living_sectors_segregated.py
```

**Output**:
- `segregated_comparison.png`: 3-panel bar chart (one per scenario)
- `segregated_stats.csv`: Per-seed statistics
- Console summary: Mean complexity by scenario

**Expected result**:
- Donor-only: C_avg < 2
- Acceptor-only: C_avg ~ 5-8
- Mixed: C_avg ~ 10-15

**Expected runtime**: ~5 minutes

---

### 4. `analyze_complexity_correlations.py`

**Purpose**: Statistical analysis of correlation between D/A indices and molecular complexity.

**Usage**:
```bash
python analyze_complexity_correlations.py
```

**Analyses performed**:
- Spearman correlation: D vs Complexity, A vs Complexity
- Group statistics by role (hub / bridge / terminator / inert)
- Scatter plots and heatmaps

**Output**:
- `complexity_summary.csv`: Full dataset (Element, D, A, Role, Avg_C, Max_C, Avg_Size)
- `correlation_stats.txt`: Statistical summary (r, p-values, group means)
- `correlation_plots.png`: 3-panel figure

**Expected runtime**: ~2 minutes

---

## Workflow Example

```bash
# 1. Basic validation
python scan_virtual_island.py
python scan_living_sectors.py

# 2. Extended analysis
python scan_virtual_island.py --p-min 0 --p-max 8 --eps-min -10 --eps-max 0 --n-eps 40
python scan_living_sectors_segregated.py

# 3. Statistical analysis
python analyze_complexity_correlations.py
```

## Troubleshooting

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'geom_atoms'`

**Solution**: Ensure `geom_atoms.py`, `grower.py`, `complexity.py` are in the same directory or add the parent directory to PYTHONPATH:
```bash
export PYTHONPATH="/path/to/book2:$PYTHONPATH"  # Linux/Mac
$env:PYTHONPATH="C:\path\to\book2;$env:PYTHONPATH"  # Windows PowerShell
```

### Slow Execution
**Problem**: Scripts take too long to run.

**Solution**: Reduce trial counts or grid resolution:
- Edit script and set `n_trials = 10` (instead of 20 or 50)
- Use coarser epsilon grid: `--n-eps 10`

### Out of Memory
**Problem**: Script crashes with memory error.

**Solution**: Reduce molecule size limits:
- Edit `GrowthParams` in script: `max_atoms=15` (instead of 25)
- Reduce grid density for virtual island scan

### AttributeError in grower.py
**Problem**: `'Molecule' object has no attribute 'adjacency_matrix'`

**Solution**: Verify that `geom_atoms.py` includes the `adjacency_matrix()` method in the `Molecule` class. This was added in a recent update.

## Understanding the Output

### Virtual Island Scan
- **Chi_X heatmap**: Shows spectral electronegativity. Green/blue = stable, white/red = unstable.
- **|F| heatmap**: Shows maximum bond energy. Lower values (dark blue) = more stable.

### Living Sectors Scan
- **Scatter plot**: Each dot is an element. Size/color = complexity.
- Look for clustering: acceptor plateau (A~1.24) should have high complexity.

### Segregated Comparison
- **Bar charts**: Compare max complexity across seeds in each scenario.
- Expect: Mixed > Acceptor-only > Donor-only

### Correlation Analysis
- **Spearman r**: -1 to 1 (0 = no correlation, +/-1 = perfect correlation)
- **p-value**: < 0.05 = statistically significant

## Citation

If you use these scripts or results, please cite:

> Paulig, R. (with OpenAI GPT-5.1 Thinking). *Geometric-Spectral Model v4.0: Laws of Virtual Elements and Living Sectors*. 2025.

## License

These scripts are part of the geom-spec research project and are provided for educational and scientific purposes.
