# INTEGRATION SCALE CONTRACT (P5.1.v1)

## 0. Purpose

Контракт фиксирует **как доказывается масштабируемость интеграции** (speedup vs N_atoms) при **обязательной корректности**.

## 1. Scope

Применимо к Evidence Pack, где выполняются:

- baseline интеграция (равномерная сетка),
- adaptive интеграция (любой backend),
- сравнение результатов baseline vs adaptive.

## 2. Definitions

### 2.1 Baseline grid

- Energy domain: (E \in [E_{min}, E_{max}])
- Baseline points: (M)
- Broadening/regularization: (\eta > 0)

### 2.2 Outputs

Для каждого образца (fixture) считаются:

- baseline result (y_b)
- adaptive result (y_a)
- timing: (t_b), (t_a)
- evaluation counts: (M), (K)

## 3. Correctness

### 3.1 Tolerance

[
tol = \varepsilon_{abs} + \varepsilon_{rel} \cdot |y_b|
]
[
pass \iff |y_a - y_b| \le tol
]

### 3.2 Per-sample correctness

[
correctness_pass = pass
]

Если сравнение не точечное, допускается L∞/L2 на кривой — но тогда контракт обязан:

1) явно указать метрику,
2) записывать её в `summary_metadata.json`.

### 3.3 Scale correctness set

Определим масштабный набор:
[
S = { sample : N_{atoms}(sample) \ge N_{gate} }
]
Где (N_{gate} = gate_n_min).

[
correctness_pass_rate_at_scale = \frac{#{s \in S: correctness_pass}}{#S}
]

**Correctness verdict:**

- `PASS_CORRECTNESS_AT_SCALE` если pass_rate ≥ correctness_gate_rate (обычно 1.0)
- `FAIL_CORRECTNESS_AT_SCALE` иначе

## 4. Speed and cost

### 4.1 Speedup

[
speedup = \frac{t_b}{t_a}
]

### 4.2 Eval ratio

[
eval_ratio = \frac{M}{K}
]

(фиксируем именно так, чтобы “больше = лучше”)

## 5. Overhead region rule

Для малых систем (N_atoms < gate_n_min) **speedup может быть < 1** и это **не провал проекта**.

## 6. Verdict logic

### 6.1 Preconditions

Если `FAIL_CORRECTNESS_AT_SCALE`, то:

- `integrator_speedup_verdict = NOT_VALID_DUE_TO_CORRECTNESS`
- speed metrics публикуются, но не считаются KPI.

### 6.2 Speedup verdict at scale

Считаем медиану speedup по S:

- `PASS_SPEEDUP_AT_SCALE` если `integrator_speedup_median_at_scale >= 1.0`
- `FAIL_SPEEDUP_AT_SCALE` если `< 1.0`
- `INCONCLUSIVE_NOT_ENOUGH_SCALE_SAMPLES` если |S| < min_scale_samples

## 7. Required artifacts (canonical names)

Evidence Pack **обязан** содержать либо эти файлы, либо `canonical_mapping.json` для них.

### Core

- `summary.csv`
- `summary_metadata.json`
- `manifest.json`
- `checksums.sha256`

### Integration audit

- `integration_compare.csv`
- `integration_speed_profile.csv`
- `adaptive_integration_trace.csv`
- `adaptive_integration_summary.json`

### Scale proof

- `fixtures_polymer_scale.csv`
- `speedup_vs_n.csv`

### Canonical mapping (если имена отличаются)

- `canonical_mapping.json`

## 8. Required metadata fields (`summary_metadata.json`)

Минимальный обязательный набор:

### Law reference

- `law_ref.contract_path`
- `law_ref.contract_commit`
- `law_ref.contract_version`

### Integrator config

- `integrator_mode` (например `"both"`)
- `integrator_energy_min`
- `integrator_energy_max`
- `integrator_energy_points` (baseline M)
- `integrator_eta`
- `integrator_eps_abs`
- `integrator_eps_rel`
- `gate_n_min`
- `correctness_gate_rate`
- `min_scale_samples`

### Results

- `integrator_valid_row_fraction`
- `integrator_correctness_pass_rate_at_scale`
- `integrator_speedup_median_at_scale`
- `integrator_eval_ratio_median_at_scale`
- `integrator_correctness_verdict`
- `integrator_speedup_verdict`
- `integrator_verdict_reason`

### Units rails

- `potential_unit_model` (строка, сейчас обязательно `"dimensionless"`)
- `potential_scale_gamma` (число)

## 9. Chain of truth

Истина существует только при цепочке:
PR → publish-run → release(zip+.sha256) → registry → lineage → main 3/3.

## 10. Non-negotiables

- Скорость без корректности — мусор.
- Симметричные fixtures не могут быть единственным доказательством.
- Любой “успех” должен читаться **изнутри zip**.

