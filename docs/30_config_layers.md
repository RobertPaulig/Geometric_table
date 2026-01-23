# Config Layers (A/B/C) — Constitution for experiments

Назначение: **жёстко разделить слой A / B / C** и сделать `experiment.yaml` **единой точкой входа** для воспроизводимых экспериментов (ACCURACY-A* / A2 и далее).

Ссылки (SoT):

- Слой A подчиняется `docs/contracts/PHYSICS_OPERATOR_CONTRACT.md` (контракт `H=L+V`, SoT `data/atoms_db_v1.json`, units rails).
- Рельсы контрактов и breaking-change policy: `docs/10_contracts_policy.md`.

## Главный принцип

**Один конфиг = один эксперимент.** Любое изменение, влияющее на результат, отражается в `experiment.yaml` (и попадает в `provenance.json` внутри evidence pack).

Минимальный шаблон: `configs/experiment.template.yaml`.

## Слои

### Layer A — PhysicsKernel (“Конституция”)

Меняет смысл модели. Изменения редкие и “дорогие”.

Если меняете A — **обязателен bump `physics_kernel.major_version`** (новая major-линия, условно `HETERO-3`).

Примеры A-изменений:

- тип лапласиана (combinatorial ↔ normalized ↔ random_walk)
- форма гамильтониана (`L_plus_V` ↔ другая)
- правила построения графа (kekulize / H policy / ordering / disallow dot-smiles)
- семейство потенциала `V` (какие признаки разрешены, какая форма допустима)
- определение весов рёбер (что означает bond order / aromatic multiplier)
- любые “единицы/нормировки” (rails)

### Layer B — FeatureExtractor (“Линза”)

Не меняет смысл A, а задаёт **как измеряем**: eps/beta/t, какие инварианты/признаки извлекаем, стандартализация, `gamma` и веса внутри разрешённой формы потенциала.

Меняете B — это **minor-итерация** (например, A1.6 / A2-r2), но без смены major-оператора.

Примеры B-изменений:

- набор спектральных инвариантов, eps/beta/t списки
- параметры стандартализации (fit-on-train)
- `gamma` (в рамках фиксированной формы `H=L+V`)
- веса компонент node-potential внутри разрешённого семейства A

### Layer C — Learner/Ranker (“то, что учится”)

Это то, что меняется от `.fit()`:

- семейство модели ранжирования (rank_ridge/logistic/linear_svm)
- регуляризация/гиперпараметры обучения (alpha/C)
- веса/коэффициенты модели (learner state)

Меняете C — это **новый обученный артефакт** (новый run), но смысл A/B должен быть прозрачен и воспроизводим.

## Версионирование (правило)

- Изменение A ⇒ bump `physics_kernel.major_version` (breaking / major).
- Изменение B ⇒ новая minor-итерация (новый tag/release, append-only).
- Изменение C ⇒ новый run/retrain (новый tag/release, append-only).

## “keypath → слой” (минимальная таблица)

| Keypath | Layer | Комментарий |
|---|---:|---|
| `physics_kernel.*` | A | Конституция оператора/графа/потенциала |
| `feature_extractor.*` | B | Линза (инварианты/eps/beta/t/gamma/стандартизация) |
| `learner.*` | C | Обучаемая часть (ранжирование/регрессия/регуляризация/веса) |
| `contracts.truth.*` | SoT | Истина датасета и её invariants (не мутировать) |
| `evaluation.*` | Protocol | LOOCV/splits/метрики/KPI rails |
| `artifacts.*` | Output | Куда писать результаты/pack |

## Запрещено (типовые ошибки)

- Менять слой A “тихо” без bump `physics_kernel.major_version`.
- Подменять смысл A через “гиперпараметры” B (например, менять тип лапласиана через B).
- Учиться на test (утечка: подгонка параметров по метрикам test).
- Менять truth CSV/sha256 в PR, который “улучшает метрики”.
- Переиспользовать release tag / перезаписывать assets (append-only дисциплина).

## Canonical `experiment.yaml` skeleton (минимум)

```yaml
schema:
  name: hetero.experiment
  version: 1

experiment:
  id: ACCURACY-A1.5
  run_id: a1_5-r1
  seed: 0

contracts:
  truth:
    id: isomer_truth.v1
    files:
      isomer_truth_csv: data/accuracy/isomer_truth.v1.csv
      raw_csv: data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv
      raw_csv_sha256: data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv.sha256
      contract_md: docs/contracts/isomer_truth.v1.md
    invariants:
      forbid_external_data: true
      forbid_truth_mutation: true
      split_policy: group_strict_loocv

physics_kernel:
  major_version: HETERO-2
  graph:
    builder: rdkit_2d
    add_hydrogens: explicit
    disallow_dot_smiles: true
  operators:
    laplacian:
      type: combinatorial
    hamiltonian:
      form: L_plus_V
      ensure_spd: true
      spd_shift: { mode: add_epsI, eps: 1.0e-6 }
  potentials:
    node_potential:
      family: epsilon_z_plus
      features_allowed: [Z, formal_charge, is_aromatic, in_ring, degree, valence_proxy]
    edge_weights:
      family: bond_order_plus
      bond_order_map: { single: 1.0, double: 2.0, triple: 3.0, aromatic: 1.5 }

feature_extractor:
  id: spectral_v2
  spectral:
    invariants:
      logdet: { enabled: true, eps_list: [1.0e-6, 1.0e-4] }
      heat_trace: { enabled: true, t_list: [0.5, 1.0, 2.0] }
  hyperparams:
    gamma: 0.28

learner:
  objective: pairwise_ranking
  model:
    family: rank_ridge
    regularization: { type: ridge, alpha: 1.0e-2 }

evaluation:
  protocol: loocv_by_group
  group_key: group_id
```
