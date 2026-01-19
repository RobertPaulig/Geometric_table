# PHYSICS-OPERATOR contract (P0) — `H = L + V` и анти-иллюзия AUC

Назначение: зафиксировать минимальный (P0) “физический” оператор для спектральных фич и правила интерпретации AUC
в присутствии decoys разной сложности (hardness bins).

## 1) Fundamental operator (P0)

### 1.1 Топологический оператор (база)

Для молекулы строится граф `G=(V,E)` по heavy-атомам (нумерация соответствует матрице смежности).

Матрица смежности `A ∈ R^{n×n}` (пока бинарная):

```
A_ij = 1, если (i,j) ∈ E
A_ij = 0, иначе
```

Матрица степеней `D = diag(d_i)`, где `d_i = Σ_j A_ij`.

Лапласиан:

```
L = D - A
```

### 1.2 Физический оператор (P0)

Вводится диагональный потенциал `V = diag(V_i)`, зависящий от типа атома `t_i`:

```
H = L + V
```

P0-инвариант: `V` диагонален, `A` бинарная, `H` симметричен.

## 2) Topological blindness (официально)

Если две молекулы имеют одинаковую топологию (одинаковая `A`), то `L` одинаковый ⇒ `spec(L)` одинаковый.
Следовательно, `spec(L)` **не может** быть единственным валидатором “химичности” при изменении типов атомов.

Критерий успеха P0 (тестовый инвариант):

- `spec(L_A) ≈ spec(L_B)` для одинаковой топологии
- `spec(H_A) != spec(H_B)` если хотя бы один тип атома отличается и `V(t)` различается

## 3) Single Source of Truth (SoT) для потенциала `V(t)`

### 3.1 Единственный источник

Параметры потенциала берутся **только** из `data/atoms_db_v1.json`.
Хардкод словарей в `.py` запрещён.

### 3.2 Схема SoT (фиксируем как есть)

`data/atoms_db_v1.json` в проекте уже используется и имеет строки вида:

- `Z` (int) — atomic number
- `name` (str) — symbol
- `epsilon` (float) — используем как `potential_V` для P0

P0-соглашение: `potential_V(t) := epsilon(Z=t)`.

### 3.3 Политика отсутствующих параметров (жёстко)

Если в `atoms_db_v1.json` нет `epsilon` для атома, встреченного в молекуле:

- в коде бросается `MissingPhysicsParams`
- в batch-выводе строка помечается как `status=ERROR` с `reason=missing_physics_params`
- evidence pack **всё равно** собирается (audit-grade, без “тихих” провалов)

### 3.4 Units & Scaling (Enterprise-safe)

#### Модель единиц (P0–P3)

Поля `epsilon` (и другие параметры в `data/atoms_db_v1.json`) — это **безразмерные параметры оператора** (dimensionless).
Они **не** являются энергией в eV, длиной связи в Å или любыми “реальными” физическими единицами.

#### Разрешённые/запрещённые claims

Разрешено (enterprise-safe):
- *relative trends / discrimination / robustness* (сравнение/различимость/ранжирование в рамках одного протокола).

Запрещено до калибровки:
- прямая интерпретация значений спектра/потенциала как eV/Å/“реальной энергии/длины”.

#### Масштаб потенциала

Вводится глобальный скаляр `potential_scale_gamma`:

```
V_scaled = gamma * V0
```

Где `V0` берётся **только** из SoT (`data/atoms_db_v1.json`), а `gamma` по умолчанию `1.0`.
До калибровки `gamma=1.0`, модель остаётся dimensionless; калибровка фиксируется отдельной процедурой/артефактами.

### 3.5 SCF / v4.0 protocol (self-tuning operator)

SCF (self-consistent field) — итеративный протокол, который строит самосогласованный потенциал `V` поверх базового `V0`
(SoT) и фиксирует трассу сходимости как audit-grade артефакт.

#### Включение протокола

SCF считается включённым, если одновременно:

- `potential_mode ∈ {"self_consistent","both"}`
- `physics_mode ∈ {"hamiltonian","both"}`

#### Оператор на итерации

Пусть `A` — (взвешенная или невзвешенная) матрица смежности, `D = diag(sum_j A_ij)`, `L = D - A`.
Базовый потенциал `V0` берётся **строго** из SoT (`data/atoms_db_v1.json`) и масштабируется:

```
V0_scaled = potential_scale_gamma * V0
```

На итерации `t` оператор имеет вид:

```
H(t) = L + diag(V(t))
```

Важно: все величины в этом протоколе **dimensionless** до отдельной калибровки.

#### Спектральная “плотность” ρ (детерминированный P3-minimal)

На шаге `t` вычисляем eigenpairs `(λ_k, u_k)` для `H(t)` и берём нижние `K = scf_occ_k` состояний.
Веса задаём softmax-правилом:

```
w_k = exp(-λ_k / scf_tau) / sum_j exp(-λ_j / scf_tau)
```

Локальная “плотность”:

```
ρ_i = sum_k w_k * |u_{ik}|^2
ρ_tilde_i = ρ_i - mean(ρ)
```

#### Обновление потенциала + смешивание

Обновление (P3-minimal, без химических claims):

```
V_proposed = V0_scaled + scf_gamma * ρ_tilde
V(t+1) = (1 - scf_damping) * V(t) + scf_damping * V_proposed
```

#### Критерий остановки

Сходимость:

```
residual_inf = max_i |V(t+1)_i - V(t)_i| <= scf_tol
```

Стоп по лимиту итераций: `t <= scf_max_iter`.

#### Артефакты (evidence pack)

При включённом SCF evidence pack обязан содержать:

- `scf_trace.csv` - трасса итераций (минимум: `iter`, `residual_inf`, `residual_mean`, `status`, `converged`)
- `potential_vectors.csv` - векторы потенциала по узлам (минимум: `V0`, `V_scaled`, `gamma`, `V_scf`, `rho_final`)
- `scf_summary.json` - run-level summary (`scf_status`, `scf_iters`, `scf_residual_final` и параметры SCF)
- `scf_audit_metrics.csv` - per-molecule audit таблица (минимум: `mol_id`, `scf_converged`, `scf_iters`, `residual_init`, `residual_final`, `deltaV_max`, `deltaV_p95`, `potential_gamma`, `operator_mode`)
- `summary_metadata.json` - ключи `scf_enabled`, `scf_status`, `potential_unit_model="dimensionless"`, `potential_scale_gamma`

`scf_status` (run-level) фиксируется как строка из:
`CONVERGED | MAX_ITER | ERROR_MISSING_PARAMS | ERROR_NUMERICAL`.

#### SCF Audit Requirements (P3.6)

SCF считается **доказанным как нетривиальный** только если это видно в артефактах и метаданных, а не по факту "код не упал".

Требования к `summary_metadata.json` (additive, без ломания других схем):

- агрегаты: `scf_iters_mean`, `scf_iters_p95`, `scf_iters_max`, `scf_converged_rate`
- residual: `residual_final_p95`, `residual_final_max`
- обновление поля (по узлам): `deltaV_max_max`, `deltaV_p95_max`
- `scf_nontrivial_rate`
- вердикт: `scf_audit_verdict`, `scf_audit_reason`

Определение `deltaV` (для SCF-audit):

- `deltaV_max` на молекулу = `max_node |V_scf - V_scaled|`
- `deltaV_p95` на молекулу = p95 по узлам `|V_scf - V_scaled|`

Критерий **nontrivial** (на молекулу):

- `scf_converged = True`
- `scf_iters >= 2`
- `deltaV_max >= eps_V`, где `eps_V = 1e-6` (dimensionless)

Вердикт прогона `scf_audit_verdict` (audit-grade, без fail-fast):

- `TRIVIAL_FIXED_POINT` если `scf_iters_max <= 1` или `deltaV_max_max < eps_V`
- `NONCONVERGED` если `scf_converged_rate < 0.95`
- `INCONCLUSIVE_INSUFFICIENT_ASYM` если в прогоне недостаточно асимметричных кейсов (по протоколу асимметрии)
- `SUCCESS` если `scf_nontrivial_rate >= 0.50` и `scf_converged_rate >= 0.95`

Жёсткое правило интерпретации:

- Если `scf_audit_verdict != SUCCESS`, SCF **запрещено** трактовать как "работает/влияет", даже если CI зелёный.

#### Verdict-правило (audit-grade, без fail-fast)

Если SCF включён, но не сошёлся (`scf_status=MAX_ITER`), пайплайн **не падает**,
а помечает результат как `outcome_verdict=INCONCLUSIVE_SCF_NOT_CONVERGED` с `outcome_reason=scf_not_converged`.

## 4) Hardness curve + запрет самообмана AUC

Декои могут быть “лёгкими” (слишком непохожими), и тогда высокая AUC может быть самообманом.

Пайплайн обязан выпускать артефакты:

- `hardness_curve.csv` (строка на пару original↔decoy)
- `hardness_curve.md` (правила биннинга + AUC по бинам + `auc_interpretation`)
- `summary_metadata.json` (агрегаты `n_*`, `auc_*`, `median_tanimoto`, `auc_interpretation`)

Политика `auc_interpretation` (P0):

- если данных в hard-бине недостаточно ⇒ `INCONCLUSIVE_DECOYS_TOO_EASY`
- если в hard-бине нет сигнала ⇒ `ILLUSION_CONFIRMED`
- если есть сигнал на hard ⇒ `SUCCESS_SIGNAL_ON_HARD`

## 5) Naming policy (обязательное)

В репозитории запрещены персоналии/фамилии: в текстах, названиях файлов, ключах JSON, названиях режимов.

