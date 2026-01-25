# ACCURACY-A4.1 — Cycle-Flux / Holonomy Observable (cycle-level scoring; graph-only; 0 DOF)

**Roadmap-ID:** `ACCURACY-A4.1 (Cycle-Flux / Holonomy Observable, cycle-level scoring)`  
**Status:** planned (Memory Fix required before any code)  
**Owner:** Executor  
**Source of Truth:** this document (SoT)

## 0) Контекст (fact-only)

* A2.* truth accepted, KPI FAIL: сохраняются инверсии (negative Spearman groups).
* A3.5 — **hypothesis FAIL** (см. `docs/specs/accuracy_a3_5_edge_coherence_condensate.md`).
* A4.0 — **hypothesis FAIL** (PR #257 closed without merge):
  compute-pack run https://github.com/RobertPaulig/Geometric_table/actions/runs/21338671447 (zip SHA256 `D32533F991155E27EC2166C63781A398E74E1972C1E14BA7F754F6B79F8A6C99`),
  `num_groups_spearman_negative_test=4`.

**Вывод:** signed edge-only линейный функционал от `K=f(H)` недостаточен ⇒ следующий минимальный класс наблюдаемой должен быть **cycle-level gauge-invariant holonomy/flux**.

---

## 1) Цель A4.1 (одна гипотеза)

**Гипотеза A4.1:** остаточные инверсии определяются **цикловой степенью свободы** (holonomy/flux), которая не редуцируется к сумме реберных signed вкладов. Если перейти к cycle-level наблюдаемой (gauge-инвариантной), то на `functional_only` (LOOCV по `group_id`) получим:

**MVP:** `num_groups_spearman_negative_test == 0`.

---

## 2) STOP/GO (жёстко)

### STOP

Запрещено писать A4.1 код, пока не сделан **Memory Fix**:

* файл SoT создан (этот),
* внесён в `docs/99_index.md` как `REF-ACCURACY-A4.1-CONTRACT`,
* добавлен в `ENTRYPOINT.md` read-order,
* добавлен STOP-гейт в `CONTEXT.md`,
* добавлена секция в `docs/ROADMAP.md`.

### GO

После мержа docs Memory Fix PR в main и main CI 3/3 success — разрешён отдельный code PR `ACCURACY-A4.1 (code)`.

---

## 3) Инварианты (нарушишь — STOP)

1. Truth CSV/sha неизменен (тот же input truth + truth-copies в ZIP).
2. Splits неизменны: LOOCV строго по `group_id`.
3. Graph-only: без 3D/stereo/конформеров/embedding.
4. Операторная часть не меняется: тот же `H`, те же веса графа, тот же `K=f(H)` как в A3.4/A3.5/A4.0 пайплайне.
5. Нет ML.
6. 0 новых DOF (первый проход): никаких параметр-сеток/κ/тюнинга.
7. Zero-leak: A2 outputs бит-в-бит неизменны.
8. Publish-chain: запрещено публиковать FAIL (release/registry/lineage) — publish gate `kpi.verdict == PASS` обязателен.

---

## 4) Каноническое определение A4.1 (формулы, без вариантов “как получится”)

### 4.1 Kernel (как уже принято)

Используем тот же `K := f(H)`, что принят в пайплайне A3.4/A3.5/A4.0 (без новых определений H/L/W/весов).

### 4.2 Gauge-invariant edge phase (подготовка)

Для ребра `(i,j) ∈ E` определяем:

* `q_ij := exp(-i * θ_ij) * K_ij`, где `θ_ij` — link phase из существующего phase-channel (A3.*)
* `r_ij := q_ij / |q_ij|` если `|q_ij| > 0`, иначе `r_ij := 1`

`r_ij` — комплексное число на единичной окружности (или 1), которое удобно для holonomy.

### 4.3 Deterministic spanning tree + фундаментальные циклы

Выбираем детерминированное spanning tree `T` на `G=(V,E)` (например BFS от минимального индекса узла; tie-break по индексу).

Для каждого chord edge `(u,v) ∈ E \\ T` строим фундаментальный цикл:

* `C_uv := path_T(u→v) + (v→u)` (ориентация фиксирована детерминированно)

### 4.4 Holonomy по циклу

Определяем holonomy angle:

* `Φ_uv := Arg( Π_{(i→j)∈C_uv} r_ij )`

Это gauge-инвариантная величина (зависит только от циклового флюкса/holonomy).

### 4.5 Cycle-level score (скаляр, edge-only запрещён)

Считаем:

* `S_cycle := Σ_{(u,v)∈E\\T} w_uv * sin^2(Φ_uv)`

где `w_uv` — существующий вес ребра (как в текущем графе/H).

Запрещено:

* любые node-агрегации (ρ_eff/κ) как финальный скор,
* возврат к edge-only линейному скору (это A4.0),
* новые параметры/сеточки.

---

## 5) KPI A4.1 (MVP)

На LOOCV(test), `functional_only`:

* **обязательно:** `num_groups_spearman_negative_test == 0`
* MVP non-regression: `num_groups_spearman_negative_test <= 4` (не хуже A3.4/A4.0 baseline)

---

## 6) Evidence Pack (обязательный состав)

В `evidence_pack.zip` обязательно:

1. `metrics.json`:
   * `schema_version=accuracy_a1_isomers_a4_1.v1`
   * `kpi.verdict`
   * `num_groups_spearman_negative_test`, `median_spearman_by_group_test`, `pairwise_order_accuracy_overall_test`, `top1_accuracy_mean_test`
2. `predictions.csv` (group_id, id, truth_rel, pred_score, pred_rank)
3. `cycle_flux_by_molecule.csv`:
   * `S_cycle`, `num_cycles`, `sum_abs_cycle_contrib` (sanity)
4. `cycle_flux_by_cycle.csv`:
   * `id`, `group_id`, `cycle_id`, `phi_uv`, `sin2_phi`, `w_uv`, `contrib`
5. `checksums.sha256` (missing=0, mismatches=0)

---

## 7) Forensics (обязательное)

### 7.1 Zero-leak

* `rg -n "accuracy_a4|a4_1|cycle_flux|holonomy" scripts --glob "accuracy_a1_isomers_a2*"` → `NO_MATCHES`

### 7.2 Bit-for-bit A2 proof

pre=`origin/main@<sha>` post=`PR head@<sha>`, одинаковая команда A2, разные out_dir.  
SHA256 должны совпасть для `metrics.json` и `predictions.csv`.

### 7.3 ZIP integrity

`python -m zipfile -t evidence_pack.zip` → Done testing  
`checksums.sha256`: missing=0, mismatches=0.

