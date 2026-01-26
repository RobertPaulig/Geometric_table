# ACCURACY-A4.2 — Cycle-Basis / SSSR Holonomy (cycle-level scoring; graph-only; 0 DOF)

**Roadmap-ID:** `ACCURACY-A4.2 (Cycle-basis / SSSR Holonomy Observable, 0 DOF)`  
**Status:** planned (Memory Fix required before any code)  
**Owner:** Executor  
**Source of Truth:** this document (SoT)

## 0) Контекст (fact-only)

* A4.1 (BFS fundamental cycle holonomy) is **hypothesis FAIL** (see evidence in `docs/specs/accuracy_a4_1_cycle_flux_holonomy.md`).
* Posthoc rescore shows `sin²` vs `1-cos` does **not** remove the remaining negative groups; function choice is not the root cause (see `docs/evidence/accuracy_a4_1_posthoc_rescore.json`).

**Вывод:** следующий минимальный шаг — сменить **координатизацию cycle-space** (BFS fundamental cycles → deterministic SSSR cycles), не меняя функцию и не добавляя DOF.

---

## 1) Цель A4.2 (одна гипотеза)

**Гипотеза A4.2:** A4.1 FAIL обусловлен тем, что holonomy считалась на фундаментальных циклах BFS-дерева, которые являются выбранным базисом cycle-space и могут быть “не теми координатами” относительно кольцевой структуры, использованной для построения phase-channel (`A`/SSSR). Если считать holonomy **по deterministic SSSR cycles**, то на `functional_only` (LOOCV по `group_id`) получим:

**MVP:** `num_groups_spearman_negative_test == 0`.

---

## 2) STOP/GO (жёстко)

### STOP

Запрещено писать A4.2 код, пока не сделан **Memory Fix**:

* файл SoT создан (этот),
* внесён в `docs/99_index.md` как `REF-ACCURACY-A4.2-CONTRACT`,
* добавлен в `ENTRYPOINT.md` read-order,
* добавлен STOP-гейт в `CONTEXT.md`,
* добавлена секция в `docs/ROADMAP.md`.

### GO

После мержа docs Memory Fix PR в main и main CI 3/3 success — разрешён отдельный code PR `ACCURACY-A4.2 (code)`.

---

## 3) Инварианты (нарушишь — STOP)

1. Truth CSV/sha неизменен (тот же input truth + truth-copies в ZIP).
2. Splits неизменны: LOOCV строго по `group_id`.
3. Graph-only: без 3D/stereo/конформеров/embedding.
4. Операторная часть не меняется: тот же `H`, те же веса графа, тот же `K=f(H)` как в A3.4/A3.5/A4.0/A4.1 пайплайне.
5. Нет ML.
6. 0 новых DOF: никаких параметр-сеток/κ/тюнинга.
7. Zero-leak: A2 outputs бит-в-бит неизменны.
8. Publish-chain: запрещено публиковать FAIL (release/registry/lineage) — publish gate `kpi.verdict == PASS` обязателен.

---

## 4) Каноническое определение A4.2 (формулы, без вариантов)

### 4.1 Kernel (как уже принято)

Используем тот же `K := f(H)`, что принят в пайплайне A3.4/A3.5/A4.0/A4.1 (без новых определений H/L/W/весов).

### 4.2 Gauge-invariant edge phase (подготовка)

Для ребра `(i,j) ∈ E` определяем:

* `q_ij := exp(-i * θ_ij) * K_ij`, где `θ_ij` — link phase из существующего phase-channel (A3.*)
* `r_ij := q_ij / |q_ij|` если `|q_ij| > 0`, иначе `r_ij := 1`
* ориентация: `r(j→i) := conj(r(i→j))` (всегда)

### 4.3 Deterministic SSSR cycles (cycle list + orientation)

Циклы берём как **deterministic SSSR** (как принято в phase-channel):

* cycle list: `sssr_cycles_from_mol(mol)` (existing deterministic implementation)
* orientation rule must be deterministic and documented:
  * start at minimal atom index in the ring,
  * pick the lexicographically-min traversal among the two directions,
  * rings sorted deterministically.

### 4.4 Holonomy по SSSR cycle

Для каждого SSSR cycle `C = [v0, v1, ..., v_{m-1}]` (в уже зафиксированной ориентации) считаем:

* `Φ_C := Arg( Π_{k=0..m-1} r(v_k → v_{(k+1) mod m}) )`

### 4.5 Cycle-level score (SSSR)

Считаем:

* `w_C := Σ_{k=0..m-1} w_{v_k v_{(k+1) mod m}}` (existing edge weights from the current graph/H)
* `S_sssr := Σ_{C∈SSSR} w_C * sin^2(Φ_C)`

Запрещено:

* любые node-агрегации (ρ_eff/κ) как финальный скор,
* возврат к edge-only линейному скору (это A4.0),
* параметры/сеточки.

---

## 5) Ceiling-test (строгая дефиниция `isomorphic_H`) — decision fork (C vs not-C)

Перед любым A4.2 code обязателен ceiling-test на bad-группах из A4.1:

* `['C11H21B1N2O4','C15H24O1','C20H22N2O2']`

### Определение `H_canon(m)` (HARD)

1. Build RDKit mol from SMILES.
2. Compute `rank = Chem.CanonicalRankAtoms(mol, breakTies=True)`.
3. Reorder the **exact same H** that is used in `K=f(H)` by `rank` to obtain `H_canon`.

### Определение `isomorphic_H(a,b)` (HARD)

`isomorphic_H(a,b) := ( ||H_canon(a) - H_canon(b)||_∞ <= 1e-12 )`.

### Автоматический критерий сценария C (STOP на graph-only)

Если существует пара `(a,b)` с `isomorphic_H=true` и при этом `delta_truth = |E_truth(a)-E_truth(b)| > 1e-6`, то:

* доказан потолок graph-only представления (Scenario C),
* A4.2 запрещён,
* следующий Roadmap-ID должен быть A6 (stereo/3D).

---

## 6) KPI A4.2 (MVP)

На LOOCV(test), `functional_only`:

* **обязательно:** `num_groups_spearman_negative_test == 0`
* non-regression: `num_groups_spearman_negative_test <= 3` (не хуже A4.1)

