# ACCURACY-A4.0 — Signed/Orientable Edge Observable (edge-only scoring, no node-mix)

**Roadmap-ID:** `ACCURACY-A4.0 (Signed/Orientable Edge Observable, Edge-only scoring)`  
**Status:** planned (Memory Fix required before any code)  
**Owner:** Executor  
**Source of Truth:** this document (SoT)

## 0) Контекст (fact-only)

* A2.* truth accepted, KPI FAIL: сохраняются инверсии (negative Spearman groups).
* A3.3 (phase→ρ) truth accepted как **hypothesis FAIL**: `num_negative_test=5`.
* A3.4 (ρ + current J) truth accepted как **hypothesis FAIL**: `num_negative_test=4`.
* A3.5 (ρ + edge coherence |K_ij| magnitude) — **hypothesis FAIL** (PR #255 closed without merge):
  compute-pack run https://github.com/RobertPaulig/Geometric_table/actions/runs/21336452397 (zip SHA256 `21B9407F4D06E0A5655510B6DD68BC55D7B1342D04828709C457B9BA0E8AB73A`),
  best fixed κ is `0.0` with `num_negative_test=5` (κ-sweep on test).

**Вывод:** `|K_ij|` magnitude → node-агрегация ухудшает/не чинит инверсии. Следующая минимальная гипотеза — **signed/orientable edge observable** и **edge-only scoring** (без node-схлопывания).

---

## 1) Цель A4.0 (одна гипотеза)

**Гипотеза A4.0:** инверсии сохраняются потому, что node-only наблюдаемые (`diag f(H)`) и magnitude-каналы на рёбрах теряют знаковую/ориентированную структуру.  
Если перейти к **signed/orientable off-diagonal наблюдаемой на рёбрах** и считать **скалярный score напрямую по рёбрам** (edge-only), то на `functional_only` (LOOCV по `group_id`) получим:

**MVP:** `num_groups_spearman_negative_test == 0`.

---

## 2) STOP/GO (жёстко)

### STOP

Запрещено писать A4.0 код, пока не сделан **Memory Fix**:

* файл SoT создан (этот),
* внесён в `docs/99_index.md` как `REF-ACCURACY-A4.0-CONTRACT`,
* добавлен в `ENTRYPOINT.md` read-order,
* добавлен STOP-гейт в `CONTEXT.md`,
* добавлена секция в `docs/ROADMAP.md`.

### GO

После мержа Memory Fix в `main` и зелёного CI 3/3 — разрешён code PR A4.0.

---

## 3) Инварианты (нарушишь — STOP)

1. Truth неизменен (тот же truth CSV и truth-copies в ZIP).
2. Split неизменен: **LOOCV строго по `group_id`**.
3. Graph-only, без 3D.
4. Оператор/веса/ядро K: тот же `K=f(H)` как в A3.4/A3.5 пайплайне (без новых фич).
5. Нет ML (никаких моделей/регрессий/обучаемых весов).
6. Нет новых DOF: **0 новых ручек** (никакого κ, никаких сеток параметров).
7. Zero-leak: A2 outputs **бит-в-бит** неизменны.
8. Publish-chain: запрещено публиковать FAIL (release/registry/lineage) — publish gate `kpi.verdict == PASS` обязателен.

---

## 4) Каноническое определение A4.0 (формулы, без вариантов “как получится”)

### 4.1 Kernel (как уже принято)

Используем тот же `K := f(H)`, что сейчас уже принят в пайплайне A3.4/A3.5 (без новых определений H/L/W/весов).  
A4.0 не вводит новый kernel и не вводит новые параметры kernel’а.

### 4.2 Edge observable (signed/orientable)

Выбираем **ровно один** вариант и фиксируем его в реализации (без “и A, и B”):

**Вариант B (с фазой/гейдж-фикс; предпочтительный):**

* `B_ij := Re(exp(-i * θ_ij) * K_ij)`

где `θ_ij` — link phase на ребре из существующего phase-channel (A3.*), а `K_ij` — off-diagonal kernel entry.

Запрещено: `|K_ij|`, `|Im|` и любые модули как основной канал A4.0.

### 4.3 Edge-only score (скаляр)

Считаем напрямую по рёбрам:

* `S_edge := Σ_{(i,j)∈E} w_ij * B_ij`

где `w_ij` — строго существующие веса рёбер из текущего графа/H (без новых фич).

Запрещено:

* любые node-агрегации как финальный скор (`Σ→узел→норма→смесь`),
* `rho_eff`, `κ`, и любые сетки параметров.

---

## 7) KPI A4.0 (MVP)

На LOOCV(test), `functional_only`:

* **обязательно:** `num_groups_spearman_negative_test == 0`

---

## 8) Evidence Pack (обязательный состав)

В `evidence_pack.zip` обязательно:

1. `metrics.json`:
   * `schema_version=accuracy_a1_isomers_a4_0.v1`
   * `kpi.verdict`
   * `num_groups_spearman_negative_test`, `median_spearman_by_group_test`, `pairwise_order_accuracy_overall_test`, `top1_accuracy_mean_test`
   * `variant=B` (фиксировано)
2. `predictions.csv` (group_id, id, truth_rel, pred_score, pred_rank)
3. `edge_score_by_molecule.csv`:
   * `S_edge`, `sum_abs_edge_contrib`, `num_edges`
4. `edge_contrib_topk.csv`:
   * top-K рёбер по вкладу `w_ij * B_ij` (для bad-групп)
5. `checksums.sha256` (missing=0, mismatches=0)

---

## 9) Forensics (обязательное)

### 9.1 Zero-leak

* `rg -n "accuracy_a4|signed_edge|phase_channel|accuracy_a3" scripts --glob "accuracy_a1_isomers_a2*"` → `NO_MATCHES`

### 9.2 Bit-for-bit A2 proof

pre=`origin/main@<sha>` post=`PR head@<sha>`, одинаковая команда A2, разные out_dir.
SHA256 должны совпасть для `metrics.json` и `predictions.csv`.

---

## 10) Реализация (минимально)

1. Новый runner: `scripts/accuracy_a1_isomers_a4_0_signed_edge.py` (opt-in)
2. Контракт-тест: `tests/test_accuracy_a1_isomers_a4_0_contract.py` (opt-in, `RUN_A4_TESTS=1`)
3. Workflows:
   * `.github/workflows/compute_accuracy_a1_isomers_a4_0.yml` (PR artifact, Actions-only truth)
   * `.github/workflows/publish_accuracy_a1_isomers_a4_0.yml` (main publish, с гейтом `kpi.verdict==PASS`)

---

## 11) Решение-вилка (что доказывает результат)

* PASS (MVP достигнут): подтверждение гипотезы “sign/orientation был недостающим минимальным каналом”.
* FAIL при соблюдении инвариантов: sign edge-only всё ещё недостаточно ⇒ следующий Roadmap-ID должен перейти к cycle-level observable или к pairwise/edge-density скорингу (новый контракт, не A4.0).
