# ACCURACY-A4.0 — Signed/Orientable Edge Observable (edge-only scoring, no node-mix)

**Roadmap-ID:** `ACCURACY-A4.0 (Signed/Orientable Edge Observable, Edge-only scoring)`  
**Status:** closed (hypothesis FAIL; PR #257 closed without merge)  
**Owner:** Executor  
**Source of Truth:** this document (SoT)

**Outcome (evidence):**
- Actions compute-pack run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21338671447
- SHA256(evidence_pack.zip): `D32533F991155E27EC2166C63781A398E74E1972C1E14BA7F754F6B79F8A6C99`
- `kpi.verdict=FAIL`
- `num_groups_spearman_negative_test=4`
- `negative_spearman_groups_test=['C13H20O1','C20H22N2O2','C21H20N6O2S1','C21H23N3O3']`
- `median_spearman_by_group_test=0.45`
- `pairwise_order_accuracy_overall_test=0.5652173913`
- `top1_accuracy_mean_test=0.4`

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

---

## 12) Архитектурная логика (неразличимость → минимальный разрушитель → развилка)

**Фиксировано (инварианты):** graph-only, `H`/`w_ij` как в A3.4/A3.5, `K=f(H)` тем же способом, truth + LOOCV по `group_id` неизменны. Меняется только **класс наблюдаемой** и **агрегация**.

### 12.1 Лемма неразличимости: diag(K) и |K_ij| — gauge-инварианты

Пусть `H` (и `K=f(H)`) эрмитовы. Для любой диагональной унитарной матрицы:

```
D = diag(e^{iα_1}, …, e^{iα_n}),   H' = D H D*,   K' = f(H') = D K D*
```

имеем:

- `K'_{ii} = K_{ii}` для всех `i` (диагональ неизменна)
- `|K'_{ij}| = |e^{i(α_i-α_j)}K_{ij}| = |K_{ij}|` для всех `(i,j)` (модуль ребра неизменен)

**Следствие:** любой скоринг, использующий только `diag(K)` и/или `|K_ij|`, **не различает** элементы одного gauge-класса `K ~ D K D*`. Это минимальный механизм, почему node-only и magnitude-edge могут давать устойчивые инверсии.

### 12.2 Почему magnitude-edge + node-агрегация особенно опасны

Даже если `K_ij` содержит “правильный” сигнал (интерференция путей, bonding/anti-bonding), переход:

`|K_ij| → Σ по узлам → нормировка → смесь`

делает наблюдаемую почти всегда **чётной по знаку** и разрушает информацию о компенсации вкладов (плюсы/минусы “съедают” друг друга на уровне рёбер, но становятся “всегда плюс” после модулей/норм).

Это объясняет A3.5 FAIL как **закрытие класса наблюдаемой**, а не как “не настроили”.

### 12.3 Минимальный разрушитель неразличимости: signed/orientable edge observable

A4.0 использует **ориентированную** реберную величину:

`B_{ij} := Re(e^{-iθ_{ij}}K_{ij}) = |K_{ij}| cos(arg K_{ij} - θ_{ij})`,

которая **не** является инвариантом по фазе (в отличие от `|K_ij|`) и потому добавляет информацию о знаке/ориентации (конструктивная vs деструктивная интерференция).

**Критично:** A4.0 запрещает node-схлопывание и считает score **сразу по рёбрам**:

`S_edge = Σ_{(i,j)∈E} w_{ij} B_{ij}`.

### 12.4 Диагностическая развилка A / B / C (одна процедура без подбора параметров)

Эта процедура делит остаточные ошибки на 3 класса и определяет следующий Roadmap-ID.

**Шаг 0 (C-detector: потолок графового представления).**  
Для каждой bad-группы проверить, существуют ли пары изомеров, у которых `H` совпадает *с точностью до перестановки вершин* (изоморфизм взвешенного помеченного графа).

- Если да, и truth ранги различаются → **Сценарий C:** информация вне графа (stereo/3D). Тогда A4.* (на `f(H)`) принципиально не поможет → следующий Roadmap не A4.1, а “3D/stereo”.

**Шаг 1 (детерминированный gauge-fix без ручек).**  
Выбрать каноническое spanning tree `T` (например BFS по канонической нумерации) и подобрать `D`, чтобы на рёбрах дерева фазы стали нулевыми: `arg(K'_{ij})=0` для `(i,j)∈T`, где `K'=DKD*`. (Это убирает произвольность вершинных фаз.)

**Шаг 2 (A vs B: edge-signed vs cycle-level).**  
Посчитать два скаляра на каждом молекулярном графе:

- **A(m)** — A4.0 edge-only:

  `A(m) = Σ_{(i,j)∈E} w_{ij} Re(e^{-iθ_{ij}}K'_{ij}).`

- **F(m)** — cycle-flux summary (если нужно для диагностики): для фундаментальных циклов относительно `T`:

  `F(m) = Σ_C sin^2(Φ_C),`

  где `Φ_C` — gauge-инвариантная “холономия” цикла (фаза произведения нормированных реберных фаз, скорректированная на `θ`).

Интерпретация:

- Если `A(m)` различает и согласуется с truth на bad-группах, а `F(m)` почти константен → **Сценарий A:** проблема была в потере знака/ориентации → A4.0 должен был бы чинить.
- Если `A(m)` не различает, но `F(m)` различает и согласуется → **Сценарий B:** нужен cycle-level observable → следующий Roadmap-ID `ACCURACY-A4.1`.
- Если сработал Шаг 0 → **Сценарий C:** нужен 3D/stereo Roadmap, не A4.1.

### 12.5 Если A4.0 FAIL при соблюдении инвариантов — что это доказывает

Если A4.0 FAIL, то это доказывает структурно (не параметрически):

> Класс наблюдаемых “линейный edge-only функционал от `K=f(H)` с восстановлением знака/ориентации” недостаточен на фиксированном graph-only представлении.

Тогда следующий минимальный шаг **не** “крутить ручки”, а смена класса наблюдаемой:

- **Roadmap-ID:** `ACCURACY-A4.1 (Cycle-Flux / Holonomy Observable)`  
  **DoD:** добавить gauge-инвариантную cycle-наблюдаемую (`Φ_C` или summary `F(m)`) в evidence pack и показать устранение остаточных инверсий на тех bad-группах, где A4.0 не смог (при тех же truth/splits/H/K).
