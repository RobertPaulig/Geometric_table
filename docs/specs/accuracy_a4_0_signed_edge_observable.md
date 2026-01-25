# ACCURACY-A4.0 — Signed Edge Observable (ρ + signed edge condensate), Φ fixed, nested κ

**Roadmap-ID:** `ACCURACY-A4.0 (Signed Edge Observable, Advanced, 1 new DOF)`  
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

**Вывод:** magnitude-канал `|K_ij|` в node-mix схеме недостаточен (и/или теряет критичную знаковую/ориентированную информацию). Следующая минимальная гипотеза — использовать **signed/oriented edge observable** (gauge-invariant) вместо magnitude.

---

## 1) Цель A4.0 (одна гипотеза)

**Гипотеза A4.0:** инверсии сохраняются потому, что magnitude-кохерентность на рёбрах (`|K_ij|`) теряет знак/ориентацию.  
Если заменить edge-канал на **gauge-invariant signed edge observable** и смешивать его с ρ через один κ, то на `functional_only` (LOOCV по `group_id`) получим:

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
4. SCF дисциплина как A2.2: safety floor → `phi -= mean(phi)` → backtracking → монотонность по финальному score.
5. Zero-leak: A2 outputs **бит-в-бит** неизменны.
6. Один новый DOF: **κ**. Никаких новых “ручек” (кроме фиксированных констант контракта).
7. Φ **фиксирован**: `Phi_fixed := π/2` (никакого выбора Φ).
8. `tau` фиксирован (тот же, что принят в A3.4; по умолчанию `tau=1.0`), не оптимизируется.

---

## 4) Определение наблюдаемых (канонические формулы)

### 4.1 Phase operator и heat kernel

Используется фазовый/магнитный оператор `L_A(Phi_fixed)` (эрмитов), как в A3.* (SSSR flux, 2π-periodicity, hermiticity, D по real weights).

Определяем тепловое ядро:

* `K := exp(-tau * L_A)`

Реализация через eigen-decomposition допустима; важно, чтобы `K_ij` на рёбрах был вычислен без построения полного NxN, если не требуется.

### 4.2 Узловая плотность ρ (как в A3.4)

* `rho_raw := diag(K)`
* `rho_imag_max := max(|Im(rho_raw)|)`; требование: `rho_imag_max < 1e-12`
* `rho := Re(rho_raw)`
* `rho := rho / sum(rho)` (жёсткая нормировка)

### 4.3 Signed edge observable (новое: sign/orientation)

Для каждого ребра `(i,j)` (существующие химические связи) считаем gauge-invariant величину:

* `q_ij := exp(-i * θ_ij) * K_ij`
  где `θ_ij` — link phase на ребре из того же phase-channel, что использовался для сборки `L_A`.

Определяем signed observable на ребре:

* `b_ij := Re(q_ij)`  (вещественная, может быть отрицательной)

### 4.4 Узловая агрегация signed edge observable

Сводим реберный сигнал в узловую величину (node-local интерфейс сохраняем):

* `B_i := Σ_{j~i} w_ij * b_ij`

где `w_ij` — реальные веса рёбер (те же, что используются для `L_base`/оператора; не новый параметр).

### 4.5 Преобразование B → распределение (без новых DOF)

Так как `B_i` может быть отрицательным, приводим к неотрицательной “плотности” детерминированно:

* `B_shift := B - min(B) + eps_shift`, где `eps_shift := 1e-12` (фиксировано)
* `b_sum := sum(B_shift)`
* если `b_sum > 0`: `b_norm := B_shift / b_sum`, иначе `b_norm := 0-vector`

Логировать сенсоры: `b_min`, `b_max`, `b_sum`, `eps_shift`.

---

## 5) Эффективная плотность (1 DOF κ)

Единственный новый DOF: κ.

* `rho_eff := (1 - kappa)*rho + kappa*b_norm`
* `rho_eff := rho_eff / sum(rho_eff)` (renorm обязателен; логировать флаг)

Дальше pipeline остаётся прежним:

* `phi := -log(rho_eff + eps)` → safety floor → `phi -= mean(phi)` → SCF/backtracking/energy как в A2.2.

---

## 6) Параметры (фиксировано) и nested selection

* `Phi_fixed := π/2` (фиксировано; не DOF)
* `tau := 1.0` (фиксировано; не DOF; должен совпасть с A3.4)
* `kappa_candidates := {0.0, 0.25, 0.5, 1.0}` (фиксировано)

Nested selection κ (train-only) внутри каждого outer LOOCV fold:

1. минимизировать `num_negative_train_inner`
2. затем максимизировать `median_spearman_train_inner`
3. tie-break: выбрать **больший κ**

Обязательные поля в `best_config.json` и `metrics.json`:

* `nested_selection=true`
* `phi_fixed=...`
* `kappa_candidates=[...]`
* `selected_kappa_by_outer_fold`
* `search_space_size=4`

---

## 7) KPI A4.0 (MVP)

На LOOCV(test), `functional_only`:

* **обязательно:** `num_groups_spearman_negative_test == 0`

---

## 8) Evidence Pack (обязательное расширение)

Как в A3.4 truth-pack + добавить/расширить:

* `rho_compare.csv`: добавить сенсоры `b_min`, `b_max`, `b_sum`, `kappa_selected`, `rho_eff_entropy`, `rho_imag_max`.
* `signed_edge_summary.csv`:
  * агрегаты per-molecule/per-group: `b_sum`, `b_min`, `b_max`, `degree_stats` (чтобы ловить корреляцию с degree).
* `search_results.csv`: inner результаты по κ для каждого outer fold.
* `metrics.json`: KPI + nested поля + sensors max.

---

## 9) Forensics (обязательное)

### 9.1 Zero-leak

* `rg -n "accuracy_a4|signed_edge|phase_channel|accuracy_a3" scripts --glob "accuracy_a1_isomers_a2*"` → `NO_MATCHES`

### 9.2 Bit-for-bit A2 proof

pre=`origin/main@<sha>` post=`PR head@<sha>`, одинаковая команда A2, разные out_dir.
SHA256 должны совпасть для `metrics.json` и `predictions.csv`.

---

## 10) Реализация (минимально)

1. Новый runner: `scripts/accuracy_a1_isomers_a4_0_signed_edge_observable.py` (opt-in)
2. Контракт-тест: `tests/test_accuracy_a4_0_contract.py` (opt-in, `RUN_A3_TESTS=1` или аналогичный флаг)
3. Workflows:
   * `compute_accuracy_a4_0.yml` (PR artifact, Actions-only truth)
   * `publish_accuracy_a4_0.yml` (main publish→release→registry→lineage)

---

## 11) Решение-вилка (что доказывает результат)

* PASS (MVP достигнут): подтверждение гипотезы “sign/orientation был недостающим минимальным каналом”.
* FAIL при соблюдении инвариантов: sign/node-mix всё ещё недостаточно ⇒ следующий Roadmap-ID должен уйти от node-mix к edge/pair scoring или к cycle-level observable.

