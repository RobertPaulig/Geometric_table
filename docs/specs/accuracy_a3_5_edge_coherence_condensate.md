# ACCURACY-A3.5 — Edge-Coherence Condensate (ρ + edge coherence), Φ fixed, nested κ

**Roadmap-ID:** `ACCURACY-A3.5 (Edge-Coherence Condensate, Advanced, 1 new DOF)`  
**Status:** closed (hypothesis FAIL; PR #255 closed without merge)  
**Owner:** Executor  
**Source of Truth:** this document (SoT)

## Evidence (Actions compute-pack; PR not merged)

- PR: https://github.com/RobertPaulig/Geometric_table/pull/255 (closed without merge)
- compute-pack run (A3.5): https://github.com/RobertPaulig/Geometric_table/actions/runs/21336452397
- tested merge-ref: `86d0cebda1c5325b1a83a8e664f4de862aabd052` (`ci/test|ci/test-chem|ci/docker=success`, run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21336452395)
- evidence_pack.zip SHA256: `21B9407F4D06E0A5655510B6DD68BC55D7B1342D04828709C457B9BA0E8AB73A`
- LOOCV(test) `functional_only` (from `metrics.json` in the evidence pack): `num_negative_test=6`, `median_spearman=-0.55`, `pairwise=0.347826...`, `top1=0.0`
- κ-sweep on test (from `kappa_sweep_test.csv`): best fixed κ is `0.0` with `num_negative_test=5` (κ=0.25→6, κ=0.5→8, κ=1.0→9)

## 0) Контекст (fact-only)

* A2.* truth accepted, KPI FAIL: сохраняются инверсии (negative Spearman groups).
* A3.3 (phase→ρ) truth accepted как **hypothesis FAIL**: `num_negative_test=5`.
* A3.4 (ρ + |Im(q_ij)| current) truth accepted как **hypothesis FAIL**: `num_negative_test=4`, κ выбран `0.25` во всех folds.

**Вывод:** добавление “тока” помогло, но не достаточно ⇒ вероятно, теряем информацию из-за:

1. **узлового сведения** (node-only),
2. **выкидывания структуры** (например, |Im| теряет ориентацию/связность),
3. нужна минимальная **парная/реберная** наблюдаемая, но остаться в “локальной плотности конденсата”.

---

## 1) Цель A3.5 (одна гипотеза)

**Гипотеза A3.5:** инверсии сохраняются, потому что диагональная плотность `ρ_i = ⟨e_i, f(H)e_i⟩` и ток `|Im(q_ij)|` не захватывают “кохерентность”/связность между вершинами.

Добавление **edge-coherence конденсата** (из off-diagonal теплового ядра) и его смешивание с `ρ` через один κ должно привести к:

**MVP:** `num_groups_spearman_negative_test == 0` (functional_only).

---

## 2) STOP/GO (жёстко)

### STOP

Запрещено писать A3.5 код, пока не сделан **Memory Fix**:

* файл SoT создан (этот),
* внесён в `docs/99_index.md`,
* добавлен в `ENTRYPOINT.md` read-order,
* добавлен STOP-гейт в `CONTEXT.md`,
* добавлена секция в `docs/ROADMAP.md`.

### GO

После мержа Memory Fix в main и зелёного CI 3/3 — разрешён code PR A3.5.

---

## 3) Инварианты (нарушишь — STOP)

1. Truth неизменен (тот же truth CSV и truth-copies в ZIP).
2. Split неизменен: **LOOCV строго по `group_id`**.
3. Graph-only, без 3D.
4. SCF дисциплина как A2.2: safety floor → `phi -= mean(phi)` → backtracking → монотонность по финальному score.
5. Zero-leak: A2 outputs **бит-в-бит** неизменны.
6. Один новый DOF: **κ**. Никаких новых “ручек” (кроме фиксированных констант контракта).
7. Φ **фиксирован** (см. §6). Никакого выбора Φ.

---

## 4) Определение наблюдаемых (канонические формулы)

### 4.1 Phase operator (как принято в A3.*)

Используется фазовый/магнитный оператор `L_A(Φ_fixed)` (эрмитов).

Определяем тепловое ядро:

* `K := exp(-tau * L_A)`  (τ — фиксированная константа как в A3.4; не DOF)

### 4.2 Узловая плотность (ρ)

* `rho_raw := diag(K)`
* `rho_imag_max := max(|Im(rho_raw)|)`
  Требование: `rho_imag_max < 1e-12`
* `rho := Re(rho_raw)`
* `rho := rho / sum(rho)`  (жёсткая нормировка)

### 4.3 Edge-Coherence (новое: реберный конденсат)

Мы добавляем минимально “парную” величину, но сводим её в **узловую плотность** (чтобы интерфейс остался node-local).

Для каждого ребра `(i,j)` графа (только для существующих связей):

* **Edge coherence magnitude:**

  * `c_ij := |K_ij|`  *(вариант v1)*
  * *(допустима эквивалентная форма v1b: `c_ij := |K_ij|^2` — но выбрать ровно одну и зафиксировать в коде; предпочтение v1 = |K_ij|)*

Сводим к узловой величине:

* `C_i := Σ_{j~i} c_ij`
* `c_sum := sum(C)`
* если `c_sum > 0`: `c_norm := C / c_sum`, иначе `c_norm := 0-vector`

**Смысл:** это “сколько кохерентности/связности у узла по ребрам”, без знака и без gauge-зависимости.

---

## 5) Эффективная плотность для скоринга (1 DOF κ)

Определяем:

* `rho_eff := (1 - kappa)*rho + kappa*c_norm`
* затем `rho_eff := rho_eff / sum(rho_eff)` (renorm обязателен; логируем флаг)

**Единственный DOF:** `kappa`.

---

## 6) Фиксированные параметры (не DOF)

* `Phi_fixed := π/2` (как в A3.4; фиксировано)
* `tau := <тот же, что принят в A3.4>` (фиксировано; не оптимизируем)
* `kappa_candidates := {0.0, 0.25, 0.5, 1.0}` (фиксированная сетка)

---

## 7) Nested selection (только κ, train-only)

Выбор κ делается **внутри каждого outer LOOCV fold**:

* Outer: test = одна группа `group_id=g*`
* Inner: оценка только на train-группах (mini-LOOCV по train-группам допустим)

Критерий выбора κ:

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

## 8) KPI A3.5 (MVP)

На LOOCV(test), `functional_only`:

**обязательно:**

* `num_groups_spearman_negative_test == 0`

дополнительно (ориентиры, не блокеры если MVP достигнут):

* `median_spearman_by_group_test >= 0.55`
* `pairwise_order_accuracy_overall_test >= 0.60`
* `top1_accuracy_mean_test >= 0.30`

---

## 9) Evidence Pack (обязательное расширение)

Как в A3.4 truth-pack + добавить/расширить:

### 9.1 rho_compare.csv (per-molecule rows)

Обязательные колонки:

* `rho_sum`, `rho_imag_max`, `rho_entropy`, `rho_renorm_applied`, `rho_renorm_delta`, `rho_floor_rate`
* `c_sum`, `c_entropy`, `c_norm_entropy`
* `kappa_selected`, `rho_eff_entropy`, `rho_eff_renorm_applied`
* `phi_fixed`

### 9.2 coherence_summary.csv

* агрегаты по молекуле/группе:

  * `mean_c_sum_by_group`, `median_c_sum_by_group`
  * `c_sum_max`, `c_entropy_stats`
  * (опционально) `degree_stats` (чтобы отследить “просто коррелирует с degree”)

### 9.3 search_results.csv

* inner результаты по κ для каждого outer fold:

  * `fold_id`, `kappa`, `num_negative_inner`, `median_spearman_inner`, `selected`

### 9.4 metrics.json

* KPI + nested поля + sensors max:

  * `rho_imag_max_max`, `c_sum_max`, `kappa_distribution`

---

## 10) Forensics (обязательное)

### 10.1 Zero-leak

Команда:

* `rg -n "accuracy_a3|a3_5|coherence" scripts --glob "accuracy_a1_isomers_a2*"`

Ожидаем:

* `NO_MATCHES`

### 10.2 Bit-for-bit A2 proof

pre=`origin/main@<sha>` post=`PR head@<sha>`, одинаковая команда A2, разные out_dir.

SHA256 должны совпасть для `metrics.json` и `predictions.csv`.

---

## 11) Реализация (минимально)

1. Новый runner: `scripts/accuracy_a1_isomers_a3_5_edge_coherence.py` (opt-in)
2. Контракт-тест: `tests/test_accuracy_a1_isomers_a3_5_contract.py` (opt-in, `RUN_A3_TESTS=1`)
3. Workflows:

   * `compute_accuracy_a1_isomers_a3_5.yml` (PR artifact, Actions-only truth)
   * `publish_accuracy_a1_isomers_a3_5.yml` (main publish→release→registry→lineage)

---

## 12) Решение-вилка (что доказывает результат)

* Если A3.5 достигает `num_negative_test==0` ⇒ подтверждение гипотезы: **off-diagonal coherence** была недостающей минимальной наблюдаемой.
* Если A3.5 FAIL при соблюдении всех инвариантов ⇒ это сильный факт, что:

  * либо требуется **ориентированная/знаковая** реберная величина (не модуль),
  * либо нужно уходить от node-сведения и переходить к **pairwise observable** в скоринге (но это уже новый Roadmap-ID, не A3.5).

---

# MEMORY FIX (копипаст-список)

В отдельном docs PR:

1. Добавить этот файл в `docs/specs/`.
2. `docs/99_index.md`: `REF-ACCURACY-A3.5-CONTRACT → docs/specs/accuracy_a3_5_edge_coherence_condensate.md`
3. `ENTRYPOINT.md`: добавить в read-order.
4. `CONTEXT.md`: STOP-гейт “прочитать REF-ACCURACY-A3.5-CONTRACT”.
5. `docs/ROADMAP.md`: секция A3.5 planned + DoD (publish truth-chain + KPI).

