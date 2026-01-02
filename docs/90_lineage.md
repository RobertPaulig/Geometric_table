# Lineage идей (`docs/90_lineage.md`)

Назначение: одно место “append-only”, где фиксируем **что изменилось, почему и чем это доказано** (эксперимент + тест + ссылка на коммит).

## Шаблон записи (copy/paste)

### YYYY-MM-DD — <краткое название>

- **Гипотеза:** ...
- **Метод:** ...
- **Эксперимент:** команды запуска + входные данные
- **Результат:** ключевые числа / графики / пути к CSV/PNG
- **Добавлен тест:** `tests/...`
- **Обновлены доки:** `docs/...`
- **Коммит(ы):** `git_sha`
- **Дальше:** ...

---

### 2026-01-02 — HETERO-1A audit: стабильный CLI + JSON-контракт

- **Гипотеза:** можно сделать один публичный “аудит”-entrypoint, который всегда выдаёт стабильный JSON (tie-aware AUC + neg-controls gate) и тем самым превращает HETERO-линию в проверяемый контракт.
- **Метод:** отдельный CLI `python -m analysis.chem.audit` + минимальная фиксированная JSON-схема; детерминизм закреплён тестом.
- **Эксперимент:** `python -m analysis.chem.audit --input tests/data/hetero_audit_min.json --seed 0 --out audit.json`
- **Результат:** JSON с ключами `version,dataset_id,n_pos,n_neg,auc_tie_aware,neg_controls,run` (см. `docs/README.md`).
- **Добавлен тест:** `tests/test_hetero_audit_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `310978c`, `41089df`
- **Дальше:** привязать audit к реальному мини-датасету из `analysis/chem` (или curated baseline) и формализовать “verdict invariants” (ε для float).

### 2026-01-02 — Mendoza-step: neg-controls exact для малых N

- **Гипотеза:** для малых выборок (N≤10) neg-controls должны быть детерминированными без Монте-Карло, чтобы гейт не “гулял” от сидов/порядка.
- **Метод:** в `analysis.chem.audit` считать `perm_q` точным перебором всех разметок при фиксированных `n_pos/n_neg`; `rand_q` приравнять `null_q`; добавить `method`/`reps_used`.
- **Эксперимент:** `python -m analysis.chem.audit --input tests/data/hetero_audit_min.json --seed 0 --neg_control_reps 999 --out audit.json`
- **Результат:** на `hetero_audit_min_v1` выставляется `method=exact`, `reps_used=0`, и `rand_q == null_q`.
- **Добавлен тест:** `tests/test_hetero_audit_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `4247e78`, `676ec59`
- **Дальше:** расширить exact-ветку до N≤12 (если нужно) и/или добавить ε-инварианты для float-полей в контракте.

### 2026-01-02 — Warning: веса vs unweighted null_q

- **Гипотеза:** при наличии весов в данных нужно явно сигнализировать, что `null_q` сейчас считается без учёта весов.
- **Метод:** добавить `schema_version` и `warnings` в JSON, зафиксировать `null_q_method`, и предупредить при неединичных весах.
- **Эксперимент:** `python -m analysis.chem.audit --input tests/data/hetero_audit_min.json --seed 0 --out audit.json`
- **Результат:** `warnings` включает `weights_used_in_auc_but_null_q_is_unweighted` на weighted-датасете.
- **Добавлен тест:** `tests/test_hetero_audit_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `315b047`, `ada4bcb`
- **Дальше:** добавить weighted-null и сверку с известными значениями.

### 2026-01-02 — Decoys v1: tree-chem без RDKit

- **Гипотеза:** можно генерировать структурные декои в tree-химии без RDKit, сохраняя degree-sequence и типы узлов.
- **Метод:** Prüfer-последовательности с фиксированными кратностями (`deg_i-1`), дедупликация по хэшу рёбер.
- **Эксперимент:** `python -m analysis.chem.decoys --input tests/data/hetero_tree_min.json --out decoys.json`
- **Результат:** детерминированный JSON с `k_generated == k_requested`, без нарушений `max_valence`.
- **Добавлен тест:** `tests/test_hetero_decoys_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `73e32dd`, `67191a3`
- **Дальше:** добавить базовую метрику разнообразия/покрытия и минимальный фильтр по структурным инвариантам.

### 2026-01-02 — Decoys v1.1: diversity/coverage + фильтр

- **Гипотеза:** для decoys нужен минимальный контроль разнообразия и дистанции от оригинала, иначе набор может быть тривиальным.
- **Метод:** метрика `dist(A,B)=|E_A Δ E_B|/(2*(n-1))`, фильтры `min_dist_to_original` и `min_pair_dist`, счетчики отброшенных кандидатов.
- **Эксперимент:** `python -m analysis.chem.decoys --input tests/data/hetero_tree_min.json --min_pair_dist 0.5 --min_dist_to_original 0.5 --max_attempts 50 --out decoys.json`
- **Результат:** JSON содержит `metrics` и `filter`; при жёстких порогах возможен warning `could_not_generate_k_decoys_under_constraints`.
- **Добавлен тест:** `tests/test_hetero_decoys_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `0e70794`, `06aaec6`
- **Дальше:** добавить рекомендации (recommendations) по подбору порогов и минимальный diversity-score.

### 2026-01-02 — Pipeline v1: decoys -> selection -> audit

- **Гипотеза:** можно склеить контуры (decoys + coverage selection + audit) в один воспроизводимый JSON-артефакт.
- **Метод:** один CLI `analysis.chem.pipeline`, два режима селекции (`firstk`, `maxmin`), детерминированный toy-score из `dist_to_original`.
- **Эксперимент:** `python -m analysis.chem.pipeline --tree_input tests/data/hetero_tree_min.json --k 10 --seed 0 --timestamp 2026-01-02T00:00:00+00:00 --select_k 5 --selection maxmin --out pipeline.json`
- **Результат:** JSON `hetero_pipeline.v1` содержит `decoys`, `selection`, `audit` в одном отчёте; `maxmin` не хуже `firstk` по `min_pairwise_dist`.
- **Добавлен тест:** `tests/test_hetero_pipeline_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `7ffe729`, `9c9506d`, `92f4fd7`
- **Дальше:** вынести toy-score в `score_mode` и добавить экспорт отчёта (CSV/MD) для пользователя.

### 2026-01-02 — Pipeline v1.1: score_mode + warnings + stable selection IDs

- **Гипотеза:** pipeline должен явно маркировать toy-скоринг, агрегировать warnings и давать устойчивые идентификаторы выбора.
- **Метод:** добавить `score_mode/score_definition`, агрегировать `warnings` из всех блоков, хранить `selected_hashes` и `index_base`.
- **Эксперимент:** `python -m analysis.chem.pipeline --tree_input tests/data/hetero_tree_min.json --k 10 --seed 0 --timestamp 2026-01-02T00:00:00+00:00 --select_k 5 --selection maxmin --out pipeline.json`
- **Результат:** JSON содержит агрегированный `warnings` и стабильные `selected_hashes`.
- **Добавлен тест:** `tests/test_hetero_pipeline_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `482f6ec`, `84a3d11`, `aeafc54`
- **Дальше:** добавить `report` CLI (md/csv) поверх pipeline JSON.
