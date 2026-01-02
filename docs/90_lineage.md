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
