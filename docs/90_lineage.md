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
