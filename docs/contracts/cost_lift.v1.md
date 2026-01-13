# Контракт `cost_lift.v1` (utility report JSON)

Назначение: зафиксировать формат `cost_lift_report.json` для “money proof” (cost&lift на expensive truth).
Отчёт считается “истиной utility”, если:

- соответствует schema `cost_lift.v1`,
- использует truth CSV по `customer_truth.v1`,
- и его ключевые поля/инварианты покрыты guardrail-тестом.

См. policy: `docs/10_contracts_policy.md`.

## Формат

- Тип: JSON
- Кодировка: UTF-8

## Schema (v1)

Top-level (object), обязательные поля:

- `report_schema` (string, required): строго `"cost_lift.v1"`.
- `truth_schema` (string, required): строго `"customer_truth.v1"`.
- `summary_csv` (string, required): provenance-путь к исходному `summary.csv` (из evidence pack).
- `truth_csv` (string, required): provenance-путь к `truth.csv`.
- `skip_policy` (string, required): для v1 фиксируем `"unknown_bucket"`.
- `seed` (int, required): seed отчёта (в т.ч. bootstrap).
- `K_requested` (int, required): запрошенный budget K.
- `K_effective` (int, required): фактически применённый K (≤ N eligible; должен быть > 0).

Агрегаты входа:

- `N_total` (int, required): всего строк в `summary.csv`.
- `N_ok` (int, required): число строк `status==OK`.
- `N_skip` (int, required): число строк `status==SKIP`.
- `N_error` (int, required): число строк `status==ERROR`.
- `N_with_truth` (int, required): число OK-строк, для которых есть truth label.
- `truth_coverage_rate` (float, required): `N_with_truth / N_ok`.
- `unknown_bucket_rate` (float, required): доля OK-строк без truth (`(N_ok-N_with_truth)/N_ok`).

Методы (object `methods`, required), обязательные ключи:

- `baseline_random`
- `baseline_score_only_topk`
- `filtered_score_plus_audit_topk`

Каждый метод — object с обязательными полями:

- `k_effective` (int, required): размер выборки (должен быть > 0).
- `hits` (int, required): число `PASS` по expensive truth внутри выбранных.
- `hit_rate` (float, required): `hits / k_effective`.
- `ci_low` (float, required): нижняя граница 95% bootstrap CI по `hit_rate`.
- `ci_high` (float, required): верхняя граница 95% bootstrap CI по `hit_rate`.

Uplift (required):

- `uplift_score_plus_audit_vs_score_only` (float, required): `hit_rate(filtered_score_plus_audit_topk) - hit_rate(baseline_score_only_topk)`
- `uplift_score_plus_audit_vs_random` (float, required): `hit_rate(filtered_score_plus_audit_topk) - hit_rate(baseline_random)`

## Семантика (v1)

- “Hit” = `expensive_label == PASS` в truth CSV.
- Eligible строки для сравнения методов:
  - `status == "OK"` в `summary.csv`
  - есть truth label
  - `gate` и `slack` парсятся как float
  - `verdict` непустой (PASS/FAIL)
- `baseline_score_only_topk` ранжирует по `auc_tie_aware`. Если колонки `auc_tie_aware` нет в `summary.csv`, используем восстановление `auc_tie_aware = gate + slack` (совместимо с `hetero_audit.v2`).
- `filtered_score_plus_audit_topk` выбирает топ-K по `slack` только среди `verdict == PASS`.

## Ошибки / FAIL-условия

Отчёт должен считаться **невалидным** (FAIL), если:

- отсутствует любой обязательный ключ,
- `report_schema != "cost_lift.v1"` или `truth_schema != "customer_truth.v1"`,
- `K_effective == 0`,
- `N_ok == 0` или нет eligible строк,
- uplift не является конечным числом (NaN/inf) или не может быть посчитан.

