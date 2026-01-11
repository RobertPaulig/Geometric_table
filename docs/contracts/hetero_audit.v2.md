# Contract: `hetero_audit.v2`

Назначение: зафиксировать смысл полей `gate/slack/verdict` так, чтобы они **реагировали на качество скоринга** и могли использоваться как метрика пользы (VALUE-M2).

## Schema version

- `schema_version`: `hetero_audit.v2`

## Обязательные поля (верхний уровень)

- `version` — содержимое `VERSION` (или `unknown`).
- `dataset_id` — строковый идентификатор набора.
- `n_pos`, `n_neg` — число позитивов/негативов в `items`.
- `auc_tie_aware` — AUC с учётом ties (tie-aware).
- `neg_controls` — блок отрицательных контролей и гейтинга.
- `run` — `{seed, timestamp, cmd}`.
- `warnings` — список строк-предупреждений.

## `neg_controls` (семантика v2)

Поля:

- `null_q` — точный q-квантиль null-AUC (Wilcoxon U) для (`n_pos`,`n_neg`) при фиксированном порядке рангов.
- `perm_q` — q-квантиль AUC при пермутациях меток (exact/MC).
- `rand_q` — (для совместимости) равно `null_q`.
- `neg_auc_max` — `max(perm_q, rand_q)` (информативное поле; в v2 **не** используется для `slack`).
- `method`, `reps_used`, `null_q_method` — provenance расчёта `perm_q/null_q`.
- `margin` — добавка к `null_q`.
- `gate` — порог: `gate = min(1.0, null_q + margin)`.
- `slack` — ключевой сигнал качества: `slack = auc_tie_aware - gate`.
- `verdict` — `PASS` если `slack >= 0`, иначе `FAIL`.

Примечание: пустые `gate/slack/margin` допускаются только в тех случаях, когда пайплайн явно выставляет `audit.neg_controls.verdict=SKIP` (например, при `missing_scores_for_all_decoys`).

