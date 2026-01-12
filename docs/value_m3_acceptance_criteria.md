# VALUE-M3 (Customer truth / proxy): acceptance criteria (контракт)

Назначение: зафиксировать **проверяемые** критерии приёмки customer/proxy evidence pack.
Этот документ определяет, когда пакет можно считать “пригоден для клиента” (GO) или требует остановки (STOP).

См. также: [CONTEXT](../CONTEXT.md), [Release checklist](95_release_checklist.md), [Artefacts registry](artefacts_registry.md), [Contracts policy](10_contracts_policy.md).

---

## 1) Вход (customer/proxy)

### 1.1. Минимальный вход

- `input.csv` (CSV): минимум колонки `id,smiles`.
- `scores.json` (JSON): **только** `schema_version == hetero_scores.v1` (см. `docs/contracts/hetero_scores.v1.md`).

Примечание про “scores-only”:
- “scores-only” означает: мы не требуем кода модели/веса/фичей, а используем внешний `scores.json`.
- При этом для воспроизводимого audit/evidence pack всё равно требуется `smiles` (для генерации decoys и расчётов).

### 1.2. Proxy-режим (для пилота VALUE-M3)

Proxy считается валидным, если:
- вход и scores генерируются детерминированно (фикс seed),
- сборка заканчивается evidence pack’ом с `ERROR=0`,
- факт принятия фиксируется по правилам “истины” (Release asset + SHA256 + registry + CI gates).

---

## 2) Обязательные свойства evidence pack (проверяемо)

### 2.1. Состав и целостность

Evidence pack обязан содержать минимум:
- `summary.csv`
- `metrics.json`
- `index.md`
- `manifest.json`
- `checksums.sha256`
- `evidence_pack.zip` (если публикация идёт в zip-режиме)

Правила проверки: см. `docs/95_release_checklist.md` (zip validate + checksums).

### 2.2. Никаких “тихих провалов”

Каждая строка `input.csv` обязана иметь итог:
- `status ∈ {OK, SKIP, ERROR}`
- `reason` (непустой для `SKIP/ERROR`)

---

## 3) Метрики и пороги (GO/STOP)

### 3.1. Жёсткие гейты (обязательные)

GO возможно **только если**:
- `metrics.json: counts.ERROR == 0`
- `rows_ok >= 1` (иначе невозможно валидировать поведение `slack/verdict` на OK-строках)
- `scores_coverage.rows_missing_scores_input == 0` (для customer/proxy pack в режиме `score_mode=external_scores`)

### 3.2. Метрики “качества скоринга” (для интерпретации)

Для подтаблицы `summary.csv` с `status == "OK"`:
- `median(slack)` считается **только по OK-строкам**
- `PASS-rate` считается **только по OK-строкам** (доля `verdict == PASS`)

Если для OK-строк `slack` не число или `verdict` пустой — это **STOP** (регресс контракта аудита).

### 3.3. Допустимые причины SKIP (customer/proxy)

Разрешены (не блокируют GO при выполнении 3.1):
- `invalid_smiles`
- `too_many_atoms`
- `disconnected`
- `no_decoys_generated`
- `missing_scores_input` (ожидаемо 0 при корректном `scores.json`, но допускается как сигнал качества входа)

Любые новые/неожиданные причины SKIP должны фиксироваться в outcome (release notes + registry) и разбираться отдельно.

---

## 4) Истина приёмки (артефакты)

VALUE-M3 (как факт) считается DONE только если:
1) есть GitHub Release asset (zip),
2) есть SHA256 на asset,
3) есть запись в `docs/artefacts_registry.md` (URL + SHA256 + команда + outcome),
4) зелёные CI gates на целевом SHA: `ci/test`, `ci/test-chem`, `ci/docker`.

