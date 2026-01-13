# Контракт `customer_truth.v1` (expensive truth labels)

Назначение: зафиксировать формат CSV-файла “expensive truth” для расчёта utility-отчёта cost&lift (VALUE-M5).
Этот файл joinится с `summary.csv` по идентификатору молекулы.

См. policy: `docs/10_contracts_policy.md`.

## Формат

- Тип: CSV (заголовок обязателен)
- Кодировка: UTF-8
- Разделитель: запятая `,`
- Дополнительные колонки допускаются (forward-compatible) и игнорируются потребителями.

## Schema (v1)

Обязательные колонки:

- `molecule_id` (string, required): ключ молекулы для join с `summary.csv`.  
  Инвариант: должен совпадать со значением колонки `id` в `summary.csv`.
- `expensive_label` (string, required): `PASS` или `FAIL`.  
  Интерпретация: `PASS` = “hit” (положительный исход).
- `truth_source` (string, required): источник/правило разметки (например: `proxy_rule_v1`, `pfizer_truth`).
- `truth_version` (string, required): версия контракта; для v1 должна быть `customer_truth.v1`.

## Инварианты / ошибки

- Нет обязательной колонки → **ERROR**
- `truth_version != "customer_truth.v1"` → **ERROR** (несовместимая схема)
- Дубликаты `molecule_id` → **ERROR**
- `expensive_label` не из {`PASS`,`FAIL`} → **ERROR**

## Минимальный пример

```csv
molecule_id,expensive_label,truth_source,truth_version
mol_001,PASS,proxy_rule_v1,customer_truth.v1
mol_002,FAIL,proxy_rule_v1,customer_truth.v1
```

