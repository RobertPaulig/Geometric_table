# Контракт `isomer_truth.v1` (DFT isomers: energy ordering truth)

Назначение: зафиксировать формат CSV-файла "expensive truth" для ACCURACY-A1 (изомеры) — сравнение порядка энергий внутри каждой группы изомеров (`group_id`) между DFT truth и физическим оператором (H = L + V).

Источник истины:
- raw: `data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv` (+ `.sha256` рядом)
- canonical (v1): `data/accuracy/isomer_truth.v1.csv` (генерируется, raw не правим)

Генератор canonical:
- `scripts/build_isomer_truth_v1.py`

См. policy: `docs/10_contracts_policy.md`.

## Формат

- Тип: CSV (заголовок обязателен)
- Кодировка: UTF-8
- Разделитель: запятая `,`
- Дополнительные колонки допускаются (forward-compatible) и игнорируются потребителями.

## Schema (v1)

Обязательные колонки:

- `id` (string, required): идентификатор молекулы (уникален в файле).
- `group_id` (string, required): идентификатор группы изомеров; метрики качества считаются внутри каждой группы.
- `smiles` (string, required): SMILES (вход для построения графа).
- `energy_rel_kcalmol` (float, required): энергия изомера относительно минимума в группе, в kcal/mol (меньше = лучше).
- `truth_source` (string, required): источник данных; для v1 должен быть `spice2_0_1_isomers_v2`.
- `truth_version` (string, required): версия контракта; для v1 должна быть `isomer_truth.v1`.

Рекомендуемые колонки (если доступны в raw):
- `formula`, `heavy_atoms`
- `formation_energy_hartree`, `formation_energy_kcalmol`
- `source` (строка-источник)

## Инварианты / ошибки

- Нет обязательной колонки → **ERROR**
- `truth_version != "isomer_truth.v1"` → **ERROR**
- `truth_source != "spice2_0_1_isomers_v2"` → **ERROR**
- Дубликаты `id` → **ERROR**
- Пустой `group_id` → **ERROR**
- Пустой `smiles` → **ERROR**
- `energy_rel_kcalmol` не парсится как float → **ERROR**
- В любой группе `group_id` меньше 2 строк → **ERROR**

## Минимальный пример

```csv
id,group_id,smiles,energy_rel_kcalmol,truth_source,truth_version
mol_001,C7H8,c1ccccc1C,0.0,spice2_0_1_isomers_v2,isomer_truth.v1
mol_002,C7H8,Cc1ccccc1,0.1,spice2_0_1_isomers_v2,isomer_truth.v1
```
