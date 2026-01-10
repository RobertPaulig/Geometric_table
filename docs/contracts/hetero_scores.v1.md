# Контракт `hetero_scores.v1` (external scores)

Назначение: формат входного `scores_input` для режима `score_mode=external_scores` (HETERO-2).

Источник истины по policy: `docs/10_contracts_policy.md`.

## Файл

- Тип: JSON
- Кодировка: UTF-8

## Schema (v1)

Top-level (object):

- `schema_version` (string, **required**): строго `"hetero_scores.v1"`.
- `original` (object, **required**):
  - `score` (number, required)
  - `weight` (number, required)
- `decoys` (object, **required**): mapping `{ "<decoy_hash>": { "score": number, "weight": number } }`

### Правила

- `schema_version != "hetero_scores.v1"` → **ERROR** (run должен падать без silent fallback).
- Для `decoys` ключи — это **хэши декоев** (`decoy_hash`) из пайплайна.
- Допускаются дополнительные (unknown) ключи на верхнем уровне и внутри `original`/`decoys[*]`: они игнорируются (forward-compatible).
- Если для части декоев нет score в `decoys`:
  - строка не пропадает,
  - это отражается warning’ом + счетчиком missing coverage.
- Если нет score **для всех** декоев — итог должен быть `SKIP` с причиной/предупреждением `skip:missing_scores_for_all_decoys` (без “тишины”).

## Minimal example

```json
{
  "schema_version": "hetero_scores.v1",
  "original": {"score": 1.0, "weight": 1.0},
  "decoys": {
    "deadbeef": {"score": 0.1, "weight": 1.0}
  }
}
```

