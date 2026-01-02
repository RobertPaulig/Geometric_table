# Пример 02: external_scores

Цель: запуск pipeline с внешними score/weight, привязанными к hash decoys.

Шаг 1: получаем decoy hashes
```bash
hetero-pipeline --tree_input tests/data/hetero_tree_min.json --k 10 --seed 0 --timestamp 2026-01-02T00:00:00+00:00 --select_k 5 --selection maxmin --out examples/out/ext_pipeline.json
```

Шаг 2: формируем `hetero_scores.v1` (пример):
```json
{
  "schema_version": "hetero_scores.v1",
  "original": {"score": 1.0, "weight": 1.0},
  "decoys": {
    "<hash_1>": {"score": 0.12, "weight": 1.0},
    "<hash_2>": {"score": 0.08, "weight": 1.0}
  }
}
```

Шаг 3: запускаем pipeline с внешними scores
```bash
hetero-pipeline --tree_input tests/data/hetero_tree_min.json --k 10 --seed 0 --timestamp 2026-01-02T00:00:00+00:00 --select_k 5 --selection maxmin --score_mode external_scores --scores_input examples/out/scores.json --out examples/out/ext_scores_pipeline.json
```

Ожидаемые файлы:
- `examples/out/ext_scores_pipeline.json`
