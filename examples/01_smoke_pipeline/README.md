# Пример 01: smoke pipeline

Цель: получить `pipeline.json` и отчёт (`.report.md` + `.decoys.csv`) на минимальном дереве.

Команды:
```bash
hetero-pipeline --tree_input tests/data/hetero_tree_min.json --k 10 --seed 0 --timestamp 2026-01-02T00:00:00+00:00 --select_k 5 --selection maxmin --out examples/out/smoke_pipeline.json
hetero-report --input examples/out/smoke_pipeline.json --out_dir examples/out --stem smoke
```

Ожидаемые файлы:
- `examples/out/smoke_pipeline.json`
- `examples/out/smoke.report.md`
- `examples/out/smoke.decoys.csv`
