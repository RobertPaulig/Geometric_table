# Release Readiness Checklist (Field)

Цель: перед тем как пометить релиз как “golden” и/или запускать масштабные прогоны на сервере,
мы обязаны доказать: (a) детерминизм/воспроизводимость, (b) целостность, (c) понятный статус по всем строкам.

## A. Code gates (обязательно до релиза)
- [ ] На целевом SHA зелёные статусы:
  - ci/test
  - ci/test-chem
  - ci/docker
- [ ] SHA зафиксирован (полный 40-символьный), и будет записан в release notes + registry.

## B. Build (локально/на ноутбуке)
- [ ] Команда запуска зафиксирована (одна строка), пример:

  hetero2-batch --input <csv> --out_dir <out_dir> --artifacts light --score_mode mock --k_decoys 2 --workers 6 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack

- [ ] Выходная папка содержит минимальный набор:
  - summary.csv
  - metrics.json
  - index.md
  - manifest.json
  - checksums.sha256
  - evidence_pack.zip

## C. Integrity (без распаковки 10k файлов)
Windows / PowerShell:
- [ ] python -m zipfile -t <out_dir>\evidence_pack.zip  → OK
- [ ] python -m zipfile -l <out_dir>\evidence_pack.zip | Select-String -Pattern "manifest.json|checksums.sha256|metrics.json|index.md|summary.csv"
- [ ] Get-FileHash <out_dir>\evidence_pack.zip -Algorithm SHA256  → записать SHA256

Дополнительно (рекомендуется): проверить checksums.sha256 внутри ZIP
(вытащить только manifest/checksums/metrics/summary/index и сверить хэши).

## D. Quality gates (что считаем “валидно”)
- [ ] metrics.json: counts.ERROR == 0
- [ ] SKIP объяснимы (ожидаемые причины: invalid_smiles / too_many_atoms / disconnected / missing_smiles / no_decoys_generated и т.п.)
- [ ] Нет “тихих провалов” (каждая строка входа отражена в summary.csv как OK/SKIP/ERROR)

## E. Determinism (минимальная проверка)
На одном и том же SHA и одном и том же input:
- [ ] Два запуска с одинаковым seed_strategy/seed дают одинаковые counts и top_reasons в metrics.json.
- [ ] (желательно) summary.csv идентичен по counts/reasons (можно сравнить агрегаты).

## F. Publish (релиз)
Правило: release = “точка истины”. После публикации мы обязаны уметь скачать и повторить проверку.

- [ ] Release tag назван однозначно (например stress-10k-YYYY-MM-DD).
- [ ] Release target_commitish == целевой SHA.
- [ ] evidence_pack.zip загружен в Assets.
- [ ] После публикации: скачали asset обратно и SHA256 совпал с записанным.

Важно:
- Не перезатирать “golden” ассеты молча. Если нашли ошибку — новый tag/release (или явно помечаем old как superseded).

## G. Registry (обязательно)
- [ ] В docs/artefacts_registry.md добавлена запись:
  - дата/тег
  - source commit SHA
  - URL на asset (прямая ссылка)
  - SHA256(evidence_pack.zip)
  - команда запуска
  - outcome (OK/SKIP/ERROR + top reasons)
- [ ] Запись в registry сделана отдельным коммитом и прошла ci gates.
