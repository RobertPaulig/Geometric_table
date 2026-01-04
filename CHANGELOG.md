# CHANGELOG

## 0.2.4

- Evidence pack усилен: `manifest.json` включает `files` (path/size/sha256), пишется `checksums.sha256` на все артефакты; опциональный `--zip_pack` создаёт `evidence_pack.zip`.
- Контрактный тест на checksums/zip (`tests/test_hetero2_evidence_pack_checksums_contract.py`).

## 0.2.3

- Evidence Pack для `hetero2-batch`: `index.md` (кликабельные ссылки на отчёты/assets/pipeline) и `manifest.json` (provenance: tool/git/python/rdkit версии, seed_strategy, guardrails, score_mode).
- Batch по умолчанию генерирует index+manifest; можно отключить флагами `--no_index/--no_manifest`.
- Контрактный тест на evidence pack (`tests/test_hetero2_batch_index_manifest_contract.py`).

## 0.2.2

- Guardrails для HETERO-2 pipeline: SKIP с причинами (`invalid_smiles`, `too_large`, `disconnected`, `missing_scores_input`) при блокировке; schema_version сохранён.
- Batch: честный summary (каждая строка = OK/SKIP/ERROR), `report_path` пуст для SKIP/ERROR, добавлен `seed_used`, предупреждения детерминированы.
- CLI knobs: `score_mode` по умолчанию mock (external_scores требует scores_input), `--seed_strategy {global,per_row}` (per_row = stable_hash(id) XOR seed), `--guardrails_max_atoms`, `--guardrails_require_connected`.

## 0.2.1

- HETERO-2: batch runner (`hetero2-batch`) для CSV (id,smiles[,scores_input])
- Dockerfile + CI docker-smoke (`ci/docker`) с rdkit
- FIX-CI: docker smoke за флагом `HETERO2_DOCKER_SMOKE=1` для локальных pytest
- FIX-DOCKER-CI: writable volume в workflow, commit statuses видимы (`ci/test`, `ci/test-chem`, `ci/docker`)
- PHI остается R&D (не включено в продукт)

## 0.2.0

- HETERO-2 CLI (`hetero2-*`) и bump версии пакета
- Отчет v2 с картинками RDKit и таблицей hard negatives (assets git-ignored)
- Контрактные тесты HETERO-2 (CLI + demo)

## 0.1.1

- Интеграционные contract-тесты для examples (smoke + external_scores)
- integration contract tests: examples smoke CLI
- integration contract tests: examples external_scores
- docs: contracts policy (schema_version / breaking rules)

## 0.1.0

- HETERO-1A: audit/decoys/pipeline/report contracts
- SDK facade (hetero1a) + packaging + CI
