# CHANGELOG

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
