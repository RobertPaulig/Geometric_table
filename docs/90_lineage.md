# Lineage идей (`docs/90_lineage.md`)

Назначение: одно место “append-only”, где фиксируем **что изменилось, почему и чем это доказано** (эксперимент + тест + ссылка на коммит).

## Шаблон записи (copy/paste)

### YYYY-MM-DD — <краткое название>

- **Гипотеза:** ...
- **Метод:** ...
- **Эксперимент:** команды запуска + входные данные
- **Результат:** ключевые числа / графики / пути к CSV/PNG
- **Добавлен тест:** `tests/...`
- **Обновлены доки:** `docs/...`
- **Коммит(ы):** `git_sha`
- **Дальше:** ...

---

### 2026-01-02 — HETERO-1A audit: стабильный CLI + JSON-контракт

- **Гипотеза:** можно сделать один публичный “аудит”-entrypoint, который всегда выдаёт стабильный JSON (tie-aware AUC + neg-controls gate) и тем самым превращает HETERO-линию в проверяемый контракт.
- **Метод:** отдельный CLI `python -m analysis.chem.audit` + минимальная фиксированная JSON-схема; детерминизм закреплён тестом.
- **Эксперимент:** `python -m analysis.chem.audit --input tests/data/hetero_audit_min.json --seed 0 --out audit.json`
- **Результат:** JSON с ключами `version,dataset_id,n_pos,n_neg,auc_tie_aware,neg_controls,run` (см. `docs/README.md`).
- **Добавлен тест:** `tests/test_hetero_audit_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `310978c`, `41089df`
- **Дальше:** привязать audit к реальному мини-датасету из `analysis/chem` (или curated baseline) и формализовать “verdict invariants” (ε для float).

### 2026-01-02 — Mendoza-step: neg-controls exact для малых N

- **Гипотеза:** для малых выборок (N≤10) neg-controls должны быть детерминированными без Монте-Карло, чтобы гейт не “гулял” от сидов/порядка.
- **Метод:** в `analysis.chem.audit` считать `perm_q` точным перебором всех разметок при фиксированных `n_pos/n_neg`; `rand_q` приравнять `null_q`; добавить `method`/`reps_used`.
- **Эксперимент:** `python -m analysis.chem.audit --input tests/data/hetero_audit_min.json --seed 0 --neg_control_reps 999 --out audit.json`
- **Результат:** на `hetero_audit_min_v1` выставляется `method=exact`, `reps_used=0`, и `rand_q == null_q`.
- **Добавлен тест:** `tests/test_hetero_audit_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `4247e78`, `676ec59`
- **Дальше:** расширить exact-ветку до N≤12 (если нужно) и/или добавить ε-инварианты для float-полей в контракте.

### 2026-01-02 — Warning: веса vs unweighted null_q

- **Гипотеза:** при наличии весов в данных нужно явно сигнализировать, что `null_q` сейчас считается без учёта весов.
- **Метод:** добавить `schema_version` и `warnings` в JSON, зафиксировать `null_q_method`, и предупредить при неединичных весах.
- **Эксперимент:** `python -m analysis.chem.audit --input tests/data/hetero_audit_min.json --seed 0 --out audit.json`
- **Результат:** `warnings` включает `weights_used_in_auc_but_null_q_is_unweighted` на weighted-датасете.
- **Добавлен тест:** `tests/test_hetero_audit_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `315b047`, `ada4bcb`
- **Дальше:** добавить weighted-null и сверку с известными значениями.

### 2026-01-02 — Decoys v1: tree-chem без RDKit

- **Гипотеза:** можно генерировать структурные декои в tree-химии без RDKit, сохраняя degree-sequence и типы узлов.
- **Метод:** Prüfer-последовательности с фиксированными кратностями (`deg_i-1`), дедупликация по хэшу рёбер.
- **Эксперимент:** `python -m analysis.chem.decoys --input tests/data/hetero_tree_min.json --out decoys.json`
- **Результат:** детерминированный JSON с `k_generated == k_requested`, без нарушений `max_valence`.
- **Добавлен тест:** `tests/test_hetero_decoys_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `73e32dd`, `67191a3`
- **Дальше:** добавить базовую метрику разнообразия/покрытия и минимальный фильтр по структурным инвариантам.

### 2026-01-02 — Decoys v1.1: diversity/coverage + фильтр

- **Гипотеза:** для decoys нужен минимальный контроль разнообразия и дистанции от оригинала, иначе набор может быть тривиальным.
- **Метод:** метрика `dist(A,B)=|E_A Δ E_B|/(2*(n-1))`, фильтры `min_dist_to_original` и `min_pair_dist`, счетчики отброшенных кандидатов.
- **Эксперимент:** `python -m analysis.chem.decoys --input tests/data/hetero_tree_min.json --min_pair_dist 0.5 --min_dist_to_original 0.5 --max_attempts 50 --out decoys.json`
- **Результат:** JSON содержит `metrics` и `filter`; при жёстких порогах возможен warning `could_not_generate_k_decoys_under_constraints`.
- **Добавлен тест:** `tests/test_hetero_decoys_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `0e70794`, `06aaec6`
- **Дальше:** добавить рекомендации (recommendations) по подбору порогов и минимальный diversity-score.

### 2026-01-02 — Pipeline v1: decoys -> selection -> audit

- **Гипотеза:** можно склеить контуры (decoys + coverage selection + audit) в один воспроизводимый JSON-артефакт.
- **Метод:** один CLI `analysis.chem.pipeline`, два режима селекции (`firstk`, `maxmin`), детерминированный toy-score из `dist_to_original`.
- **Эксперимент:** `python -m analysis.chem.pipeline --tree_input tests/data/hetero_tree_min.json --k 10 --seed 0 --timestamp 2026-01-02T00:00:00+00:00 --select_k 5 --selection maxmin --out pipeline.json`
- **Результат:** JSON `hetero_pipeline.v1` содержит `decoys`, `selection`, `audit` в одном отчёте; `maxmin` не хуже `firstk` по `min_pairwise_dist`.
- **Добавлен тест:** `tests/test_hetero_pipeline_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `7ffe729`, `9c9506d`, `92f4fd7`
- **Дальше:** вынести toy-score в `score_mode` и добавить экспорт отчёта (CSV/MD) для пользователя.

### 2026-01-02 — Pipeline v1.1: score_mode + warnings + stable selection IDs

- **Гипотеза:** pipeline должен явно маркировать toy-скоринг, агрегировать warnings и давать устойчивые идентификаторы выбора.
- **Метод:** добавить `score_mode/score_definition`, агрегировать `warnings` из всех блоков, хранить `selected_hashes` и `index_base`.
- **Эксперимент:** `python -m analysis.chem.pipeline --tree_input tests/data/hetero_tree_min.json --k 10 --seed 0 --timestamp 2026-01-02T00:00:00+00:00 --select_k 5 --selection maxmin --out pipeline.json`
- **Результат:** JSON содержит агрегированный `warnings` и стабильные `selected_hashes`.
- **Добавлен тест:** `tests/test_hetero_pipeline_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `482f6ec`, `84a3d11`, `aeafc54`
- **Дальше:** добавить `report` CLI (md/csv) поверх pipeline JSON.

### 2026-01-02 — Report v1: pipeline.json -> report.md + decoys.csv

- **Гипотеза:** продуктовый отчёт должен быть отделён от pipeline и воспроизводим из одного JSON.
- **Метод:** отдельный CLI `analysis.chem.report`, сборка summary + таблицы decoys.
- **Эксперимент:** `python -m analysis.chem.report --input pipeline.json --out_dir . --stem example`
- **Результат:** стабильные `example.report.md` и `example.decoys.csv`.
- **Добавлен тест:** `tests/test_hetero_report_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `9b8fce1`, `d81b0a9`, `e1f1bf9`
- **Дальше:** добавить `score_mode=external_scores` и формат отчёта для пользовательского скоринга.

### 2026-01-02 — Pipeline v1.2: external_scores

- **Гипотеза:** pipeline должен принимать внешние скоры по hash без привязки к индексам decoys.
- **Метод:** новый режим `score_mode=external_scores` + формат `hetero_scores.v1`; неполные скоры дают warning.
- **Эксперимент:** `python -m analysis.chem.pipeline --tree_input tests/data/hetero_tree_min.json --score_mode external_scores --scores_input scores.json --out pipeline.json`
- **Результат:** `audit` строится из внешних score/weight; `warnings` фиксирует пропуски.
- **Добавлен тест:** `tests/test_hetero_pipeline_cli.py`
- **Обновлены доки:** `docs/README.md`
- **Коммит(ы):** `21c2c60`, `60c0fc4`, `d4847ad`
- **Дальше:** вынести score_mode в отдельный блок и добавить SDK facade.

### 2026-01-02 — Examples as contract tests (v0.1.1)

- **Гипотеза:** examples должны быть воспроизводимыми контрактами, а не только документацией.
- **Метод:** интеграционные тесты для smoke (CLI) и external_scores (SDK).
- **Эксперимент:** запуск `hetero-pipeline/hetero-report` в tmp_path и `run_pipeline(score_mode=external_scores)`.
- **Результат:** проверены артефакты и warnings без записи в repo.
- **Добавлен тест:** `tests/test_examples_smoke_cli.py`, `tests/test_examples_external_scores.py`
- **Обновлены доки:** `CHANGELOG.md`
- **Коммит(ы):** `8c2f5df`, `639cd7a`, `d16e4cf`
- **Дальше:** добавить public demo command в README и закрепить schema_version policy.

### 2026-01-03 — HETERO-2: Spectral Cycle Engine (заготовка)

- **Гипотеза:** ...
- **Метод:** ...
- **Эксперимент:** ...
- **Результат:** ...
- **Добавлен тест:** ...
- **Обновлены доки:** ...
- **Коммит(ы):** ...
- **Дальше:** ...

### 2026-01-03 — HETERO-2 Sprint-1: ChemGraph + Spectral FP

- **Гипотеза:** циклы в молекулах дают отличимый спектральный отпечаток, пригодный для hard negatives.
- **Метод:** RDKit ChemGraph (adjacency/laplacian/physchem) + спектр Лапласиана и LDOS-вектор.
- **Эксперимент:** aspirin SMILES -> граф -> спектр; проверка инвариантности к перестановке узлов.
- **Результат:** детерминированный спектральный fp, инвариантный к перестановке.
- **Добавлен тест:** `tests/test_hetero2_chemgraph.py`, `tests/test_hetero2_spectral.py`
- **Обновлены доки:** `docs/04_backlog.md`, `docs/05_decision_log.md`, `docs/90_lineage.md`
- **Коммит(ы):** `7c6ee87`, `88870ae`
- **Дальше:** decoys_rewire (double-edge-swap) + RDKit sanitize + pipeline v2.

### 2026-01-03 — HETERO-2 Sprint-2: Cycle decoys + pipeline v2

- **Гипотеза:** degree-preserving rewiring + RDKit sanitize даёт hard negatives с кольцами, пригодные для аудита.
- **Метод:** double-edge swap на разрешённых связях (non-ring, non-aromatic), RDKit sanitize + канонизация SMILES; pipeline v2 на SMILES с внешними/мок-скорами; отчёт с Rings/PhysChem/Hardness.
- **Эксперимент:** Aspirin SMILES -> rewire decoys (k=5/20) -> external_scores (mock) -> audit -> report.
- **Результат:** детерминированный pipeline v2, отчёт содержит кольца, physchem, tanimoto-hardness, verdict из аудита.
- **Добавлен тест:** `tests/test_hetero2_decoys_rewire.py`, `tests/test_hetero2_pipeline_report.py`
- **Обновлены доки:** `docs/90_lineage.md`
- **Коммит(ы):** `2679e83`, `e79c503`, `7b8f808`
- **Дальше:** demo_aspirin_v2 (красная кнопка), cycle selection/hardness metrics, интеграция с decoys_v2 в pipeline/report.

### 2026-01-03 — HETERO-2 Sprint-3: Productize WOW (CLI + images + contracts)

- **Гипотеза:** публичный CLI + визуальный отчет с картинками повышает доверие к HETERO-2 как продукту.
- **Метод:** CLI entrypoints `hetero2-*`, отчёт v2 с RDKit-картинками и таблицами hard negatives, контрактные интеграционные тесты (demo+CLI).
- **Эксперимент:** Aspirin через CLI/demo -> pipeline v2 -> report v2 (md + assets) в tmp_path.
- **Результат:** детерминированный отчет с картинками, таблицей decoys, предупреждениями; CLI работает из упаковки.
- **Добавлен тест:** `tests/test_hetero2_cli_entrypoints.py`, `tests/test_hetero2_demo_aspirin_contract.py`
- **Обновлены доки:** `README.md`, `CHANGELOG.md`, `docs/90_lineage.md`
- **Коммит(ы):** `4f3af4a`, `0fcecdd`, `db1347f`
- **Дальше:** selection/hardness метрики + публичный отчетный формат для клиента.

### 2026-01-04 — HETERO-2 PHI experiment (Ray Harmony)

- **Гипотеза:** спектральная сумма (PHI) может быть маркером hardness для decoys с кольцами.
- **Метод:** RayAuditor (divisor_sum_profile), phi_from_eigs/phi_from_smiles на спектре Лапласиана, первые числа на аспирине + decoys.
- **Эксперимент:** aspirin -> phi_original vs phi_decoys (mock decoys), self-check sigma sum=8299.
- **Результат:** экспериментальный скрипт и тесты, продукт не меняется. Bench (scale=300, seed=0, k≈30): aspirin (delta=-71, count=2), acetaminophen (decoys=0), ibuprofen (delta=-1477, count=9), naproxen (delta=+1254, count=3), salicylic_acid (decoys=0); сигнал неустойчив.
- **Добавлен тест:** `tests/experimental/test_ray_constant.py`, `tests/experimental/test_ray_phi_determinism.py`, `tests/experimental/test_ray_phi_runs_on_aspirin.py`
- **Обновлены доки:** `docs/04_backlog.md`, `docs/05_decision_log.md`, `docs/90_lineage.md`
- **Коммит(ы):** `b92b70f`, `228873f`, `183bf42`, `228873f`
- **Дальше:** собрать PHI по ≥3 молекулам с кольцами, проверить критерий принятия перед включением в отчет; улучшить генерацию decoys для сложных кольцевых молекул.

### 2026-01-04 — HETERO-2 Sprint-4: Batch + Docker (industrial delivery)

- **Гипотеза:** корпоративному пользователю нужен batch/CLI/Docker контур с гарантированным RDKit и отчетами.
- **Метод:** batch CLI (`hetero2-batch`), Dockerfile с rdkit, CI docker-smoke + commit statuses, отчеты/asset per molecule.
- **Эксперимент:** CSV (id,smiles) -> batch -> summary.csv + reports; docker run demo -> report + assets.
- **Результат:** детерминированные артефакты per molecule, статусы `ci/docker`/`ci/test`/`ci/test-chem`.
- **Добавлен тест:** `tests/test_hetero2_batch_contract.py`, `tests/test_docker_smoke_contract.py`
- **Обновлены доки:** `README.md`
- **Коммит(ы):** `183bf42`, `ebfa640` (добавить коммиты Sprint-4 после пуша)
- **Дальше:** улучшить batch (параллелизм/пер-строковый seed), расширить docker smoke (external_scores), не включать PHI в продукт до принятия.

### 2026-01-04 - Release v0.2.1 (batch+docker delivery)

- **Гипотеза:** релиз 0.2.1 фиксирует batch/Docker/CI гейты без смены схем.
- **Метод:** bump версии, changelog, lineage с ссылками на CI fixes.
- **Эксперимент:** уже выполненный docker-smoke с writable volume.
- **Результат:** статусы ci/test, ci/test-chem, ci/docker зелёные; batch/Docker доступны из релиза.
- **Добавлен тест:** (покрытие осталось прежним)
- **Обновлены доки:** `CHANGELOG.md`, `docs/90_lineage.md`
- **Коммит(ы):** `1a257e3`, `fb5a39b`, `5df0a1e`, `c13cfff`, `c7cc918`
- **Дальше:** guardrails/новые фичи отдельно от релиза.

### 2026-01-05 - HETERO-2 Sprint-4b: Guardrails before Scale

- **Идея:** не терять строки batch и блокировать тяжёлые расчёты для очевидно плохих входов.
- **Эксперимент:** preflight guardrails (invalid/too_large/disconnected) вызываются в начале `run_pipeline_v2`; для SKIP возвращается валидный payload с neg-controls verdict `SKIP`.
- **Тест:** `tests/test_hetero2_guardrails_contract.py` (invalid/too_large/disconnected), обновлён `tests/test_hetero2_batch_contract.py` (пустой SMILES -> SKIP/ERROR в summary).
- **Док:** `README.md`, `docs/05_decision_log.md`, `docs/90_lineage.md`
- **Коммит(ы):** (текущий PR)
- **Дальше:** таймауты/воркеры для batch (следующий спринт), расширить policy на плотный граф, добавить CLI-флаги для порогов.

### 2026-01-05 - HETERO-2 Sprint-5: Batch control knobs

- **Идея:** управляемый batch перед масштабом: явные режимы seed и guardrails без скрытых fallback'ов.
- **Эксперимент:** `--seed_strategy {global,per_row}` с детерминированным `stable_hash(id)=crc32`; per_row использует `seed XOR stable_hash(id)` и пишет `seed_used` в summary. Default `score_mode=mock` (external_scores без файла -> SKIP). Guardrails пороги доступны через CLI.
- **Тест:** `tests/test_hetero2_batch_contract.py::test_batch_seed_strategy_per_row`, `tests/test_hetero2_guardrails_contract.py::test_pipeline_default_mock_and_guardrail_limit`.
- **Коммит(ы):** ceac0b5, (текущий PR)
- **Дальше:** таймауты/воркеры для batch, расширить доки по seed-replay.
