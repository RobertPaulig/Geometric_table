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
- **Коммит(ы):** ceac0b5, a219cc5
- **Дальше:** таймауты/воркеры для batch, расширить доки по seed-replay.

### 2026-01-05 - HETERO-2 Sprint-6: Evidence Pack (index + manifest)

- **Идея:** отдавать клиенту готовый evidence-pack (summary + кликабельный index + manifest с provenance).
- **Эксперимент:** `hetero2-batch` всегда пишет `index.md` (таблица с ссылками на report/assets/pipeline) и `manifest.json` (tool/git/python/rdkit версии, seed_strategy, guardrails, score_mode); флаги `--no_index/--no_manifest` для отключения.
- **Тест:** `tests/test_hetero2_batch_index_manifest_contract.py` (проверка наличия index/manifest и ключевых полей).
- **Коммит(ы):** 0743dc1, c1b20eb, 0aafa62
- **Дальше:** Stress harness (timeout/worker), html/pdf отчёты (опционально).

### 2026-01-05 - HETERO-2 Sprint-7: Evidence Pack Hardening

- **Идея:** сделать evidence pack самопроверяемым и переносимым (хэши + опциональный zip).
- **Эксперимент:** `checksums.sha256` на все артефакты; `manifest.json["files"]` с path/size/sha256; флаг `--zip_pack` для `evidence_pack.zip`.
- **Тест:** `tests/test_hetero2_evidence_pack_checksums_contract.py` (наличие checksums, manifest.files, zip содержит index/manifest).
- **Коммит(ы):** 278da6d, c4c240e, 555af7d
- **Дальше:** Stress harness (timeout/worker), pdf/html отчёты.

### 2026-01-05 - HETERO-2 Sprint-8: Stress Harness (resume/metrics)

- **Идея:** надёжный batch на 100k+: параллельность, resume, таймауты, метрики.
- **Эксперимент:** streaming summary с `flush+fsync`, `multiprocessing.Pool(maxtasksperchild=100)`, `resume/overwrite`, `metrics.json` (counts/top_reasons/runtime/config) интегрирован в manifest/checksums, `index.md` сортируется по id.
- **Тест:** `tests/test_hetero2_batch_resume_contract.py`, `tests/test_hetero2_metrics_contract.py`.
- **Коммит(ы):** b2279f5, 6d4d93d, 39a23dc (релиз v0.2.5)
- **Дальше:** Gate-Stress 10k–50k; Sprint-9 external scores.

### 2026-01-07 - CI permissions + publish asset path + stress-10k-2026-01-07 release

- **Links:** workflow fix `d70394661474acd7c1b74d581471fa7fb10bb263`, release `https://github.com/RobertPaulig/Geometric_table/releases/tag/stress-10k-2026-01-07`, asset `https://github.com/RobertPaulig/Geometric_table/releases/download/stress-10k-2026-01-07/evidence_pack.zip`
- **SHA256:** `DF8EF1412990461CD52FFE35019E8F8BA3A40A7BDEACBE3AB1EEF00191D3AC35`

### 2026-01-07 - Spectral stability metrics (experimental)

- **Scope:** added spectral_gap, spectral_entropy, spectral_entropy_norm (no gating).
- **Examples (SMILES -> gap, entropy, entropy_norm):**
  - CC -> 2.0, -0.0, 0.0
  - CCC -> 1.0, 0.56233514, 0.81127812
  - c1ccccc1 -> 1.0, 1.47350239, 0.91553851

### 2026-01-07 - Sprint-9: external scores contract + provenance

- **What:** enforce `hetero_scores.v1` schema for external scores; record provenance (scores_input_id, scores_input_sha256, scores_schema_version).
- **Refs:** PR #3, merge `1cc33ffa7c2dc36a64df195e452fd579bf055328`.

### 2026-01-08 - Sprint-10: pilot quickstart + scores coverage

- **What:** pilot quickstart (Docker + minimal inputs), scores coverage in metrics/index.
- **Refs:** PR (TBD), merge (TBD).

### 2026-01-09 - Pilot release r2 (recommended demo)

- **Release:** https://github.com/RobertPaulig/Geometric_table/releases/tag/pilot-2026-01-08-r2
- **Asset:** https://github.com/RobertPaulig/Geometric_table/releases/download/pilot-2026-01-08-r2/pilot_evidence_pack.zip
- **SHA256:** BB564070C2B02087B589A511FB12AD4DEDC13C1EE06A58BC793DD5CF51D3B2A8
- **Merge SHA:** 84aa13aa352eb4dbd63c4c3f36f00ba81b307470

### 2026-01-10 — Roadmap: VALUE-TRACK (Pfizer-ready facts) + SaaS milestones
- Added `docs/ROADMAP.md` with VALUE-TRACK (VALUE>SAAS: доказательство пользы через релизные артефакты + SHA256 + registry).
- Linked ROADMAP from `docs/99_index.md` (REF-ROADMAP).
- Note: PR CI contexts (ci/test, ci/test-chem, ci/docker) are posted on GitHub merge-ref SHA for pull_request runs; merge policy follows the tested SHA for PR gating.

### 2026-01-10 — Freeze `hetero_scores.v1` (contract + compat tests + pointers)
- PR: https://github.com/RobertPaulig/Geometric_table/pull/20
- Head SHA: 7aa0d8f40d0991e2a78d5cf67b20b012ba974fcc
- Added contract doc `docs/contracts/hetero_scores.v1.md` + backward-compat tests + pointers in `CONTEXT.md` and `docs/99_index.md`.

### 2026-01-10 - VALUE-M1 ring-suite evidence pack (release asset + SHA256 + registry)
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-ring-suite-2026-01-10
- Source commit: ba63273ea6f9b3f8c87cf0791b372fb7fc5d2871
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-ring-suite-2026-01-10/value_ring_suite_evidence_pack.zip
- SHA256(value_ring_suite_evidence_pack.zip): 912071F3927D733FF4F5EDA1AB5A3158F83D18EBA4E99B1C2CC174FD6EE08274
- Outcome: OK=60, SKIP=140, ERROR=0; top_skip_reasons: no_decoys_generated: 140; share_rows_with_n_decoys_gt_0: 30.0%
- PRs: https://github.com/RobertPaulig/Geometric_table/pull/21, https://github.com/RobertPaulig/Geometric_table/pull/22

### 2026-01-11 - DOCS-STRICT-1: индекс как обязательный + comms protocol + guardrail
- What: added `docs/20_comms_protocol.md`; made `docs/99_index.md` cover обязательные документы; aligned `CONTEXT.md` load order to real источники; restored Tom I truth as `docs/name3.md`; added docs-index integrity test.
- PR: https://github.com/RobertPaulig/Geometric_table/pull/28

### 2026-01-11 - VALUE-M2 known bad/good evidence packs (release assets + SHA256 + registry)
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-known-bad-good-2026-01-11
- Source commit: 706aaaf32c52e2df9b79bc611421d57af3cbecb4
- Assets:
  - BAD-constant: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-11/value_known_bad_good_BAD-constant_evidence_pack.zip
    - SHA256: 043940CC6FE275D7393BD9F1AAB8A6CB8033890430F5B495782D226FB97CD5DF
  - BAD-random: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-11/value_known_bad_good_BAD-random_evidence_pack.zip
    - SHA256: 38393053ABDF710D3AB4BAE68C7EA1A55547A8F984B0600E17411953B65294C1
  - GOOD-synthetic: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-11/value_known_bad_good_GOOD-synthetic_evidence_pack.zip
    - SHA256: DF27F9CA9CA4A74089EF1966D9591FEDDE7F9C452CD62BDE94F4D384F09F27B3
- Run: https://github.com/RobertPaulig/Geometric_table/actions/runs/20895097355
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/30 (merge: d477d56db5facd9aab2ba8853c8d4d8c1f096a93)
- Separation facts (computed on status==OK rows only; no auto-threshold gating): Delta_median_slack(GOOD - BAD-constant)=0.000000; Delta_PASS_rate(GOOD - BAD-constant)=0.000000; Delta_median_slack(GOOD - BAD-random)=0.000000; Delta_PASS_rate(GOOD - BAD-random)=0.000000

### 2026-01-11 - VALUE-M2 known bad/good evidence packs (r2; separation restored via hetero_audit.v2)
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-known-bad-good-2026-01-11-r2
- Source commit: 8110a1b78f2a67d684d64343c996e64d218f99e4
- Assets:
  - BAD-constant: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-11-r2/value_known_bad_good_BAD-constant_evidence_pack.zip
    - SHA256: 5B117E204E9E98128EE4C6BEBC609D4282862DBF3BEB935FF432076809F0046A
  - BAD-random: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-11-r2/value_known_bad_good_BAD-random_evidence_pack.zip
    - SHA256: E4255428FC9EEE082D54B04D6A01E7EE98F5F59717CBA259590D6457F1064916
  - GOOD-synthetic: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-11-r2/value_known_bad_good_GOOD-synthetic_evidence_pack.zip
    - SHA256: 228E5B0515316831DE5D208BEE624240973215BDAC85236C1583DEC1B7EA6B5C
- Run: https://github.com/RobertPaulig/Geometric_table/actions/runs/20897298904
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/34 (merge: 1e8079ad7ca1a71f00b186574865ecac853d68c0)
- Separation facts (computed on status==OK rows only; no auto-threshold gating): Delta_median_slack(GOOD - BAD-constant)=0.500000; Delta_PASS_rate(GOOD - BAD-constant)=1.000000; Delta_median_slack(GOOD - BAD-random)=0.000000; Delta_PASS_rate(GOOD - BAD-random)=0.333333

### 2026-01-11 - VALUE-M4: standard claims (doc + guardrail tests)
- What: added `docs/standard_claims.md` (что гарантируем/не гарантируем); added guardrail tests for frozen `schema_version` and evidence-pack required files; added `REF-STANDARD-CLAIMS` in `docs/99_index.md`; updated `docs/10_contracts_policy.md`.
- PR: https://github.com/RobertPaulig/Geometric_table/pull/37
- Head SHA: 4b41e6ae7ff9fcb7cfdba511b8f20169f80b0c01

### 2026-01-11 - VALUE-M3: acceptance criteria contract (proxy)
- What: added `docs/value_m3_acceptance_criteria.md`; added `REF-VALUE-M3-ACCEPT` in `docs/99_index.md`; marked VALUE-M3 as in-progress in `docs/ROADMAP.md`.
- PR: https://github.com/RobertPaulig/Geometric_table/pull/38
- Head SHA: 568c1dc3e7f4d54a08aa5ec2b7a4f0b0f7b2f6c6

### 2026-01-12 - VALUE-M3: fix publish-value-customer-proxy workflow facts step quoting
- What: fix SyntaxError in facts step to unblock publish run for VALUE-M3.
- PR: https://github.com/RobertPaulig/Geometric_table/pull/40
- Merge commit: 2554cfe4d0d342656a080d6a0e0209ffd6cd757d
- CI run (main): https://github.com/RobertPaulig/Geometric_table/actions/runs/20913938972

### 2026-01-12 - VALUE-M3: customer proxy evidence pack (release asset + SHA256 + registry)
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-customer-proxy-2026-01-12
- Source commit: 6951804e7892b208a38b877e102df643a7d8e30d
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-customer-proxy-2026-01-12/value_customer_proxy_evidence_pack.zip
- SHA256(value_customer_proxy_evidence_pack.zip): FE7AA762DCC6D512771DA40E90EB58557B32D6A3641033C65546D0553C16B225
- Acceptance criteria (contract): docs/value_m3_acceptance_criteria.md
- Run: https://github.com/RobertPaulig/Geometric_table/actions/runs/20914040368
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/41 (merge: b3e2602bc08a1e9164ce6cad9fb322f65585db2c)
- Facts (from summary.csv; computed on status==OK rows only):
  - rows_total: 200
  - rows_ok: 60
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons: no_decoys_generated: 140
  - share_rows_with_n_decoys_gt_0: 0.300 (30.0%)
  - median_slack: 0.000000
  - pass_rate: 0.666667

### 2026-01-12 - CI-AUTO-PR-1: auto CI for automation PRs (no allow-empty commits)
- What: `pytest.yml` supports `workflow_dispatch`; publish workflows that create automation PRs now dispatch `pytest.yml` on the automation branch and wait for completion, so `ci/test|ci/test-chem|ci/docker` appear without manual empty commits.
- PRs: https://github.com/RobertPaulig/Geometric_table/pull/49 (initial), https://github.com/RobertPaulig/Geometric_table/pull/51 (fix run-url logging).
- Live proof: publish run https://github.com/RobertPaulig/Geometric_table/actions/runs/20937794112 created automation PR https://github.com/RobertPaulig/Geometric_table/pull/52 and triggered pytest https://github.com/RobertPaulig/Geometric_table/actions/runs/20937822351 on tested SHA 6f8f505ef04de0b41172a95f2fc475ba3638072e.

### 2026-01-12 - VALUE-M3: customer proxy evidence pack (r2; rows_missing_scores_input gate)
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-customer-proxy-2026-01-12-r2
- Source commit: 2bd92404e139804cc3cc088766ede94106962ead
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-customer-proxy-2026-01-12-r2/value_customer_proxy_evidence_pack.zip
- SHA256(value_customer_proxy_evidence_pack.zip): C2A8350EFA0D8BEB957E65DE42C0591080085F614B10E255430774B463F67029
- Acceptance criteria (contract): docs/value_m3_acceptance_criteria.md
- Run: https://github.com/RobertPaulig/Geometric_table/actions/runs/20922529046
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/46 (merge: 5064d50cddc88622872a3faf67655dd395701342)
- Facts (from summary.csv; computed on status==OK rows only):
  - rows_total: 200
  - rows_ok: 60
  - scores_coverage.rows_missing_scores_input: 0
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons: no_decoys_generated: 140
  - share_rows_with_n_decoys_gt_0: 0.300 (30.0%)
  - median_slack: 0.000000
  - pass_rate: 0.666667

### 2026-01-13 - CORE-UTILITY-1: customer truth contract + cost&lift report (VALUE-M5 utility baseline)
- What: added `docs/contracts/customer_truth.v1.md`; added `scripts/cost_lift.py` (outputs `cost_lift_report.json` / schema `cost_lift.v1`); added guardrail test `tests/test_cost_lift_report_contract.py`; added refs in `docs/99_index.md`.
- PR: https://github.com/RobertPaulig/Geometric_table/pull/55
- Merge commit: 021eb815091e390b668622b3f4608e6e4ca0a047
- CI run (main): https://github.com/RobertPaulig/Geometric_table/actions/runs/20946389319

### 2026-01-13 - VALUE-M5 utility proxy evidence pack (proxy truth + cost&lift)
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-utility-proxy-2026-01-13
- Source commit: 97d1e2e24b31defded76bf74618409eb611d92bc
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/20952518490
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-proxy-2026-01-13/value_utility_proxy_evidence_pack.zip
- SHA256(value_utility_proxy_evidence_pack.zip): C1AFC8992DDB88B3528030395D8D1E69DB395C7EE89AA5B902EC300A761A3FD4
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/63 (merge: 6a8b38a1a03870f0266be7a6ba61d6d4d3f99fce)
- Facts:
  - rows_total: 200
  - rows_ok: 60
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons: no_decoys_generated: 140
  - truth_source: proxy_rule_v1 (deterministic, no-leakage)
  - utility (K_effective=60; skip_policy=unknown_bucket):
    - baseline_random_hit_rate: 0.333333 (ci: 0.216667..0.450000)
    - baseline_score_only_hit_rate: 0.333333 (ci: 0.216667..0.433333)
    - filtered_hit_rate: 0.500000 (ci: 0.350000..0.675000)
    - uplift_vs_random: 0.166667
    - uplift_vs_score_only: 0.166667

### 2026-01-14 - VALUE-M6: utility real-truth ingest evidence pack (external truth.csv + sha256)
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-utility-realtruth-2026-01-14-r1
- Source commit: 72720901439cc5f3e2b559f5e606568a8d40bece
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/20983871401
- truth.csv (customer_truth.v1):
  - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  - SHA256(truth.csv): 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  - truth_source: external (PASS=11, FAIL=189)
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-14-r1/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): 65A00E8879B9B03BF558F630C85EABFC0C285C1B8DF3635D231B4A90DD7D816B
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/71 (merge: bc7c45d771a16bc8b387e11bcdaac1c18fe79207)
- Facts:
  - rows_total: 200
  - rows_ok: 60
  - scores_coverage.rows_missing_scores_input: 0
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons: no_decoys_generated: 140
  - share_rows_with_n_decoys_gt_0: 0.300 (30.0%)
  - utility (skip_policy=unknown_bucket; K_requested=10000; K_effective=60; N_with_truth=60):
    - baseline_random_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - baseline_score_only_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - filtered_hit_rate: 0.075000 (ci: 0.000000..0.150000; k_effective=40)
    - uplift_vs_random: 0.008333
    - uplift_vs_score_only: 0.008333

### 2026-01-14 - VALUE-M7: real scores ingest evidence pack (external hetero_scores.v1 pinned by sha256)
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-utility-realtruth-2026-01-14-r2
- Source commit: 4b89a5a464bdc5e547649dd610ee8af24b250368
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/20997753050
- scores.json (hetero_scores.v1; external):
  - scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340466779
  - SHA256(scores.json): 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  - score_key: external_ci_rule_v1
- truth.csv (customer_truth.v1):
  - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  - SHA256(truth.csv): 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-14-r2/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): 18E54A8CDE6550DCE9E965711189331E8D0F05DA44C6A4CAB5A5A8FEED64D5B9
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/75 (merge: 43c56a13349452ccc0af80e0853f1d3abb7c55d1)
- Facts:
  - rows_total: 200
  - rows_ok: 60
  - scores_coverage.rows_missing_scores_input: 0
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons: no_decoys_generated: 140
  - share_rows_with_n_decoys_gt_0: 0.300 (30.0%)
  - utility (scores_source=external; skip_policy=unknown_bucket; K_requested=10000; K_effective=60; N_with_truth=60):
    - baseline_random_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - baseline_score_only_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - filtered_hit_rate: 0.066667 (ci: 0.016667..0.150000)
    - uplift_vs_random: 0.000000
    - uplift_vs_score_only: 0.000000

### 2026-01-14 - COVERAGE-DECoys-1: decoy fallback restored OK-coverage (utility realtruth r3)
- What: introduced `decoy_strategy.v1` with fallback (strict -> fallback) to reduce `SKIP=no_decoys_generated` and raise OK-coverage (PR #78).
- Decoy fallback PR: https://github.com/RobertPaulig/Geometric_table/pull/78 (merge: cdc081b4d18b3e3f1d63d7f6ac335e4fb0f8d437)
- Scores release (external, pinned): https://github.com/RobertPaulig/Geometric_table/releases/tag/scores-external-ci-rule-v2-2026-01-14
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-utility-realtruth-2026-01-14-r3
- Source commit: cdc081b4d18b3e3f1d63d7f6ac335e4fb0f8d437
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21004714316
- scores.json (hetero_scores.v1; external):
  - scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340539600
  - SHA256(scores.json): D22A19B51EAEBDE1A778B2FE69E10F9E78BA726F64CBF9A643ADD235D167D157
  - score_key: external_ci_rule_v2
- truth.csv (customer_truth.v1):
  - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  - SHA256(truth.csv): 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-14-r3/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): 704B1F82933799E65F5F33D982A0D3EEBC34DA06BE8001670500B869CE3C5A00
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/79 (merge: 12d6d719e7ea3f426dab072665c271f09f1853c9)
- Facts:
  - rows_total: 200
  - rows_ok: 200
  - scores_coverage.rows_missing_scores_input: 0
  - status_counts: OK=200, SKIP=0, ERROR=0
  - share_rows_with_n_decoys_gt_0: 1.000 (100.0%)
  - utility (scores_source=external; skip_policy=unknown_bucket; K_requested=10000; K_effective=200; N_with_truth=200):
    - baseline_random_hit_rate: 0.055000 (ci: 0.025000..0.085000)
    - baseline_score_only_hit_rate: 0.055000 (ci: 0.025000..0.085000)
    - filtered_hit_rate: 0.055000 (ci: 0.025000..0.085000)
    - uplift_vs_random: 0.000000
    - uplift_vs_score_only: 0.000000

### 2026-01-17 - COVERAGE-DECOYS-2: RELAX-A/RELAX-B decoy strategies + coverage smoke tests
- What: extended `decoy_strategy.v1` chain with intermediate strategies (strict -> RELAX-A -> RELAX-B -> aromatic-as-single fallback) to reduce `SKIP:no_decoys_generated` without changing audit/contracts; added deterministic smoke tests for baseline molecules.
- PR: https://github.com/RobertPaulig/Geometric_table/pull/82 (head: 214394681c0a139b223fbca530a6bc6c57862c1b; merge: a07d5274c00fe375eca2b4b6ea66b701cbfc8c18)
- CI run (3/3 ci/* on merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21091698355

### 2026-01-17 - LIVE-PROOF-UTILITY-REAL-SCORES-POST-COVERAGE-1: utility realtruth (external truth+scores pinned) after COVERAGE-DECOYS-2
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-utility-realtruth-2026-01-17-r1
- Source commit: 0673c7d44192fd591a5910c7352d1c37aa1718d4
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21093723229
- scores.json (hetero_scores.v1; external pinned):
  - scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340466779
  - SHA256(scores.json): 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  - score_key: external_ci_rule_v1
- truth.csv (customer_truth.v1; external pinned):
  - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  - SHA256(truth.csv): 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-17-r1/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): 43B489FBFBF813AB0FE5E62FC9038649A9DD3C5A75D9D038241DE6800FACFF1F
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/88 (merge: 134bd6f17947327603f3425d21b67cb5da9c8cca)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21093888729
- Facts:
  - rows_total: 200
  - rows_ok: 200
  - scores_coverage.rows_missing_scores_input: 0
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - share_rows_with_n_decoys_gt_0: 1.000 (100.0%)
  - utility (scores_source=external; skip_policy=unknown_bucket; K_requested=10000; K_effective=60):
    - baseline_random_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - baseline_score_only_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - filtered_hit_rate: 0.066667 (ci: 0.016667..0.150000)
    - uplift_vs_random: 0.000000
    - uplift_vs_score_only: 0.000000
  - comparison_vs_value-utility-realtruth-2026-01-14-r2 (same pinned truth+scores; pre-coverage):
    - status_counts: OK=60, SKIP=140, ERROR=0 -> OK=200, SKIP=0, ERROR=0
    - top_skip_reasons: no_decoys_generated: 140 -> (none)
    - coverage_ok_rate: 0.300000 -> 1.000000
    - share_rows_with_n_decoys_gt_0: 0.300000 -> 1.000000
    - selection_K_effective: 60 -> 60

### 2026-01-17 - UTILITY-K-EFFECTIVE-UNLOCK-1: explain selection_K_effective via eligibility breakdown (utility realtruth r2)
- What: added explicit `eligibility` decomposition to `cost_lift_report.json` (why `K_effective` < `K_requested`) and recorded it in registry for a pinned rerun.
- Code PR: https://github.com/RobertPaulig/Geometric_table/pull/91 (merge: 2a37d0aab5c3f7f66ac340bfa816966d377b45d3)
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-utility-realtruth-2026-01-17-r2
- Source commit: 2a37d0aab5c3f7f66ac340bfa816966d377b45d3
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21097389307
- scores.json (hetero_scores.v1; external pinned):
  - scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340466779
  - SHA256(scores.json): 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  - score_key: external_ci_rule_v1
- truth.csv (customer_truth.v1; external pinned):
  - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  - SHA256(truth.csv): 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-17-r2/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): 1AE60548048E02B321FDE969B8540A88BE1B8D3B34C66CF23A125946E0D60785
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/92 (merge: 6f0a073c66d1d078a13acf732f08c577151c2f06)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21097528092
- Facts:
  - status_counts: OK=200, SKIP=0, ERROR=0
  - utility: selection_K_requested=10000, selection_K_effective=60
  - eligibility (from cost_lift_report.json):
    - rows_truth_known=200, rows_verdict_pass_fail=60, rows_eligible_for_cost_lift=60
    - K_effective_reason_top: verdict_not_pass_fail=140 (share_total=0.700000)

### 2026-01-17 - UTILITY-K-EFFECTIVE-UNLOCK-2: decoy scores coverage sync (n_decoys_scored) to explain verdict=SKIP and K_effective
- What: added `scores_coverage` aggregation to `cost_lift_report.json` from `summary.csv` (`n_decoys_generated`, `n_decoys_scored`) and recorded it in registry via a pinned utility realtruth rerun.
- Code PR: https://github.com/RobertPaulig/Geometric_table/pull/94 (merge: dd0af1f9c36297a196ac5df3472c07dcd6c7df6a)
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-utility-realtruth-2026-01-17-r3
- Source commit: dd0af1f9c36297a196ac5df3472c07dcd6c7df6a
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21099962560
- scores.json (hetero_scores.v1; external pinned):
  - scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340466779
  - SHA256(scores.json): 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  - score_key: external_ci_rule_v1
- truth.csv (customer_truth.v1; external pinned):
  - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  - SHA256(truth.csv): 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-17-r3/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): 815A9E3E1D8BBBE6BB16801A3BBC27C2CCD10E93D40168D10DD4A882C84B5236
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/95 (merge: a58bd04dff39fce00b38cb0ea510ea7f3afdc994)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21100128809
- Facts (from summary.csv):
  - n_decoys_scored histogram: 0=140, 1-5=40, 6-10=20
  - verdict x (n_decoys_scored==0 vs >0): PASS(gt0)=60; SKIP(eq0)=140
  - scores_coverage (from cost_lift_report.json):
    - rows_with_decoys_scored_gt0=60, rows_with_decoys_scored_eq0=140
    - decoys_scored_total=240, decoys_missing_total=640

### 2026-01-17 - DECOY-SCORES-REQUEST-PACK-1: export missing decoy hashes/smiles for external rescoring
- What: added a request-pack artefact `missing_decoy_scores.csv` (decoy_hash + decoy_smiles + count_rows_affected) to make external scores refresh actionable when `missing_scores_for_all_decoys` dominates.
- Code PR: https://github.com/RobertPaulig/Geometric_table/pull/97 (merge: 8d5a38c2dcc71a25dbcab6c9e00929d679e2018a)
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-utility-realtruth-2026-01-17-r4
- Source commit: 8d5a38c2dcc71a25dbcab6c9e00929d679e2018a
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21100791630
- scores.json (hetero_scores.v1; external pinned):
  - scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340466779
  - SHA256(scores.json): 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  - score_key: external_ci_rule_v1
- truth.csv (customer_truth.v1; external pinned):
  - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  - SHA256(truth.csv): 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-17-r4/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): CEA2E0599355CC5A31CA4B2318EC707AF85BE4298196E2AEB672F32C9A9C29AA
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/98 (merge: f5060f1638a7c46efee772c3c6cde31859903fd7)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21101010888
- Facts:
  - request-pack file: missing_decoy_scores.csv
  - scores_coverage.unique_missing_decoy_hashes: 32
  - scores_coverage.missing_decoy_hashes_top10: see `missing_decoy_hashes_top10` in `metrics.json` and the registry entry for the tag.

### 2026-01-17 - DECOY-SCORES-ASSET-REFRESH-1: refresh external scores asset to cover missing decoy_hashes and unlock K_effective
- What: published an updated external scores asset (same `score_key=external_ci_rule_v1`, `hetero_scores.v1`) that covers the missing decoy hashes from the r4 request-pack, then reran utility realtruth to confirm `K_effective` unlock.
- External scores asset:
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/scores-external-ci-rule-v1-2026-01-17-r1
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/scores-external-ci-rule-v1-2026-01-17-r1/hetero_scores_external_ci_rule_v1.json
  - SHA256(hetero_scores_external_ci_rule_v1.json): E3A00B22B7419D87DE059E66045C8466F8871FBE8380D7A4EC4F3F6B4CCA87C0
- Utility rerun:
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-utility-realtruth-2026-01-17-r5
  - Source commit: d74e46ee23e98d49fbc8a37bcae32fdacbcf49ec
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21101681012
  - scores.json (hetero_scores.v1; external pinned):
    - scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/342035099
    - SHA256(scores.json): E3A00B22B7419D87DE059E66045C8466F8871FBE8380D7A4EC4F3F6B4CCA87C0
    - score_key: external_ci_rule_v1
  - truth.csv (customer_truth.v1; external pinned):
    - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
    - SHA256(truth.csv): 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-17-r5/value_utility_realtruth_evidence_pack.zip
  - SHA256(value_utility_realtruth_evidence_pack.zip): A637058199BBD523B69C95680BAF0D7D768293CBCE1FEAC7237F6478F1304BB1
  - Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/100 (merge: 239ccee6241d92cd0f1aac19833d57e7fbc5b7ce)
  - CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21101797258
  - Facts:
    - scores_coverage.unique_missing_decoy_hashes: 0
    - scores_coverage.rows_with_decoys_scored_eq0: 0 (was 140 in r4)
    - selection_K_effective: 200 (was 60 in r4)
