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

### 2026-01-18 - DECOY-REALISM + CHEM-OPERATOR FOUNDATION-1: hard decoys mode, hardness curve, and opt-in H-operator
- What:
  - Added a hard-decoys accept/reject mode based on Morgan Tanimoto (`--decoy_hard_mode`) to prevent "too-easy decoys" from inflating AUC.
  - Added "hardness curve" artefacts (`hardness_curve.csv`, `hardness_curve.md`) and `metrics.json.decoy_realism` with `auc_interpretation`.
  - Added opt-in chemistry-aware operator features (`--operator_mode h_operator`) and exported `operator_features.csv`.
- Code PR: https://github.com/RobertPaulig/Geometric_table/pull/102 (merge: 64ab419118e85229760a0935a43ec2f05ac4a839)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21103259389
- Live proof (ring-suite):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-ring-suite-2026-01-18
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-ring-suite-2026-01-18/value_ring_suite_evidence_pack.zip
  - SHA256(value_ring_suite_evidence_pack.zip): 3BFB1865AE6C6A0163F8F729E7B9BBFAF61B96D8099BC7E9F8B35C0A6B3D0030
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21103307012
  - Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/103 (merge: 039a23734df91dc946fed35ffa6983eefbb82fba)
  - CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21103438952
  - Facts (from `metrics.json.decoy_realism`):
    - tanimoto_median: 0.119643
    - hardness bins (pairs_total): easy=760, medium=120, hard=0
    - auc_interpretation: INCONCLUSIVE (decoys_too_easy)

### 2026-01-18 - PHYSICS-OPERATOR-RAILS-1: live proof publish→release→registry (stress pack)
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-rails-2026-01-18-r1
- Source commit: 2a9bd703abee100d9ee0fdafe3e89acc42c1316f
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21116814765
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-rails-2026-01-18-r1/evidence_pack.zip
- SHA256(evidence_pack.zip): 20980360782EBE926F1F4E448369D8F692431059EE3863489EE3AD27805773D1
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/110 (merge: 8314c0656ab6f6aa8eadff20c32d9cce61ca4473)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21116867958

### 2026-01-18 - PHYSICS-WEIGHTED-EDGES-1: weighted edges (bond order + Δχ) live proof (stress pack)
- Code PR: https://github.com/RobertPaulig/Geometric_table/pull/113 (merge: 7c1d14207316e8130ae734ff2c07f80ae9a6b4e9)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21118368856
- Workflow PR: https://github.com/RobertPaulig/Geometric_table/pull/114 (merge: 80fbe9895609b8c43a758c301bd933c4e87ef38f)
- CI run (3/3 ci/* on workflow merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21118477746
- Live proof (stress pack):
  - Inputs: physics_mode=both; edge_weight_mode=bond_order_delta_chi
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-weights-2026-01-18-r1
  - Source commit: 80fbe9895609b8c43a758c301bd933c4e87ef38f
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21118526044
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-weights-2026-01-18-r1/evidence_pack.zip
  - SHA256(evidence_pack.zip): 388FA597A852B0CC881136B0A45FA089CE2797E2E0E6BACDB7B3FA47D9158F4F
  - Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/115 (merge: 86ea9afcc48d138b18db8373b990612790895fc2)
  - CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21118668233

### 2026-01-18 - PHYSICS-P2-LDOS-GREEN-1: DOS/LDOS artifacts (stress pack)
- Code PR: https://github.com/RobertPaulig/Geometric_table/pull/117 (merge: 8a5a1fe2aecfee07942493bd815a4a65a7252f8e)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21119463290
- Live proof (stress pack):
  - Inputs: physics_mode=both; edge_weight_mode=unweighted
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-ldos-2026-01-18-r1
  - Source commit: 8a5a1fe2aecfee07942493bd815a4a65a7252f8e
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21119526003
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-ldos-2026-01-18-r1/evidence_pack.zip
  - SHA256(evidence_pack.zip): 6589FA29A9ABC3BA3CD65446EA45ABA2033C663F6AF61AC38B01BFEEAC00C652
  - Evidence pack artifacts: dos_curve.csv, ldos_summary.csv, summary_metadata.json (dos_grid_n=128, dos_eta=0.05)
  - Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/118 (merge: 1b9a67d56e2d3ca58e2877447e4de69fb58dcc1e)
  - CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21119709311

### 2026-01-19 - PHYSICS-P3-SELF-CONSISTENT-POTENTIAL-1: self-consistent potential (SCF) artifacts (stress pack)
- Code PR: https://github.com/RobertPaulig/Geometric_table/pull/120 (merge: 5cc2b4d3ff2d9438f9b236c59056124458ea0cc7)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21121323194
- Workflow PR (publish inputs): https://github.com/RobertPaulig/Geometric_table/pull/121 (merge: 8e9ab74dc1a44c91c09d082c03c314a448ed9a02)
- CI run (3/3 ci/* on workflow merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21121518323
- Live proof (stress pack):
  - Inputs: physics_mode=both; edge_weight_mode=bond_order_delta_chi; potential_mode=both; scf_max_iter=50; scf_tol=1e-6; scf_damping=0.5; scf_occ_k=5; scf_tau=1.0
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-scf-2026-01-19-r1
  - Source commit: 8e9ab74dc1a44c91c09d082c03c314a448ed9a02
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21121557871
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-scf-2026-01-19-r1/evidence_pack.zip
  - SHA256(evidence_pack.zip): ADFFF664035F103661E011FE5EF8FB490D4A48449BCC8BA101B9D71BB17061A4
  - Evidence pack artifacts: scf_trace.csv, potential_vectors.csv, summary_metadata.json (scf_schema=hetero2_scf.v1; scf_converged=true; scf_iters=1; scf_residual_final=3.469446951953614e-17)
  - Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/122 (merge: b949c180a98d9e5cbc23ec520855728ca9875a3f)
  - Registry facts PR (SCF summary): https://github.com/RobertPaulig/Geometric_table/pull/123 (merge: 6316631c13b332624f62892cfed0139f2342098a)

### 2026-01-19 - PHYSICS-UNITS-CALIBRATION-RAILS-1: dimensionless units model + potential_scale_gamma (stress pack)
- What:
  - Formalized units model: operator parameters are dimensionless (no physical Å/eV claims without calibration).
  - Added global potential scaling parameter `potential_scale_gamma` (default 1.0) and logged it into evidence pack metadata.
- Code PR: https://github.com/RobertPaulig/Geometric_table/pull/125 (merge: 587fd8f816d4e66cc2569815f4210ca972bc3525)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21122560272
- Live proof (stress pack):
  - Inputs: physics_mode=both; edge_weight_mode=unweighted; potential_mode=both; potential_unit_model=dimensionless; potential_scale_gamma=1.0
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-units-gamma-2026-01-19-r1
  - Source commit: 587fd8f816d4e66cc2569815f4210ca972bc3525
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21122692884
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-units-gamma-2026-01-19-r1/evidence_pack.zip
  - SHA256(evidence_pack.zip): 6529B5356345576F858D333F71737332D049EB745A6191AD6AD10775A93FA4BD
  - Evidence pack artifacts: potential_vectors.csv (V0,V_scaled,gamma), summary_metadata.json (potential_unit_model, potential_scale_gamma)
  - Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/126 (merge: 670cb69c3e6eddc108975cc5343d04e32f215a67)
  - CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21122839905

### 2026-01-19 - PHYSICS-P3-SCF-RAILS-1: SCF artifacts (trace+summary) + registry facts (stress pack)
- Code PR: https://github.com/RobertPaulig/Geometric_table/pull/128 (merge: dfc5b72301bfdaea4e5d5c7834c8e0202c1dfddf)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21129760355
- Live proof (stress pack):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-scf-2026-01-19-r2
  - Source commit: dfc5b72301bfdaea4e5d5c7834c8e0202c1dfddf
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21129844747
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-scf-2026-01-19-r2/evidence_pack.zip
  - SHA256(evidence_pack.zip): E89E6E99972840A1900C776529065C2009EF87D5A386545289DA15C71F020179
  - Evidence pack artifacts: scf_trace.csv, scf_summary.json, summary_metadata.json (potential_unit_model=dimensionless, potential_scale_gamma=1.0, scf_enabled=true, scf_status=CONVERGED)
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/129 (merge: 00e97107bee82c2030b19280a936591e8c8692de)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21130094003
- Registry facts PR (SCF fields): https://github.com/RobertPaulig/Geometric_table/pull/130 (merge: 65513a25e56f552473fc7a0269f4b180cacfe4e2)
- CI run (3/3 ci/* on registry facts merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21130322458

### 2026-01-19 - P3.5-SCF-NONTRIVIAL-PROOF-1: SCF audit verdict (nontriviality proof on asym fixture set)
- Code PR: https://github.com/RobertPaulig/Geometric_table/pull/132 (merge: 69ac40d21eaa7bc22fc77d9d99ec139da4283ca7)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21133533715
- Live proof (asym fixture set; local build, release assets pinned):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-scf-audit-2026-01-19-r1
  - Source commit: 69ac40d21eaa7bc22fc77d9d99ec139da4283ca7
  - Input CSV asset (asym_ ids): https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-scf-audit-2026-01-19-r1/scf_audit.csv
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-scf-audit-2026-01-19-r1/evidence_pack.zip
  - SHA256(evidence_pack.zip): 01E630207A05DE92DB405DDB3248061F634691E9592C15AA7DDE42A22D158B21
  - SCF audit verdict: SUCCESS (reason: nontrivial_on_asym_fixture; iters_median=15.0; delta_V_p95=0.0880012216377013; residual_init_mean=0.037387575359043745 → residual_final_mean=7.321790598292799e-07)
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/133 (merge: c2f71b288ee81a30da3bd720ccaa0f102e13f592)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21134356078

### 2026-01-19 - PHYSICS-P3.6-SCF-NONTRIVIAL-GATES-1: SCF audit metrics + run-level nontrivial verdict (asym fixture set)
- Code PR: https://github.com/RobertPaulig/Geometric_table/pull/135 (merge: 220f00773a6e87a3af642ebb3c6e4eb35ebd0042)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21136385185
- Live proof (asym fixture set; local build, release assets pinned):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-scf-audit-2026-01-19-r2
  - Source commit: 220f00773a6e87a3af642ebb3c6e4eb35ebd0042
  - Input CSV asset (asym_ ids): https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-scf-audit-2026-01-19-r2/scf_audit.csv
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-scf-audit-2026-01-19-r2/evidence_pack.zip
  - SHA256(evidence_pack.zip): 3409C1137495CE450D22D6D23C4EB174FDF696EE317712124CB5248F5C18BD7E
  - Evidence pack artifacts: scf_trace.csv, scf_summary.json, scf_audit_metrics.csv, summary_metadata.json (scf_audit_verdict=SUCCESS; scf_iters_mean=14.75; scf_iters_max=15; deltaV_max_max=0.1567902269999999; scf_converged_rate=1.0; scf_nontrivial_rate=1.0)
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/136 (merge: 8582b0316c93f316fc42ed990746e0b07cd1111c)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21136630958

### 2026-01-19 - P4.0 INTEGRATION-BASELINE-RAILS-1: baseline интегратор (energy grid) + benchmark артефакты (stress pack)
- Code PR: https://github.com/RobertPaulig/Geometric_table/pull/138 (merge: eb0fa851d685f79d561f92f439fac7c3000cd1c9)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21140329965
- Live proof (stress pack):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-integration-baseline-2026-01-19-r1
  - Source commit: eb0fa851d685f79d561f92f439fac7c3000cd1c9
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21140462233
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-integration-baseline-2026-01-19-r1/evidence_pack.zip
  - SHA256(evidence_pack.zip): C07DC02484C1EB75751A6CEE3BE83C82664E51980DAC9192DA25BEEC95F6140B
  - Evidence pack artifacts: integration_benchmark.csv, integration_benchmark.md, summary_metadata.json (integrator_mode=baseline; energy_range=[-0.15, 4.15]; energy_points=128; eta=0.05; integration_walltime_ms_median=2.3876529999995455)
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/139 (merge: 639d79edb98181b9715ef0f8d01adc8cd9cc2784)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21141045407

### 2026-01-19 - P4.1 ADAPTIVE-INTEGRATION-1: adaptive integrator + correctness comparison + speedup verdict (stress pack)
- Code PR: https://github.com/RobertPaulig/Geometric_table/pull/141 (merge: 19b21d980e968525783e635b87689aa854403128)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21144672211
- Bugfix PR (correctness): https://github.com/RobertPaulig/Geometric_table/pull/143 (merge: 5600faa7debce8647e574ee2f858b3ddd534a9c3)
- CI run (3/3 ci/* on bugfix merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21145366452
- Live proof r1 (stress pack; adaptive correctness failed):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-integration-adaptive-2026-01-19-r1
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21144735538
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-integration-adaptive-2026-01-19-r1/evidence_pack.zip
  - SHA256(evidence_pack.zip): 6C2F3ED358220DB60F725AF202B310051CD9F901FC454420F3B7A9FD08464C0C
  - integrator_verdict=FAIL_CORRECTNESS; integrator_correctness_pass_rate=0.666667; integrator_speedup_median=0.322449
- Live proof r2 (stress pack; adaptive correctness restored):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-integration-adaptive-2026-01-19-r2
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21145429269
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-integration-adaptive-2026-01-19-r2/evidence_pack.zip
  - SHA256(evidence_pack.zip): C76F584884E28CA04EF792D19D5BE2A0B13F3B67FAD526EEA36DD21CED09028C
  - integrator_verdict=SUCCESS; integrator_correctness_pass_rate=1.0; integrator_speedup_median=0.231419; integrator_speedup_verdict=NO_SPEEDUP_YET
  - dos_L_segments_used=20; dos_L_evals_total=975; dos_L_error_est_total=9.46613e-05
- Evidence pack artifacts: adaptive_integration_trace.csv, adaptive_integration_summary.json, integration_compare.csv, integration_benchmark.csv, integration_benchmark.md, summary_metadata.json
- Registry PR r2: https://github.com/RobertPaulig/Geometric_table/pull/144 (merge: 6687fde3cb32a548dc4f0a59a58c563b40f5d667)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21145777488
- Registry PR r1 (historical r1 facts): https://github.com/RobertPaulig/Geometric_table/pull/142 (merge: 1fcfdaaa7f04e0c344670102c3ee39177a8b278d)
- CI run (3/3 ci/* on registry r1 merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21146286010

### 2026-01-19 - P4.2 ADAPTIVE-INTEGRATION-SPEEDUP-1: prefer-split heuristic + live proof r2 (speedup still not achieved)
- Code PR (prefer split for wide baseline-aware segments): https://github.com/RobertPaulig/Geometric_table/pull/148 (merge: 31babd0f08fd62f731bb2af3de2750c2ae8b9e57)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21150551480
- Live proof r2 (stress pack; integrator_mode=both):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-integration-adaptive-speedup-2026-01-19-r2
  - Source commit: 31babd0f08fd62f731bb2af3de2750c2ae8b9e57
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21150618113
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-integration-adaptive-speedup-2026-01-19-r2/evidence_pack.zip
  - SHA256(evidence_pack.zip): 46E9F6088A06002D767D36E39F585F5763230A1CB5FD41976DE647D84327C1B4
  - Evidence pack artifacts: integration_speed_profile.csv, integration_compare.csv, adaptive_integration_trace.csv, adaptive_integration_summary.json, summary_metadata.json
  - integrator_correctness_pass_rate=1.0; integrator_speedup_median=0.237091; integrator_speedup_verdict=NO_SPEEDUP_YET
  - dos_L: adaptive_evals_total=223 (eval_ratio=1.7422); segments_used=20; cache_hit_rate=0.4802; speedup=0.2371
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/149 (merge: 8ac10f0b9da8089fe34098782be78123c1b25f15)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21150874235

### 2026-01-19 - P4.2.1 SPEEDUP-TRUTH-AND-FIX-1: metrics integrity for integration speed profile + live proof r3
- Code PR: https://github.com/RobertPaulig/Geometric_table/pull/151 (merge: 72ea11636462ac29f39c31ec44ab8723f56df788)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21152886985
- Live proof r3 (stress pack; integrator_mode=both; metrics integrity restored):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-integration-adaptive-speedup-2026-01-19-r3
  - Source commit: 72ea11636462ac29f39c31ec44ab8723f56df788
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21152980769
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-integration-adaptive-speedup-2026-01-19-r3/evidence_pack.zip
  - SHA256(evidence_pack.zip): B0A669402074032C910C8C32B973146DDA710B6172EAF1A3F5DBF69BCF951D61 (verified by download)
  - Evidence pack artifacts: integration_speed_profile.csv, integration_compare.csv, adaptive_integration_trace.csv, adaptive_integration_summary.json, integration_benchmark.csv, integration_benchmark.md, summary_metadata.json
  - integrator_verdict=SUCCESS; integrator_correctness_pass_rate=1.0; integrator_eval_ratio_median=1.9609375; integrator_speedup_median=0.203809; integrator_speedup_verdict=FAIL_SPEEDUP
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/152 (merge: 7754ac0a2dcb91dcd540a38a2ab9e2d05669b175)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21153056730

### 2026-01-20 - P4.3R INSTRUMENT+SELECT-TRUTH-1: integration profiling + mode selector (baseline for small n_atoms)
- Code PR: https://github.com/RobertPaulig/Geometric_table/pull/154 (merge: f2950c205da43a09e95bd0a4de9ebd0c994b6817)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21154789330
- Live proof r1 (stress pack; integrator_mode=both; selector enabled):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-integration-select-truth-2026-01-20-r1
  - Source commit: f2950c205da43a09e95bd0a4de9ebd0c994b6817
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21154848940
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-integration-select-truth-2026-01-20-r1/evidence_pack.zip
  - SHA256(evidence_pack.zip): AC47AD2A3BC3D8EDE69CF804D8B3A2B7F5664127E6F8EA69F538D135B9A9AFAA (verified by download)
  - Evidence pack artifacts: integration_profile.csv, integration_timing_breakdown.json, integrator_select_summary.json, summary_metadata.json
  - Facts (summary_metadata.json): integrator_valid_row_fraction=1.0; integrator_correctness_pass_rate=1.0; integrator_eval_ratio_median=1.9609375; integrator_speedup_median=0.282874; integrator_speedup_verdict=PASS_SCALING_READY; integrator_selected_fraction_baseline=1.0; integrator_selected_fraction_adaptive=0.0
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/155 (merge: f956f427f2a52ea526b3e2c8d371a4f696b9762b)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21155008818

### 2026-01-20 - PHYSICS-P5-LARGE-SCALE-PROOF-1: large-scale integration proof (speedup vs n_atoms)
- Code PR (scale fixtures + speedup-vs-n evidence pack builder): https://github.com/RobertPaulig/Geometric_table/pull/157 (merge: 665343708c53828baa40144bf1dcd64cbc7c8fd9)
- Publish workflow PR: https://github.com/RobertPaulig/Geometric_table/pull/158 (merge: c1e4df300883811021f818a527cdece7e1ecf38b)
- Workflow fix PR (publish input validation): https://github.com/RobertPaulig/Geometric_table/pull/159 (merge: 77f98b9c5fd892a27c2c8cc929132824b36a4f77)
- Live proof r1 (integrator_mode=both; dos_eta=0.2):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r1
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21166148106
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r1/physics_large_scale_evidence_pack.zip
  - SHA256(physics_large_scale_evidence_pack.zip): C651C583D893C37A91E25AFC5D3FD049933E7A8ABA3D6E5AE47E6DB57FFF6653 (verified by download)
  - Evidence pack artifacts: fixtures_polymer_scale.csv, speedup_vs_n.csv, speedup_vs_n.md, summary_metadata.json
  - Facts (summary_metadata.json): scale_n_atoms_max=800; scale_speedup_median_at_maxN=0.201066; scale_break_even_n_estimate=null; scale_speedup_verdict=FAIL_SPEEDUP_AT_SCALE
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/160 (merge: e736d2b84e4fc9c3539f647d02f31a48d08b443c)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21166311914

### 2026-01-20 - PHYSICS-P5.1-INTEGRATION-SCALE-LAW-1: freeze integration scale law (contract + gates + canonical artifacts)
- Code PR (P5.1 law + contract tests): https://github.com/RobertPaulig/Geometric_table/pull/162 (merge: e7f576d7bfdbea31f8229ae90c5806ff7508331d)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21184133870
- Live proof r2 (large-scale; integrator_mode=both; dos_eta=0.2):
  - Law ref (from summary_metadata.json): docs/contracts/INTEGRATION_SCALE_CONTRACT.md @ e7f576d7bfdbea31f8229ae90c5806ff7508331d (p5.1.v1)
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r2
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21184833947
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r2/physics_large_scale_evidence_pack.zip
  - SHA256(physics_large_scale_evidence_pack.zip): BB8A54751BFA98D8A68C719A19B7B8A0284977BED591459CF0D029878654F999 (verified by download)
  - Evidence pack artifacts: fixtures_polymer_scale.csv, speedup_vs_n.csv, speedup_vs_n.md, integration_compare.csv, integration_speed_profile.csv, adaptive_integration_trace.csv, adaptive_integration_summary.json, summary_metadata.json
  - Facts (summary_metadata.json): gate_n_min=200; integrator_correctness_pass_rate_at_scale=1.0; integrator_speedup_median_at_scale=0.11588066997158922; integrator_eval_ratio_median_at_scale=1.1962616822429906; integrator_correctness_verdict=PASS_CORRECTNESS_AT_SCALE; integrator_speedup_verdict=FAIL_SPEEDUP_AT_SCALE; scale_speedup_verdict=FAIL_SPEEDUP_AT_SCALE
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/163 (merge: 1646c3bdbecf4cfb8113e54331b37457d5500ca0)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21185587833

### 2026-01-20 - PHYSICS-P5.2-COST-DECOMPOSITION-1: cost decomposition for integration runtime (timing breakdown + bottleneck verdict)
- Code PR (P5.2 timing breakdown + gates): https://github.com/RobertPaulig/Geometric_table/pull/165 (merge: f465e9be203e39ac0d6c98c91cad080322fe487c)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21188158945
- Live proof r3 (large-scale; integrator_mode=both; dos_eta=0.2):
  - Law ref (from summary_metadata.json): docs/contracts/INTEGRATION_SCALE_CONTRACT.md @ f465e9be203e39ac0d6c98c91cad080322fe487c (p5.1.v1)
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r3
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21188218951
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r3/physics_large_scale_evidence_pack.zip
  - SHA256(physics_large_scale_evidence_pack.zip): 8A1999E0DB0E03A59B6AB1318698B002A6594FE842E71CCED250FEF1947E84CE (verified by download)
  - Evidence pack artifacts: fixtures_polymer_scale.csv, speedup_vs_n.csv, speedup_vs_n.md, timing_breakdown.csv, integration_compare.csv, integration_speed_profile.csv, adaptive_integration_trace.csv, adaptive_integration_summary.json, summary_metadata.json
  - Facts (summary_metadata.json): gate_n_min=200; integrator_correctness_verdict=PASS_CORRECTNESS_AT_SCALE; integrator_speedup_median_at_scale=0.11754749203803679; integrator_speedup_verdict=FAIL_SPEEDUP_AT_SCALE; cost_bottleneck_verdict_at_scale=BOTTLENECK_IS_INTEGRATOR; cost_median_dos_ldos_eval_ms_at_scale=2.487639000023023; cost_median_integration_logic_ms_at_scale=4.544935500007341
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/166 (merge: 53307fe347c20a195b39cea181dce2619d17ddde)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21188380534

### 2026-01-21 - PHYSICS-P5.3-INTEGRATION-LOGIC-OPT-1: optimize integration_logic runtime (P5.3 KPI fields + publish gate)
- Code PR (integration_logic optimization + KPI gates): https://github.com/RobertPaulig/Geometric_table/pull/169 (merge: 540a46511a1275fa8358aac00cd2e85cb36092a5)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21193354363
- Live proof r4 (large-scale; integrator_mode=both; dos_eta=0.2):
  - Law ref (from summary_metadata.json): docs/contracts/INTEGRATION_SCALE_CONTRACT.md @ 540a46511a1275fa8358aac00cd2e85cb36092a5 (p5.1.v1)
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r4
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21193382219
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r4/physics_large_scale_evidence_pack.zip
  - SHA256(physics_large_scale_evidence_pack.zip): 94DE3D0A457F8A2129DBE2A577291862B60486F5959AC1D03FCF4D239CFD75A9 (verified by download)
  - Evidence pack artifacts: fixtures_polymer_scale.csv, speedup_vs_n.csv, speedup_vs_n.md, timing_breakdown.csv, integration_compare.csv, integration_speed_profile.csv, adaptive_integration_trace.csv, adaptive_integration_summary.json, summary_metadata.json
  - Facts (summary_metadata.json): gate_n_min=200; integrator_correctness_verdict=PASS_CORRECTNESS_AT_SCALE; integrator_speedup_median_at_scale=0.21174035241485198; integrator_eval_ratio_median_at_scale=1.1531531531531531; integrator_speedup_verdict=FAIL_SPEEDUP_AT_SCALE; cost_bottleneck_verdict_at_scale=MIXED; cost_median_dos_ldos_eval_ms_at_scale=2.5924949999875935; cost_median_integration_logic_ms_at_scale=2.218445499991617; cost_median_integration_logic_ms_at_scale_before=4.544935500007341; cost_median_integration_logic_ms_at_scale_after=2.218445499991617; cost_integration_logic_speedup_at_scale=2.048702796631477; cost_integration_logic_opt_verdict_at_scale=PASS
  - P5.3 KPI thresholds (registry rails): cost_integration_logic_speedup_gate_break_even=1.0; cost_integration_logic_speedup_gate_strong=2.0
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/171 (merge: 2bf2e6da7b8423118ebb67967d2fec81a0e0ccf7)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21193563936
- Registry addendum PR (P5.3 thresholds): https://github.com/RobertPaulig/Geometric_table/pull/172 (merge: 993f6f303462f33b106206cd6fb15dbcde89b28b)
- CI run (3/3 ci/* on addendum merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21193677005

### 2026-01-21 - PHYSICS-P5.4-PERF-HARDNESS-TOPOLOGY-1: ring-suite topology hardness in P5-scale (anti-illusion)
- Code PR (ring-suite + topology hardness gates): https://github.com/RobertPaulig/Geometric_table/pull/170 (merge: 300c8657ef2c808462662cbc05f1c4245f8fe71b)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21204483269
- Live proof r5 (large-scale; integrator_mode=both; dos_eta=0.2):
  - Law ref (from summary_metadata.json): docs/contracts/INTEGRATION_SCALE_CONTRACT.md @ 300c8657ef2c808462662cbc05f1c4245f8fe71b (p5.1.v1)
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r5
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21204574412
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r5/physics_large_scale_evidence_pack.zip
  - SHA256(physics_large_scale_evidence_pack.zip): 2414FBBFEC48A920E061FA3037BA0626FBFDD28F905C3DE874BBF1CCCFE8AF48 (verified by download)
  - Evidence pack artifacts: fixtures_polymer_scale.csv, fixtures_ring_scale.csv, speedup_vs_n.csv, speedup_vs_n_by_family.csv, speedup_vs_n.md, timing_breakdown.csv, integration_compare.csv, integration_speed_profile.csv, adaptive_integration_trace.csv, adaptive_integration_summary.json, summary_metadata.json
  - Facts (summary_metadata.json): topology_families=['polymer', 'ring']; topology_gate_n_min=200; speedup_median_at_scale_polymer=0.0879949862363698; speedup_verdict_at_scale_polymer=FAIL_SPEEDUP_AT_SCALE; speedup_median_at_scale_ring=0.07868455761041604; speedup_verdict_at_scale_ring=FAIL_SPEEDUP_AT_SCALE; topology_hardness_verdict=NO_SPEEDUP_YET; topology_hardness_reason=polymer(verdict=FAIL_SPEEDUP_AT_SCALE, median=0.0879949862363698) ring(verdict=FAIL_SPEEDUP_AT_SCALE, median=0.07868455761041604) gate_n_min=200; integrator_correctness_verdict=PASS_CORRECTNESS_AT_SCALE
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/174 (merge: 4e65ee4c30db55fcff450523a847690c1430c954)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21204853592

### 2026-01-21 - PHYSICS-P5.5-RING-PERF-FOLLOW-UP-1: ring per-family cost profile (polymer vs ring) + verdict rails
- Code PR (per-family timing breakdown + gates): https://github.com/RobertPaulig/Geometric_table/pull/176 (merge: 93145ec561ce6565b149c4e1b5536f7e778731db)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21208619062
- Live proof r6 (large-scale; integrator_mode=both; dos_eta=0.2):
  - Law ref (from summary_metadata.json): docs/contracts/INTEGRATION_SCALE_CONTRACT.md @ 93145ec561ce6565b149c4e1b5536f7e778731db (p5.1.v1)
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r6
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21208683069
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r6/physics_large_scale_evidence_pack.zip
  - SHA256(physics_large_scale_evidence_pack.zip): 7A933610708BEEF60FC69DD37BC6A200679E9BD3E92BBBF4CA981C4C7CBED530 (verified by download)
  - Evidence pack artifacts: fixtures_polymer_scale.csv, fixtures_ring_scale.csv, speedup_vs_n.csv, speedup_vs_n_by_family.csv, speedup_vs_n.md, timing_breakdown.csv, timing_breakdown_by_family.csv, integration_compare.csv, integration_speed_profile.csv, adaptive_integration_trace.csv, adaptive_integration_summary.json, summary_metadata.json
  - Facts (summary_metadata.json): gate_n_min=200; integrator_correctness_verdict=PASS_CORRECTNESS_AT_SCALE; integrator_speedup_median_at_scale=0.12003553852247553; topology_hardness_verdict=NO_SPEEDUP_YET; topology_ring_cost_gap_verdict_at_scale=RING_SLOWER_DUE_TO_BUILD_OPERATOR; cost_median_total_ms_at_scale_polymer_estimate=7.468396366668155; cost_median_total_ms_at_scale_ring_estimate=8.9061203666598; cost_ratio_ring_vs_polymer_total_ms_at_scale_estimate=1.192507725809557
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/177 (merge: f534b1fab00dfdb64e982d7ef1e963e2dee37cf1)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21208898844

### 2026-01-21 - PHYSICS-P5.6-RING-SPEEDUP-LAW-1: ring speedup law (contract + gates; registry-grade ring KPI)
- Code PR (ring speedup contract + P5.6 gates): https://github.com/RobertPaulig/Geometric_table/pull/179 (merge: 4b4c33eb82673b214ac9ad8e50f5f0d64791dec0)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21211015374
- Ring law ref: docs/contracts/RING_SPEEDUP_CONTRACT.md @ 4b4c33eb82673b214ac9ad8e50f5f0d64791dec0 (p5.6.v1)
- Live proof r7 (large-scale; integrator_mode=both; dos_eta=0.2):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r7
  - Publish run (failed; P5.6 gate SyntaxError): https://github.com/RobertPaulig/Geometric_table/actions/runs/21211077151
  - Fix PR (publish gate python quoting): https://github.com/RobertPaulig/Geometric_table/pull/180 (merge: f01d32149faaac2d8d70872895e28ca7273be55d)
  - Publish run (success): https://github.com/RobertPaulig/Geometric_table/actions/runs/21211317639
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r7/physics_large_scale_evidence_pack.zip
  - SHA256(physics_large_scale_evidence_pack.zip): D9DF8097C8C6EA639400ACBCC80F32694E02550C68E3DDBF07911C8683F12666 (verified by download)
  - Evidence pack artifacts: speedup_vs_n_by_family.csv, timing_breakdown_by_family.csv, summary_metadata.json
  - Facts (summary_metadata.json): ring_speedup_median_at_scale=0.09713153126539123; ring_eval_ratio_median_at_scale=0.8951048951048951; ring_correctness_pass_rate_at_scale=1.0; ring_speedup_verdict_at_scale=NO_SPEEDUP_YET; topology_ring_cost_gap_verdict_at_scale=RING_SLOWER_DUE_TO_BUILD_OPERATOR
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/181 (merge: fda19830603ee43982803a5db38c78f72db3fd9e)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21211537548

### 2026-01-21 - PHYSICS-P5.7-BUILD_OPERATOR-OPT-RING-1: build_operator optimization (ring) (vectorize+caches)
- Code PR (ring build_operator optimization): https://github.com/RobertPaulig/Geometric_table/pull/184 (merge: 8689913741dffd5aee094af591c6339fc1605f26)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21215717226
- Live proof r8 (large-scale; integrator_mode=both; dos_eta=0.2):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r8
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21215792704
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r8/physics_large_scale_evidence_pack.zip
  - SHA256(physics_large_scale_evidence_pack.zip): 441D58EB7D389E81BFD2434777B8FC33CB1698B99C9494B27FE66ABDD18665EC (verified by download)
  - Evidence pack artifacts: fixtures_polymer_scale.csv, fixtures_ring_scale.csv, speedup_vs_n.csv, speedup_vs_n_by_family.csv, speedup_vs_n.md, timing_breakdown.csv, timing_breakdown_by_family.csv, integration_compare.csv, integration_speed_profile.csv, adaptive_integration_trace.csv, adaptive_integration_summary.json, summary_metadata.json
  - Facts (summary_metadata.json): ring_speedup_median_at_scale=0.09606983162926894; ring_speedup_verdict_at_scale=NO_SPEEDUP_YET; topology_ring_cost_gap_verdict_at_scale=RING_SLOWER_DUE_TO_INTEGRATION_LOGIC; cost_ratio_ring_vs_polymer_build_operator_ms_at_scale=1.1281262332542437; cost_ratio_ring_vs_polymer_total_ms_at_scale_estimate=1.1214741766515028
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/185 (merge: 9012c80c54c3e7310a483ca19b0aa08aa4e7b922)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21215829997

### 2026-01-21 - PHYSICS-P5.8-INTEGRATION_LOGIC-OPT-RING-1: integration_logic optimization (ring) (adaptive overhead)
- Code PR (adaptive integration_logic optimization + P5.8 KPI gates): https://github.com/RobertPaulig/Geometric_table/pull/187 (merge: d5be05323c4de4954d4570785d0cf4a2fd61fa37)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21219672343
- Live proof r9 (large-scale; integrator_mode=both; dos_eta=0.2):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r9
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21219734082
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r9/physics_large_scale_evidence_pack.zip
  - SHA256(physics_large_scale_evidence_pack.zip): E1381DC4DB2D099C69973370BE4C53CC6390FAE23CF960217DC500B599DC713B (verified by download)
  - Evidence pack artifacts: fixtures_polymer_scale.csv, fixtures_ring_scale.csv, speedup_vs_n.csv, speedup_vs_n_by_family.csv, speedup_vs_n.md, timing_breakdown.csv, timing_breakdown_by_family.csv, integration_compare.csv, integration_speed_profile.csv, adaptive_integration_trace.csv, adaptive_integration_summary.json, summary_metadata.json
  - Facts (summary_metadata.json): integrator_correctness_verdict=PASS_CORRECTNESS_AT_SCALE; topology_ring_cost_gap_verdict_at_scale=RING_SLOWER_DUE_TO_INTEGRATION_LOGIC; cost_ratio_ring_vs_polymer_integration_logic_ms_at_scale=1.2641339636812614; cost_median_integration_logic_ms_at_scale_ring_before=2.6423465000817714; cost_median_integration_logic_ms_at_scale_ring_after=3.0608639999840648; cost_integration_logic_speedup_at_scale_ring=0.8632681818256308; cost_integration_logic_opt_verdict_at_scale_ring=FAIL; ring_speedup_median_at_scale=0.08512662027971973; ring_speedup_verdict_at_scale=NO_SPEEDUP_YET; cost_ratio_ring_vs_polymer_build_operator_ms_at_scale=1.130292975349101; cost_ratio_ring_vs_polymer_total_ms_at_scale_estimate=1.1967770953457784
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/188 (merge: 017445dd55c54c3143fcaeb49e818f4b12550429)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21219765841

### 2026-01-21 - VALUE-M1 ring-suite evidence pack (chem coverage; ERROR=0)
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-ring-suite-2026-01-21
- Source commit: 632d5f1b231288ae1308a338cf52f9299eec70db
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21222158878
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-ring-suite-2026-01-21/value_ring_suite_evidence_pack.zip
- SHA256(value_ring_suite_evidence_pack.zip): 97A3181F897A89A0559E0099A40AF7537D8A7E08A5B3D7D6377514E042E27509 (verified by download)
- Facts (summary.csv): rows_total=200; status_counts OK=200, SKIP=0, ERROR=0; top_skip_reasons=(none); share_rows_with_n_decoys_gt_0=1.000 (100%)
- Decoy strategy distribution (summary.csv): rewire_strict_v1=60; rewire_relax_a_v1=40; rewire_fallback_aromatic_as_single_v1=100
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/190 (merge: d5bd4a15d7b51167556f03d143078d71f8d92f31)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21222210835

### 2026-01-21 - VALUE-M1.1-DECOYS-COVERAGE: ring-suite decoys coverage gate (coverage>=0.95; no_decoys_generated<=10)
- Code PR (M1.1 gates + anti-regression test): https://github.com/RobertPaulig/Geometric_table/pull/192 (merge: 4aa99224109d1f4a9418c775e1e0a3acd9d9694c)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21224871070
- Code PR (allow tag suffix -m1_1(-rN)?): https://github.com/RobertPaulig/Geometric_table/pull/193 (merge: 3ee006204c4fb5635c779c54254384c4132d98ed)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21225130000
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-ring-suite-2026-01-21-m1_1
- Source commit: 3ee006204c4fb5635c779c54254384c4132d98ed
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21225180059
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-ring-suite-2026-01-21-m1_1/value_ring_suite_evidence_pack.zip
- SHA256(value_ring_suite_evidence_pack.zip): A250164C1D47EDF650846C0D8E1F9B043D3530C6E962C7309255F0B50F629E96 (verified by download)
- Facts (summary.csv): rows_total=200; status_counts OK=200, SKIP=0, ERROR=0; top_skip_reasons=(none); no_decoys_generated=0; share_rows_with_n_decoys_gt_0=1.000 (100%)
- Decoy strategy distribution (summary.csv; status=OK): rewire_strict_v1=60; rewire_relax_a_v1=40; rewire_fallback_aromatic_as_single_v1=100
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/194 (merge: c997129056254f2f7e85c9d87fba20bce7970d9d)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21225233053

### 2026-01-21 - VALUE-M2 known bad/good evidence packs (release assets + SHA256 + registry)
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-known-bad-good-2026-01-21
- Source commit: cd8113bbe269caa3d171df19d5fe417b125ba92a
- Assets:
  - BAD-constant: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-21/value_known_bad_good_BAD-constant_evidence_pack.zip
    - SHA256: 6D4C12D4523AADC35CB65EAB5A0FB8E8E2EE01626769E74AD0E62B4D7BF182BF
  - BAD-random: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-21/value_known_bad_good_BAD-random_evidence_pack.zip
    - SHA256: DC723348B495F0E6AC29ABF749D8858F023762FA947DBF75066BCB05D62B3046
  - GOOD-synthetic: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-21/value_known_bad_good_GOOD-synthetic_evidence_pack.zip
    - SHA256: 4FC460FEE5712FC3349CD44B8EF3D6ACF43BD4D98EDBDBD7DD7F01DC5C74AB25
- Run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21228738744
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/196 (merge: a3e8ad4071640a42caf503a12d16d8f38d952032)
- Separation facts (computed on status==OK rows only; no auto-threshold gating): Delta_median_slack(GOOD - BAD-constant)=0.500000; Delta_PASS_rate(GOOD - BAD-constant)=1.000000; Delta_median_slack(GOOD - BAD-random)=0.000000; Delta_PASS_rate(GOOD - BAD-random)=0.400000

### 2026-01-22 - VALUE-M2.1-SLACK-SIGNAL-REPAIR-1: known bad/good evidence packs (r2; slack distribution stats)
- Code PR (add slack distribution stats): https://github.com/RobertPaulig/Geometric_table/pull/198 (merge: 6cbcc31cbd20749cecc0f62d4b68986801c801e8)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21230339035
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-known-bad-good-2026-01-21-r2
- Source commit: 6cbcc31cbd20749cecc0f62d4b68986801c801e8
- Run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21230392968
- Assets:
  - BAD-constant: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-21-r2/value_known_bad_good_BAD-constant_evidence_pack.zip
    - SHA256: AC12456914248E6D1D0A44AD1827E367532B2A9452B77CA22D26A8A77BB84EE8
  - BAD-random: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-21-r2/value_known_bad_good_BAD-random_evidence_pack.zip
    - SHA256: CE1DC0C1CD03DDA63F0078ECC5F4B71F3CE74B1BE4A9643790B3EE9A552676B5
  - GOOD-synthetic: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-21-r2/value_known_bad_good_GOOD-synthetic_evidence_pack.zip
    - SHA256: 84F5215A583D73481DB7612FA92D90BD54AB5BF2D310ECE54BE6667E5FFECEE9
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/199 (merge: 527749a81c331521b20f602890da0aeaa5bc5c71)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21230473360
- Separation facts (computed on status==OK rows only; OK=200): Δ_PASS_rate(GOOD - BAD-random)=0.400000; Δ_median_slack(GOOD - BAD-random)=0.000000; Δ_mean_slack(GOOD - BAD-random)=0.311111; Δ_p25_slack(GOOD - BAD-random)=1.000000; Δ_p75_slack(GOOD - BAD-random)=0.000000

### 2026-01-22 - ACCURACY-A1 isomers baseline (DFT truth ordering vs H_trace proxy)

- Code PR (truth contract + runner): https://github.com/RobertPaulig/Geometric_table/pull/201 (merge: 517a1f125afb43b54b6ee961ff48b55c29af5335)
- Code PR (publish workflow + provenance in pack): https://github.com/RobertPaulig/Geometric_table/pull/202 (merge: 59a9461ec7ebfd2f13e97f321375d2b549baac49)
- Code PR (publish workflow fix: facts step): https://github.com/RobertPaulig/Geometric_table/pull/203 (merge: 0efe6602c636bc587745f6acfa95c6295dc12f0c)
- Publish run (failed; fixed in PR #203): https://github.com/RobertPaulig/Geometric_table/actions/runs/21242357438
- Publish run (success): https://github.com/RobertPaulig/Geometric_table/actions/runs/21242635179
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/accuracy-a1-isomers-2026-01-22-r1
- Source commit: 0efe6602c636bc587745f6acfa95c6295dc12f0c
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-22-r1/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): A44624D1B705CA000FEC2DB92EE871E29ACD052E51F30B1C06F7468CF8258A89 (verified by download)
- Pack contains (provenance): data/accuracy/isomer_truth.v1.csv; docs/contracts/isomer_truth.v1.md; data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv (+ .sha256)
- Facts (metrics.json): rows_total=35; groups_total=10; mean_spearman_pred_vs_truth=0.0533647488893285; pairwise_order_accuracy_overall=0.5; top1_accuracy_mean=0.2
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/204 (merge: ee836c9b0dc432e8eaf6a4311853eb4a67338409)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21242765078

### 2026-01-22 - ACCURACY-A1.2 isomers signal repair sweep (operator-spectrum predictors)

- Code PR (sweep runner + publish workflow): https://github.com/RobertPaulig/Geometric_table/pull/206 (merge: 89c8e9d87ead70063a2b2bdb532a3d37d245cbd1)
- Code PR (publish workflow fix: facts step): https://github.com/RobertPaulig/Geometric_table/pull/207 (merge: 13c2ee2d66bad98a811962181b4198c5f271a9d8)
- Publish run (failed; fixed in PR #207): https://github.com/RobertPaulig/Geometric_table/actions/runs/21246384599
- Publish run (success): https://github.com/RobertPaulig/Geometric_table/actions/runs/21246627533
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/accuracy-a1-isomers-2026-01-22-r2
- Source commit: 13c2ee2d66bad98a811962181b4198c5f271a9d8
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-22-r2/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): E04117E5AB26B7248507AEA21159F98512274B8051ADFFD30D0ADA98F2D4A0D4 (verified by download)
- Pack contains (provenance): data/accuracy/isomer_truth.v1.csv; docs/contracts/isomer_truth.v1.md; data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv (+ .sha256)
- Facts (metrics.json): rows_total=35; groups_total=10; verdict=SIGNAL_OK (mean_spearman_pred_vs_truth>=0.2); baseline_mean_spearman=-0.022816061641181683; baseline_pairwise_order_accuracy_overall=0.5434782608695652; baseline_top1_accuracy_mean=0.3; best_mean_spearman=0.3799999999999999; best_pairwise_order_accuracy_overall=0.6956521739130435; best_top1_accuracy_mean=0.5; best_config(predictor=logdet_shifted_eps, edge_weight_mode=unweighted, potential_mode=static, gamma=0.25, beta=None)
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/208 (merge: 47f723f7e1e84864e1fe585052a9e4e9aa491adc)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21246722991

### 2026-01-22 - ACCURACY-A1.3 narrow calibration sweep (logdet_shifted_eps gamma/eps/shift grid)

- Code PR (A1.3 sweep params + group-aware metrics + publish workflow): https://github.com/RobertPaulig/Geometric_table/pull/211 (merge: 14970cf76d9f44d8f18e3a1df503e454353717e9)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21250567759
- Publish run (success): https://github.com/RobertPaulig/Geometric_table/actions/runs/21250626844
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/accuracy-a1-isomers-2026-01-22-a1_3-r1
- Source commit: 14970cf76d9f44d8f18e3a1df503e454353717e9
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-22-a1_3-r1/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): D5DE1211A7C6ADF78419A6AA9ADCB8F530E5B6C68985363F70715CAE159361A5 (verified by download)
- Pack contains (provenance): data/accuracy/isomer_truth.v1.csv; docs/contracts/isomer_truth.v1.md; data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv (+ .sha256); provenance.json
- Facts (metrics.json): rows_total=35; groups_total=10; verdict=SIGNAL_OK (mean_spearman_pred_vs_truth>=0.2); baseline_mean_spearman=0.0533647488893285; baseline_median_spearman_by_group=0.3054092553389459; baseline_pairwise_overall=0.5; baseline_pairwise_by_group_mean=0.42000000000000004; baseline_top1=0.2; best_mean_spearman=0.3899999999999999; best_median_spearman_by_group=0.5499999999999998; best_pairwise_overall=0.6739130434782609; best_pairwise_by_group_mean=0.6633333333333333; best_top1=0.5; best_config(predictor=logdet_shifted_eps, edge_weight_mode=unweighted, potential_mode=static, gamma=0.28, eps=1e-6, shift=0.0)
- KPI A1.3 gates (mean>=0.55 & median>=0.50): FAIL (mean=0.3899999999999999, median=0.5499999999999998)
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/212 (merge: 9364fcc58fb79d93e4bfe16b288dd2a75b2e58b8)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21250755897

### 2026-01-22 - ACCURACY-A1.4 feature upgrade (chemistry-aware operator + multi-feature ridge; train/test split)

- Parent release (A1.3): https://github.com/RobertPaulig/Geometric_table/releases/tag/accuracy-a1-isomers-2026-01-22-a1_3-r1
- Code PR (A1.4 runner + tests + publish workflow + roadmap): https://github.com/RobertPaulig/Geometric_table/pull/214 (merge: dacfb1512c54b033da783a1149e59805e4ed0a64)
- Code PR (publish workflow fix: facts step quoting): https://github.com/RobertPaulig/Geometric_table/pull/215 (merge: bdc279f492ccf4c76b469158867797879fa6af40)
- Code PR (publish workflow fix: facts output newlines): https://github.com/RobertPaulig/Geometric_table/pull/216 (merge: b6498550160e7e4b3d998ffb0878f4d88027ada9)
- Code PR (publish workflow fix: GITHUB_OUTPUT newlines): https://github.com/RobertPaulig/Geometric_table/pull/217 (merge: 04eefabd05c262646d29ce93830951514646ab52)
- Publish run (failed): https://github.com/RobertPaulig/Geometric_table/actions/runs/21256741669
- Publish run (failed): https://github.com/RobertPaulig/Geometric_table/actions/runs/21256956558
- Publish run (failed): https://github.com/RobertPaulig/Geometric_table/actions/runs/21257134909
- Publish run (failed; required contexts gate): https://github.com/RobertPaulig/Geometric_table/actions/runs/21257304691
- Publish run (success): https://github.com/RobertPaulig/Geometric_table/actions/runs/21257398707
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/accuracy-a1-isomers-2026-01-22-a1_4-r1
- Source commit: 04eefabd05c262646d29ce93830951514646ab52
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-22-a1_4-r1/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): 785DD76FCD254EB46E447693BF10FC4C97BD33468BF3AE7FF850D6201DED864B (verified by download)
- Pack contains (provenance): data/accuracy/isomer_truth.v1.csv; docs/contracts/isomer_truth.v1.md; data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv (+ .sha256); data/atoms_db_v1.json; predictions.csv; group_metrics.csv; best_config.json
- Facts (metrics.json): rows_total=35; groups_total=10; split(train_groups=7, test_groups=3); kpi_verdict=FAIL; test_mean_spearman_by_group=0.5999999999999999; test_median_spearman_by_group=0.4999999999999999; test_pairwise_order_accuracy_overall=0.75; test_top1_accuracy_mean=0.3333333333333333
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/218 (merge: c8144c8d12781fa18fa8db140f4e0dc7dd42e291)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21257564605

### 2026-01-22 - ACCURACY-A1.5 pairwise ranking (within-group ranking; LOOCV by group_id)

- Parent release (A1.4): https://github.com/RobertPaulig/Geometric_table/releases/tag/accuracy-a1-isomers-2026-01-22-a1_4-r1
- Code PR (A1.5 runner + tests + publish workflow + roadmap): https://github.com/RobertPaulig/Geometric_table/pull/220 (merge: 9f61b4e55dda142e6fed8668fe074532c4c53d10)
- CI run (3/3 ci/* on code merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21261301899
- Publish run (failed; required contexts gate): https://github.com/RobertPaulig/Geometric_table/actions/runs/21261320907
- Publish run (success): https://github.com/RobertPaulig/Geometric_table/actions/runs/21261442088
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/accuracy-a1-isomers-2026-01-22-a1_5-r2
- Source commit: 9f61b4e55dda142e6fed8668fe074532c4c53d10
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-22-a1_5-r2/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): 101D6FED30C26B3A2B049203C270E51E118E9EB4164F39A579E3FEDF9FBFD7A1 (verified by download)
- Pack contains (provenance): data/accuracy/isomer_truth.v1.csv; docs/contracts/isomer_truth.v1.md; data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv (+ .sha256); data/atoms_db_v1.json; predictions.csv; fold_metrics.csv; group_metrics.csv; best_config.json; provenance.json (source_sha_main=9f61b4e55dda142e6fed8668fe074532c4c53d10)
- Facts (metrics.json): rows_total=35; groups_total=10; cv_method=LOOCV_GROUP_ID; model_type=pairwise_logistic_l2; kpi_verdict=FAIL; mean_spearman_by_group=0.36999999999999994; median_spearman_by_group=0.5499999999999998; pairwise_order_accuracy_overall=0.6956521739130435; top1_accuracy_mean=0.5; worst_groups=[C11H21B1N2O4, C15H24O1, C21H23N3O3]
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/221 (merge: 6d9a2304c35339b7ccd43437149fc721e1a54c4e)
- CI run (3/3 ci/* on registry merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21261555324

### 2026-01-23 - ACCURACY-A2 isomers self-consistent functional (graph-only)

- Source commit: 48e530faffd70a67905be03de0ac6a2d85cc0c55
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21270487891
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/accuracy-a1-isomers-2026-01-23-a2-r1
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-23-a2-r1/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): D407C9FD85DFE87130D092E571B49F42B115836801BB80EF0B0C3843DB6E7A72
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/225
- Outcome (facts from metrics.json):
  - rows_total: 35
  - groups_total: 10
  - kpi.verdict: FAIL
  - loocv_test:
    - mean_spearman_by_group: 0.15999999999999998
    - median_spearman_by_group: 0.3999999999999999
    - pairwise_order_accuracy_overall: 0.5869565217391305 (27/46)
    - top1_accuracy_mean: 0.4
  - worst_groups:
    - C13H20O1: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C15H24O1: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C22H22N4O2: spearman=-0.4999999999999999, top1=0.0, pairwise_acc=0.3333333333333333

### 2026-01-23 - ACCURACY-A2.1 isomers full functional v1 (heat-kernel diag + SCF)

- Source commit: 089cd1ed09003ed2df0feea038992c0ca107ed34
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21272574463
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/accuracy-a1-isomers-2026-01-23-a2_1-r1
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-23-a2_1-r1/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): 99109A00B060F39C2C83028EB0D57CD2BC1CB227E74B7741B4000E732D41D2AC
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/228
- Outcome (facts from metrics.json):
  - rows_total: 35
  - groups_total: 10
  - kpi.verdict: FAIL
  - loocv_test_functional_only:
    - mean_spearman_by_group: -0.019999999999999997
    - median_spearman_by_group: 0.24999999999999994
    - pairwise_order_accuracy_overall: 0.4782608695652174 (22/46)
    - top1_accuracy_mean: 0.1
    - num_groups_spearman_negative: 4
  - loocv_test_calibrated_linear:
    - mean_spearman_by_group: -0.059999999999999984
    - median_spearman_by_group: 0.24999999999999994
    - pairwise_order_accuracy_overall: 0.5217391304347826 (24/46)
    - top1_accuracy_mean: 0.2
    - num_groups_spearman_negative: 4
  - worst_groups:
    - C15H24O1: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C22H22N4O2: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C20H25N3O2: spearman=-0.7999999999999998, top1=0.0, pairwise_acc=0.16666666666666666

### 2026-01-23 - ACCURACY-A2.2 isomers full functional v1 (variationally-stable SCF)

- Source commit: 8ee3c424dbe39fd1daab842d3ce2f2786704e80c
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/21274484646
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/accuracy-a1-isomers-2026-01-23-a2_2-r1
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-23-a2_2-r1/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): 24504681610E2C0382C08B2310D93AEFB9C6E54C4059CD5D0CC6ACF2DBDA257A
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/231
- Outcome (facts from metrics.json):
  - rows_total: 35
  - groups_total: 10
  - kpi.verdict: FAIL
  - loocv_test_functional_only:
    - mean_spearman_by_group: -0.039999999999999994
    - median_spearman_by_group: 0.3499999999999999
    - pairwise_order_accuracy_overall: 0.5 (23/46)
    - top1_accuracy_mean: 0.2
    - num_groups_spearman_negative: 4
  - loocv_test_calibrated_linear:
    - mean_spearman_by_group: 0.009999999999999998
    - median_spearman_by_group: 0.14999999999999997
    - pairwise_order_accuracy_overall: 0.5217391304347826 (24/46)
    - top1_accuracy_mean: 0.1
    - num_groups_spearman_negative: 4
  - worst_groups:
    - C21H23N3O3: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C15H24O1: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C22H22N4O2: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
