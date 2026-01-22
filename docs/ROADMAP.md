# VALUE-TRACK (P0): Pfizer-ready FACTS — доказываем пользу, не “маркетинг”
Принцип: SaaS имеет смысл только если VALUE доказан артефактами.

## Общие правила (обязательные)
- Любая VALUE-веха считается DONE только если:
  1) есть GitHub Release asset (zip),
  2) есть SHA256 на asset,
  3) есть запись в [Artefacts registry](artefacts_registry.md) (URL + SHA256 + команда + outcome),
  4) выполнены CI gates на целевом SHA: ci/test + ci/test-chem + ci/docker.
  См. [CONTEXT](../CONTEXT.md), [Release checklist](95_release_checklist.md), [Artefacts registry](artefacts_registry.md), [Backlog](04_backlog.md).
- "Тихих провалов" быть не должно: каждая строка входа отражена как OK/SKIP/ERROR.
- Полезность измеряем не "ощущениями", а полями verdict/gate/slack/margin в summary.csv (они уже пишутся).

---

## VALUE-M0 — Pipeline Truth (уже есть, держим как baseline)
Цель: доказать, что evidence pack стабильно строится, ERROR=0, determinism проверяем, SHA256 фиксируем.

DoD:
- Выполняется release checklist: pack содержит summary.csv/metrics.json/index.md/manifest.json/checksums.sha256/evidence_pack.zip
- metrics.json: counts.ERROR == 0
- Registry entry добавлен (URL + SHA256 + команда + outcome)

---

## VALUE-M1 — Chem Coverage Suite (“кольца/ароматика/гетероциклы”) как факт
Цель: доказать, что на химическом подмножестве с кольцами пайплайн не имеет слепых зон и даёт предсказуемые SKIP-reasons.

Scope (v1 минимально):
- Используем “seed” список из pilot_generate_input.py (есть benzene=c1ccccc1 + aromatics)
- Генерация decoys идёт в режиме lock_aromatic=True, allow_ring_bonds=False (фиксируем как часть контракта suite)

DoD (жёсткий):
- Release tag: value-ring-suite-YYYY-MM-DD
- Pack собран с --zip_pack и валидирован “без распаковки” (zipfile -t и grep обязательных файлов)
- Quality: ERROR=0
- Метрики/факты, которые фиксируем в release notes + registry:
  - OK/SKIP/ERROR
  - top_reasons SKIP (invalid_smiles/too_many_atoms/disconnected/no_decoys_generated/…)
  - доля строк с n_decoys>0 (из summary.csv)
- Registry entry обязателен

---

## VALUE-M2 — Known Bad / Known Good (доказать, что мы “ловим плохую модель”)
Статус: [ ] planned  [ ] in-progress  [x] done (r2)

Proof (r2, факты/истина):
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-known-bad-good-2026-01-11-r2
- Assets (zip):
  - BAD-constant: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-11-r2/value_known_bad_good_BAD-constant_evidence_pack.zip
    - SHA256: 5B117E204E9E98128EE4C6BEBC609D4282862DBF3BEB935FF432076809F0046A
  - BAD-random: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-11-r2/value_known_bad_good_BAD-random_evidence_pack.zip
    - SHA256: E4255428FC9EEE082D54B04D6A01E7EE98F5F59717CBA259590D6457F1064916
  - GOOD-synthetic: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-11-r2/value_known_bad_good_GOOD-synthetic_evidence_pack.zip
    - SHA256: 228E5B0515316831DE5D208BEE624240973215BDAC85236C1583DEC1B7EA6B5C
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/34 (merge: 1e8079ad7ca1a71f00b186574865ecac853d68c0)
- Lineage PR: https://github.com/RobertPaulig/Geometric_table/pull/35 (merge: 2c9516d8c1db2c8e0b40fb935067dfb165dc5f69)
- Separation facts (OK-only; no auto-threshold gating):
  - Δ_median_slack(GOOD - BAD-constant): 0.500000
  - Δ_PASS_rate(GOOD - BAD-constant): 1.000000
  - Δ_PASS_rate(GOOD - BAD-random): 0.333333

Примечание (честность): r1 (`value-known-bad-good-2026-01-11`) имел separation Δ=0.000000; r2 восстановил separation через `hetero_audit.v2` (см. CORE-AUDIT-FIX-1 ниже).
Цель: показать, что verdict/gate/slack реально реагируют на деградацию скоринга, а не являются шумом.

Как меряем “пользу”:
- Берём один и тот же suite input (из M1).
- Прогоняем несколько scores-режимов, и сравниваем распределения:
  - доля gate=PASS/FAIL,
  - медиана slack/margin,
  - доля “verdict=FAIL” и т.п.
  Эти поля уже в summary.csv.

Набор режимов (v1):
1) BAD-constant: все scores одинаковые
2) BAD-random: случайные scores (фикс seed)
3) GOOD-synthetic: “оригинал выше, decoys ниже” (синтетический эталон)
4) (опц.) REAL: внешний scores.json от клиента (позже)

DoD (жёсткий, измеримый):
- Выпущены минимум 3 релиза-ассета (или один релиз с 3 ассетами) с тегом:
  value-known-bad-good-YYYY-MM-DD
- Для BAD vs GOOD есть разделение:
  - медиана slack(GOOD) > медиана slack(BAD) минимум на Δ (старт: Δ=0.05, подстраиваем по факту),
  - и/или PASS-rate(GOOD) - PASS-rate(BAD) ≥ 20 п.п.
- ERROR=0 во всех пакетах
- Registry entry на каждый asset (или явное перечисление asset_url+sha256 по каждому)

---

## VALUE-M3 — Customer Truth (Pfizer / proxy) + acceptance criteria
Статус: [ ] planned  [ ] in-progress  [x] done

Acceptance criteria (контракт): [`docs/value_m3_acceptance_criteria.md`](value_m3_acceptance_criteria.md)
Цель: перейти от синтетики к реальной проверке пользы на данных/скорах клиента.

DoD:
- Согласованные acceptance criteria (в терминах gate/slack/FAIL-rate на их наборах)
- Выпуск evidence pack (возможно private) с тем же форматом истины: manifest/checksums/zip + SHA256
- Политика хранения/доступа (если private) описана в docs, но формат и верификация такие же

Proof:
- r1 (без явного гейта `scores_coverage.rows_missing_scores_input` в артефакте):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-customer-proxy-2026-01-12
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-customer-proxy-2026-01-12/value_customer_proxy_evidence_pack.zip
  - SHA256(value_customer_proxy_evidence_pack.zip): FE7AA762DCC6D512771DA40E90EB58557B32D6A3641033C65546D0553C16B225
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/20914040368
  - Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/41 (merge: b3e2602bc08a1e9164ce6cad9fb322f65585db2c)
  - Lineage PR: https://github.com/RobertPaulig/Geometric_table/pull/42 (merge: acfb13224f7727e8b89a638708f17ddf44b5e7e5)
- r2 (с гейтом `scores_coverage.rows_missing_scores_input==0` и фиксацией в фактах):
  - Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-customer-proxy-2026-01-12-r2
  - Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-customer-proxy-2026-01-12-r2/value_customer_proxy_evidence_pack.zip
  - SHA256(value_customer_proxy_evidence_pack.zip): C2A8350EFA0D8BEB957E65DE42C0591080085F614B10E255430774B463F67029
  - Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/20922529046
  - Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/46 (merge: 5064d50cddc88622872a3faf67655dd395701342)
  - Lineage PR: https://github.com/RobertPaulig/Geometric_table/pull/47 (merge: 457a14ce898522c417f2d76db6faf7892736f93e)

---

## VALUE-M4 — “Мы претендуем на стандарт”
Цель: формализовать стандарт не как “мы молодцы”, а как:
- контракт формата (manifest/checksums/registry),
- контракт метрик/полей (summary.csv schema),
- и минимальные acceptance tests.

DoD:
- Документ “Standard claims” в docs (что именно гарантируем)
- Backward-compat тесты на summary.csv и hetero_scores.v1
- Процедура изменения (breaking change policy)

---

## VALUE-M5 - Utility (cost&lift) pipeline: proxy truth -> cost_lift_report.json -> release/registry/lineage
Статус: [ ] planned  [ ] in-progress  [x] done

Цель: публиковать utility-артефакт “Claim-2 money proof” (cost&lift на expensive truth) как воспроизводимую цепочку (truth -> report -> pack -> registry -> lineage).

Proxy truth rule (v1, no-leakage):
- truth зависит только от input (`molecule_id` + canonical_smiles) и детерминированен (byte-for-byte)
- truth.csv валиден по `customer_truth.v1` и содержит `truth_source=proxy_rule_v1`
- truth не читает `summary.csv/metrics.json/audit` и не импортирует audit-модули

DoD:
- Release tag: `value-utility-proxy-YYYY-MM-DD(-rN)`
- evidence_pack.zip содержит `truth.csv` и `cost_lift_report.json` (+ zip validate и обязательные файлы pack)
- `cost_lift_report.json` валиден по `cost_lift.v1` (schema + методы + bootstrap CI) и использует `skip_policy=unknown_bucket`
- Quality gates: `ERROR=0`, `scores_coverage.rows_missing_scores_input==0`
- Истина: Release asset + SHA256 + registry entry + lineage entry
- Code gates: `ci/test|ci/test-chem|ci/docker` = success на source SHA

Proof:
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-utility-proxy-2026-01-13
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-proxy-2026-01-13/value_utility_proxy_evidence_pack.zip
- SHA256(value_utility_proxy_evidence_pack.zip): C1AFC8992DDB88B3528030395D8D1E69DB395C7EE89AA5B902EC300A761A3FD4
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/20952518490
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/63 (merge: 6a8b38a1a03870f0266be7a6ba61d6d4d3f99fce)
- Lineage PR: https://github.com/RobertPaulig/Geometric_table/pull/64 (merge: 85611c948c5cae89acecc468bcd459818ed37d11)

Примечание: uplift/дельты не гейтим (фиксируем как факт в outcome + registry), гейтим только валидность/воспроизводимость/no-leakage.

---

## VALUE-M6 - REAL-TRUTH-INGEST-1: external truth.csv -> cost_lift_report.json -> release/registry/lineage
Статус: [ ] planned  [ ] in-progress  [x] done

Цель: считать `cost_lift.v1` на внешней “дорогой” truth (Pfizer/proxy expensive truth) без изменения форматов и без proxy_rule.

Scope (v1):
- truth приходит снаружи (не генерируется в workflow) и проходит строгую верификацию загрузки: truth_url + truth_sha256
- truth.csv валиден по `customer_truth.v1`, `truth_source=external`
- uplift/дельты не гейтим (фиксируем как факт в outcome + registry), гейтим только валидность/воспроизводимость/честность полей

DoD:
- Release tag: `value-utility-realtruth-YYYY-MM-DD(-rN)`
- evidence_pack.zip содержит `truth.csv` и `cost_lift_report.json` (+ zip validate и обязательные файлы pack)
- truth.csv: SHA256 совпадает с переданным `truth_sha256`; валидация `customer_truth.v1` проходит (колонки/типы/дубликаты/coverage)
- `cost_lift_report.json` валиден по `cost_lift.v1` (schema + методы + bootstrap CI) и фиксирует `skip_policy` в отчёте
- Quality gates: `ERROR=0`, `scores_coverage.rows_missing_scores_input==0`
- Истина: Release asset + SHA256 + registry entry + lineage entry
- Code gates: `ci/test|ci/test-chem|ci/docker` = success на source SHA

Proof:
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-utility-realtruth-2026-01-14-r1
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-14-r1/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): 65A00E8879B9B03BF558F630C85EABFC0C285C1B8DF3635D231B4A90DD7D816B
- Publish run: https://github.com/RobertPaulig/Geometric_table/actions/runs/20983871401
- truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
- SHA256(truth.csv): 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/71 (merge: bc7c45d771a16bc8b387e11bcdaac1c18fe79207)
- Lineage PR: https://github.com/RobertPaulig/Geometric_table/pull/72 (merge: 207affa63f7b246b9dd5ac2a52aa8cd6b4d9b27c)

## VALUE-M7 - REAL-SCORES-INGEST-1: external scores.json (hetero_scores.v1) -> utility pack (pinned by sha256)
Статус: [ ] planned  [ ] in-progress  [x] done

Цель: сделать ingest внешних raw scores (frozen `hetero_scores.v1`) как "истину загрузки": `scores_url + scores_sha256`, без генерации scores внутри workflow, с тем же конвейером utility-артефакта (truth -> cost_lift -> pack -> release -> registry -> lineage).

Scope (v1):
- scores приходят снаружи (не генерируются в workflow) и проходят строгую верификацию загрузки: scores_url + scores_sha256
- scores.json валиден по `hetero_scores.v1` (schema_version + required fields); score_key (если присутствует) фиксируется как факт в outcome
- внешний scores-файл кладём в evidence pack и фиксируем provenance (manifest/checksums + release notes + registry outcome)

DoD:
- workflow `publish_value_utility_realtruth.yml` поддерживает optional `scores_url/scores_sha256` и при наличии печатает:
  - `Gate OK: scores.json downloaded + sha256 verified + hetero_scores.v1 validated`
- batch использует внешний `--scores_input` (а не proxy scores)
- evidence_pack.zip содержит `truth.csv`, `cost_lift_report.json`, и внешний scores-файл (например `scores_external.json`)
- Quality gates: `ERROR=0`, `scores_coverage.rows_missing_scores_input==0`
- Истина: Release asset + SHA256 + registry entry + lineage entry
- Code gates: `ci/test|ci/test-chem|ci/docker` = success на source SHA

Proof:
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
- Lineage PR: https://github.com/RobertPaulig/Geometric_table/pull/76 (merge: 8029baf6df12236d99e778845748e40058706272)

---

## Как это превращается в реальные "вехи-факты" через ваши workflows (без нового велосипеда)

Вы уже умеете делать “истину” через publish workflow:

* `publish_stress_pack.yml` генерит вход, гонит `hetero2-batch --zip_pack`, валидирует zip, гейтит ERROR=0, считает SHA256, публикует release asset и делает PR в registry.
* `publish_pilot_pack.yml` делает то же самое, но в режиме external_scores и с `scripts/pilot_generate_input.py`.

**Следствие:** VALUE-M1/M2 делаем либо:

1. расширением/клоном этих workflow (лучше: отдельные `publish_value_*`), либо
2. параметризацией (suite + scores_variant), но всё равно с тем же DoD: zip validate + ERROR=0 + SHA256 + registry.

---

## P0 порядок работ (без SaaS, пока не доказали пользу)

1. **Закрыть V1-CHECKLIST-3 (freeze hetero_scores.v1)** — это prerequisite, иначе внешние scores нельзя называть контрактом.
2. **VALUE-M1** (ring-suite pack)
3. **VALUE-M2** (known bad/good separation packs)
4. Только потом обсуждаем “SaaS масштабирование”, как упаковку уже доказанного ядра.

---

## CORE-AUDIT-FIX-1 — Audit v2 (slack/verdict зависят от качества скоринга)
Статус: [x] done

Proof:
- PR: https://github.com/RobertPaulig/Geometric_table/pull/32
- Merge commit: https://github.com/RobertPaulig/Geometric_table/commit/c2d8b80a6cef402fc3015aff2cc95a82790521e5
- CI run (3/3 ci/* на merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/20897172830

## CI-AUTO-PR-1 — Auto CI для automation PR (без allow-empty commit)
Статус: [x] done

Proof:
- PR #49 (workflow_dispatch + trigger CI): https://github.com/RobertPaulig/Geometric_table/commit/53a61417fb1b07d81506cdc539fd797151af805c
- Publish run verify (лог содержит `pytest run url:` + `tested_sha:`): https://github.com/RobertPaulig/Geometric_table/actions/runs/20937794112
- Automation PR #52 (live proof, merged): https://github.com/RobertPaulig/Geometric_table/commit/f089a00b235567b6af09ad09b21ee0883150ab64
- Lineage PR #53 (запись в docs/90_lineage.md): https://github.com/RobertPaulig/Geometric_table/commit/0691248c135c11486d7c82a607386b51266bd359

---

## Physics Roadmap (operator + anti-иллюзия)

Цель: сделать признаки/скоринг чувствительными к химии (тип атома) и запретить “красивую AUC” на лёгких ловушках.
Все “успехи” фиксируются артефактами evidence pack + registry + lineage.

### PHYSICS-P0 (текущий): `H = L + V` + hardness curve + `auc_interpretation`
Статус: [ ] planned  [x] in-progress  [ ] done

DoD (фактами):
- В коде есть режим `physics_mode=topological|hamiltonian|both`, а `H` строится как `H = L + diag(V)`.
- SoT параметров `V(t)` — только `data/atoms_db_v1.json` (см. `docs/contracts/PHYSICS_OPERATOR_CONTRACT.md`), missing params → строка `ERROR` (audit-grade, без “тихих” провалов).
- В evidence pack есть `hardness_curve.csv` (пары) + `hardness_curve.md` + `summary_metadata.json` с `auc_interpretation`.
- Тест `tests/test_physics_operator_blindness.py` зелёный: `spec(L)` слеп к типам атомов, `spec(H)` реагирует.
- Замкнута цепочка publish→release(+sha256)→registry→lineage на main (3/3 ci/*).

### PHYSICS-P1: weighted adjacency (edge weights) без поломки P0
Статус: [ ] planned  [ ] in-progress  [ ] done

DoD:
- Добавлен `A_w`/`L_w=D_w-A_w` (включается флагом, не ломает P0).
- Evidence pack содержит сравнение `L` vs `L_w` vs `H` и метрики на hard-bin (не только overall).

### PHYSICS-P2: локальные спектральные признаки (DOS/LDOS/heat-kernel)
Статус: [ ] planned  [ ] in-progress  [ ] done

DoD:
- Добавлены локальные спектральные фичи (например LDOS/heat-kernel) с фиксированными параметрами/seed policy.
- Улучшение различимости на hard-ловушках без деградации hardness coverage.

### PHYSICS-P3: ускорение solver (итеративный/асинхронный) с детерминизмом
Статус: [ ] planned  [ ] in-progress  [ ] done

DoD:
- Итеративный solver логирует сходимость (итерации/остаток) и сохраняет артефакт в evidence pack.
- Детерминизм подтверждён через rerun и совпадающие checksums.

### PHYSICS-P3.1: CALIBRATION-GAMMA-1 (масштаб потенциала, без физ-claims)
Статус: [ ] planned  [ ] in-progress  [ ] done

Entry criteria:
- PHYSICS-P3 “self-consistent / trace / verdict” работает и детерминизм подтверждён.
- В evidence pack явно зафиксированы `potential_unit_model="dimensionless"` и `potential_scale_gamma` (см. контракт).

DoD:
- Есть скрипт `scripts/calibrate_gamma.py`, который принимает calibration set (CSV) и подбирает `potential_scale_gamma`.
- Скрипт пишет `calibration_report.json` с параметрами/метриками/идентификатором набора.
- Результат калибровки фиксируется как “истина” через evidence pack + registry entry (dataset-id + gamma + отчёт).

Hard rule:
- Пока PHYSICS-P3.1 не DONE, запрещены claims про физические единицы (Å/eV). Разрешены только dimensionless/trend claims.

### PHYSICS-P4 (опционально): ускорение интеграции (SADDLa)
Статус: [ ] planned  [ ] in-progress  [ ] done

Entry criteria (обязательные факты):
- В пайплайне есть конкретный модуль/функция, где считается высокоразмерный интеграл/квадратура (ссылка/путь фиксируются).
- В evidence pack есть профиль исполнения (pstats/json/аналог), и интегральный шаг занимает ≥ 25% runtime.
- Бейзлайн-метод интеграции (например MC/QMC/др.) не проходит по стоимости/точности (фиксируется фактами).

DoD:
- Добавлен режим “integration acceleration”, включаемый флагом (дефолтное поведение без флага не меняется).
- Evidence pack содержит профиль “до/после” и подтверждение детерминизма (seed policy + checksums).
- Улучшение фиксируется на hard-bin метриках (не только overall), без ухудшения hardness coverage.
- Замкнута цепочка publish→release(+sha256)→registry→lineage на main (3/3 ci/*).

### PHYSICS-P5.1-INTEGRATION-SCALE-LAW-1: integration scale law (contract + gates + canonical evidence pack)
Status: [ ] planned  [ ] in-progress  [x] done

Closed:
- PR #162 (code): https://github.com/RobertPaulig/Geometric_table/pull/162
- Publish run (large-scale proof): https://github.com/RobertPaulig/Geometric_table/actions/runs/21184833947
- Release r2 (zip+.sha256): https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r2
- PR #163 (registry): https://github.com/RobertPaulig/Geometric_table/pull/163
- PR #164 (lineage): https://github.com/RobertPaulig/Geometric_table/pull/164

Goal:
- Freeze "what counts as a win" for integration scaling: correctness-first; speedup is evaluated only at scale (N_atoms >= gate_n_min); speedup<1 on small N is allowed (overhead region).

Entry criteria:
- P5 large-scale pack builder / publish workflow exists (speedup vs N_atoms).
- CI gate infrastructure exists: `ci/test|ci/test-chem|ci/docker` (tested merge-ref).

DoD (law is DONE only when the truth-chain is closed):
- Contract doc merged to main: `docs/contracts/INTEGRATION_SCALE_CONTRACT.md` (p5.1.v1).
- Evidence pack contains canonical artifacts (or `canonical_mapping.json`); `summary_metadata.json` includes `law_ref.*` and P5.1 verdict fields.
- Contract-tests enforce: ZIP contents + required metadata fields + verdict logic (correctness gates speed metrics; scale-only gating).
- Publish workflow has a fail-fast gate for P5.1 metadata/verdict (anti-drift).
- Live proof chain-of-truth closed: publish-run -> release(zip+.sha256) -> registry PR -> lineage PR (append-only) -> main 3/3.

Hard rule:
- No new integration optimizations/claims until P5.1 is merged to main; speed metrics are NOT KPI if FAIL_CORRECTNESS_AT_SCALE.

### PHYSICS-P5.2-COST-DECOMPOSITION-1: cost decomposition for integration runtime (timing breakdown + bottleneck verdict)
Status: [ ] planned  [ ] in-progress  [x] done

Closed:
- PR #165 (code): https://github.com/RobertPaulig/Geometric_table/pull/165
- Publish run r3 (large-scale proof): https://github.com/RobertPaulig/Geometric_table/actions/runs/21188218951
- Release r3 (zip+.sha256): https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r3
- PR #166 (registry): https://github.com/RobertPaulig/Geometric_table/pull/166
- PR #167 (lineage): https://github.com/RobertPaulig/Geometric_table/pull/167
- PR #168 (registry addendum: P5.2 cost facts): https://github.com/RobertPaulig/Geometric_table/pull/168

Goal:
- Stop optimizing blind: decompose runtime per fixture into operator build vs DOS/LDOS eval vs integrator logic vs I/O, and publish a bottleneck verdict.

Entry criteria:
- PHYSICS-P5.1 is DONE on main (truth-chain closed; evidence pack + registry + lineage).
- P5 large-scale pack builder / publish workflow exists (speedup vs N_atoms).

DoD:
- Evidence pack contains `timing_breakdown.csv` with per-sample rows and N_atoms-bin aggregates.
- `timing_breakdown.csv` schema (per-sample): `build_operator_ms`, `dos_ldos_eval_ms`, `integration_logic_ms`, `io_ms`, `total_ms`.
- `summary_metadata.json` includes a bottleneck verdict at scale: `BOTTLENECK_IS_DOS_LDOS | BOTTLENECK_IS_INTEGRATOR | BOTTLENECK_IS_IO | MIXED`.
- Contract-tests / publish gates validate presence and schema of timing breakdown (anti-drift).

Hard rule:
- No new integration optimizations/claims until P5.2 is merged; next actions must be justified by the bottleneck verdict.

### PHYSICS-P5.3-INTEGRATION-LOGIC-OPT-1: integration logic optimization (reduce integration_logic_ms; KPI + gates)
Status: [ ] planned  [ ] in-progress  [x] done

Closed:
- PR #169 (code): https://github.com/RobertPaulig/Geometric_table/pull/169
- Publish run r4 (large-scale proof): https://github.com/RobertPaulig/Geometric_table/actions/runs/21193382219
- Release r4 (zip+.sha256): https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r4
- PR #171 (registry): https://github.com/RobertPaulig/Geometric_table/pull/171
- PR #172 (registry addendum: P5.3 thresholds rails): https://github.com/RobertPaulig/Geometric_table/pull/172
- PR #173 (lineage): https://github.com/RobertPaulig/Geometric_table/pull/173

Goal:
- Optimize only integration_logic (adaptive integrator overhead), because P5.2 bottleneck verdict at scale is `BOTTLENECK_IS_INTEGRATOR`.

Entry criteria:
- PHYSICS-P5.2 truth-chain is DONE (publish-run → release(zip+.sha256) → registry + lineage on main).

DoD:
- Integration correctness at scale stays PASS (P5.1 correctness verdict unchanged).
- Evidence pack metadata includes new P5.3 KPI fields:
  - `cost_median_integration_logic_ms_at_scale_before`
  - `cost_median_integration_logic_ms_at_scale_after`
  - `cost_integration_logic_speedup_at_scale`
  - `cost_integration_logic_opt_verdict_at_scale = PASS|FAIL|INCONCLUSIVE`
- Contract-tests and publish gates validate presence of the P5.3 fields (anti-drift) and keep `timing_breakdown.csv` schema intact.

Hard rule:
- No other integration optimizations/claims until P5.3 is merged and truth-chain is closed.

### PHYSICS-P5.4-PERF-HARDNESS-TOPOLOGY-1: ring-suite topology hardness in P5-scale (anti-illusion)
Status: [ ] planned  [ ] in-progress  [x] done

Closed:
- PR #170 (code): https://github.com/RobertPaulig/Geometric_table/pull/170
- Publish run r5 (large-scale proof): https://github.com/RobertPaulig/Geometric_table/actions/runs/21204574412
- Release r5 (zip+.sha256): https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r5
- PR #174 (registry): https://github.com/RobertPaulig/Geometric_table/pull/174
- PR #175 (lineage): https://github.com/RobertPaulig/Geometric_table/pull/175

Goal:
- Add ring-suite to P5-scale and publish speedup profiles per `topology_family` to prevent chain-only (tridiagonal) proof illusions.

Entry criteria:
- PHYSICS-P5.1 is DONE on main (truth-chain closed; evidence pack + registry + lineage).
- P5 large-scale pack builder / publish workflow exists (speedup vs N_atoms).

DoD (law is DONE only when the truth-chain is closed):
- Evidence pack contains both fixture suites:
  - `fixtures_polymer_scale.csv`
  - `fixtures_ring_scale.csv`
- Evidence pack contains per-family speed profile:
  - `speedup_vs_n_by_family.csv` with `family, n_atoms, n_samples, median_speedup, median_eval_ratio, correctness_pass_rate`.
- `summary_metadata.json` includes P5.4 fields and verdict logic:
  - `topology_families = ["polymer","ring"]`
  - `speedup_median_at_scale_polymer`, `speedup_median_at_scale_ring`
  - `speedup_verdict_at_scale_polymer`, `speedup_verdict_at_scale_ring`
  - `topology_hardness_verdict`, `topology_hardness_reason`
- Contract-tests / publish gates validate ring-suite artifacts + P5.4 metadata/verdict (anti-drift).
- Live proof chain-of-truth closed: publish-run -> release(zip+.sha256) -> registry PR -> lineage PR (append-only) -> main 3/3.

Hard rule:
- No scalability conclusions (or single combined speedup claims) without ring-suite and per-family speed profiles.

### PHYSICS-P5.5-RING-PERF-FOLLOW-UP-1: ring performance follow-up (per-family cost profile + verdict rails)
Status: [ ] planned  [ ] in-progress  [x] done

Closed:
- PR #176 (code): https://github.com/RobertPaulig/Geometric_table/pull/176
- Publish run r6 (large-scale proof): https://github.com/RobertPaulig/Geometric_table/actions/runs/21208683069
- Release r6 (zip+.sha256): https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r6
- PR #177 (registry): https://github.com/RobertPaulig/Geometric_table/pull/177
- PR #178 (lineage): https://github.com/RobertPaulig/Geometric_table/pull/178

Goal:
- Produce per-family cost decomposition (polymer vs ring) at scale and a factual verdict for where ring is slower.

Entry criteria:
- PHYSICS-P5.4 truth-chain is DONE (ring-suite + per-family speed profile on main).
- P5 large-scale pack builder / publish workflow exists (evidence pack + gates).

DoD (law is DONE only when the truth-chain is closed):
- Evidence pack contains `timing_breakdown_by_family.csv` (bin-level per-family aggregates).
- `timing_breakdown_by_family.csv` schema includes:
  - `family, n_atoms, n_samples, median_build_operator_ms, median_dos_ldos_eval_ms, median_integration_logic_ms, median_io_ms, median_total_ms`.
- `summary_metadata.json` includes required P5.5 fields:
  - `cost_median_total_ms_at_scale_polymer_estimate`, `cost_median_total_ms_at_scale_ring_estimate`
  - `cost_ratio_ring_vs_polymer_total_ms_at_scale_estimate`
  - `topology_ring_cost_gap_verdict_at_scale`, `topology_ring_cost_gap_reason_at_scale`
- Contract-tests / publish gates validate presence + schema of per-family timing breakdown and P5.5 metadata/verdict (anti-drift).
- Live proof chain-of-truth closed: publish-run -> release(zip+.sha256) -> registry PR -> lineage PR (append-only) -> main 3/3.

Hard rule:
- No ring performance optimizations/claims without per-family cost profile (P5.5).

### PHYSICS-P5.6-RING-SPEEDUP-LAW-1: ring speedup law (contract + gates; no polymer-only claims)
Status: [ ] planned  [ ] in-progress  [x] done

Closed:
- PR #179 (code): https://github.com/RobertPaulig/Geometric_table/pull/179
- Publish run r7 (large-scale proof): https://github.com/RobertPaulig/Geometric_table/actions/runs/21211317639
- Release r7 (zip+.sha256): https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r7
- PR #181 (registry): https://github.com/RobertPaulig/Geometric_table/pull/181
- PR #182 (lineage): https://github.com/RobertPaulig/Geometric_table/pull/182
- PR #183 (lineage incident note): https://github.com/RobertPaulig/Geometric_table/pull/183

Goal:
- Freeze "what counts as ring speedup at scale" as a contract (no interpretation drift).

Entry criteria:
- PHYSICS-P5.4 truth-chain is DONE (ring-suite + per-family speed profile on main).
- PHYSICS-P5.5 truth-chain is DONE (per-family timing + ring cost gap verdict on main).

DoD (law is DONE only when the truth-chain is closed):
- Contract doc merged to main: `docs/contracts/RING_SPEEDUP_CONTRACT.md` (p5.6.v1).
- Evidence pack contains: `speedup_vs_n_by_family.csv`, `timing_breakdown_by_family.csv`, `summary_metadata.json`.
- `summary_metadata.json` includes P5.6 fields + verdict rails:
  - `ring_speedup_median_at_scale`, `ring_eval_ratio_median_at_scale`, `ring_correctness_pass_rate_at_scale`
  - `ring_speedup_verdict_at_scale`, `ring_speedup_verdict_reason_at_scale`
  - linkage rail: `topology_ring_cost_gap_verdict_at_scale` (P5.5)
- Contract-tests / publish gates validate presence + verdict logic (anti-drift).
- Live proof chain-of-truth closed: publish-run -> release(zip+.sha256) -> registry PR -> lineage PR (append-only) -> main 3/3.

Hard rule:
- No chemistry-wide scalability claims without P5.6 ring-law fields in a registry-grade evidence pack.

### PHYSICS-P5.7-BUILD_OPERATOR-OPT-RING-1: build_operator optimization (ring) (narrow; reason-driven)
Status: [ ] planned  [ ] in-progress  [x] done

Closed:
- PR #184 (code): https://github.com/RobertPaulig/Geometric_table/pull/184
- Publish run r8 (large-scale proof): https://github.com/RobertPaulig/Geometric_table/actions/runs/21215792704
- Release r8 (zip+.sha256): https://github.com/RobertPaulig/Geometric_table/releases/tag/physics-operator-large-scale-2026-01-20-r8
- PR #185 (registry): https://github.com/RobertPaulig/Geometric_table/pull/185
- PR #186 (lineage): https://github.com/RobertPaulig/Geometric_table/pull/186

Goal:
- Reduce ring build_operator cost at scale without changing laws/fixtures/baseline parameters.

Entry criteria:
- PHYSICS-P5.6 truth-chain is DONE (ring speedup law accepted).
- Latest registry-grade evidence indicates: `topology_ring_cost_gap_verdict_at_scale = RING_SLOWER_DUE_TO_BUILD_OPERATOR`.

DoD (optimization is DONE only when the truth-chain is closed):
- Code PR merged (main CI 3/3).
- P5 large-scale proof publish-run on main succeeds and produces a new evidence pack + `.sha256`.
- Registry + lineage entries record updated build_operator ring-vs-polymer cost facts (numbers + verdict reasons), with stable law refs.

Hard rule:
- No P5.6 contract/threshold/fixtures/baseline parameter changes in P5.7 PR (only build_operator optimization).

### PHYSICS-P5.8-INTEGRATION_LOGIC-OPT-RING-1: integration_logic optimization (ring) (narrow; reason-driven)
Status: [ ] planned  [x] in-progress  [ ] done

Goal:
- Reduce ring integration_logic overhead at scale without changing laws/fixtures/baseline parameters.

Entry criteria:
- PHYSICS-P5.7 truth-chain is DONE (build_operator optimization accepted).
- Latest registry-grade evidence indicates: `topology_ring_cost_gap_verdict_at_scale = RING_SLOWER_DUE_TO_INTEGRATION_LOGIC`.

DoD (optimization is DONE only when the truth-chain is closed):
- Code PR merged (main CI 3/3).
- P5 large-scale proof publish-run on main succeeds and produces a new evidence pack + `.sha256`.
- Registry + lineage entries record updated ring-vs-polymer integration_logic cost facts (numbers + verdict reasons), with stable law refs.

Hard rule:
- No P5.6 contract/threshold/fixtures/baseline parameter changes in P5.8 PR (only integration_logic optimization).

---

## Accuracy Roadmap (expensive truth / calibration)

Цель: зафиксировать "expensive truth" (DFT) и получить воспроизводимый baseline-контур, который можно калибровать без дрейфа протокола.

### ACCURACY-A1 (Isomers) — DFT truth + baseline operator proxy

#### ACCURACY-A1.0 — Truth & Contract
Статус: [ ] planned  [ ] in-progress  [x] done

Proof:
- PR #201 (code): https://github.com/RobertPaulig/Geometric_table/pull/201
- Merge commit: https://github.com/RobertPaulig/Geometric_table/commit/517a1f125afb43b54b6ee961ff48b55c29af5335
- CI run (3/3 ci/* on merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21242006439

DoD:
- Raw truth CSV tracked at `data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv` with adjacent `.sha256`.
- Canonical truth generated and tracked: `data/accuracy/isomer_truth.v1.csv`.
- Contract doc merged: `docs/contracts/isomer_truth.v1.md`.
- Runner produces `summary.csv/metrics.json/index.md/manifest.json/checksums.sha256/evidence_pack.zip`: `scripts/accuracy_a1_isomers_run.py`.

#### ACCURACY-A1.1 - Evidence pack release (truth-chain closure)
Статус: [ ] planned  [ ] in-progress  [x] done

Proof:
- Publish run (success): https://github.com/RobertPaulig/Geometric_table/actions/runs/21242635179
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/accuracy-a1-isomers-2026-01-22-r1
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/204 (merge: ee836c9b0dc432e8eaf6a4311853eb4a67338409)
- Lineage PR: https://github.com/RobertPaulig/Geometric_table/pull/205 (merge: 3ce46f46f706e785495719e8789ab7e647803e8e)
- CI run (3/3 ci/* on lineage merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21243152174

DoD (A1 is DONE only when the truth-chain is closed):
- publish-run on main produces an evidence pack asset + `.sha256`.
- release tag + asset URL + SHA256 recorded in `docs/artefacts_registry.md`.
- lineage entry added to `docs/90_lineage.md` (append-only).
- main CI 3/3 green on merge SHAs of registry + lineage PRs.

#### ACCURACY-A1.2 - Isomers: signal repair (sweep predictors + SCF/weights)
Статус: [ ] planned  [ ] in-progress  [x] done

Proof:
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/accuracy-a1-isomers-2026-01-22-r2
- Publish run (success): https://github.com/RobertPaulig/Geometric_table/actions/runs/21246627533
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/208 (merge: 47f723f7e1e84864e1fe585052a9e4e9aa491adc)
- Lineage PR: https://github.com/RobertPaulig/Geometric_table/pull/209 (merge: df4b0f9278a5e3be8d4adab17fd26f22d3d593d6)
- CI run (3/3 ci/* on lineage merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21246973597

Goal:
- Improve isomer ordering signal vs baseline H_trace, without changing the truth dataset and without any large downloads.

DoD (truth-chain closure + best-config facts):
- Sweep runner produces `sweep_results.csv` + `best_config.json` and an evidence pack zip with provenance inputs.
- publish-run on main produces a release asset + `.sha256`.
- registry entry records baseline metrics + best metrics + best_config (append-only).
- lineage entry records the same tag/asset/SHA256 + references to PRs/runs (append-only).
- main CI 3/3 green on merge SHAs of registry + lineage PRs.

Hard rule:
- No large dataset downloads (SPICE hdf5 etc). Only input is `data/accuracy/isomer_truth.v1.csv`.

#### ACCURACY-A1.3 - Narrow calibration around A1.2 best-config (logdet_shifted_eps)
Status: [ ] planned  [ ] in-progress  [x] done

Proof:
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/accuracy-a1-isomers-2026-01-22-a1_3-r1
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/212 (merge: 9364fcc58fb79d93e4bfe16b288dd2a75b2e58b8)
- Lineage PR: https://github.com/RobertPaulig/Geometric_table/pull/213 (merge: 789deed5a4386b8c45db2379e6e2e0efdb2d38c1)
- CI run (3/3 ci/* on lineage merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21251081388

Outcome (facts):
- Release tag: `accuracy-a1-isomers-2026-01-22-a1_3-r1`
- SHA256(zip): `D5DE1211A7C6ADF78419A6AA9ADCB8F530E5B6C68985363F70715CAE159361A5`
- KPI: `mean_spearman_by_group = 0.39` (<0.55; FAIL), `median_spearman_by_group = 0.55` (>=0.50; PASS)
- Best config (fact): `gamma=0.28, eps=1e-6, shift=0.0`

Goal:
- Narrow sweep around A1.2 best predictor (`logdet_shifted_eps`, `gamma≈0.25`) using group-aware metrics (per `group_id`).

DoD (facts):
- Truth dataset unchanged; canonical truth remains reproducible from raw.
- Evidence pack includes `sweep_results.csv`, `best_config.json`, `metrics.json`, `provenance.json`, `checksums.sha256`, `manifest.json`.
- `metrics.json` includes group-aware fields: `spearman_by_group`, `mean_spearman_by_group`, `median_spearman_by_group`, `pairwise_order_accuracy_by_group_mean`, `top1_accuracy_mean`.
- KPI targets (facts, not a gate): `mean_spearman_by_group >= 0.55` and `median_spearman_by_group >= 0.50`.
- Truth-chain closure: publish-run → release(zip+.sha256) → registry → lineage → main CI 3/3.

Hard rule:
- No large dataset downloads (SPICE hdf5 etc). Only input is `data/accuracy/isomer_truth.v1.csv`.

#### ACCURACY-A1.4 - Feature Upgrade (chemistry-aware operator + multi-feature predictor)
Status: [ ] planned  [ ] in-progress  [x] done

Proof:
- Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/accuracy-a1-isomers-2026-01-22-a1_4-r1
- Registry PR: https://github.com/RobertPaulig/Geometric_table/pull/218 (merge: c8144c8d12781fa18fa8db140f4e0dc7dd42e291)
- Lineage PR: https://github.com/RobertPaulig/Geometric_table/pull/219 (merge: c5f31b1c847486e1201e2d5de7576162a468854b)
- CI run (3/3 ci/* on lineage merge SHA): https://github.com/RobertPaulig/Geometric_table/actions/runs/21257778533

Outcome (facts):
- Release tag: `accuracy-a1-isomers-2026-01-22-a1_4-r1`
- SHA256(zip): `785DD76FCD254EB46E447693BF10FC4C97BD33468BF3AE7FF850D6201DED864B`
- KPI verdict: `FAIL`
- Test metrics (facts): `mean_spearman_by_group_test=0.60`, `median_spearman_by_group_test=0.50`, `pairwise_order_accuracy_overall_test=0.75`, `top1_accuracy_mean_test=0.333...`

Goal:
- Improve isomer ordering quality with a chemistry-aware operator and a multi-feature predictor, evaluated on a holdout split by `group_id`.

DoD (facts):
- Truth dataset unchanged; canonical truth remains reproducible from raw.
- Train/test split is by `group_id` (7 train groups / 3 test groups, fixed seed).
- Evidence pack includes `predictions.csv`, `group_metrics.csv`, `metrics.json` (train/test/overall), `best_config.json`, `provenance.json`, `manifest.json`, `checksums.sha256`.
- KPI gates (test split): `mean_spearman_by_group_test >= 0.55`, `median_spearman_by_group_test >= 0.55`, `pairwise_order_accuracy_overall_test >= 0.65`, `top1_accuracy_mean_test >= 0.40`.
- Truth-chain closure: publish-run → release(zip+.sha256) → registry → lineage → main CI 3/3.

Hard rule:
- Do not modify truth files: `data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv`, `data/accuracy/isomer_truth.v1.csv`, `docs/contracts/isomer_truth.v1.md`.

#### ACCURACY-A1.5 - Pairwise ranking + LOOCV by group_id
Status: [ ] planned  [x] in-progress  [ ] done

Goal:
- Improve stability on worst groups and top-1 selection by training a pairwise ranking model within each `group_id`, evaluated via leave-one-group-out CV (LOOCV).

DoD (facts):
- Truth dataset unchanged; canonical truth remains reproducible from raw.
- CV method is LOOCV by `group_id` (every group is a test fold once; seed fixed for fold order).
- Evidence pack includes `predictions.csv` (out-of-sample), `fold_metrics.csv`, `metrics.json`, `best_config.json`, `provenance.json`, `manifest.json`, `checksums.sha256`.
- KPI gates (LOOCV test folds): `mean_spearman_by_group_test >= 0.55`, `median_spearman_by_group_test >= 0.55`, `pairwise_order_accuracy_overall_test >= 0.70`, `top1_accuracy_mean_test >= 0.40`.
- Truth-chain closure: publish-run → release(zip+.sha256) → registry → lineage → main CI 3/3.

Hard rule:
- Do not modify truth files: `data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv`, `data/accuracy/isomer_truth.v1.csv`, `docs/contracts/isomer_truth.v1.md`.

#### ACCURACY-A2 - Self-consistent functional (graph-only)
Status: [x] planned  [ ] in-progress  [ ] done

Goal:
- Fix worst-group inversions by scoring isomers with a self-consistent functional on the 2D graph (no 3D, no new datasets).

DoD (facts):
- Truth dataset unchanged; canonical truth remains reproducible from raw.
- CV method is LOOCV by `group_id` (every group is a test fold once; seed fixed for fold order).
- Evidence pack includes `predictions.csv` (out-of-sample), `fold_metrics.csv`, `group_metrics.csv`, `metrics.json`, `best_config.json`, `provenance.json`, `manifest.json`, `checksums.sha256`.
- KPI gates (LOOCV test folds): `mean_spearman_by_group_test >= 0.55`, `median_spearman_by_group_test >= 0.55`, `pairwise_order_accuracy_overall_test >= 0.70`, `top1_accuracy_mean_test >= 0.40`.
- Truth-chain closure: publish-run ¢Å' release(zip+.sha256) ¢Å' registry ¢Å' lineage ¢Å' main CI 3/3.

Hard rule:
- Do not modify truth files: `data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv`, `data/accuracy/isomer_truth.v1.csv`, `docs/contracts/isomer_truth.v1.md`.

# ROADMAP - HETERO-2 как SaaS (Pfizer-ready evidence pipeline)

Назначение: зафиксировать целевую картину SaaS и вести разработку через обязательные вехи (milestones).
Этот документ - "истина по развитию продукта" (в отличие от сырого склада идей в backlog).

---

## 0) Правила ведения разработки (обязательные)

### 0.1. “Веха = контракт”
Каждая веха (Milestone) имеет:
- цель,
- список обязательных эпиков/работ,
- Definition of Done (DoD),
- артефакты/доказательства (Artifacts/Proof).

**Запрещено** объявлять веху “готовой”, если не выполнен её DoD и не приложены артефакты.

### 0.2. “Каждый PR привязан к Roadmap-ID”
Каждый PR обязан:
- ссылаться на конкретный Roadmap-ID (например: `SAAS-M1`, `SAAS-E2`),
- обновлять прогресс в ROADMAP (галочки / status),
- проходить CI-гейты проекта.

### 0.3. “Никаких тихих провалов”
В SaaS-пайплайне любая ошибка/пропуск обязан быть:
- явным (ERROR / SKIP),
- объяснённым (reason),
- агрегированным в метриках/отчёте.

### 0.4. Evidence pack — единица ценности
Единица результата: **evidence pack** (zip/dir), содержащий manifest и checksums.
Система должна уметь:
- повторить run по manifest,
- доказать целостность (sha256),
- объяснить SKIP/ERROR.

---

## 1) North Star: что продаём

HETERO-2 = “CI/CD для молекулярных ML-моделей”:
загрузил данные/предсказания → Run → получил воспроизводимый evidence pack + понятные failure modes →
можно шарить внутри организации и прикладывать к аудиту.

### Commercial Goal (Ultra-fast validation of generated molecules)

#### The market problem
Modern generative chemistry can produce **millions** of “drug-like” molecules, but the bottleneck is **validation**.
Running expensive downstream evaluation (3D docking / MD / QM / wet lab) on a large candidate set is costly and slow.
Pure ML scoring is often misleading: high headline metrics can hide dataset bias and “Clever Hans” effects.

#### Our commercial goal (what we sell)
**HETERO-2 is an ultra-fast, reproducible "first truth" filter** for large generated libraries:
it helps teams **decide what is worth paying for downstream** and **detect when scoring is self-deceptive**.

Target (order-of-magnitude, workflow-dependent):
- Reduce early validation cost for ~1,000,000 generated candidates from **~$500k-scale** workflows to **~$10k-scale** workflows,
  by providing a fast 2D screening + audit layer *before* expensive stages.

#### What HETERO-2 does
- **Ultra-fast 2D screening signal** (graph/operator-based descriptors) to rank / filter candidates at scale.
- **Matched decoys / hard negatives** to stress-test scoring and avoid “easy negatives” that inflate metrics.
- **Audit-grade outputs**: deterministic reports with explicit SKIP/ERROR reasons, plus negative controls to detect leakage/bias.
- **Truth via artifacts**: results are published as evidence packs with checksums and recorded in the artefacts registry.

#### What HETERO-2 does NOT do (Non-Goals)
- We do **not** replace docking / MD / QM / wet lab experiments.
- We do **not** claim clinical validity, binding affinity truth, or “discovering drugs”.
- We do **not** guarantee absolute accuracy; we guarantee **reproducible computation + transparent failure modes**.

#### Pilot DoD (2–4 weeks, one live proof)
A pilot is successful when we can produce a single, auditable before/after report on the customer’s workflow:

**Inputs**
- A generated library (or a recent generative campaign output).
- The customer’s current “baseline” early filter/scoring approach (whatever they use today).

**Outputs**
- A HETERO-2 run producing:
  - ranked candidates + matched decoys,
  - audit report with negative controls,
  - evidence pack asset + `.sha256`,
  - registry entry with OK/SKIP/ERROR and top SKIP reasons.

**Measured outcomes**
- Downstream efficiency:
  - fewer candidates sent to expensive downstream stages (top-N reduction),
  - with equal or improved downstream hit confirmation rate (as measured by the customer’s existing downstream stage).
- Risk reduction:
  - negative controls remain near-random (no suspicious “always-high” metrics),
  - audit report flags failure modes clearly (no silent failures).
- Reproducibility:
  - rerunning on the same inputs produces the same verdict and matching checksums.

Commercially, we win when the customer can say:
> “We spend less on downstream evaluation and trust the screening decision more.”

---

## 2) Референс-архитектура (минимальные компоненты)

### 2.1. Domain-модель SaaS
- Org (tenant)
- Project
- User
- Role (RBAC)
- Job / Run
- Artifact (versioned)
- AuditEvent (append-only)

### 2.2. Компоненты платформы (минимум)
1) **Auth + Org/Project/RBAC**
2) **API Gateway** (всё через API; UI — клиент)
3) **Job Orchestrator** (submit/status/cancel/retry)
4) **Queue** (приоритеты; idempotency keys)
5) **Worker Pool** (CPU/GPU; autoscale; лимиты времени/памяти)
6) **Artifact Store** (versioned; signed URLs)
7) **Metadata DB** (jobs, params, manifests, billing usage)
8) **Audit Log Service** (append-only события)
9) **Observability** (logs/metrics/traces/alerts)
10) **Billing + Metering** (после пилотов, но заложить модель)

### 2.3. Поток данных (обязательный)
Upload (или scores-only) → Job → Workers → Artifacts (evidence pack) → Viewer/Download → Share/Export.

---

## 3) Definition of Done “SaaS-ready” (обязательные свойства)

### A) Безопасность и доверие (must-have для фармы)
- TLS in transit + шифрование at rest
- tenant isolation (логика + storage paths + ключи)
- secrets management
- data lifecycle (retention + delete)
- rate limiting / базовые защитные меры
- “scores-only” режим как опция снижения барьеров

### B) Воспроизводимость (killer-feature)
- manifest: версии/контейнеры, параметры, seeds, hashes входов, job_id/run_id
- checksums на артефакты
- versioning артефактов и возможность восстановить отчёт
- явные SKIP reasons

### C) Надёжность/масштабирование
- очередь + воркеры + retries + idempotency
- quotas/лимиты (по планам)
- cost controls: max time/mem, cancel/pause, приоритеты

### D) Наблюдаемость
- метрики: latency, error rate, queue depth, worker utilization, cost
- логи: структурированные, с tenant_id/job_id
- алерты: рост ошибок, деградация времени, падение воркеров/очереди

### E) Продаваемость
- API-first + UI
- free tier + paywall (квоты)
- report sharing внутри org
- admin: пользователи/роли/квоты/аудит
- metering/billing (после пилотов — но проектировать заранее)

---

## 4) Вехи (Milestones) — обязательные

### SAAS-M0 — Evidence pipeline v1 стабилен (база)
Статус: [ ] planned  [x] in-progress  [ ] done

Цель: “одна команда → evidence_pack.zip”, честные SKIP/ERROR, детерминизм, manifest+checksums.
(Это соответствует текущей линии evidence pack hardening.)

DoD:
- docker e2e “one command → local evidence_pack.zip”
- freeze `hetero_scores.v1` (compat tests + doc pointer)
- CI гейты проекта зелёные

Артефакты:
- команды запуска + пример pack
- тесты совместимости
- ссылки на релизные ассеты + registry (если используется)

---

### SAAS-M1 — Pilot SaaS MVP (чтобы начать пилоты)
Статус: [ ] planned  [ ] in-progress  [ ] done

Цель: любой пользователь в организации может запустить аудит и получить evidence pack без ручной магии.

Обязательные эпики:
- SAAS-E1 Multi-tenant core (Org/Project/RBAC)
- SAAS-E2 Jobs + Queue + Workers (минимально)
- SAAS-E3 Artifact store + Metadata DB
- SAAS-E4 Reproducibility manifest/checksums как default
- SAAS-E5 Scores-only режим
- SAAS-E6 Observability baseline (logs+metrics)

DoD (проверяемо):
- 1 кнопка / 1 API вызов “Run audit” → результат (pack + manifest + checksums)
- RBAC реально запрещает доступ чужим org/project
- любой run воспроизводим по manifest (в пределах обещаний)
- 95% типовых джоб завершаются успешно на стандартных размерах

Артефакты:
- OpenAPI/Swagger или эквивалент (минимум endpoints: submit/status/results/download)
- пример job/run_id + скачанный pack
- документированные поля manifest + пример

---

### SAAS-M2 — Public Beta (self-serve + cost control)
Статус: [ ] planned  [ ] in-progress  [ ] done

Цель: внешняя команда сама проходит путь до отчёта; free tier не сжигает инфраструктуру.

Обязательные эпики:
- SAAS-E7 API-first + стабильные ответы ошибок
- SAAS-E8 Quotas + priorities (free vs paid)
- SAAS-E9 UI dashboard (status/ETA/cancel/retry)
- SAAS-E10 Share report внутри org
- SAAS-E11 Data retention (auto-delete policy)
- SAAS-E12 Metering counters (usage)

DoD:
- квоты enforce на API и в оркестраторе
- cancel/retry работает, idempotency гарантирована
- retention реально удаляет данные/артефакты по политике
- есть понятные метрики использования per org/project

---

### SAAS-M3 — Sales v1 (для сделок)
Статус: [ ] planned  [ ] in-progress  [ ] done

Цель: закрыть блокеры продаж и аудита.

Обязательные эпики:
- SAAS-E13 Audit log (append-only)
- SAAS-E14 Расширенный RBAC (project-level: view/run/admin)
- SAAS-E15 SLO + алерты + трассировка (request→job→worker)
- SAAS-E16 Billing минимальный (планы/инвойсы/ограничения)

DoD:
- audit log: кто/что/когда для submit/run/download/share/delete
- экспорт evidence packs стандартизирован (zip) + проверка целостности
- определён SLO (например: 99% jobs < N часов) и наблюдаемость подтверждена

---

### SAAS-M4 — Enterprise readiness (фарма)
Статус: [ ] planned  [ ] in-progress  [ ] done

Цель: проходить vendor security review и procurement.

Эпики:
- SAAS-E17 SSO (SAML/OIDC) + (желательно) SCIM
- SAAS-E18 Single-tenant / VPC deployment опция
- SAAS-E19 Data residency (EU/US)
- SAAS-E20 Export audit logs в SIEM
- SAAS-E21 SOC2-ready процесс/контроли (trajectory)

DoD:
- типовой security questionnaire без “красных”
- документированы процессы incident/access/log retention
- есть путь к private deployment

---

## 5) Эпики (детализация, чтобы вести задачами)

### SAAS-E1 — Multi-tenant core (Org/Project/RBAC)
- [ ] org/project сущности в metadata DB
- [ ] RBAC роли (минимум Admin/Member)
- [ ] tenant_id/project_id в каждом job/log/artifact path

### SAAS-E2 — Jobs/Queue/Workers
- [ ] submit job → queued → running → done/failed/canceled
- [ ] retries + idempotency key
- [ ] лимиты времени/памяти на job

### SAAS-E3 — Artifact store + metadata
- [ ] versioned storage: `{org}/{project}/{job}/{run}/...`
- [ ] signed URLs / доступ по RBAC
- [ ] индексация артефактов и связи с runs

### SAAS-E4 — Reproducibility default
- [ ] manifest обязателен
- [ ] checksums обязательны
- [ ] “rerun by manifest” сценарий (внутренний)

### SAAS-E5 — Scores-only
- [ ] режим: вход = scores.json (+ минимальные идентификаторы)
- [ ] пайплайн не требует “исходников модели/структур” если выбран режим

### SAAS-E6 — Observability baseline
- [ ] structured logs: tenant_id, project_id, job_id
- [ ] метрики очереди/воркеров/ошибок/времени
- [ ] минимальные алерты на error-rate и queue backlog

---

## 6) Метрики успеха (чтобы управлять продуктом)
- % успешных jobs (по типовым размерам)
- медиана/95p runtime
- error-rate по причинам
- доля “scores-only” у клиентов
- cost per run (CPU/GPU)
- time-to-first-value (от регистрации до первого pack)

---
