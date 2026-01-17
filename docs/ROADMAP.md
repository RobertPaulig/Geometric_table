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

# ROADMAP — HETERO-2 как SaaS (Pfizer-ready evidence pipeline)

Назначение: зафиксировать целевую картину SaaS и вести разработку через обязательные вехи (milestones).
Этот документ — “истина по развитию продукта” (в отличие от сырого склада идей в backlog).

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
