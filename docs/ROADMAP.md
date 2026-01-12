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

## Как это превращается в реальные “вехи-факты” через ваши workflows (без нового велосипеда)

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
