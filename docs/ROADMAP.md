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
