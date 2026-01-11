# ROADMAP — HETERO-2 (Pfizer-ready evidence pipeline → затем SaaS)

Этот документ — **истина по развитию продукта**: что делаем, в каком порядке, и **какими артефактами доказываем** “пользу/качество/готовность”.

**Принцип:** SaaS имеет смысл только если **VALUE доказан фактами** (evidence packs + registry + воспроизводимость + гейты).

---

## 0) Правила разработки (обязательные)

### 0.1. Веха = контракт

Каждая веха (Milestone) обязана иметь:

* **Goal** (зачем),
* **Scope** (что входит),
* **DoD** (Definition of Done — проверяемо),
* **Proof** (артефакты/ссылки/хэши),
* **Risks** (что может сломаться/обмануть нас).

Запрещено объявлять “DONE”, если DoD и Proof не выполнены.

### 0.2. Истина = Release asset + SHA256 + Registry

Любая “факт-веха” считается DONE **только если**:

1. есть **GitHub Release asset** (zip),
2. есть **SHA256** на asset,
3. есть запись в `docs/artefacts_registry.md` (**URL + SHA256 + команда + outcome**),
4. CI-гейты зелёные на целевом SHA (см. 0.3).

**Никаких zip/out_* в git.** В git — только реестр и документы.

### 0.3. STOP/GO и “какой SHA считается истинным”

**Железное правило:** пока нет **3 зелёных контекстов** → **STOP**.

Контексты: `ci/test`, `ci/test-chem`, `ci/docker`.

**Чтобы не зависать на GitHub merge-ref:**

* На PR-этапе “SHA, который реально тестировался”, может быть **merge-ref SHA**, если workflow ставит статусы туда.
  → В отчёте всегда указываем **оба**: `head SHA` и `tested SHA (merge-ref)` и даём ссылки на run.
* Финальная истина после мержа — **merge SHA в `main`**, и **там обязаны быть 3 зелёных**.

Если это неудобно — отдельная тех-веха: “починить постинг статусов на head SHA”.

### 0.4. Никаких “тихих провалов”

Каждая строка входа должна иметь итог: **OK / SKIP / ERROR** + reason.
Тишина = баг.

### 0.5. Обязательное ведение `docs/90_lineage.md`

Каждый merge, релиз, и VALUE-pack:

* добавляет запись в `docs/90_lineage.md` (дата, SHA, что изменилось, ссылки на релизы/ассеты, SHA256).

Это не бюрократия — это аудит-дорожка.

### 0.6. Каждый PR привязан к Roadmap-ID

Каждый PR обязан:

* ссылаться на Roadmap-ID (пример: `VALUE-M2`, `CORE-FREEZE-1`, `SAAS-M1`),
* обновлять статус в ROADMAP (checkbox / short note),
* проходить CI-гейты проекта.

---

## 1) Что продукт делает “в реальности” (без маркетинга)

**Вход (Pfizer/team):** файл(ы) с данными и/или `scores.json` (scores-only режим — приоритет).
**Выход:** `evidence_pack.zip` с manifest + checksums + отчётами.

Мы **не “удаляем молекулы” молча**. Мы:

* помечаем проблемные случаи как **SKIP/ERROR** (с причиной),
* даём агрегированные метрики качества и “failure modes”,
* и (ключевое) показываем, что **плохой скоринг/модель ловится** измеримыми сигналами (gate/slack/verdict).

---

# TRACK A — VALUE-TRACK (P0): Pfizer-ready FACTS

## VALUE-M0 — Pipeline Truth (baseline)

**Goal:** доказать, что pack строится стабильно: ERROR=0, детерминизм, целостность.

**DoD:**

* pack содержит минимум: `summary.csv`, `metrics.json`, `index.md`, `manifest.json`, `checksums.sha256`, `evidence_pack.zip`
* `metrics.json`: `counts.ERROR == 0`
* Release asset + SHA256 + registry entry

**Proof (baseline):**
* Registry: `docs/artefacts_registry.md` → `stress_10k (light) - 2026-01-05`
* Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/stress-10k-2026-01-05/evidence_pack.zip
* SHA256(evidence_pack.zip): `458EF5315D52B17B7797F67F3B89E2A091A12058406C9876DCC6D9925E95C76B`

**Status:** [ ] planned  [ ] in-progress  [x] done

---

## VALUE-M1 — Chem Coverage Suite (“кольца/ароматика/гетероциклы”) как факт

**Goal:** доказать, что на химически “опасном” подмножестве (кольца) нет слепых зон, а SKIP-reasons предсказуемы.

**Scope v1:**

* фиксируем suite (seed-список) и режим генерации (например: `lock_aromatic`, запреты/разрешения на ring-bonds) **как контракт**
* suite должен быть воспроизводим из `manifest`

**DoD (жёсткий):**

* Release tag: `value-ring-suite-YYYY-MM-DD`
* pack собран с `--zip_pack`
* zip-валидация: `zipfile -t` + проверка наличия обязательных файлов
* `ERROR=0`
* в Release notes + registry фиксируем факты:

  * `OK/SKIP/ERROR`
  * `top_reasons` для SKIP
  * доля строк с `n_decoys > 0` (из `summary.csv`)

**Proof (DONE):**
* Registry: `docs/artefacts_registry.md` → `value-ring-suite-2026-01-10`
* Release tag: https://github.com/RobertPaulig/Geometric_table/releases/tag/value-ring-suite-2026-01-10
* Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-ring-suite-2026-01-10/value_ring_suite_evidence_pack.zip
* SHA256(value_ring_suite_evidence_pack.zip): `912071F3927D733FF4F5EDA1AB5A3158F83D18EBA4E99B1C2CC174FD6EE08274`
* Outcome (facts): `OK=60, SKIP=140, ERROR=0; top_skip_reasons: no_decoys_generated=140; share_rows_with_n_decoys_gt_0=30.0%`

**Status:** [ ] planned  [ ] in-progress  [x] done

---

## VALUE-M2 — Known Bad / Known Good (ловим “плохую модель”)

**Goal:** доказать, что наши поля `verdict/gate/slack/margin` **реагируют на деградацию скоринга** (а не шум).

**Метод:**

* один и тот же input suite (из M1)
* разные variants `scores.json`:

  1. `BAD-constant` (все scores одинаковые)
  2. `BAD-random` (случайные, seed фиксирован)
  3. `GOOD-synthetic` (оригинал выше, decoys ниже)
  4. (позже) `REAL-client` (внешний)

**DoD (измеримый):**

* минимум 3 ассета (или 1 релиз с 3 ассетами) tag: `value-known-bad-good-YYYY-MM-DD`
* есть разделение BAD vs GOOD по метрике (фиксируем пороги как “starting point”, потом калибруем фактами):

  * `median(slack_GOOD) - median(slack_BAD) >= Δ` (старт: `Δ=0.05`)
  * и/или `PASS_rate(GOOD) - PASS_rate(BAD) >= 20 pp`
* `ERROR=0` во всех пакетах
* registry entry для каждого asset

**Status:** [x] planned  [ ] in-progress  [ ] done

---

## VALUE-M3 — Customer Truth (Pfizer/proxy) + acceptance criteria

**Goal:** перейти от синтетики к реальным данным/скорам клиента с заранее согласованными критериями.

**DoD:**

* зафиксированы acceptance criteria в терминах `FAIL-rate`, `slack`, `gate`, coverage
* выпущен evidence pack (может быть private), но с тем же форматом истины:

  * manifest + checksums + zip + SHA256
* описана политика доступа/retention (если private)

**Status:** [x] planned  [ ] in-progress  [ ] done

---

## VALUE-M4 — “Мы претендуем на стандарт”

**Goal:** оформить стандарт как **контракты и тесты**, а не лозунги.

**DoD:**

* документ `docs/standard_claims.md`:

  * что гарантируем (формат, воспроизводимость, честная маркировка, минимальные проверки)
  * что **не** гарантируем (границы применимости)
* backward-compat тесты:

  * `summary.csv` schema
  * `hetero_scores.v1` schema/version
* политика изменений (breaking change policy)

**Status:** [x] planned  [ ] in-progress  [ ] done

---

# TRACK B — CORE (ядро пайплайна и контракты)

## CORE-FREEZE-1 — Freeze `hetero_scores.v1` (prereq для VALUE-M2/M3)

**Goal:** внешние scores нельзя называть контрактом, пока не зафиксировано.

**DoD:**

* документ контракта (schema + semantics)
* тест backward-compat
* ссылка/указатель в `CONTEXT.md`
* зелёные CI-гейты

**Proof (DONE):**
* PR: https://github.com/RobertPaulig/Geometric_table/pull/20
* Merge SHA (main): `ba9752e2145ba77f8afad5902ea0b2454e91a545`
* Contract doc: `docs/contracts/hetero_scores.v1.md` (policy: `docs/10_contracts_policy.md`)

**Status:** [ ] planned  [ ] in-progress  [x] done

---

## CORE-SUITES-1 — Suite framework (воспроизводимые наборы)

**Goal:** suites должны быть “включаемыми” и воспроизводимыми через manifest.

**DoD:**

* suite_id → фиксированный генератор входа
* manifest хранит suite_id + параметры + seed
* локальный и CI-workflow умеют собрать pack по suite_id

**Status:** [x] planned  [ ] in-progress  [ ] done

---

# TRACK C — SaaS (включается только после VALUE-M2)

**Правило запуска SaaS-работ:**
Если `VALUE-M2` не DONE — SaaS считается преждевременным.

## SAAS-M0 — Evidence pipeline v1 стабилен (база)

**Goal:** “одна команда → evidence_pack.zip” (E2E), честные SKIP/ERROR, воспроизводимость.

**DoD:**

* Docker E2E (one command → local `evidence_pack.zip`)
* `CORE-FREEZE-1` DONE
* 3 CI-гейта зелёные на merge SHA в main

**Status:** [x] planned  [ ] in-progress  [ ] done

---

## SAAS-M1 — Pilot SaaS MVP (пилоты)

(и далее SAAS-M2/M3/M4 — как упаковка и масштабирование)

> Здесь оставляем SaaS-часть короче и “по делу”, чтобы не раздувать документ до платформенной фантазии до доказанной пользы.

**Status:** [x] planned  [ ] in-progress  [ ] done

---

# TRACK D — “Научная честность” и контроль физичности (guardrails)

Это ключ к требованию: **использовать новаторов можно**, но **в продукте живёт только то, что прошло проверку**.

## SCI-RULES-1 — Любая “новая идея” = гипотеза под A/B

Любая эвристика/идея допускается только если:

* она включается флагом (или отдельным suite/variant),
* даёт **измеримый выигрыш** на VALUE-наборах,
* не ухудшает стабильность/воспроизводимость,
* и не меняет “смысл” результата скрытно.

## SCI-PHYS-1 — Инвариантность и “не ломаем физику”

Если мы меняем численные/дискретизационные детали, то DoD включает:

* сравнение на фиксированных наборах: baseline vs new
* критерии: **сходимость/стабильность** и отсутствие регрессий в ключевых физических/химических инвариантах, которые мы обещаем
* отчёт в evidence pack (добавляется секция “Sensitivity/Robustness”)

---

## Приложение: соглашения об именовании релизов (обязательное)

* `value-ring-suite-YYYY-MM-DD`
* `value-known-bad-good-YYYY-MM-DD`
* `value-customer-YYYY-MM-DD`
* `standard-claims-YYYY-MM-DD` (если релизим pack-проверку стандарта)

---

## Статусная таблица (коротко)

* [x] CORE-FREEZE-1
* [x] VALUE-M0
* [x] VALUE-M1
* [ ] VALUE-M2
* [ ] VALUE-M3
* [ ] VALUE-M4
* [ ] SAAS-M0
* [ ] SAAS-M1

---

## Важно про “не потеряем честность?”

Как правило: **мы не продаём “новую физику”**. Мы продаём:

* воспроизводимый аудит,
* честный отчёт,
* и измеримую способность отличать “плохой скоринг” от “хорошего”.

Любая “спектральная” идея допустима **только как математическое/численное улучшение**, пока она:

1. проходит VALUE-вехи,
2. не ухудшает инвариантность/контракты,
3. фиксируется артефактами (release+sha256+registry),
4. и не требует веры.

