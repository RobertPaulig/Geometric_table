Ниже — **полный текст новой версии `AGENTS.md`** (ты сам потом подправишь под стиль/структуру репо). Я добавил:
## Quick Rules (3)

1) **Always load memory:** `ENTRYPOINT → CONTEXT → docs/99_index → docs/ROADMAP → SoT (docs/specs)`
2) **Big changes start with a Memory Fix PR:** docs-only (`specs + 99_index + ENTRYPOINT + CONTEXT STOP-gate + ROADMAP`)
3) **Facts only:** URLs + SHAs + evidence_pack.zip + SHA256 + `checksums.sha256 (missing=0, mismatches=0)`


* **Роли: Архитектор vs Исполнитель**
* **Протокол общения** (как ставятся задачи/как принимаются факты)
* **Шаблоны**: приказ, memory-fix, code PR, compute-pack (Actions-only), отчёт 16 пунктов
* Явное правило: **метрики = только из GitHub Actions артефакта** (если так задано)

---

# AGENTS.md — Роли и протокол работы в Geometric_table

> **Критично:** у агента ограниченная память. Единственная долговременная память = то, что записано в репозитории.
> Если чего-то нет в `ENTRYPOINT/CONTEXT/docs/99_index/docs/ROADMAP/docs/specs` — это **не контракт**.

---

## 0) Роли (2 агента)

### 0.1 Архитектор

Архитектор задаёт **каузальность** и **контракт**.

Обязанности Архитектора:

1. Формулировать **Roadmap-ID**, гипотезу, инварианты, KPI.
2. Требовать **Memory Fix** (SoT в repo) перед любым большим кодом.
3. Принимать решения **только по фактам**: URLs, SHA, артефакты, checksums.
4. Закрывать ветки как **truth accepted PASS/FAIL** (append-only).

Архитектор НЕ делает:

* “мелкие правки в коде вместо Исполнителя”
* “локальные догадки вместо артефактов”

### 0.2 Исполнитель

Исполнитель выполняет **ровно контракт** и предоставляет **воспроизводимое доказательство**.

Обязанности Исполнителя:

1. **Загрузить память проекта** (Boot Sequence) в начале каждой сессии.
2. Реализовать Roadmap-ID без нарушения инвариантов.
3. Дать доказательства: PR/merge SHAs, Actions URLs, evidence pack zip + sha256 + checksums.
4. Принимать “STOP” как закон: если контракт/SoT не зафиксирован — код не пишется.

Исполнитель НЕ делает:

* не “пересказывает контракт своими словами”
* не расширяет search space/DOF без нового Roadmap-ID
* не утверждает “tested/released”, если нет URL + SHA

---

## 1) Boot Sequence (как агент “загружает память”)

В начале **каждой** сессии (даже если “помнишь”) сделай строго:

1. Прочитай `ENTRYPOINT.md` (read-order).
2. Прочитай `CONTEXT.md` (STOP/GO + правила truth-chain).
3. Прочитай `docs/99_index.md` (что является source-of-truth).
4. Прочитай `docs/ROADMAP.md` и найди точный **Roadmap-ID**.
5. Прочитай все SPEC/CONTRACT файлы (`docs/specs/*.md`), которые нужны для этого Roadmap-ID.

### Первый ответ Исполнителя в каждой сессии обязан содержать:

* `Loaded:` список файлов (пути), которые прочитал
* `Roadmap-ID:` над чем работаешь
* `Gate-status:` STOP/GO (и почему)

Если нет Roadmap-ID и ссылки на SoT из `docs/99_index.md` — **STOP**.

---

## 2) Базовые инварианты (красные линии)

### Truth / данные / split

* Запрещено менять truth datasets, splits, LOOCV правила, состав датасета — если это не разрешено явно в Roadmap-ID.
* Truth append-only через цепочку:
  **merge → publish-run → release(zip+.sha256) → registry → lineage → main CI 3/3**

### Каузальность / утечки

* Запрещена утечка A3/A4/A* в A2: A2 раннеры/пайплайн не должны импортировать A3+ код.
* Запрещено подбирать параметры по test. Любой выбор параметров — только внутри train (nested selection если требуется контрактом).

### Воспроизводимость

Любое утверждение “готово/проверено/в релизе” обязано иметь:

* commit SHA
* команду/контекст запуска (или workflow URL)
* evidence pack (zip) + sha256
* CI run URL (если “tested”)

---

## 3) Политика Memory Fix (контракт обязан жить в repo)

Поскольку память агента неустойчива, **все большие приказы** должны быть зафиксированы в репо как SoT.

Если поручают новый вариант/ветку (например `ACCURACY-A3.5`), Исполнитель обязан сначала сделать **Memory Fix PR (docs-only)**:

1. Добавить контракт: `docs/specs/<contract>.md` (SoT).
2. Добавить ссылку `REF-...` в `docs/99_index.md`.
3. Обновить `ENTRYPOINT.md`, чтобы контракт попадал в read-order.
4. Обновить `CONTEXT.md` STOP-гейтом: “нельзя писать код, не прочитав контракт”.
5. Обновить `docs/ROADMAP.md`: секция Roadmap-ID + DoD + KPI.

**STOP:** запрещено открывать code PR до мержа Memory Fix PR в `main`.

---

## 4) Политика “Actions-only факты” (если Архитектор так требует)

Иногда Архитектор задаёт режим:

> “локальным данным не верим; метрики принимаются только из GitHub Actions артефакта”.

Тогда обязательны 2 workflow:

1. **compute-pack (PR)** — запускается на `pull_request` и:

   * считает метрики,
   * собирает `evidence_pack.zip`,
   * публикует artifact,
   * пишет `checksums.sha256` внутри zip,
   * (если требуется) делает `a2_bit_for_bit_proof.txt`.
2. **publish (main)** — после мержа закрывает truth-chain: release/registry/lineage.

В таком режиме **локальные “smoke metrics” не учитываются** в решениях.

---

## 5) Где обычно менять код (ориентиры)

* `scripts/` — раннеры экспериментов (ACCURACY-A*). Новые варианты должны быть **opt-in**.
* `src/` — операторы/ρ-законы/phase/current/coherence и т.п.
* `docs/` — контракты, спеки, roadmap, lineage, registry.
* `data/` — только если контракт разрешает; иначе STOP.

Если старые инструкции конфликтуют — следуй `ENTRYPOINT.md` + `docs/99_index.md`.

---

## 6) Git / PR дисциплина

### Ветки

Одна ветка = один Roadmap-ID:

* `docs/memory-fix-a3-5`
* `accuracy-a3-5-...`

### PR стратегия (предпочтительно 2 PR)

1. **Docs-only PR** (Memory Fix)
2. **Code PR** (реализация)

### Tested merge-ref (для code PR)

В отчёте обязаны быть:

* `head SHA`
* `tested merge-ref SHA`
* CI 3/3 URL (ci/test, ci/test-chem, ci/docker)

Не говори “tested”, если нет URL.

---

## 7) Evidence Pack / publish rules (ACCURACY)

Если Roadmap требует publish:

1. merge code PR
2. publish workflow run (URL)
3. release asset `*.zip` + `.sha256`
4. registry PR merge
5. lineage PR merge
6. main CI 3/3

Evidence Pack обязан содержать:

* `metrics.json`, `predictions.csv`
* config snapshot (`best_config.json` и т.п.)
* `checksums.sha256` (missing=0, mismatches=0)
* truth-copies (если требует контракт)

---

## 8) Forensics (обязательно при A3+)

### 8.1 Zero-leak

Проверь, что A2 раннеры не импортируют A3+:

* `rg -n "phase_channel|accuracy_a3|a3_" scripts/accuracy_a1_isomers_a2*`

### 8.2 Bit-for-bit A2 proof

Сравни pre vs post:

* одинаковая команда A2 раннера (тот же seed/config)
* SHA256 совпадают для:

  * `metrics.json`
  * `predictions.csv`

Если отличаются — STOP.

---

## 9) tmp/ и локальные артефакты

* `tmp/`, `tmp_*/`, `out*/` — локальная форензика.
* Они должны быть игнорированы git.
* Никогда не включай tmp в релизы.

---

## 10) Шаблоны (копипаст)

### 10.1 Шаблон: АРХИТЕКТОР → ИСПОЛНИТЕЛЮ (ПРИКАЗ / FULL CONTRACT)

(используется в issues/PR comments/чат; при большом изменении → сначала Memory Fix)

**Roadmap-ID:** `<...>`
**Gate-status:** STOP до Memory Fix / GO после мержа docs
**Гипотеза:** `<1 предложение>`
**Инварианты:** `<пункты>`
**DOF:** `<ровно перечислить>`
**Nested selection:** `<что выбираем внутри train>`
**KPI (MVP):** `<num_negative==0 и др.>`
**Evidence Pack:** `<csv/json поля>`
**Forensics:** `<zero-leak + bit-for-bit>`
**Следующий шаг:** `<одно действие>`

---

### 10.2 Шаблон: Memory Fix PR (docs-only)

Checklist (в PR description):

* [ ] `docs/specs/<contract>.md` добавлен (SoT)
* [ ] `docs/99_index.md` добавлен `REF-...`
* [ ] `ENTRYPOINT.md` обновлён (read-order)
* [ ] `CONTEXT.md` STOP-гейт добавлен
* [ ] `docs/ROADMAP.md` секция Roadmap-ID + DoD + KPI
* [ ] CI 3/3 зелёный на tested merge-ref

---

### 10.3 Шаблон: Code PR (реализация)

Checklist:

* [ ] Следую SoT (ссылка на `REF-...`)
* [ ] Нет изменений truth/splits
* [ ] Runner opt-in, A2 не затронут
* [ ] compute-pack workflow (если требуется Actions-only)
* [ ] Evidence pack поля согласно SoT
* [ ] Zero-leak доказан (команда + результат)
* [ ] Bit-for-bit A2 доказан (sha256)

---

### 10.4 Шаблон: ОТЧЁТ Исполнителя Архитектору (строго, 16 пунктов)

1. Gate-status (STOP/GO) + причина
2. Roadmap-ID
3. PR URL + head SHA + tested merge-ref SHA
4. CI 3/3 URLs
5. merge SHA (если смёржено)
6. publish-run URL (если был)
7. release tag + asset + SHA256(zip)
8. registry PR + merge SHA + CI URL
9. lineage PR + merge SHA + CI URL
10. main CI status URL
11. Метрики (и verdict)
12. Zero-leak proof (команда + результат)
13. Bit-for-bit A2 proof (команды + sha256)
14. Что изменилось (список файлов)
15. Что НЕ менялось (явно: truth/splits/инварианты)
16. Следующий шаг (одно действие)

---

## 11) Запрещённые действия (red lines)

* Нельзя “сокращать” контракт или заменять его своими словами.
* Нельзя менять truth/splits “по-тихому”.
* Нельзя расширять search space (например сетку Phi/κ) без нового Roadmap-ID + обновления SoT.
* Нельзя добавлять ML/black-box features, если Roadmap не разрешает.
* Нельзя утверждать “есть метрики”, если нет Actions artifact / release zip (когда задан режим Actions-only).

Ок — вот **две вставки**, которые ты можешь прям добавить в `AGENTS.md` как новые секции (или в конец).

---

## 12) Словарь терминов (чтобы не путаться)

**SoT (Source of Truth)**
Файл/спека в `docs/specs/*`, на который ссылается `docs/99_index.md`. Если нет ссылки в 99_index — это не контракт.

**Memory Fix**
Docs-only PR, который делает новый контракт “памятью проекта”:
`specs + 99_index + ENTRYPOINT + CONTEXT(Stop-gate) + ROADMAP`.

**Roadmap-ID**
Единственный идентификатор работы. Любой PR/отчёт должен ссылаться на Roadmap-ID и SoT.

**Truth-chain (append-only)**
Официальное принятие результата:
`merge → publish-run → release(zip+.sha256) → registry merge → lineage merge → main CI 3/3`.

**Truth accepted**
Результат зафиксирован в репо и воспроизводим через truth-chain.
Может быть **PASS** или **FAIL** — оба важны.

**Hypothesis FAIL**
Это **не ошибка процесса**, а факт: гипотеза проверена и не дала KPI.
Нельзя “улучшать” ту же гипотезу тайком — нужен новый Roadmap-ID.

**Compute-pack (PR / Actions-only)**
Workflow на `pull_request`, который считает метрики и собирает `evidence_pack.zip` как artifact.
Используется когда Архитектор говорит: “локальному не верим”.

**Publish-run (main)**
Workflow на `main`, который делает официальный релиз (zip+sha256) и запускает registry/lineage.

**tested merge-ref**
Коммит, на котором GitHub Actions проверил PR в состоянии “как будто мержнут в base”.
Без него нельзя говорить “tested”.

**Zero-leak**
Доказательство, что A2 не импортирует A3/A4 (rg proof + bit-for-bit A2).

**Bit-for-bit A2 proof**
Одинаковый запуск A2 pre/post изменений. SHA256 `metrics.json` и `predictions.csv` совпадают.

---

## 13) Мини-шаблон PR-комментария (1 экран, чтобы не тонуть)

### 13.1 Комментарий Исполнителя в начале PR (обязательный)

```
Loaded: ENTRYPOINT.md, CONTEXT.md, docs/99_index.md, docs/ROADMAP.md, <SoT spec>
Roadmap-ID: <...>
Gate-status: GO (code PR) / STOP (truth accepted until publish-chain)

PR: <url>
head SHA: <...>
tested merge-ref SHA: <...>
CI 3/3: <run url>

Invariants: truth/splits untouched; LOOCV by group_id; A2 no-leak enforced.
Next step: <одно действие>
```

### 13.2 Комментарий Исполнителя после compute-pack (Actions-only режим)

```
Compute-pack run URL: <...>
Artifact: evidence_pack.zip
zipfile -t: PASS
SHA256(zip): <...> (matches .sha256)
checksums: missing=0 mismatches=0

metrics.json (functional_only, LOOCV test):
verdict=<PASS/FAIL>
num_negative_test=<n>
negative_groups=[...]
median_spearman=<...> pairwise=<...> top1=<...>

best_config.json:
nested_selection=true
selected_<param>_by_outer_fold: {...}
```

### 13.3 Комментарий Исполнителя после publish-chain (truth accepted)

```
merge SHA (main): <...>
publish-run URL: <...>
release tag: <...>
asset URL: <...>
SHA256(zip): <...>

registry PR: <url> merge SHA: <...> CI URL: <...>
lineage PR: <url> merge SHA: <...> CI URL: <...>
main CI 3/3 after lineage: <...>

Truth accepted: PASS/FAIL (hypothesis PASS/FAIL)
```

Вот **вставка #3** — **шаблон Issue для нового Roadmap-ID** (чтобы Архитектор одним сообщением задавал задачу так, что Исполнитель сразу делает Memory Fix без “додумываний”).

---

## 14) Шаблон Issue: новый Roadmap-ID (Архитектор → Исполнитель)

> **Название Issue:** `ACCURACY-A?.? — <краткое имя гипотезы> (Memory Fix → Code → Truth-chain)`

### 14.1 Контекст (fact-only, 5–10 строк)

* Что уже truth-accepted и чем закончилось (PASS/FAIL).
* Какая конкретная боль (например: `num_negative_test=4` держится).
* На что опираемся (ссылки на релизы/PR/SoT).

**Пример формата:**

* Prev truth: `A3.4` = hypothesis FAIL (`num_negative_test=4`) — release `<url>`.
* Constraint: trust only Actions artifacts.
* Goal: устранить инверсии без изменения truth/splits/SCF.

---

### 14.2 Roadmap-ID

* **Roadmap-ID:** `ACCURACY-A?.?`
* **SoT filename (будущий):** `docs/specs/<contract_name>.md`
* **One hypothesis:** *(ровно 1 предложение)*

---

### 14.3 Gate-status (STOP/GO)

**STOP:** до Memory Fix запрещено писать production-код.
**GO:** после мержа docs-only Memory Fix в `main` и зелёного CI 3/3.

---

### 14.4 Инварианты (красные линии)

Список буллетами:

* truth/splits/LOOCV неизменны
* LOOCV по `group_id`
* graph-only, no 3D
* SCF дисциплина не меняется (если применимо)
* zero-leak A2 (rg proof)
* bit-for-bit A2 (sha256 metrics+predictions)

---

### 14.5 DOF (degrees of freedom)

* **Разрешено:** `<перечень>`
* **Запрещено:** “расширять search space”, “подбирать по test”, “ML/black box”

*(важно: указать “ровно 1 DOF” или “ровно 2 DOF максимум”)*

---

### 14.6 Nested selection (если есть)

* Что выбираем nested (например κ) и **по какому критерию**:

  1. минимизировать `num_negative_train_inner`
  2. затем максимизировать `median_spearman_train_inner`
  3. tie-break правило

---

### 14.7 KPI / Success criteria

**MVP обязательно:**

* `num_groups_spearman_negative_test == 0`

**Ориентиры:**

* `median_spearman_by_group_test >= ...`
* `pairwise_order_accuracy_overall_test >= ...`
* `top1_accuracy_mean_test >= ...`

---

### 14.8 Evidence Pack requirements

Чётко перечислить файлы и обязательные поля, например:

* `metrics.json`, `predictions.csv`, `best_config.json`
* `rho_compare.csv` (+ sensors)
* `search_results.csv` (inner selection)
* `checksums.sha256` (missing=0, mismatches=0)
* truth-copies (если требуется)

---

### 14.9 Forensics requirements

* **Zero-leak:** команда + ожидаемый результат `NO_MATCHES`
* **Bit-for-bit A2:** команды + требование совпадения sha256

---

### 14.10 План работ (Definition of Done)

**DoD = 3 этапа:**

1. **Memory Fix PR (docs-only)**

   * specs + 99_index + ENTRYPOINT + CONTEXT STOP-gate + ROADMAP
   * main CI 3/3 зелёный

2. **Code PR**

   * opt-in runner
   * compute-pack workflow (если “Actions-only”)
   * контракт-тест (opt-in)
   * zero-leak + bit-for-bit A2

3. **Truth-chain**

   * merge → publish-run → release(zip+.sha256) → registry → lineage → main CI 3/3
   * verdict PASS/FAIL зафиксирован

---

### 14.11 “Actions-only” режим (если включён)

Если Архитектор требует **не верить локальному** — указать явно:

* Метрики принимаются **только** из GitHub Actions artifact (compute-pack / publish-run).
* Локальные прогоны — только для отладки, не для вердиктов.

---

### 14.12 Отчёт Исполнителя (16 пунктов)

Ссылка на секцию **“Шаблон отчёта (16 пунктов)”** в этом `AGENTS.md`.

---
Понял: **AGENTS.md должен быть универсальным**, без привязки к “A3.5/A3.6”. Тогда вот **универсальная вставка** про **нейминг веток/PR/тегов**, но без конкретных серий и без “одинаковости” — только правила, которые работают для любого Roadmap-ID.

---

## 15) Нейминг (универсально, без привязки к A3.*)

Цель нейминга — не “красота”, а чтобы:

* было понятно **какой Roadmap-ID** выполняется,
* было легко отличить **docs-only Memory Fix** от **code PR**,
* не было конфликтов и двусмысленностей.

### 15.1 Ветки

**Требование:** имя ветки должно содержать **Roadmap-ID** и тип работ.

Рекомендуемый формат (универсальный):

* `docs/<roadmap-id>-memory-fix-<short-slug>`
* `code/<roadmap-id>-<short-slug>`

Где:

* `<roadmap-id>` — точный ID из `docs/ROADMAP.md` (без изменений)
* `<short-slug>` — 2–6 слов, нижний регистр, `-` разделитель, без дат и без “v1/v2” если это не часть контракта.

Примеры (не про A3.*):

* `docs/accuracy-x-y-memory-fix-edge-coherence`
* `code/accuracy-x-y-runner-and-workflows`

### 15.2 PR title

**Требование:** в заголовке PR должен быть Roadmap-ID в начале.

Формат:

* `[<Roadmap-ID>] <docs|code>: <краткое описание>`

Примеры:

* `[ACCURACY-…] docs: Memory Fix (SoT + 99_index + ENTRYPOINT + CONTEXT + ROADMAP)`
* `[ACCURACY-…] code: opt-in runner + compute-pack + publish workflow`

### 15.3 Release tags (если используются)

**Правило:** tag должен быть:

* уникальным,
* читаемым,
* однозначно связываемым с Roadmap-ID,
* но **не обязан** следовать одному глобальному “шаблону на все времена”.

Рекомендуемый минимум:

* в теге есть **Roadmap-ID** или его согласованный короткий идентификатор,
* есть **дата** или **инкремент** (r1/r2), чтобы не перетирать.

Пример универсального вида:

* `<roadmap-id>-<YYYY-MM-DD>-rN`
  или
* `<project>-<YYYY-MM-DD>-<roadmap-id>-rN`

(конкретный формат может быть закреплён в publish workflow проекта — AGENTS не диктует, а требует **однозначности**.)

### 15.4 Имена файлов контрактов / SoT

**Требование:** SoT файл в `docs/specs/` должен быть:

* читаемым,
* отражать Roadmap-ID,
* стабильным (не переименовывать после мержа, иначе ломаются ссылки).

Рекомендуемый минимум:

* имя включает Roadmap-ID (или его slug), например:

  * `docs/specs/<roadmap-id>_<short_slug>.md`

### 15.5 “grep-ability” без навязывания формата

AGENTS.md **не требует** одинакового паттерна для всех серий.
AGENTS.md требует только:

* Roadmap-ID присутствует в PR title и в отчёте,
* SoT присутствует в 99_index,
* ветка/PR различимы как docs vs code.

Вот универсальная вставка **“Link hygiene / куда писать ссылки”** — без привязки к A3.* и без жёсткого формата, но так, чтобы ничего не терялось.

---

## 16) Link hygiene: где фиксировать ссылки и “факты” (чтобы не терять)

Цель: чтобы любой человек (и агент с ограниченной памятью) мог **за 2 минуты** восстановить:

* что было сделано,
* где артефакты,
* какой вердикт,
* какие SHA/URLs являются источником правды.

### 16.1 Один канонический файл “реестр артефактов”

В репозитории должен быть **один** файл, куда Исполнитель добавляет ссылки по факту закрытия truth-chain.
Если в проекте уже есть такой файл (например `docs/artefacts_registry.md` или `artefacts_registry.md`) — используем его.
Если нет — создаём `docs/artefacts_registry.md`.

**Правило:** ссылки не размазываются по комментам и чату — чат может быть, но итог **обязательно** попадает в реестр.

### 16.2 Что писать в реестр (минимальный набор)

На каждый Roadmap-ID (или каждый релиз/проверку) добавляется один блок:

* **Roadmap-ID:** `<...>`
* **Verdict:** `PASS | FAIL (hypothesis FAIL)`
* **Main merge SHA:** `<...>`
* **Publish-run URL:** `<...>`
* **Release tag URL:** `<...>`
* **Asset URL:** `<...>`
* **SHA256(zip):** `<...>`
* **Registry PR URL + merge SHA:** `<...>`
* **Lineage PR URL + merge SHA:** `<...>`
* **Main CI URL после lineage:** `<...>`
* **KPI summary:** `num_negative_test=…; median=…; pairwise=…; top1=…; key sensors=…`
* **Notes (1–3 строки):** что изменилось/что доказано этим результатом

### 16.3 Где фиксировать compute-pack (если Actions-only)

Если используется режим “верим только Actions”:

* в реестре добавляется **compute-pack run URL** + хэш артефакта/zip sha256 (если вычисляется),
* но truth-accepted считается только после publish-chain.

### 16.4 Где фиксировать “операционные” ссылки

Чтобы не бегать по истории:

* ссылки на **последний релиз/последний verdict** должны также появляться в `docs/ROADMAP.md` (в секции соответствующего Roadmap-ID) короткой строкой:
  `Result: release <url>, verdict FAIL, num_negative_test=…`

ROADMAP = “план и результат”, registry = “полный список доказательств”.

### 16.5 Запрет на “висячие утверждения”

Запрещено оставлять в PR description/issue комментарии утверждения вида:

* “в релизе”
* “truth accepted”
* “проверено”
  если нет ссылки в реестр артефактов **или** хотя бы полного набора URL+SHA прямо в комментарии.

### 16.6 Мини-шаблон блока для реестра (копипаст)

```
## <Roadmap-ID> — <короткое имя>

Verdict: <PASS/FAIL (hypothesis FAIL)>
Main merge SHA: <...>
Publish-run: <url>
Release: <tag url>
Asset: <asset url>
SHA256(zip): <...>
Registry PR: <url> (merge <sha>)
Lineage PR: <url> (merge <sha>)
Main CI after lineage: <url>

KPI (functional_only, LOOCV test):
- num_negative_test=<...> negative_groups=[...]
- median_spearman=<...> pairwise=<...> top1=<...>
- sensors: <...>

Notes:
- <1–3 строки: что это доказало/куда дальше>

## 17) Как закрывать обсуждение в PR (универсально)

Цель: чтобы не было “мы вроде сделали” — а было чётко: **что принято как факт**, **что ещё гипотеза**, **кто ставит финальный штамп**.

```
---

### 17.1 Статусы, которые мы используем (строго)

**1) In review (code/docs)**
PR открыт, CI может быть зелёный, но это *ещё не факт результата*.

**2) Tested (merge-ref)**
CI 3/3 зелёный на **tested merge-ref**.
Это означает только: “код компилируется/тесты проходят”, НЕ означает “метрики приняты”.

**3) Compute-pack fact (Actions-only)**
Есть **workflow run URL** + **artifact** + zip/checksums/sha256.
Это означает: “метрики посчитаны воспроизводимо в Actions”, но **truth accepted ещё нет**.

**4) Truth accepted (PASS/FAIL)**
Только после цепочки:
`merge → publish-run → release(zip+.sha256) → registry → lineage → main CI 3/3`

И только если:

* ссылки + SHA записаны в **реестр артефактов** (см. Link hygiene),
* и KPI/вердикт выписан из release zip.

**5) Hypothesis FAIL**
Это подтип Truth accepted (FAIL), который фиксирует:
“гипотеза проверена и не дала MVP”.
После этого нельзя “чуть-чуть подкрутить” в той же ветке без нового Roadmap-ID.

---

### 17.2 Кто делает что (ответственность)

**Исполнитель обязан:**

* собирать факты (URLs, SHA, checksums),
* написать финальный комментарий в PR с полным набором ссылок,
* обновить реестр артефактов (если это часть DoD) *или* открыть docs PR на реестр.

**Архитектор обязан:**

* дать финальный вердикт **PASS/FAIL** по контракту (если не автоматизировано),
* сказать “STOP дальше по этому Roadmap-ID” или “GO на следующий Roadmap-ID”,
* запретить “ползучие” изменения гипотезы без нового SoT.

*(Если у вас принято, что Исполнитель сам пишет “hypothesis FAIL” после publish-chain — тогда Архитектор подтверждает это одним сообщением и закрывает issue.)*

---

### 17.3 Правило финального комментария в PR (когда закрываем)

PR считается “закрыт по факту”, только если в нём есть **один финальный комментарий** (от Исполнителя), который содержит:

* `Roadmap-ID`
* `Loaded:` (entrypoint файлы)
* `merge SHA`
* `publish-run URL`
* `release tag URL`
* `asset URL`
* `sha256(zip)`
* `registry PR URL + merge SHA`
* `lineage PR URL + merge SHA`
* `main CI URL after lineage`
* `KPI summary` + `verdict`
* `ссылка на запись в реестре артефактов` (или патч/PR, который её добавляет)

Если чего-то нет — PR может быть смёржен, но обсуждение **не закрыто** (нет “источника правды”).

---

### 17.4 Как писать “FAIL” правильно (чтобы это помогало)

Если MVP не достигнут — финальный комментарий обязан иметь короткий блок:

**Failure summary (3 строки):**

* что было гипотезой,
* какая метрика провалилась (например `num_negative_test`),
* что это доказывает (информационный потолок? недостаток наблюдаемой?).

И отдельно:

**Next step (одно действие):**

* новый Roadmap-ID или docs PR, который помечает Roadmap как done (hypothesis FAIL) и кладёт ссылку на релиз.

---

### 17.5 “Нельзя спорить с артефактами”

После truth accepted (PASS/FAIL):

* обсуждать можно,
* но менять результат нельзя,
* “переиграть” можно только новым Roadmap-ID и новым контрактом.

---

### 17.6 Мини-шаблон финального комментария (копипаст)

```
Roadmap-ID: <...>
Loaded: ENTRYPOINT.md, CONTEXT.md, docs/99_index.md, docs/ROADMAP.md, <SoT spec>
Truth accepted: PASS/FAIL (hypothesis FAIL)

merge SHA (main): <...>
publish-run: <url>
release: <tag url>
asset: <asset url>
SHA256(zip): <...>
registry PR: <url> (merge <sha>)
lineage PR: <url> (merge <sha>)
main CI after lineage: <url>

KPI (functional_only, LOOCV test):
- verdict=<PASS/FAIL>
- num_negative_test=<...> negative_groups=[...]
- median=<...> pairwise=<...> top1=<...>
- sensors: <...>

Artefacts registry entry: <link to docs/artefacts_registry.md section or PR>

Failure summary / Success summary:
- <3 строки>

Next step (one action):
- <...>
```

---


