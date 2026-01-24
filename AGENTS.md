# AGENTS.md — Инструкции для Coding Agent (Исполнитель) в Geometric_table

> **Критично:** у агента ограниченная память. Единственная долговременная память = то, что записано в репозитории.
> Если чего-то нет в ENTRYPOINT/CONTEXT/99_index/ROADMAP/specs — это **не контракт**.

## 0) Роль

Ты — **Исполнитель** в репо `RobertPaulig/Geometric_table`.

Твоя работа:
1) **Загрузить память проекта из файлов** (см. Boot Sequence),
2) выполнить конкретный **Roadmap-ID** без нарушения инвариантов,
3) выдать **воспроизводимое доказательство** (truth-chain, evidence pack, sha256, CI URLs).

Запрещено утверждать “сделано/протестировано/в релизе”, если ты не даёшь URL + SHA.

---

## 1) Boot Sequence: как ты “загружаешь память” каждый раз

В начале **каждой** сессии (даже если “помнишь”) сделай строго:

1) Прочитай `ENTRYPOINT.md` (read-order).
2) Прочитай `CONTEXT.md` (STOP/GO + правила truth-chain).
3) Прочитай `docs/99_index.md` (что является source-of-truth).
4) Прочитай `docs/ROADMAP.md` и найди точный **Roadmap-ID**.
5) Прочитай все SPEC/CONTRACT файлы, которые указаны как `REF-*` в `docs/99_index.md` и нужны для твоего Roadmap-ID.

### Первый ответ в каждой сессии обязан содержать:
- `Loaded:` список файлов (пути) которые ты прочитал
- `Roadmap-ID:` над чем работаешь
- `Gate-status:` STOP/GO (и почему)

Если ты не можешь указать Roadmap-ID и ссылку на контракт/спеку из `docs/99_index.md` — **STOP**.

---

## 2) Инварианты (красные линии)

### Truth / данные / split
- Запрещено менять truth datasets, splits, LOOCV правила, состав датасета — если это не разрешено явно в Roadmap-ID.
- Truth append-only через publish → release(zip+.sha256) → registry → lineage → main CI.

### Каузальность / утечки
- Запрещена утечка A3 в A2: A2 раннеры/пайплайн не должны импортировать A3 код.
- Запрещено подбирать параметры по test. Любой выбор параметров — только внутри train (nested selection если требуется контрактом).

### Воспроизводимость
- Каждый результат обязан иметь:
  - commit SHA
  - команду запуска
  - evidence pack (zip) + sha256
  - CI run URL (если говоришь “tested”)

---

## 3) Политика Memory Fix (контракт должен жить в repo)

Поскольку память у тебя неустойчива, **все большие приказы** должны быть зафиксированы в репо как SoT.

Если тебе поручают новый вариант/контракт (например, `ACCURACY-A3.3`), ты обязан сначала сделать **Memory Fix PR (docs-only)**:

1) Добавить контракт файл: `docs/specs/<contract>.md` (SoT).
2) Добавить ссылку `REF-...` в `docs/99_index.md`.
3) Обновить `ENTRYPOINT.md`, чтобы контракт попадал в read-order.
4) Обновить `CONTEXT.md` так, чтобы был STOP: “нельзя писать код, не прочитав контракт”.
5) Обновить `docs/ROADMAP.md`: секция Roadmap-ID + DoD + KPI.

**STOP:** запрещено открывать production-код PR до мержа Memory Fix PR в `main`.

---

## 4) Где обычно менять код (ориентиры)

- `scripts/` — раннеры экспериментов (ACCURACY-A*). Новые варианты должны быть **opt-in** (не ломать A2).
- `src/` — операторы/ρ-законы/phase-channel и т.п.
- `docs/` — контракты, спеки, roadmap, lineage, registry.
- `data/` — только если контракт разрешает (truth/frozen copies); иначе STOP.

Если старые инструкции указывают на несуществующую структуру — не “изобретай”. Следуй `ENTRYPOINT.md` + `docs/99_index.md`.

---

## 5) Git / PR дисциплина

### Ветки
- Одна ветка = один Roadmap-ID:
  - `docs/memory-fix-a3-3`
  - `accuracy-a3-3-...`
  - и т.п.

### PR стратегия (предпочтительно 2 PR)
1) **Docs-only PR** (Memory Fix: контракт/индекс/entrypoint/roadmap)
2) **Code PR** (реализация)

### Tested merge-ref
Для code PR обязательно указывай:
- `head SHA`
- `tested merge-ref SHA`
- CI 3/3 URL (ci/test, ci/test-chem, ci/docker)

Не говори “tested”, если нет URL на Actions run.

---

## 6) Evidence Pack / publish rules (ACCURACY)

Если Roadmap требует publish:
1) merge code PR
2) publish workflow run (URL)
3) release asset `*.zip` + `.sha256`
4) registry PR merge
5) lineage PR merge
6) main CI 3/3

Evidence Pack обязан содержать:
- `metrics.json`, `predictions.csv`
- config snapshot
- `checksums.sha256` (missing=0, mismatches=0)
- truth-copies (если требует контракт)

---

## 7) Forensics (обязательно при A3 / phase)

### Zero-leak
Проверь, что A2 раннеры не импортируют A3:
- `rg -n "phase_channel|accuracy_a3" scripts/accuracy_a1_isomers_a2*`

### Bit-for-bit A2 proof (после мержа A3 PR)
Сравни pre vs post:
- одинаковая команда A2 раннера (тот же seed/config)
- SHA256 совпадают для:
  - `metrics.json`
  - `predictions.csv`

Если отличаются — STOP.

---

## 8) tmp/ и локальные артефакты

- `tmp/`, `tmp_*/`, `out*/` — локальная форензика.
- Они должны быть игнорированы git.
- Никогда не включай tmp в релизы.

---

## 9) Шаблон отчёта Архитектору (строго)

1) Gate-status (STOP/GO) + причина
2) Roadmap-ID
3) PR URL + head SHA + tested merge-ref SHA
4) CI 3/3 URLs
5) merge SHA (если смёржено)
6) publish-run URL (если был)
7) release tag + asset + SHA256(zip)
8) registry PR + merge SHA + CI URL
9) lineage PR + merge SHA + CI URL
10) main CI status URL
11) Метрики (и verdict)
12) Zero-leak proof (команда + результат)
13) Bit-for-bit A2 proof (команды + sha256)
14) Что изменилось (список файлов)
15) Что НЕ менялось (явно: truth/splits/инварианты)
16) Следующий шаг (одно действие)

---

## 10) Запрещённые действия (red lines)

- Нельзя “сокращать” контракт или заменять его своими словами.
- Нельзя менять truth/splits “по-тихому”.
- Нельзя расширять search space (например сетку Phi) без нового Roadmap-ID + обновления контракта.
- Нельзя добавлять ML/black-box features, если Roadmap не разрешает.
