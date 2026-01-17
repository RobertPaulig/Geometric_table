# Geometric_table — CONTEXT (единая точка входа)

Этот файл — **операционный entrypoint**: что читать, как проверять “истину”, как мерджить по STOP/GO.

## Commercial TL;DR (enterprise-safe)

- Боль рынка: миллионы кандидатов, bottleneck — дорогая валидация (docking/MD/QM/wet lab).
- Что продаём: ultra-fast “first truth” фильтр + audit, чтобы решить “что стоит платить дальше” и ловить самообман скоринга.
- Что не продаём: не заменяем docking/QM/wet lab и не заявляем клиническую истинность/аффинити/“drug discovery”.
- Гарантия: воспроизводимый расчёт + явные OK/SKIP/ERROR + failure modes (без “тихих провалов”).
- Истина результатов: release asset (zip) + `.sha256` + запись в `docs/artefacts_registry.md`.
- Источник формулировки и DoD: `docs/ROADMAP.md` → §1 “North Star: что продаём”.

## Быстрый старт контекста (STOP если не сделано)

1) Прочитать `CONTEXT.md` (этот файл) и `docs/99_index.md` (обязательные REF-ы).  
2) Прочитать `docs/ROADMAP.md` (VALUE-first приоритеты и DoD).  
3) Открыть `docs/90_lineage.md` (последние управленческие изменения и релизы).  
4) Вставить 1 цитату (3–8 строк) из `docs/95_release_checklist.md` из раздела D или F (для контроля “что считаем релизом”).  
5) Прочитать `docs/pilot_quickstart.md` (demo сценарий + команды).  

## Порядок загрузки контекста (строгий)

1. `README.md` — общий обзор.
2. `docs/README.md` — документация и карта.
3. Том I (Конституция/ТЗ, **source of truth**): `docs/name3.md`  
   Производные артефакты (если присутствуют): `docs/name3.tex`, `docs/name3.pdf`.
4. Том II (devline/R&D, **source of truth**): `docs/name4.tex`  
   Производный артефакт (если присутствует): `docs/name4.pdf`.
5. `docs/04_backlog.md` и `docs/05_decision_log.md`.
6. `docs/90_lineage.md` — append-only история.
7. `docs/99_index.md` — обязательный индекс документов/REF-ов.
8. `docs/95_release_checklist.md` — checklist релиза.
9. `VERSION` — baseline версии.

## Репо-структура (коротко)

- `core/` — модель / ядро.
- `analysis/` — сканы, тесты, визуализации.
- `data/` — индексы и агрегированные таблицы.
- `results/` — отчёты.
- `docs/` — документация (entrypoint, roadmap, lineage, registry, contracts).
- `run_pipeline.py` — основной пайплайн.

## Pilot canonical artifact (r2)

- Release: https://github.com/RobertPaulig/Geometric_table/releases/tag/pilot-2026-01-08-r2
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/pilot-2026-01-08-r2/pilot_evidence_pack.zip
- SHA256: `BB564070C2B02087B589A511FB12AD4DEDC13C1EE06A58BC793DD5CF51D3B2A8`

## How to verify truth

1) Download asset from the release.
2) Compute SHA256 and compare to `docs/artefacts_registry.md` and the release `.sha256`.
3) Registry entry must match the asset URL + SHA256 (see `docs/artefacts_registry.md`).

## Contracts (frozen)

- `hetero_scores.v1`: `docs/contracts/hetero_scores.v1.md` (see also `docs/10_contracts_policy.md`).

## How to run the pilot

- Pilot quickstart: `docs/pilot_quickstart.md` (demo scenario + commands).

## Gates (STOP/GO)

- Required contexts: `ci/test`, `ci/test-chem`, `ci/docker` on the exact SHA.
- PR gating note: for `pull_request` runs, these commit statuses are posted on GitHub’s **merge-ref SHA** (the tested SHA), not the PR head SHA.
- Reporting rule: in Gate-status always show **both** `head SHA` and `tested SHA (merge-ref)` for PRs, plus run links.
- If any are missing or red: **STOP** (no merge).

См. также `docs/20_comms_protocol.md` (форма обращений/отчётов).
