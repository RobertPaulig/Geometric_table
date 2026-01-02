# ENTRYPOINT — Geometric_table / geom-spec

## 0. Start Here
1. Прочитать `docs/00_project_vision.md` (миссия таблицы, артефакты v1).
2. Пройти `docs/README.md` (deps, тесты, workflow, troubleshooting).
3. Для подробной карты `analysis/` см. `docs/analysis_structure.md`.
4. Журнал решений/линию наследования держим в `docs/05_decision_log.md`, а backlog гипотез — в `docs/04_backlog.md`.

## 1. Версии и зоны ответственности
- **Stable Core:** QSG v5.0 / «Том I» (`docs/name3.pdf`, текстовая версия `docs/name3.md`).
- **R&D:** QSG v6.x+/Spectral Lab и связанные ветки (см. `docs/name4.tex`).
- Любые правки ядра обязаны фиксироваться в decision log, остальные — через backlog/отчёты.

## 2. Минимальный запуск / проверка
```
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -q
```
Репорты по тестам и окружению — в `REPORT*.md` (например, `REPORT_wsZ1.md`).

## 3. Полный snapshot для ИИ/аудита
Используем `collect_project_snapshot.py`:
```
python collect_project_snapshot.py -o project_snapshot.txt --root .
```
В snapshot попадают дерево проекта и содержимое текстовых файлов (исключая бинарные/PNG/PDF). Tom I доступен в виде `docs/name3.md`, поэтому ИИ видит базу.

## 4. Правило развития (инвариант)
Каждая задача оформляется так: `HYPOTHESIS -> ANALYSIS SCRIPT -> RESULTS ARTEFACT -> запись в Decision Log`.
Т.е. делаем скрипт/запуск в `analysis/`, сохраняем артефакт (`results/...` или `docs/REPORT_*.md`) и фиксируем решение в `docs/05_decision_log.md` + backlog.

## 5. Где смотреть отчёты/utls
- `analysis/chem/*` — контур HETERO-1A (см. новые отчёты `results/hetero_*`).
- `analysis/geom/*`, `analysis/ws/*` — спектральные и геометрические эксперименты (см. `docs/analysis_structure.md`).
- Утилита `collect_project_snapshot.py` и README в корне содержат актуальные инструкции.
