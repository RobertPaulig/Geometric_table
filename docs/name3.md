# Том I (Конституция/ТЗ): HETERO-2 как Pfizer-ready evidence pipeline

Назначение: зафиксировать “конституционные” правила истины/артефактов/гейтов, чтобы развитие шло по VALUE-треку и было проверяемым.

Source of truth: `docs/name3.md` (этот файл).  
Производные артефакты (если присутствуют): `docs/name3.tex`, `docs/name3.pdf`.

---

## 1) Истина артефакта (обязательное правило)

**Истина артефакта** = скачать **Release asset** → посчитать **SHA256** → сверить и зафиксировать в `docs/artefacts_registry.md` (URL + SHA256 + команда + outcome).

Запрещено:
- коммитить `out_*`/zip/большие результаты в git;
- объявлять “DONE”, если нет release asset + SHA256 + registry entry.

---

## 2) STOP/GO (гейты)

Пока на точном SHA нет **3 зелёных контекстов** → **STOP**.

Контексты: `ci/test`, `ci/test-chem`, `ci/docker`.

Подробности: `../CONTEXT.md` (секция `## Gates (STOP/GO)`).

---

## 3) Никаких “тихих провалов”

Пайплайн не имеет права “терять строки”.
Каждая строка входа обязана получить явный статус: **OK / SKIP / ERROR** + `reason`.

---

## 4) Evidence pack = единица результата

Единица результата: `evidence_pack.zip` (или распакованный каталог), который содержит минимум:

- `summary.csv`
- `metrics.json`
- `index.md`
- `manifest.json`
- `checksums.sha256`
- `evidence_pack.zip` (если был режим `--zip_pack`)

Правила релиза: `docs/95_release_checklist.md`.

---

## 5) Точка входа и управление развитием

- Entry point: `../CONTEXT.md`
- Управляющий документ (VALUE-first): `docs/ROADMAP.md`
- Истина артефактов: `docs/artefacts_registry.md`
- История изменений: `docs/90_lineage.md`
