## Geom-Mendeleev v2 — гипотезы и проверки

### Hypothesis S/P (строгая D/A-периодичность s/p-блоков)

**Описание.**

- D/A-индексы для s-доноров, s-мостов, p-semihub и p-акцепторов не зависят от периода
  (2-й, 3-й, 4-й …), а только от geom-класса.
- Проверяем на текущем диапазоне элементов, который уже есть в `element_indices_with_dblock.csv`.

**Задачи.**

- Написать анализ-скрипт `analysis/test_geom_periodicity_sp.py`.
- Зафиксировать разброс D/A внутри каждого geom-класса по периодам.

### Hypothesis D-layer (универсальный d-слой)

**Описание.**

- d-металлы разных периодов (3d, 4d, 5d) принадлежат одному геометрическому классу `d_octa`
  и реализуют единый «металлический слой» по сложности и D/A.
- На этапе v2 работаем только с теми d-элементами, которые уже есть в `geom_atoms`/
  `element_indices_with_dblock.csv`.

**Задачи.**

- Написать анализ-скрипт `analysis/test_d_block_universality.py`.
- Сравнить D/A и сложность по периодам внутри `d_octa`, где данные есть.

### Geom-Mendeleev v3 — расширение по Z

#### Hypothesis SP-extended (s/p до 6-го периода)

Описание:
- Геометрическая периодичность D/A для s-доноров, s-мостов, p-semihub и p-акцепторов
  сохраняется при добавлении элементов 6-го периода (Cs, Ba, Tl, Pb, Bi, Po, At, Rn),
  описанных как геометрические клоны соответствующих классов.

Задачи:
- поддерживать в `geom_atoms.py` прототипы тяжёлых s/p как клонов существующих классов;
- написать анализ-скрипт, сравнивающий D/A и роли новых элементов с существующими периодами.

#### Hypothesis D-layer-extended (d-слой 4d/5d)

Описание:
- При добавлении d-элементов 4d/5d в класс `d_octa` средние D/A и ростовая сложность
  остаются в узком диапазоне вокруг значений 3d-слоя, то есть d-блок сохраняет
  характер универсального металлического слоя, либо выявляется естественное разбиение
  на подслои (ранние/поздние d-металлы).

Задачи:
- расширить d-слой (при необходимости) в `geom_atoms.py`;
- написать анализ-скрипт, сравнивающий D/A и сложности по периодам внутри `d_octa`.

- [DATA-CLEAN-ROOT-1] Принять решение по CSV в корне (`geom_nuclear_map.csv`, `periodic_after.csv`):
  - Вариант A: перенести в `data/` и обновить чтение/пайплайн.
  - Вариант B: оставить legacy в корне и зафиксировать fallback-логику как инвариант.
  - Результат зафиксировать в `docs/05_decision_log.md`.

- [SPECTRAL-SEMICL-1] Добавить semiclassical (phase-space) оценку F_levels:
  - Реализовать estimator в `core/f_levels_1d.py`.
  - Сравнить с exact-spectrum и текущими FDM-proxy в `analysis/spectral1d/*`.
  - Выгрузить CSV в `results/` и добавить короткий отчёт.

- [CLEAN-ROOT-ARTIFACTS-1] Перенести артефакты из корня (PNG/TXT) в `results/` и обновить ссылки в analysis-скриптах.
- [DEV-TESTS-1] Добавить `requirements-dev.txt` (pytest) и секцию в README для запуска тестов.
- [DEV-TESTS-2] Починить запуск `pytest -q` без `ModuleNotFoundError`:
  - добавить `tests/conftest.py` для bootstrap PYTHONPATH,
  - зафиксировать, какие тесты реально падают после устранения импорт-ошибок.
- [DEPS-1] Зафиксировать runtime/dev зависимости:
  - добавить `requirements.txt` (PyYAML как runtime),
  - сделать `requirements-dev.txt` надстройкой над runtime,
  - обновить `pytest.ini` для регистрации `@pytest.mark.slow`.
- [DEV-TESTS-3] Довести `pytest -q` до “содержательных” падений для роста:
  - Molecule: восстановить совместимый API (nodes/depth) через shim в `core/geom_atoms.Molecule`,
  - growth-сканы: синхронизировать тесты с текущим CLI (`tests/test_growth_scans_small.py`).

- [SPECTRAL-DENSITY-1] WS_Spectrum -> rho(r) -> FDM: подключить радиальную плотность из nuclear_spectrum_ws в estimate_atom_energy_fdm через ThermoConfig.density_source и coupling_density (с кешированием и масштабированием).
- [SPECTRAL-DENSITY-2] Следующий шаг: угловая часть (Y_lm) и вывод port_geometry из интерференции (не из JSON).
- [TOPO-3D-1] Добавить 3D force-directed layout и 3D-оценку пересечений/запутанности как альтернативу circle-proxy; сравнить 2D crossing proxy и 3D entanglement на типовых графах и сохранить результаты в results/topo3d_compare.csv.
- [DBLOCK-SPEC-1] Вывести d-block свойства из заполнения d-уровней спектра без Pauling.
- [SPECTRAL-WS-Z-1] Включить Z-зависимое радиусное масштабирование WS-потенциала (coupling_ws_Z, ws_Z_ref, ws_Z_alpha) и сделать ws_sp_gap(Z), rho_ws(r;Z) чувствительными к Z при сохранении legacy-режима при coupling_ws_Z=0.
- [SPECTRAL-GEOM-1C] После SPECTRAL-WS-Z-1: сделать ws_sp_gap действительно элемент-специфичным: centered hybrid_strength(gap_ref, scale), скан gap(Z), калибровка на B/C/N/O/Si/P/S.
- [SPECTRAL-DENSITY-1C] После SPECTRAL-WS-Z-1: добавить shape-sensitive observables (⟨r⟩, ⟨r²⟩, r_rms, kurtosis) для rho_gauss vs rho_ws_scaled; при необходимости R&D-терм в FDM energy, чувствительный к форме.
- [SPECTRAL-REPORT-CLI-1] Привести compare_density_sources и compare_port_geometry_sources к поддержке --thermo-config и CLI override’ов (в т.ч. coupling_ws_Z), печатать Effective ThermoConfig в консоль, добавить --out в scan_ws_gap_vs_Z чтобы не затирать результаты.
## HETERO-2 (P0) — Spectral Cycle Engine (WOW hard negatives)

### [HETERO2-EPIC-0] Цель продукта (стандарт отчета)
Описание:
- HETERO-report становится обязательным артефактом проверки модели на hard negatives с кольцами.
DoD:
- demo_aspirin_v2.py (красная кнопка) + aspirin_report.md
- pytest -q зеленый, CI зеленый

### [HETERO2-RDKIT-1] RDKit SMILES -> Graph + базовые дескрипторы
Задачи:
- ChemGraph(SMILES) -> adjacency/laplacian
- physchem: MW/LogP/TPSA (для отчета)

### [HETERO2-REWIRE-1] Cycle decoys: degree-preserving double-edge-swap + RDKit sanitize
Задачи:
- генерация K hard negatives с сохранением degrees
- фильтры валидности (rdkit sanitize), дедуп, not-isomorphic

### [HETERO2-SPECTRAL-1] Spectral fingerprint (Laplacian spectrum / LDOS)
Задачи:
- spectrum/ldos vector детерминированно
- тест: инвариантность к порядку, стабильность при фиксированных seed

### [HETERO2-PHI-EXP-0] Ray Harmony (эксперимент)
Задачи:
- Реализовать RayAuditor (divisor_sum_profile) + phi_from_eigs/phi_from_smiles (эксперимент)
- Собрать первые числа по 3+ молекулам с кольцами; критерий принятия: устойчивый сдвиг phi_decoys vs phi_original

## HETERO-2: v1.0.0 candidate checklist

- [V1-CHECKLIST-1] CONTEXT.md entrypoint: include links to pilot r2 release, registry entry, and the checklist itself.
- [V1-CHECKLIST-2] End-to-end Docker scenario: one command -> local evidence_pack.zip (2-minute run).
- [V1-CHECKLIST-3] Freeze hetero_scores.v1 contract: minimal backward-compat tests + doc pointer in CONTEXT.md.
