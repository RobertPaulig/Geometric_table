## QSG Versions (project-level)

- `QSG v7.1` — baseline for resumed version tracking; tag `qsg-v7.1` (see `VERSION`).

### [Geom-Mendeleev v2] старт гипотез

**Решение.**

- Развивать геометрическую таблицу как систему «гипотеза → скрипт → отчёт», а не как статическую картинку.
- Первая пара гипотез:
  - Hypothesis S/P: строгая D/A-периодичность для s/p до текущего диапазона \(Z\).
  - Hypothesis D-layer: универсальный d-слой (3d/4d/5d) в рамках доступных данных.

**Мотивация.**

- Это делает проект воспроизводимым и проверяемым.
- Любое следующее развитие (расширение по \(Z\), введение f-слоя) будет на тех же рельсах.

### [Geom-Mendeleev v3] расширение по Z

**Решение.**

- Расширять геометрическую таблицу по \(Z\) геометрическими клонами уже известных классов
  (s/p/d), фиксируя каждую гипотезу отдельным анализ-скриптом и отчётом.
- На этапе v3 не вводить ещё f-слой, а сосредоточиться на s/p до 6-го периода и d-слое 4d/5d.

**Мотивация.**

- Это позволяет отделить проверуемую экстраполяцию (s/p/d) от более спекулятивного f-слоя.

### [Heavy s/p χ_Pauling] референс и оверлей

**Решение.**

- Введён файл `data/pauling_heavy_sp_reference.csv` с χ_Pauling для Cs, Ba, Tl, Pb, Bi, Po, At
  по стандартной шкале Полинга (WebElements / Wikipedia).
- Эти значения используются только для проверки (оверлея) и не входят в текущий набор калибровки
  `fit_chi_to_pauling.py` (CALIB_SET остаётся прежним).

## [Geom–nuclear–FDM] Первое объединение слоёв (NB-1/NB-1_fix)

**Решение.**

- Введена комбинированная таблица `data/geom_nuclear_complexity_summary.csv`,
  объединяющая геометрический слой (роли, $D/A$, $\chi_{\mathrm{spec}}$),
  FDM-комплексность и ядерные индексы (`band_width`, `N_best`,
  `Nbest_over_Z`, `neutron_excess`).
- Исправлен поиск ядерных данных в
  `analysis/analyze_geom_nuclear_complexity.py`: файлы
  `geom_isotope_bands.csv` и `geom_nuclear_map.csv` теперь ищутся
  сначала в `data/`, затем в корне репозитория.

**Статус.**

- Для элементов H–Xe ядерные колонки заполнены для 38 из 44 элементов
  (остальные остаются пустыми и требуют расширения ядерного датасета).
- Скрипт `analysis/analyze_fdm_vs_nucleus.py` даёт следующие
  Spearman-корреляции между нормированной FDM-комплексностью
  `C_norm_fdm_mean` и ядерными параметрами (n=29):

  - глобально: `band_width` (r ≈ 0.33, p ≈ 0.08),
    `Nbest_over_Z` (r ≈ -0.43, p ≈ 0.020),
    `neutron_excess` (r ≈ -0.43, p ≈ 0.021);
  - по ролям/группам заметен только эффект для bridge (n=6):
    `Nbest_over_Z` (r ≈ 0.85, p ≈ 0.034),
    остальные связи статистически хрупкие.

- На данный момент связи между FDM-комплексностью и ядерными
  характеристиками рассматриваются как R\&D-уровень; для строгих
  утверждений требуется более полный и однородный ядерный датасет.

## [FDM Tree Complexity v1] выбор параметров λ, q

**Решение.**

- На основе сеточной калибровки `analysis/calibrate_fdm_params.py` выбран первый рабочий режим
  FDM-слоя сложности деревьев (Fgeom-FDM).
- Параметры зафиксированы как:
  - `lambda_fdm = 0.60`
  - `q_fdm = 1.50`

**Критерий выбора.**

- Для каждой пары (`lambda_fdm`, `q_fdm`) по ролям (`terminator`, `bridge`, `hub`) считались:
  - средние значения нормированной сложности `C2_norm_mean`,
  - стандартные отклонения `C2_norm_std`,
  - зазоры `delta_tb = m_bridge - m_term`, `delta_bh = m_hub - m_bridge`,
  - `min_gap = min(delta_tb, delta_bh)`,
  - `max_std = max(C2_norm_std по ролям)`.
- Введён score:
  - `score = min_gap - β * max_std` с β ≈ 0.7,
  - пары ранжировались по убыванию score.
- Требование к рабочему режиму v1:
  - разумный `min_gap > 0` (строгая иерархия по средним) на части сетки,
  - умеренный `max_std` относительно `min_gap`,
  - близость к максимуму score на текущей сетке.

**Интеграция.**

- Параметры зафиксированы в `core/complexity_fdm.py` как
  `LAMBDA_FDM_DEFAULT = 0.6`, `Q_FDM_DEFAULT = 1.5`.
- Эти значения используются во всех вызовах
  `compute_complexity_features_v2(..., backend="fdm" | "hybrid")`
  и во всех связанных R&D-скриптах (`analysis/test_tree_growth_law_fdm.py`,
  `analysis/calibrate_fdm_params.py`).

## [QSG v5.0] Silicon softness in tree growth

**Решение.**

- В `AtomGraph` добавлен параметр `softness`, отражающий мягкость атомного каркаса в росте деревьев.
- Для C: `softness = 0.0` (жёсткий базовый каркас).
- Для Si: `softness = 0.30`. При росте деревьев `grow_molecule_christmas_tree` глобальная вероятность
  продолжения ветвления умножается на `(1 - softness)` для всех шагов, если семя = Si.

**Эффект.**

- Средний размер и нормированная сложность Si-деревьев ниже, чем у C, при тех же параметрах роста,
  что отражает меньшую устойчивость длинных Si–Si цепочек по сравнению с C–C.

**Версия.**

- С этого шага модель обозначается как **QSG v5.0**.

## [QSG v5.0] Cycle statistics and crossing number

**Результат.**

- Скрипт `analysis/analyze_cycle_stats.py` показывает, что для текущих параметров `GrowthParams`
  подавляющее большинство сгенерированных молекул (по стандартному набору семян) являются деревьями
  с цикломатическим числом `cyclomatic = 0`.

**Решение.**

- На уровне QSG v5.0 не вводится явный crossing-number-штраф в Fgeom/FDM-слое.
- В `docs/04_backlog.md` остаётся отдельная задача на будущий R&D:
  при расширении генератора до графов с циклами реализовать crossing-proxy в FDM-слое.

## [QSG v5.0] Tree capacity constant

**Результат.**

- По данным `results/tree_growth_law_fdm_stats.txt` оценена константа `alpha_tree`
  в законе `C ~ alpha_tree * n log(1+n)`:
  значения `alpha_tree_v1` и `alpha_tree_fdm` записаны в `results/tree_capacity_fit.txt`.

**Интерпретация.**

- Это численный аналог предельной плотности сложности для деревьев
  в модели роста "Christmas Tree" при параметрах QSG v5.0.
- Константа фиксируется как R&D-инвариант и может использоваться для сравнения
  разных версий генератора и FDM-слоя.

## [Spectral Lab v1] Минимальный оператор $\hat{H}$ (QSG v6.0)

**Решение.**

- Введён минимальный ``Spectral Lab'' для 1D-оператора малых колебаний
  $\hat{H}$ на решётке: модуль `core.spectral_lab_1d` строит оператор
  $H = -(1/2m)\,\mathrm{d}^2/\mathrm{d}x^2 + V(x)$ на равномерной сетке
  и рассчитывает спектр, плотность состояний (DOS) и локальную плотность
  LDOS.
- На основе спектра определяется игрушечный функционал уровней
  `F_levels` (сумма весов по собственным значениям) и его FDM-прокси
  по потенциальной решётке (`core.f_levels_1d`).
- Скрипт `analysis/test_spectral_lab_1d.py` создаёт простые стенды
  (ящик, гармонический осциллятор), рассчитывает спектр, DOS/LDOS и
  сравнивает `F_levels` по спектру и по FDM-приближению, записывая
  результаты в `results/spectral_lab_1d_*.txt`.

**Статус.**

- Spectral Lab v1 реализует первый честный пример оператора $\hat{H}$
  и функционала $F_{\mathrm{levels}}$ в виде 1D-модели на решётке.
- Связь с основным FDM-слоем и Complexity пока носит характер R\&D:
  цель — проверить согласованность спектральных и FDM-приближений на
  простейших системах.
- Расширение Spectral Lab до 2D/3D и использование его для атомных
  конфигураций и геометрического атома отнесено к жёлтой зоне QSG v6.x.

## [QSG v6.x] Том II: развитие модели и Spectral Lab R&D

**Решение.**

- Введён отдельный документ `docs/name4.tex` как Том II модели QSG.
  В нём фиксируются направления развития версий v6.x и далее, не
  затрагивающие стабильное ядро QSG v5.0 (описанное в `name3.tex`).
- В Том II вынесены:
  - Spectral Lab v1 как R\&D-площадка для оператора $\hat{H}$ и
    функционала уровней $F_{\mathrm{levels}}$;
  - экспериментальные планы SL-1 (resolution scan, сравнение
    $F_{\mathrm{levels}}^{(\mathrm{spec})}$ и
    $F_{\mathrm{levels}}^{(\mathrm{FDM})}$) и SL-2 (игрушечная
    спектральная жёсткость $\chi_{\mathrm{spec}}^{(1D)}$);
  - дорожная карта по расширению Spectral Lab до 2D/3D, развитию
    d-блока, генератора с циклами и ядерного моста.

**Статус.**

- QSG v5.0/FDM v1 и Spectral Lab v1 задокументированы и считаются
  частью стабильного ядра (Том I, `name3.tex`).
- Том II (`name4.tex`) находится в стадии активного R\&D; все новые
  эксперименты и гипотезы для версий QSG v6.x и далее оформляются
  через Spectral Lab и соответствующие разделы Тома II.

## [CHEM-VALIDATION-0] C4 butane skeleton (n-butane vs isobutane)

**Конфиг.**

- Скрипт: `analysis/chem/chem_validation_0_butane.py`.
- Режим роста: `grow_molecule_christmas_tree` с параметрами
  `stop_at_n_atoms=4`, `allowed_symbols=["C"]` (ровно 4 атома углерода, без H).
- Атомный слой: стандартный `AtomGraph` для C (softness=0, жёсткий каркас).
- Термоконфиги (через `override_thermo_config`):
  - Mode A: только FDM-комплексность (`compute_complexity_features_v2(..., backend="fdm")`, `coupling_topo_3d=0`).
  - Mode B: FDM + topo3d layout/entanglement (`backend="fdm_entanglement"`, `coupling_topo_3d=1`).
- Запуск CHEM-VALIDATION-0:
  - пример: `python -m analysis.chem.chem_validation_0_butane --n_runs 50 --seeds 0 1 --modes A B`.
  - артефакты: `results/chem_validation_0_butane.csv`, `results/chem_validation_0_butane.txt` (git-ignored).

**Метрики.**

- Топология определяется по отсортированным степеням вершин `deg_sorted` (граф — чистое дерево по C–C):
  - `deg_sorted = [1,1,2,2]` → `topology = "n_butane"`.
  - `deg_sorted = [1,1,1,3]` → `topology = "isobutane"`.
  - любое другое дерево на 4 вершинах → `topology = "other"`.
- Для каждой траектории роста сохраняются:
  - `mode`, `seed`, `run_idx`, `n_atoms` (ожидается 4),
  - `topology`, `deg_sorted`,
  - `complexity_fdm`, `complexity_fdm_entanglement`, `total_score`, `runtime_sec`.
- В `results/chem_validation_0_butane.txt` сводятся:
  - частоты `P(n_butane)`, `P(isobutane)` по каждому режиму,
  - `median` / `p90` по `total_score` для каждой топологии,
  - оценка стабильности `log(P(iso)/P(n))` (по суммарным счётчикам; разбивка по seed может быть добавлена позже).

**Наблюдения (первый прогон, n_runs=50, seeds=[0,1]).**

- Mode A (только FDM):
  - `P(isobutane) ≈ 0.97` (97/100), `P(n_butane) ≈ 0.03` (3/100).
  - `score[isobutane]: median ≈ 2.84, p90 ≈ 2.84`.
  - `score[n_butane]: median ≈ 3.09, p90 ≈ 3.09`.
  - `log(P(iso)/P(n)) ≈ 3.48` (сильный перекос сопряжённой цепочки в пользу изобутана по частоте роста).
- Mode B (FDM + topo3d entanglement):
  - `P(isobutane) ≈ 0.93` (93/100), `P(n_butane) ≈ 0.07` (7/100).
  - `score[isobutane]: median ≈ 2.84, p90 ≈ 2.84`.
  - `score[n_butane]: median ≈ 3.09, p90 ≈ 3.09`.
  - `log(P(iso)/P(n)) ≈ 2.59`.

**Выводы.**

- В текущей конфигурации CHEM-VALIDATION-0 модель явно предпочитает изобутановый каркас по частоте генерации (`P(iso) >> P(n)`).
- При этом FDM-слой сложности (как в чистом виде, так и с topo3d-модификацией) даёт более высокий `total_score` для n-butane (`deg=[1,1,2,2]`), чем для isobutane.
- Это означает, что на уровне C4-деревьев текущий ростовой механизм и сложностной функционал не согласованы с интуитивным химическим порядком стабильностей (цепь vs ветвление):
  - генератор почти всегда растит более ветвлённый каркас,
  - но complexity/entanglement трактует линейный каркас как более «сложный» объект.
- Следующий шаг после CHEM-VALIDATION-0:
  - отделить вклад proposal-политики роста от энергий/сложности (CHEM-VALIDATION-1 на C5/C6),
  - ввести более жёсткую проверку на согласованность порядка стабильностей по нескольким изомерным сериям.

## [CHEM-VALIDATION-1A] C5 pentane skeleton (n / iso / neo)

**Конфиг.**

- Скрипт: `analysis/chem/chem_validation_1a_pentane.py`.
- Режим роста: `grow_molecule_christmas_tree` с параметрами
  `stop_at_n_atoms=5`, `allowed_symbols=["C"]`, `max_depth=5`.
- ALKANE-VALIDITY-0 (конструктивная валидность, без rejection-sampling по траектории):
  - в `core/grower.py` добавлен флаг `GrowthParams.enforce_tree_alkane=True`,
    который строит **связный tree-only** C-скелетон (ровно 1 ребро на новый атом, без циклов) с валентностью `deg(C)<=4`;
  - при `stop_at_n_atoms=5` всегда получается валидный C5-алкановый каркас, поэтому `P(other)=0` и `resample_attempts_used==1`.
- Терморежимы (абляции, через `override_thermo_config`):
  - `R` (proposal-only): `grower_use_mh=False`, все couplings = 0.
  - `A` (FDM-only): MH on, `compute_complexity_features_v2(..., backend="fdm")`, `coupling_topo_3d=0`.
  - `B` (FDM + topo3d): как `A` + `compute_complexity_features_v2(..., backend="fdm_entanglement")`, `coupling_topo_3d=1`.
  - `C` (FDM + topo3d + shape): как `B` + `coupling_shape_softness=1`, `coupling_shape_chi=1` (shape-часть пока R&D-заглушка в отчёте, `score_shape=0`).
- SPEED-FIX-1: для `B/C` включён topo3d prefilter на деревьях/малых n (`topo3d_prefilter_tree=True`, `topo3d_prefilter_min_n=10`), чтобы не запускать layout/entanglement на C5-деревьях.
- PROGRESS-UX-1: добавлен `analysis/utils/progress.py` и флаг `--progress` (default=True) для отображения прогресса (tqdm или fallback “N/Total”).
- Запуск CHEM-VALIDATION-1A (полный):
  - `python -m analysis.chem.chem_validation_1a_pentane --n_runs 1000 --seeds 0 1 2 3 4 --modes R A B C --progress`.
  - артефакты (git-ignored): `results/chem_validation_1a_pentane.csv`, `results/chem_validation_1a_pentane.txt`.

**Классификация топологий (degree multiset).**

- `deg_sorted = [1,1,2,2,2]` → `topology="n_pentane"`.
- `deg_sorted = [1,1,1,2,3]` → `topology="isopentane"`.
- `deg_sorted = [1,1,1,1,4]` → `topology="neopentane"`.
- иначе → `topology="other"`.

**Результаты частот (n_runs=1000, seeds=[0..4], всего 5000 на режим; tree-only, `other=0`).**

- Mode `R`:
  - `P(iso)=0.5762`, `P(n)=0.3372`, `P(neo)=0.0866`.
  - `log(P(iso)/P(n))=0.5358`, `log(P(neo)/P(n))=-1.3594`.
- Mode `A`:
  - `P(iso)=0.5790`, `P(n)=0.3394`, `P(neo)=0.0816`.
  - `log(P(iso)/P(n))=0.5341`, `log(P(neo)/P(n))=-1.4254`.
- Mode `B`:
  - `P(iso)=0.5720`, `P(n)=0.3364`, `P(neo)=0.0916`.
  - `log(P(iso)/P(n))=0.5308`, `log(P(neo)/P(n))=-1.3009`.
- Mode `C`:
  - `P(iso)=0.5810`, `P(n)=0.3336`, `P(neo)=0.0854`.
  - `log(P(iso)/P(n))=0.5548`, `log(P(neo)/P(n))=-1.3626`.

**Производительность/тайминги (полный прогон 20k runs).**

- `elapsed_sec ≈ 43.1`, `runs_done=20000`.
- `resample_attempts_used`: mean=1.000, p90=1, p99=1 (конструктивный генератор).
- CSV содержит честные тайминги: `t_growth_sec`, `t_scoring_sec`, `t_total_sec`.

**Детерминированные эталоны (fixed adjacency; A/B/C).**

- Edges:
  - n-pentane (path): `(0-1,1-2,2-3,3-4)`
  - isopentane: `(1-0,1-2,1-4,2-3)`
  - neopentane (star): `(0-1,0-2,0-3,0-4)`
- Scores/Δ:
  - Mode `A`: `score(n)=4.2985`, `score(iso)=4.0633`, `score(neo)=3.1399`;
    `Δ(n-iso)=0.2351`, `Δ(n-neo)=1.1586`, `Δ(iso-neo)=0.9235`.
  - Mode `B`: `score(n)=4.2985`, `score(iso)=4.0804`, `score(neo)=3.1399`;
    `Δ(n-iso)=0.2180`, `Δ(n-neo)=1.1586`, `Δ(iso-neo)=0.9406`.
  - Mode `C`: `score(n)=4.2985`, `score(iso)=4.0804`, `score(neo)=3.1399`;
    `Δ(n-iso)=0.2180`, `Δ(n-neo)=1.1586`, `Δ(iso-neo)=0.9406`.

**Sign-check (что делает MH).**

- В коде MH (`core/mh.py`) всегда принимается `ΔG<=0`, и для `ΔG>0` используется вероятность `exp(-ΔG/T)`.
  С учётом `ΔG = coupling*(E_new - E_old)` и `E=compute_energy=complexity.total`, это означает:
  **MH минимизирует score/energy** (меньший `score_total` считается стабильнее).
- Для эталонов (A/B/C) это предсказывает порядок `neo > iso > n` в конвенции “min-score best”,
  тогда как по частотам на tree-only состояниях наблюдается доминирование `iso` (и `neo` не доминирует).
  Несостыковка “частоты/acceptance vs детерминированные score” сохраняется и требует отдельной калибровки/пересмотра proposal.

**CORE-1 (degeneracy-aware прогноз частот).**

- Для tree-only изомеров частоты определяются не только энергией, но и дегенерацией по разметкам (числом labeled-скелетов) `g(topology)`.
- В отчётах `chem_validation_0_butane.py` и `chem_validation_1a_pentane.py` добавлен блок:
  - `P_pred(topology) ∝ g(topology) * exp(-coupling_delta_G * E_ref(topology) / temperature_T)`,
  - сравнение `P_pred` с наблюдаемыми `P_obs` и разности по `log(P(iso)/P(n))` (и `log(P(neo)/P(n))` для C5).
  - референсные дегенерации: C4 `g(n)=12`, `g(iso)=4`; C5 `g(n)=60`, `g(iso)=60`, `g(neo)=5`.

## [CHEM-VALIDATION-CORE-2] Fit λ (energy scale) for degeneracy-aware model

**Решение.**

- Добавлен модуль `analysis/chem/core2_fit.py`, который подбирает масштаб энергии `λ` по сетке,
  минимизируя `KL(P_obs || P_pred(λ))` для модели:
  - `P_pred(topo;λ) ∝ g(topo) * exp(-λ * E_ref(topo) / temperature_T)`.
- В `analysis/chem/chem_validation_0_butane.py` и `analysis/chem/chem_validation_1a_pentane.py`
  добавлен блок `CORE-2: Fit lambda (degeneracy-aware)` с `λ*`, `KL_min`, `P_obs/P_pred(λ*)` и Δlog-отношениями.

**Пример результатов (n_runs=1000, seeds=[0..4]).**

- C4 / Mode A: `λ*≈0.5852`, `KL_min≈0` (для 2-топологий λ может подогнать частоты практически точно).
- C5 / Mode A: `λ*≈0.9225`, `KL_min≈0.0126` (остаточное расхождение указывает на вклад kernel/динамики роста помимо энергии+degeneracy).

## [MH-KERNEL-1] Fixed-N tree MCMC with Hastings correction

**Решение.**

- Добавлен контрольный фикс-N MCMC для labeled деревьев (N=4/5) с обратимыми move’ами leaf-rewire
  и явной Hastings-коррекцией `q(y→x)/q(x→y)`:
  - модуль `analysis/chem/topology_mcmc.py`,
  - runner `analysis/chem/mh_kernel_1_bench.py` (артефакты в `results/`, git-ignored).
- Цель: сравнивать `P_mcmc_fixedN` с `P_pred(λ*)` (и косвенно с `P_obs_growth` из chem_validation),
  чтобы диагностировать bias ростового kernel’а.

**Пример (mode A, steps=50k, burnin=5k, thin=10).**

- N=4: при `λ*=0.5852` получено `KL(P_mcmc||P_pred)≈0.0192`.
- N=5: при `λ*=0.9225` получено `KL(P_mcmc||P_pred)≈0.0257`.

## [MH-KERNEL-2] Exact baseline (Prüfer enumeration) + CORE-3 (label-averaged prediction)

**Решение.**

- Добавлен перебор всех labeled деревьев через Prüfer sequences:
  - модуль `analysis/chem/exact_trees.py` (`N=4 → 16`, `N=5 → 125`).
- Добавлен exact-бейзлайн:
  - скрипт `analysis/chem/mh_kernel_2_exact.py`, который считает точное распределение
    `P_exact(state) ∝ exp(-λ E(state)/T)` и агрегирует до `P_exact(topology)`,
    а также печатает:
    - `P_pred` (CORE-2: `g*exp(-λE_ref/T)`),
    - `P_mcmc` (из MH-KERNEL-1 artifacts),
    - `KL(P_mcmc||P_exact)` и `KL(P_pred||P_exact)`,
    - статистики `E_min/E_mean/E_std/E_max` по labeled состояниям внутри топологии (диагностика label-dependence).
- CORE-3 определяется как label-averaged prediction:
  - `P_pred3(topology) = sum_{state in topo} exp(-λ E(state)/T) / Z`;
  - при использовании тех же state-энергий `P_pred3` совпадает с `P_exact` по определению.

**Результаты (Mode A, λ* из CORE-2, N=4/5).**

- N=4:
  - (до фикса инвариантности) `KL(P_mcmc||P_exact) ≈ 0.000123`, `KL(P_pred||P_exact) ≈ 0.017267`, `E_std≈0.25`.
- N=5:
  - (до фикса инвариантности) `KL(P_mcmc||P_exact) ≈ 0.000129`, `KL(P_pred||P_exact) ≈ 0.028293`, `E_std≈0.32–0.33`.

**Вывод.**

- Основной источник расхождения `P_mcmc` vs `P_pred(λ*)` — не mixing и не Hastings,
  а **label-dependence энергии**: энергия `E(state)` варьирует внутри одной и той же unlabeled топологии,
  поэтому `P_pred = g*exp(-λE_ref(topology)/T)` по одному представителю не является точной моделью.

## [LABEL-INVARIANCE-1] Canonical relabeling for tree-only FDM energy

**Решение.**

- Введена канонизация деревьев (AHU) и её применение в FDM-энергии:
  - `core/tree_canonical.py`: `canonical_relabel_tree(adj)` (центр/кодирование AHU, детерминированная перестановка вершин).
  - `core/complexity_fdm.py`: для connected tree (`m=n-1`) adjacency сначала канонизируется, затем считается FDM.
- Добавлен тест `tests/test_energy_label_invariance.py`: 200 случайных перестановок для C5 (n/iso/neo) в Mode A дают `std(E)<=1e-6`.

**Результат.**

- После канонизации `E_std(topology)=0` (в пределах float) на N=4 и N=5, и `KL(P_pred||P_exact)≈0`.
- MCMC при тех же `λ*` теперь согласуется с `P_pred` на уровне малого KL:
  - N=4, Mode A: `KL(P_mcmc||P_pred)≈1.5e-4`.
  - N=5, Mode A: `KL(P_mcmc||P_pred)≈5.4e-4`.

## [CHEM-VALIDATION-1B] C6 hexane skeleton (5 tree isomers)

**Конфиг.**

- Скрипт: `analysis/chem/chem_validation_1b_hexane.py`.
- Рост: tree-only C6 (C-skeleton), `stop_at_n_atoms=6`, `allowed_symbols=["C"]`, `enforce_tree_alkane=True`.
- Режимы `R/A/B/C` как в CHEM-VALIDATION-1A, topo3d prefilter включён (`topo3d_prefilter_tree=True`, `topo3d_prefilter_min_n=10`).
- Классификатор 5 изомеров:
  - `n_hexane`: `deg_sorted=[1,1,2,2,2,2]`
  - `2_methylpentane` vs `3_methylpentane`: `deg_sorted=[1,1,1,2,2,3]` и `dist(deg2,deg2)` = 1 vs 2
  - `2,2_dimethylbutane`: `deg_sorted=[1,1,1,1,2,4]`
  - `2,3_dimethylbutane`: `deg_sorted=[1,1,1,1,3,3]`
  - DoD: `P(other)=0` (tree-only генерация).

**CORE-1 (degeneracy).**

- `g(topology)=6!/|Aut|`:
  - `g(n_hexane)=360`, `g(2_methylpentane)=360`, `g(3_methylpentane)=360`,
  - `g(2,2_dimethylbutane)=120`, `g(2,3_dimethylbutane)=90`,
  - `sum_g=1290` (все labeled деревья N=6 кроме star).

**CORE-2 (λ-fit).**

- В отчёте печатаются `λ*`, `KL_min`, `P_obs` vs `P_pred(λ*)` и Δlog-отношения относительно `n_hexane`.

**Наблюдение (красный флаг).**

- В CHEM-VALIDATION-1B наблюдается `KL(P_obs||P_pred) ~ 0.09–0.11` и слабая зависимость частот от режимов `R/A/B/C`.
- Это несовместимо с интерпретацией `P_obs` как equilibrium-распределения по топологиям при фиксированном `N=6`:
  - после [LABEL-INVARIANCE-1] энергия для tree-only стала функцией **топологии**,
  - значит семейство `P_eq(topo;λ) ∝ g(topo) * exp(-λ E(topo)/T)` должно описывать равновесие (при корректном fixed-N sampler’е).
- Следствие: ростовой процесс (growth) даёт **kernel-biased proposal distribution** `Q(topology)` и не является корректным fixed-N sampler’ом по финальным деревьям.

## [MH-KERNEL-3] C6 exact baseline (Prüfer) + fixed-N MCMC vs growth

**Решение.**

- Для `N=6` введён exact baseline по labeled деревьям (через Prüfer, всего `6^(6-2)=1296`) с фильтром алкан-валидности `deg<=4` (остаётся `1290`):
  - `analysis/chem/mh_kernel_3_c6_exact.py`: считает `P_exact(state) ∝ exp(-λ E(state)/T)` и агрегирует до `P_exact(topology)` (Mode A).
- Введён fixed-N MCMC на деревьях `N=6` с move `leaf-rewire` + Hastings и ограничением `deg<=4`:
  - `analysis/chem/mh_kernel_3_c6_mcmc.py`: считает `P_mcmc(topology)` и печатает `KL(P_mcmc||P_exact)`.
- Сделано сравнение `P_obs_growth` (из CHEM-VALIDATION-1B) vs `P_mcmc` vs `P_exact`, плюс контроль `P_pred(λ)` (CORE-2).

**Результаты (Mode A, N=6).**

- Энергетическая модель (после LABEL-INVARIANCE-1) согласована с exact:
  - `P_pred ≈ P_exact` (по топологиям).
- MCMC корректен:
  - `KL(P_mcmc||P_exact) ≈ 1.6e-4`.
- Growth-частоты не равновесные:
  - `KL(P_obs_growth||P_exact) ≈ 0.096`,
  - `P_obs_growth` почти не меняется между `R/A/B/C` → доминирует **kernel bias** ростового процесса.

**Вывод.**

- Частоты топологий, полученные “по финальным деревьям” из growth, нельзя интерпретировать как термодинамические.
- Growth остаётся как диагностический слой для `Q(topology)` (proposal/kernal bias), а equilibrium оценивается fixed-N MCMC (или exact там, где возможно).

## [TIMING-UX-1] Wall-clock + breakdown (growth/scoring/io)

**Решение.**

- Введена общая утилита тайминга `analysis/utils/timing.py`:
  - `now_iso()`, `format_sec(x)`, контекст `timed(name, acc)`.
- Скрипты CHEM-VALIDATION / MH-KERNEL / benches печатают финальные строки:
  - `Wall-clock: start=<...> end=<...> elapsed_total_sec=<...>`
  - `Breakdown: growth=<...> scoring=<...> io=<...>` (где применимо),
  - и записывают эти значения в TXT-артефакты в `results/`.

**Ключевой вывод (почему “equilibrate-per-run” запрещён как стандарт).**

- Baseline C6 (Mode A, `5000` runs): `elapsed_total≈18.4s`, из них `growth≈14.6s`.
- Equilibrated-per-run (добавить `equilibrate_steps=2000` на каждый run): `elapsed_total≈1181.6s` (~19m 42s), почти всё — рост (fixed-N MCMC внутри каждого run).
- Стоимость растёт как `O(#runs × steps)`, поэтому per-run equilibration не подходит как стандартный протокол CHEM-VALIDATION.

## [CHEM-VALIDATION-EQ-2] Equilibrium runner (fixed-N) + сравнение с growth

**Решение (новый протокол CHEM-VALIDATION).**

- Разделяем две сущности:
  1) `Q(topology)` — proposal distribution от growth (kernel bias диагностируется, не “термодинамика”).
  2) `P_eq(topology)` — equilibrium distribution от fixed-N MCMC при заданной энергии (термодинамический объект).
  3) Для `N<=6`: сравнение `P_eq` с `P_exact` как ground truth.
- Добавлен раннер равновесия:
  - `analysis/chem/chem_validation_eq_runner.py` (`--N`, `--mode`, `--steps`, `--burnin`, `--thin`, `--chains`, `--progress`, стартовые топологии).
- CHEM-VALIDATION-1B обновлён так, что печатает:
  - `P_growth` (proposal),
  - `P_eq` (из eq-runner),
  - `KL(P_growth||P_eq)`,
  - и для C6 Mode A (если доступен exact): `KL(P_eq||P_exact)`.

## [EQ-TARGET-1] Steps→KL калибровка (worst-case starts)

**Решение.**

- Введён скан “сколько шагов нужно для заданного качества”:
  - `analysis/chem/eq_target_1_scan.py`: прогоняет fixed-N MCMC на сетке `steps` и печатает `KL_max/KL_mean` по “плохим стартам”.
- Guardrail: старт фиксируем явно, чтобы KL не зависел от удачного старта.
  - Дефолтные “плохие” старты для `N=6`: `n_hexane` и `2,2_dimethylbutane`.

**Наблюдение (Mode A, N=6).**

- Уже при `steps=2000` достигается `KL_max ≈ 0.00236` за ~2 секунды.
- При `steps=20000`: `KL ≈ 4.9e-4` за ~6 секунд (убывающая отдача).
- Следствие: для `N=6` стандартный бюджет может быть `steps=2000` (KL≪0.01), а для `N>=7` сначала делается аналогичный “EQ-TARGET scan” и выбирается `steps` по таргету.

## [ENERGY-CACHE-1] Memoization энергии в fixed-N MCMC (tree-only)

**Решение.**

- В fixed-N MCMC добавлено кэширование `energy(state)` по каноническому коду/топологии (для tree-only после LABEL-INVARIANCE-1 энергия топологична).
- В отчётности MCMC/eq-runner печатаются:
  - `steps/sec`,
  - `cache_hit_rate` (hits/misses),
  - acceptance.

**Статус.**

- Кэш обязателен для масштабирования на `N=7/8+` и для режимов с более дорогим scoring.

## [EQ-TARGET-2] Mixing budget guardrail for N=7/8 (pairwise-start KL)

**Решение.**

- Для `N=7` и `N=8` введён инженерный “датчик смешивания” fixed-N MCMC без exact baseline:
  - скрипт `analysis/chem/eq_target_2_scan.py` (Mode A по умолчанию),
  - стартовые состояния задаются как **структурные** спецификации, независимые от строковых имён изомеров:
    - `path` (линейная цепь),
    - `max_branch` (сильно разветвлённое дерево при ограничении `deg<=4`).
- На сетке `steps ∈ {2k, 5k, 10k, 20k}` (опционально `50k`) считаются:
  - wall-clock: `elapsed_total_sec`,
  - throughput: `STEPS_TOTAL`, `STEPS_PER_SEC_TOTAL`,
  - `energy_cache_hit_rate`,
  - guardrail: `KL_max/KL_mean` как **pairwise** KL между оценками распределений,
    полученных из разных “плохих стартам” (для 2 стартов: `max(KL(P1||P2), KL(P2||P1))`).

**Вывод.**

- Для `N>=7`, где exact baseline недоступен, бюджет шагов выбирается по стабилизации:
  - `KL_max` между “плохими стартам” и/или стабильности ключевых log-ratio,
  - при обязательном логировании `steps/sec` и `cache_hit_rate` (инженерная стоимость).

**Результаты (Mode A, λ=1.0, chains=3, thin=10, burnin_frac=0.1, starts=[path,max_branch]).**

- N=7:
  - steps=2k:  `KL_max=0.006028`, elapsed=0.579s, steps/sec=20734, hit=1.000
  - steps=5k:  `KL_max=0.000984`, elapsed=1.389s, steps/sec=21591, hit=1.000
  - steps=10k: `KL_max=0.000333`, elapsed=2.696s, steps/sec=22252, hit=1.000
  - steps=20k: `KL_max=0.000068`, elapsed=5.431s, steps/sec=22097, hit=1.000
- N=8:
  - steps=2k:  `KL_max=0.083180`, elapsed=0.627s, steps/sec=19146, hit=0.999
  - steps=5k:  `KL_max=0.002120`, elapsed=1.601s, steps/sec=18742, hit=1.000
  - steps=10k: `KL_max=0.001495`, elapsed=3.173s, steps/sec=18912, hit=1.000
  - steps=20k: `KL_max=0.001115`, elapsed=6.352s, steps/sec=18892, hit=1.000

**Интерпретация.**

- Для `N=7` уже `steps=5000` даёт очень малый guardrail (`KL_max≈1e-3`).
- Для `N=8` видно “плохой старт”-эффект на `steps=2000` (высокий `KL_max`), после чего `steps>=5000` стабилизирует оценку (`KL_max≈0.002` и ниже).

## [ALKANE-EXACT-1] Exact equilibrium for tree-only alkanes (N=7/8)

**Решение.**

- Для tree-only алканов (фильтр `deg<=4`) введён exact baseline по **unlabeled** топологиям без MCMC:
  - скрипт `analysis/chem/alkane_exact_1.py`.
- Алгоритм:
  - перечисляем все labeled деревья через Prüfer (`N=7: 7^5=16807`, `N=8: 8^6=262144`),
  - фильтруем `max_degree<=4`,
  - коллапсируем до уникальных unlabeled топологий по AHU-коду (`N=7: 9`, `N=8: 18`),
  - считаем размер автоморфизмов `|Aut|` и дегенерацию `g=n!/|Aut|` (проверка: `g == labeled_count`),
  - считаем энергии `E(topo)` (Mode A/B/C) и exact-меру
    `P_exact(topo) ∝ g(topo) * exp(-λ E(topo)/T)` при заданных `λ,T`.
- Артефакты пишутся в `results/alkane_exact_1_N{7,8}.csv|txt` (git-ignored).

**Результаты (λ=1.0, T=1.0).**

- N=7:
  - labeled total=16807, alkane(deg<=4)=16590, уникальных топологий=9.
  - `KL(P_eq||P_exact)=0.002760` (sanity: MCMC близок к exact при eq_steps=5000).
  - `KL(P_growth||P_exact)=0.199514` (истинная мера kernel/proposal bias роста).
- N=8:
  - labeled total=262144, alkane(deg<=4)=255920, уникальных топологий=18.
  - `KL(P_eq||P_exact)=0.004798`.
  - `KL(P_growth||P_exact)=0.283005`.

**Вывод.**

- Для `N=7/8` exact baseline дешевле и точнее, чем EQ-цепи (энергия после LABEL-INVARIANCE-1 топологична на деревьях).
- Ростовой sampler остаётся существенно смещённым относительно равновесия, и этот bias растёт с N.

## [CHEM-VALIDATION-EXACT-UX-1] P_exact as default reference for N<=8 + per-topology bias table

**Решение.**

- Скрипты `analysis/chem/chem_validation_1c_heptane.py` и `analysis/chem/chem_validation_1d_octane.py` читают
  exact baseline из `results/alkane_exact_1_N{7,8}.csv` (если файл существует) и печатают дополнительно:
  - `KL(P_eq||P_exact)` и `KL(P_growth||P_exact)` по каждому mode,
  - таблицу bias по топологиям: `bias = log(P_growth/P_exact)` (top-K по `|bias|`).
- Таким образом, для `N<=8` “истина” фиксируется как `P_exact`, а `P_eq` остаётся sanity/диагностикой.

## [GROWTH-KERNEL-BIAS-1] Bias vector + correlations vs simple tree invariants

**Решение.**

- Добавлен диагностический скрипт `analysis/chem/growth_kernel_bias_1.py`, который строит:
  - bias-вектор `log(P_growth/P_exact)` по топологиям,
  - Spearman-корреляции bias с простыми инвариантами деревьев:
    `diameter`, `max_degree`, `n_leaves`, `Wiener index`.
- Это даёт быстрый ориентир, *какие структурные факторы* систематически пере/недо-производит ростовой kernel.

## [CHEM-VALIDATION-2] C9/C10 equilibrium-first (tree-only) + self-consistency vs P_pred

**Решение.**

- Для `N=9` (C9, 35 топологий) и `N=10` (C10, 75 топологий) введены equilibrium-first прогоны:
  - `analysis/chem/chem_validation_2_nonane.py` (C9),
  - `analysis/chem/chem_validation_2_decane.py` (C10),
  - общая логика в `analysis/chem/chem_validation_2_common.py`.
- Fixed-N MCMC (leaf-rewire + Hastings) запускается с двумя “плохими стартами” `path/max_branch` и несколькими цепями,
  затем проверяется:
  - coverage (`n_unique_eq` равно ожидаемым `35/75`),
  - guardrail `KL_max_pairwise/KL_mean_pairwise` между стартами,
  - self-consistency: `KL(P_eq||P_pred)` для
    `P_pred(t) ∝ g(t) * exp(-λ E(t)/T)`, где `g=N!/|Aut|`.

**Результаты (Mode A, λ=1.0, T=1.0).**

- C9 (N=9):
  - coverage: `n_unique_eq=35`.
  - guardrail: `KL_max_pairwise=0.000763`, `KL_mean_pairwise=0.000760` (target ≤ 0.002).
  - self-consistency: `KL(P_eq||P_pred)=0.000256` (target ≤ 0.01).
  - бюджет: авто-повышение шагов до `steps=80000` (chains=5, starts=2) для выполнения guardrail.
- C10 (N=10):
  - coverage: `n_unique_eq=75`.
  - guardrail: `KL_max_pairwise=0.003181`, `KL_mean_pairwise=0.003120` (target ≤ 0.005).
  - self-consistency: `KL(P_eq||P_pred)=0.000737` (target ≤ 0.02).
  - бюджет: `steps=50000` достаточен (chains=5, starts=2).

**Вывод.**

- Для `N=9/10` критерии “корректности внутри модели” выполняются без внешнего эталона:
  coverage + convergence guardrail + согласие `P_eq` с `g*exp(-E)` (малый KL).

## [CHEM-VALIDATION-3] C11/C12 equilibrium-first + diagnostics (TECH-DEBT-1)

**Решение.**

- Добавлена диагностика производительности и “чистота метрик” для fixed-N MCMC:
  - `energy_cache_misses` трактуется как число **уникальных топологий, впервые увиденных в цепи**
    (а не “cache miss по compute”), чтобы прогрев кэша (например, при авто-эскалации steps) не обнулял misses.
  - Добавлен микро-профайлер (сэмплирование одного шага каждые `profile_every`):
    `t_move_avg`, `t_energy_avg`, `t_canon_avg`.
  - В `run_equilibrium_with_guardrail` шаги/принятия/кэш теперь агрегируются **по всем attempts**,
    чтобы throughput (`steps/sec`) был честным при авто-эскалации (это объясняет “аномалию” C9).
- Добавлены прогоны CHEM-VALIDATION-3 (equilibrium-first) для:
  - `analysis/chem/chem_validation_3_undecane.py` (C11),
  - `analysis/chem/chem_validation_3_dodecane.py` (C12).

**Результаты (Mode A, λ=1.0, T=1.0; starts=path/max_branch; chains=5).**

- C11 (N=11, target 159):
  - coverage: `n_unique_eq=159/159`.
  - guardrail: `KL_max_pairwise=0.003665` (target ≤ 0.005).
  - self-consistency: `KL(P_eq||P_pred)=0.000855`.
  - micro-profiler (`profile_every=100`): `t_move_avg≈7.40e-05s`, `t_canon_avg≈1.02e-04s`, `t_energy_avg≈9.44e-07s`.
  - “shape sanity”: SpearmanCorr(Energy, Wiener) `ρ≈0.7587` (энергия растёт с Wiener → более “растянутые” деревья менее вероятны).
  - timing: `STEPS_TOTAL=1,000,000`, `STEPS_PER_SEC_TOTAL≈5.18k`.
- C12 (N=12, target 355):
  - coverage: `n_unique_eq=355/355`.
  - guardrail: `KL_max_pairwise=0.003939` (target ≤ 0.005), потребовалась авто-эскалация до `steps=200k`.
  - self-consistency: `KL(P_eq||P_pred)=0.001075`.
  - micro-profiler (`profile_every=100`): `t_move_avg≈9.62e-05s`, `t_canon_avg≈1.34e-04s`, `t_energy_avg≈9.10e-07s`.
  - “shape sanity”: SpearmanCorr(Energy, Wiener) `ρ≈0.7536`.
  - timing: `STEPS_TOTAL=2,000,000`, `STEPS_PER_SEC_TOTAL≈2.65k`.

**Вывод.**

- Coverage на C11/C12 достигнут (159/355) без внешней “подсказки”.
- Guardrail сходится в пределах таргетов, а `P_eq` согласуется с `P_pred ∝ g*exp(-E)` (малый KL), т.е. MCMC даёт корректную меру внутри модели.
- Профилирование показывает, что “горячая” часть шага — `move` + `canonicalization`, а вычисление энергии почти полностью прячется кэшем (t_energy_avg ~ 1e-6s).

## [CHEM-VALIDATION-4] C13/C14 equilibrium-first (tree-only) + coverage/guardrail/self-consistency gates

**Решение.**

- Расширить equilibrium-first пайплайн на `N=13/14` в `Mode A` с теми же внутренними критериями корректности:
  - coverage gate: `n_unique_eq == expected_unique(N)` (табличный expected для алканов с `deg<=4`),
  - guardrail: `KL_max_pairwise` между стартами `path/max_branch`,
  - self-consistency: `KL(P_eq||P_pred)` для `P_pred ∝ g*exp(-λE/T)` при `λ=1.0`.
- Встроить ожидаемые counts в одном месте: `analysis/chem/alkane_expected_counts.py` (для `N=7..14`).
- Для удержания времени добавить OPT-2 канонизации: MCMC использует AHU-канонический ключ (tuple edge-key),
  без плотной `NxN` канонизации внутри шага.

**DoD команды.**

- `python -m analysis.chem.chem_validation_4_tridecane --mode A --progress`
- `python -m analysis.chem.chem_validation_4_tetradecane --mode A --progress`
- `python -m analysis.bench.invariance_bench_2`

## [MIXING-DIAGNOSTICS-1] Fixed-N MCMC mixing diagnostics (beyond guardrail)

**Решение.**

- Добавлен модуль `analysis/chem/mixing_diagnostics_1.py`, который даёт количественные признаки:
  - chain agreement: `KL_max_pairwise/KL_mean_pairwise` по `P(chain)`,
  - split-chain stationarity: `KL_split_max/KL_split_mean`,
  - `Rhat_energy` и `ESS_energy` по энерго-трассам (без scipy),
  - coverage dynamics: `n_unique_vs_step` (биннинг).
- Эти метрики используются как stop/auto-escalation критерии для больших N, чтобы отличать “guardrail прошёл” от
  “цепи реально стабилизировались”.

## [CHEM-VALIDATION-5] C15/C16 growth vs equilibrium (Mode A) + mixing diagnostics

**Решение.**

- Добавлены скрипты:
  - `analysis/chem/chem_validation_5_pentadecane.py` (C15),
  - `analysis/chem/chem_validation_5_hexadecane.py` (C16),
  - общая логика: `analysis/chem/chem_validation_5_common.py`.
- В отчётах разделяются:
  - `P_growth` (конструктивный рост; kernel-biased),
  - `P_eq` (fixed-N MCMC; multi-start `path/max_branch` + mixing diagnostics).
- Expected counts расширены до `N=15/16` в `analysis/chem/alkane_expected_counts.py`:
  - `N=15 -> 4347`, `N=16 -> 10359`.

## [EQ-DIST-1] Distributed EQ work units (GIMPS-style) for N=16 (resilient to time limits)

**Проблема.**

- Монолитный `eq_target_3_scan` пишет CSV/TXT только после завершения точки `steps_per_chain`; при внешних тайм-лимитах/убийстве процесса прогресс точки теряется.
- Решение: считать независимые цепи как отдельные work units и собирать их агрегацией (как GIMPS).

**Решение.**

- Введён distributed-контур (work units → submissions → aggregate):
  - генератор задач: `analysis/chem/eq_distributed_make_tasks.py`,
  - воркер одной задачи: `analysis/chem/eq_distributed_worker.py`,
  - агрегатор по точкам `steps_per_chain`: `analysis/chem/eq_distributed_aggregate.py`.
- Каждая задача соответствует одной независимой цепи (фиксированный seed, старт, chain_idx, budget) и может считаться на любой машине; результат — один JSON-файл submission.
- Для детерминизма seed фиксируется формулой:
  - `seed = seed_base + 10_000*steps_per_chain + 1_000_000*start_spec_idx + 101*chain_idx`.
- Агрегатор печатает `POINT`-блоки в машинном формате и пишет CSV/TXT по мере готовности точек; для неполных точек выводит `MISSING ...`, не падает.

**Структура каталогов (инвариант).**

- `results/dist/N16_modeA/tasks/` — `tasks.jsonl` и `task_*.json`
- `results/dist/N16_modeA/submissions/` — `task_XXXXXX__HOST__GITSHA.json`
- `results/dist/N16_modeA/aggregates/` — итоговые `eq_target_3_N16_modeA.csv|txt`

**DoD команды (N=16).**

- Мастер: генерация задач (6 точек × 2 старта × 3 цепи = 36 tasks):
  - `python -m analysis.chem.eq_distributed_make_tasks --N 16 --mode A --steps_grid 2000000 4000000 8000000 12000000 16000000 24000000 --chains 3 --start_specs path max_branch --thin 10 --burnin_frac 0.1 --seed_base 12345 --git_sha_expected <GIT_SHA> --out_dir results/dist/N16_modeA`
- Волонтёр/любая машина: выполнить одну задачу:
  - `python -m analysis.chem.eq_distributed_worker --task results/dist/N16_modeA/tasks/task_000001.json --out_dir results/dist/N16_modeA/submissions --energy_cache_path results/dist/N16_modeA/cache_<HOST>.pkl --git_sha <GIT_SHA>`
- Мастер: агрегация (можно запускать хоть каждые 10 минут; готовые точки сразу дадут `POINT`):
  - `python -m analysis.chem.eq_distributed_aggregate --tasks_jsonl results/dist/N16_modeA/tasks/tasks.jsonl --submissions_dir results/dist/N16_modeA/submissions --expected_unique 10359 --out_csv results/dist/N16_modeA/aggregates/eq_target_3_N16_modeA.csv --out_txt results/dist/N16_modeA/aggregates/eq_target_3_N16_modeA.txt`
- Верификация (как у GIMPS, опционально): для `steps in {8M,16M}` требовать 2 submissions с совпадающим `counter_hash`:
  - `python -m analysis.chem.eq_distributed_aggregate ... --require_double_for_verify --verify_steps 8000000 16000000`

## [KL-SPLIT-FLOOR-1] Statistical floor for KL_SPLIT_MAX at large K

**Контекст.**

- В EQ-TARGET-3 mixing-диагностике `KL_SPLIT_MAX` считается как симметричный KL между двумя половинами retained-window для каждой цепи (после burn-in и thinning), затем берётся максимум по цепям/стартам.

**Наблюдение.**

- При большом числе категорий `K` (у нас для C16: `K≈EXPECTED_UNIQUE_EQ=10359`) даже полностью стационарная цепь даёт ненулевой “статистический пол” для split-KL из-за конечной выборки.

**Оценка пола (инженерная).**

- При `n_half` независимых сэмплов в каждой половине порядок величины:
  - `E[KL_sym] ~ (K-1)/n_half`.
- Поэтому нельзя механически повышать `burnin_frac` (уменьшая `n_half`) и ожидать улучшения `KL_SPLIT_MAX`: можно упереться в floor даже при идеальном mixing.

**Практическое правило (N=16).**

- Для инженерного решения “pass/fail” для `KL_SPLIT_MAX` использовать gate:
  - `KL_split_max ≤ max(0.010, 1.1 * (K-1) / n_half)`,
  где:
  - `K` — число возможных категорий (для алканов: `EXPECTED_UNIQUE_EQ(N)`),
  - `n_half` — число retained-сэмплов в каждой половине окна.
- При типичных настройках EQ-DIST (`thin`, `burnin_frac`) можно оценивать:
  - `n_samples ≈ floor((steps_per_chain * (1 - burnin_frac)) / thin)`,
  - `n_half ≈ floor(n_samples / 2)`.
- Следствие: механическое увеличение `burnin_frac` уменьшает `n_half` и может сделать порог `0.010` статистически недостижимым даже при стационарной цепи.

**Пример (C16, Wave5).**

- `K=10359`, `steps_per_chain=32M`, `burnin_frac=0.30`, `thin=10`:
  - `n_samples ≈ 32M * 0.70 / 10 = 2.24M`, `n_half ≈ 1.12M`,
  - floor ≈ `(K-1)/n_half ≈ 10358/1.12M ≈ 0.00925`,
  - gate ≈ `max(0.010, 1.1*0.00925) ≈ 0.01017`.

## [INVARIANCE-BENCH-0] Overhead canonical tree relabeling (AHU)

**Решение.**

- Добавлен микробенч `analysis/bench/invariance_bench_0.py`:
  - N ∈ {6,10,20,40,80,160}, по 200 деревьев (случайные Prüfer).
  - метрики: `t_canonicalize`, `t_energy_fdm` (текущее, с каноном), `t_energy_fdm_raw` (без канона).

**Результат (mean overhead = t_canonicalize / t_energy_fdm).**

- Наблюдаемая доля overhead заметная и растёт с N (порядка ~21% на N=6 до ~55% на N=160).
- Это приемлемо как плата за строгую инвариантность в tree-only слое, но для больших N может потребовать оптимизаций
  (ускорение канонизации/кэширование/перенос части вычислений в numpy).

## [MH-KERNEL-3] C6 exact baseline (deg≤4) vs fixed-N MCMC vs growth

**Мотивация.**

- В CHEM-VALIDATION-1B наблюдается `KL(P_obs||P_pred) ~ 0.09–0.11` и слабая зависимость частот от режимов R/A/B/C → это типичный симптом того, что ростовой процесс не является корректным fixed-N sampler по финальным деревьям.

**Метод.**

- Перечислить все labeled деревья N=6 через Prüfer (1296) и отфильтровать алканы `deg≤4` (ожидаемо 1290, исключая star).
- Посчитать точное распределение `P_exact(state) ∝ exp(-λ E(state)/T)` тем же кодом энергии, что и в MCMC.
- Запустить leaf-rewire fixed-N MCMC с Hastings при ограничении `deg≤4` и сравнить `P_mcmc(topology)` с `P_exact(topology)`.
- Сравнить с `P_obs_growth(topology)` из `chem_validation_1b_hexane.csv` и с `P_pred(topology) ∝ g*exp(-λ E_ref/T)`.

**Интерпретация DoD.**

- Если `P_mcmc ≈ P_exact`, но `P_obs_growth` далеко, то рост — это generator/proposal, а не равновесный sampler по финальным деревьям, и частоты из CHEM-VALIDATION нужно оценивать через fixed-N MCMC/exact baseline.

**Результаты (C6, Mode A).**

- Использован `λ*=0.949` (из CORE-2 в `results/chem_validation_1b_hexane.txt`), `T=1`.
- Проверка алкан-ограничения: Prüfer N=6 даёт 1296 labeled деревьев, фильтр `deg≤4` оставляет 1290 (исключает только star `K1,5`).
- Получено:
  - `KL(P_mcmc || P_exact) = 1.642e-4` (MCMC совпадает с exact → Hastings + mixing корректны),
  - `KL(P_obs_growth || P_exact) = 9.602e-2` (ростовые частоты далеки от равновесия),
  - `KL(P_pred || P_exact) ≈ 0` (CORE-2 на деревьях корректен при label-invariant энергии).
- Вывод: CHEM-VALIDATION-1B частоты по финальным топологиям **нельзя** интерпретировать как equilibrium-частоты для фиксированного N (kernel bias роста доминирует); для “химической статистики” нужен fixed-N MCMC / exact baseline.

## [INVARIANCE-OPT-1] Avoid dense NxN relabeling in tree-canonical FDM path

**Мотивация.**

- INVARIANCE-BENCH-0 показал overhead до ~55–60% на N=160, что указывает на доминирование затрат плотной перестановки adjacency (O(N²)), аллокаций и `np.where`-сканов.

**Решение.**

- Ввести AHU-канонизацию как перестановку вершин и применять её к adjacency-list (O(N+E)), без построения плотной NxN матрицы.
- Перевести BFS/остовное дерево для FDM на adjacency-list, избегая `np.where` по строкам.
- Добавить INVARIANCE-BENCH-1 для сравнения legacy dense baseline vs оптимизированного пути.

**Результаты (INVARIANCE-BENCH-1).**

- На N=160 (200 деревьев): скорость относительно legacy dense выросла примерно до `~3.0x` (пример: `dense_mean≈0.00418s` → `opt_mean≈0.00137s`).
- Доля “канонизация vs raw” остаётся высокой: `canon_overhead_mean≈0.41` на N=160 (в среднем ~40% времени в `backend="fdm"` уходит на tree-canonical слой).
- Статус: таргет `≤15–20%` не достигнут; для дальнейшего снижения overhead нужны кэширование/инкрементальность (для ростовых траекторий) либо FDM-энергия, построенная из permutation-invariant инвариантов без канонизации.

## [Spectral Lab v1] Эксперимент SL-1 (resolution scan)

**Решение.**

- Добавлен скрипт `analysis/scan_spectral_lab_1d_resolution.py`, который
  для фиксированного гармонического потенциала сканирует размеры
  решётки N и сравнивает значения $F_{\mathrm{levels}}^{(\mathrm{spec})}$
  и $F_{\mathrm{levels}}^{(\mathrm{FDM})}$.
- Результаты пишутся в
  `results/spectral_lab_1d_resolution.txt` и
  `results/spectral_lab_1d_resolution.csv` с относительной ошибкой
  `rel_error`.

**Статус.**

- Для гармонического потенциала на отрезке \([-5,5]\) получено, что
  $F_{\mathrm{levels}}^{(\mathrm{spec})} \approx 1.25$ стабильно по N,
  тогда как $F_{\mathrm{levels}}^{(\mathrm{FDM})} \approx 0.30$, и
  относительная ошибка $\varepsilon(N) \approx 0.75$ практически не

## [CHEM-VALIDATION-0.2] Proposal vs Energy (C4 butane skeleton)

**Конфиг.**

- Базовый тест: тот же C4-скелет с ростом по `grow_molecule_christmas_tree` и параметрами
  `stop_at_n_atoms=4`, `allowed_symbols=["C"]` (чистое дерево из 4 атомов углерода).
- Скрипт: `analysis/chem/chem_validation_0_butane.py` с режимами:
  - Mode R (proposal-only): `grower_use_mh = False`, все энергетические couplings = 0.
  - Mode A (FDM-only): `grower_use_mh = True`, `coupling_complexity = 1.0`, `coupling_topo_3d = 0`.
  - Mode B (FDM + topo3d): `grower_use_mh = True`, `coupling_complexity = 1.0`, `coupling_topo_3d = 1.0`.
  - Mode C (FDM + topo3d + shape): как B, но с ненулевыми `coupling_shape_softness`, `coupling_shape_chi`
    в `ThermoConfig` (для C4-скелета shape не дифференцирует топологии, но режим фиксируется для будущих тестов).
- Запуск CHEM-VALIDATION-0.2:
  - пример: `python -m analysis.chem.chem_validation_0_butane --n_runs 50 --seeds 0 1 --modes R A B C`.
  - артефакты: `results/chem_validation_0_butane.csv`, `results/chem_validation_0_butane.txt`.

**Метрики.**

- Для каждого режима собираются:
  - частоты топологий `P(mode, topology)` и `log(P(iso)/P(n))`;
  - декомпозиция score по термам: `score_fdm`, `score_topo3d`, `score_shape`, `score_total`;
  - MH-статистика по топологиям: `mh_proposals`, `mh_accepted`, `mh_accept_rate`.
- Дополнительно в отчёте выводятся детерминированные эталоны:
  - `score(path)` и `score(star)` для двух фиксированных adjacency (цепочка C4 vs звезда C3–C),
  - `Δscore(n-iso) = score(path) - score(star)` по режимам R/A/B/C.

**Наблюдения (пробный прогон, n_runs=50, seeds=[0,1]).**

- Частоты топологий (P и log-отношения):
  - Mode R (proposal-only):
    - `P_R(isobutane) ≈ 0.94` (94/100), `P_R(n_butane) ≈ 0.06` (6/100).
    - `log(P_R(iso)/P_R(n)) ≈ 2.75` — чистый генератор сильно предпочитает изобутан.
  - Mode A (FDM-only, MH on):
    - `P_A(isobutane) ≈ 0.19`, `P_A(n_butane) ≈ 0.23`, `P_A(other) ≈ 0.58`.
    - `log(P_A(iso)/P_A(n)) ≈ -0.19` — после учёта энергии дерево чуть чаще стабилизируется в линейной топологии.
  - Mode B (FDM + topo3d):
    - `P_B(isobutane) ≈ 0.15`, `P_B(n_butane) ≈ 0.17`, `P_B(other) ≈ 0.68`.
    - `log(P_B(iso)/P_B(n)) ≈ -0.13` — topo3d почти не меняет баланс n vs iso по сравнению с A.
  - Mode C (FDM + topo3d + shape):
    - `P_C(isobutane) ≈ 0.08`, `P_C(n_butane) ≈ 0.19`, `P_C(other) ≈ 0.73`.
    - `log(P_C(iso)/P_C(n)) ≈ -0.87` — изобутан становится ещё менее вероятен; shape-режим усиливает bias против ветвления.

- MH-acceptance по топологиям (A/B/C):
  - В Mode A:
    - `acc_iso ≈ 0.69`, `acc_n ≈ 0.51`, `acc_other ≈ 0.04`.
  - В Mode B:
    - `acc_iso ≈ 0.73`, `acc_n ≈ 0.52`, `acc_other ≈ 0.02`.
  - В Mode C:
    - `acc_iso ≈ 0.67`, `acc_n ≈ 0.55`, `acc_other ≈ 0.04`.
  - Во всех режимах A/B/C деревья типа `other` почти всегда отклоняются (`accept_rate` на порядок ниже, чем у n/iso).

- Детерминированные эталоны:
  - Для всех режимов R/A/B/C:
    - `score(path) ≈ 3.59`, `score(star) ≈ 2.84`,
    - `Δscore(n-iso) ≈ +0.74` — линейный C4-скелет всегда получает более высокий `score_total`, чем изобутановая звезда.

**Выводы.**

- Генератор (Mode R) сам по себе сильно тянет в сторону изобутанового каркаса (`P_R(iso) >> P_R(n)`).
- Энергетический слой (FDM-комплексность + topo3d + shape) систематически считает линейный каркас более «выгодным»
  (`Δscore(n-iso) > 0` и `log(P_{A/B/C}(iso)/P_{A/B/C}(n)) < 0`), то есть толкает систему в сторону n-butane.
- Таким образом, CHEM-VALIDATION-0.2 разделяет:
  - **proposal-bias**: предпочитает изобутан (ветвление),
  - **energy-bias**: на уровне C4 детерминированно и в MH-acceptance отдаёт приоритет линейному n-butane.
- Для дальнейших этапов (CHEM-VALIDATION-1 на C5/C6) можно использовать Mode R как референс генератора,
  а Modes A/B/C — как тест того, насколько энергетический слой исправляет или усиливает proposal-bias по более богатому набору изомеров.
  меняется при N = 100, 200, 400, 800.
- Эксперимент SL-1 показывает, что текущая FDM-аппроксимация для
  $F_{\mathrm{levels}}$ даёт сильное систематическое смещение и не
  улучшает согласие при увеличении размерности решётки. Нужен новый
  дизайн FDM-прокси или переподбор функционала.

## [Spectral Lab v1] Калибровка линейного FDM-прокси для F_levels (SL-1b)

**Решение.**

- Введена параметрическая модель FDM-прокси
  $F_{\mathrm{levels}}^{(\mathrm{FDM, lin})} = a \cdot F_{\mathrm{naive}} + b$,
  где $F_{\mathrm{naive}}$ — прежняя FDM-аппроксимация по решётке.
- Скрипт `analysis/calibrate_f_levels_fdm_1d.py` подбирает коэффициенты
  $(a,b)$ по данным Spectral Lab v1 (гармонический потенциал,
  размеры решётки N = 100, 200, 400, 800) методом наименьших квадратов
  и записывает результат в `results/f_levels_fdm_1d_calibration.txt`.

**Статус.**

- Получены первые калиброванные коэффициенты
  $a \approx -6.33\times 10^{-3}$,
  $b \approx 1.2557$, уменьшающие систематическое смещение между
  $F_{\mathrm{levels}}^{(\mathrm{spec})}$ и
  $F_{\mathrm{levels}}^{(\mathrm{FDM})}$ в одномерной модели
  (RMSE порядка $6\times 10^{-4}$ на тех же N).
- Калибровка носит R\&D-характер и пока не интегрирована как дефолтный
  FDM-функционал для $F_{\mathrm{levels}}$; параметры хранятся в
  `results/f_levels_fdm_1d_calibration.txt` и могут использоваться
  в дальнейших экспериментах.

## [Spectral Lab v1] Эксперимент SL-2 (игрушечная спектральная жёсткость χ_spec^{(1D)})

**Решение.**

- Введён эксперимент SL-2 для семейства одномерных гармонических
  потенциалов $V(x;\lambda) = \tfrac{1}{2}\lambda x^2$.
- Для каждого значения $\lambda$ вычисляются различные кандидаты
  на спектральную ``жёсткость'' $\chi_{\mathrm{spec}}^{(1D)}(\lambda)$:
  энергия основного состояния, средняя энергия первых уровней,
  энергетический ``хвост'' и спектральный функционал
  $F_{\mathrm{levels}}^{(\mathrm{spec})}$.
- Эксперимент реализован скриптом
  `analysis/explore_toy_chi_spec_1d.py`; результаты записываются в
  `results/toy_chi_spec_1d_harmonic.txt` и
  `results/toy_chi_spec_1d_harmonic.csv`.

**Статус.**

- Получена первая табличка $\lambda \mapsto \chi_{\mathrm{spec}}^{(1D)}$
  для нескольких кандидатов. По результатам SL-2 в качестве игрушечного
  прототипа спектральной жёсткости в одномерной модели фиксирована
  величина
  $\chi_{\mathrm{spec}}^{(1D)}(\lambda) = \chi_{\mathrm{avg},10}(\lambda)$
  (средняя энергия первых десяти уровней), а спектральный функционал
  $F_{\mathrm{levels}}^{(\mathrm{spec})}$ при фиксированном весе
  трактуется как показатель спектральной мягкости
  $s_{\mathrm{spec}}^{(1D)}(\lambda)$, убывающий при росте жёсткости
  потенциала.

## [Geom–nuclear–FDM] Первое объединение слоёв (NB-1/NB-1_fix)

**Решение.**

- Введена комбинированная таблица `geom_nuclear_complexity_summary.csv`,
  объединяющая геометрический слой (роли, $D/A$, $\chi_{\mathrm{spec}}$),
  FDM-комплексность и ядерные индексы (`band_width`, `N_best`,
  `Nbest_over_Z`, `neutron_excess`).
- В `analysis/analyze_geom_nuclear_complexity.py` исправлён поиск
  ядерных данных: файлы `geom_isotope_bands.csv` и
  `geom_nuclear_map.csv` теперь ищутся сначала в `data/`, затем в
  корне репозитория.

## [TOPO-3D-2] 3D entanglement backend vs 2D crossing proxy

**Решение.**

- Реализован backend `fdm_entanglement` в `compute_complexity_features_v2`, который масштабирует FDM-сложность по формуле
  `total_entangled = total_fdm * (1 + coupling_topo_3d * topo_3d_beta * E_3d)`, где `E_3d` — entanglement_score по
  force-directed layout (seed=42).
- Добавлен экспериментальный скрипт `analysis/topo3d/experiment_entanglement_penalty.py`, который измеряет
  `crossing_density`, `E_3d`, `total_fdm`, `total_entangled` и `penalty_factor_3d` на смешанном датасете:
  учебные молекулы (H₂, H₂O, CH₄, NH₃, CO₂, C₂H₄, C₆H₆, NaCl, SiO₂, chain_C3, K4_C) и grower-графы (деревья и loopy).
- Численная проверка: `max|formula_residual| ≈ 2.4e-11`, `median = 0` (формула backend’а реализована корректно);
  `corr(penalty_factor_3d, E_3d) = 1.0` (как и ожидается для аффинной связи).
- На учебных молекулах 3D-штраф практически нулевой (`penalty_factor_3d ≈ 1.000–1.003`), т.е. базовая химия
  не «ломается» при включённом 3D-entanglement.
- На K4 и части loopy-grower-графов entanglement даёт заметный штраф (`E_3d ≈ 0.35–0.60`, `penalty_factor_3d ≈ 1.35–1.60`),
  что подтверждает чувствительность 3D-оси к действительно плотным/запутанным графам.
- Корреляция `E_3d` с 2D `crossing_density` близка к нулю (`≈ -0.045`), т.е. 3D-entanglement даёт независимую ось сложности,
  а не «улучшенную версию» 2D crossing-proxy.

**Инварианты.**

- При `coupling_topo_3d=0` или `topo_3d_beta=0` backend `fdm_entanglement` сводится к чистому FDM (`backend="fdm"`), так что
  зелёная зона по умолчанию не меняется.
- В R&D-режимах `topo_3d_beta` подбирается по хвосту распределения `E_3d`, чтобы p95-penalty на grower_loopy-графах оставался
  в фиксированном коридоре (например, ≤ 1.2–1.5).

## [PROCESS] Стандарт CONSOLE REPORT для Исполнителя

**Решение.**

- Для любого патча (конфиги/формулы/тесты) Исполнитель формирует CONSOLE REPORT, который включает:
  - команды и их вывод (`python -m pytest -q`, запуск ключевых скриптов);
  - краткий список изменённых файлов и суть правок (1–2 строки на файл);
  - явную проверку ключевого инварианта (например, `gain=0 → эффект=0`);
  - итоговый коммит (сообщение и ветка/хэш), без просьбы “посмотри в git”.
- Для экспериментальных скриптов (R&D) CONSOLE REPORT дополнительно содержит:
  - команду запуска с аргументами;
  - главные метрики (корреляции, квантили, топ-N по ключевому скору) прямо в тексте;
  - sanity-check формулы (например, `max|formula_residual|`, `median|residual|`);
  - пути к артефактам (`results/*.csv`, `results/*.txt`) и top-строки/таблицы.

**Инварианты.**

- Архитектор принимает решения по данным из CONSOLE REPORT; просмотр git-истории и diff’ов используется только для
  code-review и разборов расхождений, а не как основной источник метрик.

**Статус.**

- Для элементов H--Kr ядерные колонки `band_width`, `N_best`,
  `Nbest_over_Z` и `neutron_excess` заполнены для 38 из 44 элементов
  (остальные остаются пустыми и требуют расширения ядерного датасета).
- Скрипт `analysis/analyze_fdm_vs_nucleus.py` даёт следующие
  Spearman-корреляции между нормированной FDM-комплексностью
  `C_norm_fdm_mean` и ядерными параметрами (глобально по элементам с
  данными, n ≈ 29):
  - `C_norm_fdm_mean` vs `band_width`:      r ≈  0.33, p ≈ 0.08;
  - `C_norm_fdm_mean` vs `Nbest_over_Z`:    r ≈ -0.43, p ≈ 0.02;
  - `C_norm_fdm_mean` vs `neutron_excess`:  r ≈ -0.43, p ≈ 0.02.
- По ролям и группам связи менее стабильны из‑за малого числа точек;
  для hubs видна умеренная положительная связь с `band_width` (r ≈ 0.56,
  p ≈ 0.02 при n ≈ 16), остальные подвыборки в пределах шума.
- На данный момент geom–nuclear–FDM-связи рассматриваются как
  R\&D-уровень; для строгих утверждений требуется более полный и
  однородный ядерный датасет.

## [ARCH-1] Введение конфигов роста (GrowthConfig / growth_*.yaml)

**Решение.**

- Введён модуль `core/growth_config.py` с классом `GrowthConfig` и
  функцией `load_growth_config(...)`, которые позволяют задавать
  параметры роста (`GrowthParams`) через конфигурационные файлы
  (YAML/JSON) с секциями `growth`, `roles`, `physics`, `loops`.
- Добавлены конфиги `configs/growth_baseline_v5.yaml`,
  `configs/growth_cy1a.yaml`, `configs/growth_cy1b.yaml` для
  воспроизведения основных режимов (деревьевый v5.0, CY-1-A/B).
- R\&D-скрипт `analysis/analyze_cycle_stats.py` научен принимать
  параметр `--config=...`: при его наличии параметры роста берутся
  из конфига, при отсутствии используется прежний зашитый режим
  `GrowthParams(max_depth=4, max_atoms=25)`.

**Статус.**

- Конфигурационный путь пока используется только в R\&D-скриптах;
  базовая зелёная зона и продакшн-режимы сохраняют прежнее поведение
  при отсутствии `--config`. В рамках ARCH-1/step2 дополнительно
  переведён на GrowthConfig скрипт `analysis/calibrate_fdm_params.py`,
  а сеточный скрипт `analysis/scan_cycles_vs_params.py` использует
  конфиг как базовую точку для параметр-скана (при отсутствии конфига
  оба скрипта воспроизводят прежний деревьевый v5.0 baseline).

## [ARCH-1/step3] Конфигурируемые топологические штрафы FDM

**Решение.**

- Введён модуль `core/complexity_config.py` с классами
  `LoopyPenaltyConfig`, `CrossingPenaltyConfig`,
  `ComplexityPenaltyConfig` и функциями
  `get_current_penalties()`, `set_current_penalties(...)`,
  `load_complexity_penalties(...)`, задающими параметры
  топологических штрафов FDM (`alpha_cycle`, `alpha_load`,
  `beta_cross`, `max_cross_n`) через конфиг.
- Backends `"fdm_loopy"` и `"fdm_loopy_cross"` в
  `core/complexity.py` теперь читают коэффициенты штрафа из
  `ComplexityPenaltyConfig` (через `get_current_penalties()`),
  а не из жёстко зашитых констант.
- Добавлены конфиги `configs/complexity_penalties_default.yaml`
  (совпадает с прежними значениями 0.3/1.0/1.0/8) и
  `configs/complexity_penalties_soft.yaml` с более мягкими
  штрафами для R\&D.
- Скрипт `analysis/analyze_loopy_fdm_penalty.py` принимает
  параметр `--complexity-config=...`, загружает конфиг штрафов
  и применяет его через `set_current_penalties(...)` перед
  расчётом FDM-сложностей.

**Статус.**

- При отсутствии вызова `set_current_penalties(...)` поведение
  FDM-штрафов идентично прежним константам (0.3/1.0/1.0/8).
  Базовый FDM-режим v5.0, зелёная зона и продакшн-цепочка не
  зависят от новых конфигов.
 - R\&D-скрипты CY-1 могут запускаться с различными профилями
   топологических штрафов, указывая `--complexity-config`, без
   правки ядра.

## [TEMP-1] Температура среды и шум роста (QSG v6.x)

**Решение.**

- Параметр `temperature` в `GrowthParams` и `GrowthConfig` используется
  как управляющий параметр роста: эффективная вероятность продолжения
  ветви деформируется по температуре и сводится к базовому значению
  при `T = 1`.
- Добавлены конфигурации среды `configs/growth_env_cold.yaml` и
  `configs/growth_env_hot.yaml` для "холодных" и "горячих" режимов
  роста в loopy-режиме CY-1.
- Введён R\&D-скрипт `analysis/scan_temperature_effects.py`, который
  исследует зависимость средней длины деревьев, доли графов с циклами
  и нормированной FDM-сложности от температуры для стандартных семян
  (C, Si, O, S) относительно базового конфига роста.

**Статус.**

- При `T = 1` воспроизводится baseline-режим QSG v5.0 / CY-1 с той же
  статистикой роста (при запуске без конфигов или с growth_baseline_v5).
- При `T < 1` средний размер и устойчивость структур изменяются в
  ожидаемую сторону (для "холодных" режимов рост становится более
  стабильным), при `T > 1` --- менее стабильным; параметр температуры
  готов к дальнейшей физической интерпретации как "шум среды" и базис
  для будущей термодинамики роста.

## [CLEAN-1] Очистка корня репозитория и выравнивание data/results

**Решение.**

- Сделать `data/` каноническим местом для агрегированных CSV и убрать
  дубликаты из корня: файлы `element_indices_with_dblock.csv`,
  `geom_isotope_bands.csv`, `geom_nuclear_map.csv`,
  `segregated_stats.csv` теперь читаются и записываются только через
  пути внутри `data/` (соответствующие скрипты в `analysis/` обновлены).
- Вынести чисто R\&D-выгрузки `nuclear_spectrum_toy_levels.csv` и
  `tuned_nuclear_magic_scan.csv` из git, оставив их регенерацию за
  R\&D-скриптами (`core/nuclear_spectrum_fdm.py`,
  `analysis/tune_nuclear_spectrum.py`).
- Нормализовать импорты ядерного функционала на `core.nuclear_island`
  вместо неявного `from nuclear_island import ...` во всех скриптах.
- Обеспечить, чтобы новые отчёты и картинки сохранялись в `results/`,
  а не в корень (скорректированы пути сохранения для части plotting-
  скриптов и segregated-скана).

**Статус.**

- Корневая директория содержит только управляющие скрипты и конфиги
  (`run_pipeline.py`, экспорт/fit-*), без дублей таблиц и временных
  отчётов; данные и отчёты выровнены по каталогам `data/` и `results/`.
- Зеленая зона v5.0 и пайплайн `run_pipeline.py` сохраняют прежнее
  поведение; анализы по d-блоку, изотопным полосам и ядерным сканам
  продолжают работать с обновлёнными путями.

## [DATA-1] Экспорт и загрузка базы атомов из JSON

**Решение.**

- Добавлен скрипт `analysis/export_atoms_db.py`, который экспортирует
  текущую базу атомов (`AtomGraph`) из `core/geom_atoms._make_base_atoms`
  в файл `data/atoms_db_v1.json` (одноразовый снимок базы v1).
- В `core/geom_atoms.py` введена вспомогательная функция
  `_load_atoms_from_json(path)`, а конструктор базы атомов разделён на
  `_make_base_atoms_legacy()` (старый код, оставленный как fallback) и
  `_make_base_atoms()`, который сначала пытается загрузить
  `data/atoms_db_v1.json`, а при ошибке или отсутствии файла
  возвращается к legacy-реализации.

**Статус.**

- При наличии `data/atoms_db_v1.json` база атомов создаётся из JSON,
  при этом экспорт таблицы элементов (CLI `python -m core.geom_atoms
  --export-periodic`) воспроизводит прежний набор данных (с точностью
  до порядка строк, если в CSV нет явной сортировки).
- Legacy-реализация `_make_base_atoms_legacy` помечена как обратное
  совместимое fallback-хранилище; любые будущие расширения базы атомов
  (новые элементы, 4d/5d-слой и т.п.) должны вноситься через JSON.
## [CY-1] Loopy-growth и первые режимы с циклами (QSG v6.x)

**Решение.**

- Сохранить QSG v5.0 Grower в деревьевом режиме как базовый функционал
  для зелёной зоны (cyclomatic = 0 для стандартных семян и параметров).
- Расширить `GrowthParams` полями `allow_cycles`, `max_extra_bonds`,
  `p_extra_bond`, по умолчанию выключенными (`False`, `0`, `0.0`), чтобы
  сохранить деревьевый режим QSG v5.0.
- Ввести R\&D-слой loopy overlay: после роста дерева вызывается
  `_add_loopy_bonds(...)`, который добавляет до `max_extra_bonds`
  дополнительных рёбер между существующими узлами при `allow_cycles=True`.
- В R\&D-скрипте `analysis/scan_cycles_vs_params.py` включить loopy-режим
  и исследовать зависимость частоты циклов и размера графа от параметров
  роста (`GrowthParams`).
- На базе `results/cycle_param_scan.csv` выбрать два режима
  `CY-1-A` и `CY-1-B` (различные комбинации параметров роста), используемые
  как эталонные конфигурации с ненулевой долей циклов.
- Добавить скрипт `analysis/analyze_loopy_modes.py`, который измеряет
  `frac_cycles`, `max_cyclomatic`, средний размер `n_mean` и
  `cycle_load_mean` для режимов CY-1-A/B по стандартным семенам.
 - Ввести toy crossing-number для малых графов в модуле
   `core/crossing.py` (модель «вершины на окружности, рёбра — хорды»)
   и сравнить его с proxy-метриками (cyclomatic, cycle_load,
   fdm_loopy-пенальти) в R\&D-скрипте `analysis/analyze_crossing_proxy.py`.

**Статус.**

- Деревьевый режим QSG v5.0 подтверждён (`cyclomatic = 0` для базовых
  `GrowthParams`; см. `analysis/analyze_cycle_stats.py`).
- Loopy-режим активируется только через явный флаг `allow_cycles=True`
  и используется в R\&D (QSG v6.x); для удобства введён явный API
  `grow_molecule_loopy(...)`, который инкапсулирует канонический
  loopy-режим (близкий к CY-1-A).
- Для сетки параметров в `scan_cycles_vs_params` получены режимы с
  ненулевой долей циклов и различной циклонагрузкой; из них выбраны
  конфигурации CY-1-A/B для дальнейшего анализа (`analyze_loopy_modes.py`).
- В функции `compute_complexity_features_v2` добавлен backend
  `"fdm_loopy"`, который на основе базового FDM-компонента вводит
  мультипликативный штраф по цикломатическому числу и циклонагрузке;
  для деревьев (`cyclomatic = 0`) он совпадает с `"fdm"`.
- Скрипт `analysis/analyze_loopy_fdm_penalty.py` показал, что в режимах
  CY-1-A/B нормированная FDM-комплексность заметно растёт с ростом
  `cycle_load` (корреляция penalty vs cycle_load порядка 0.7), что
  делает циклы «дорогими» в FDM-законе при сохранении базового
  поведения для деревьев.
- Скрипт `analysis/analyze_crossing_proxy.py` на малых графах (n≤8)
  показал, что toy crossing-number `cr_circle(G)` в модели окружности
  даёт разумную, но не тривиальную связь с proxy:
  глобально `corr(crossing_density, cycle_load) ~ 0.64` и
  `corr(crossing_density, penalty_factor) ~ 0.63` по режимам CY-1-A/B.
  Это подтверждает, что выбранные proxy (cyclomatic, cycle_load,
  fdm_loopy) монотонно согласуются с crossing-cost и могут служить
  основанием для дальнейшей калибровки топологического штрафа.
 - В `compute_complexity_features_v2` добавлен отдельный R\&D-backend
   `"fdm_loopy_cross"`, вводящий дополнительный множитель
   $(1 + \beta_{\mathrm{cross}} \cdot \mathrm{crossing\_proxy})$, где
   для малых графов proxy строится по toy crossing-number, а для
   больших — по cycle\_load; базовые режимы `"fdm"` и `"fdm_loopy"`
   при этом не затронуты.
 - Скрипт `analysis/explore_loopy_cross_beta.py` использует уже
   посчитанные `crossing_proxy_CY1A/B.csv` и сканирует β\_cross
   (например, 0–3); результаты записаны в
   `results/loopy_cross_beta_scan.txt`. На текущих данных видно,
   что рост β\_cross повышает корреляцию между суммарным штрафом и
   плотностью пересечений при умеренном росте среднего penalty для
   hubs и terminators/bridges, но окончательный выбор β\_cross оставлен
   на дальнейшее R\&D.

## [CLEAN-2] Централизация семян роста и CLI-обёртки

**Решение.**

- Введён модуль `analysis/seeds.py` с базовым набором семян `GROWTH_SEEDS`, используемым во всех ростовых скриптах (CY-1, TEMP-1 и др.).
- Введён helper `analysis/growth_cli.py::make_growth_params_from_config_path`, унифицирующий обработку флага `--config` и базового режима v5.0.
- Скрипт `analysis/analyze_crossing_proxy.py` теперь берёт параметры режимов CY-1-A/B из конфигов `configs/growth_cy1a.yaml` / `configs/growth_cy1b.yaml`, а не из захардкоженных чисел.

**Статус.**

- При отсутствии `--config` все ростовые и калибровочные скрипты воспроизводят прежнее поведение v5.0.
- Режимы CY-1-A/B целиком определяются конфигами и не расходятся между анализ-скриптами.

## [NUC-1] Конфиги ядра и единый ядерный слой

**Решение.**
- Введён `core/nuclear_config.py` с `NuclearShellConfig` / `NuclearConfig`,
  глобальным `get_current_nuclear_config` / `set_current_nuclear_config`
  и контекстным менеджером `override_nuclear_config`.
- Параметры ядра (λ_shell, σ_p, σ_n, a_p) больше не прокидываются в
  `nuclear_functional` явными аргументами; они читаются только из `NuclearConfig`.
- Основные ядерные скрипты (`scan_isotope_band`, `analyze_isotope_bands`,
  `compare_nuclear_v02_to_experiment`, `geom_vs_ws_magic`,
  `geom_band_vs_ws_magic`, `scan_shell_bias`, `scan_valley`,
  `map_geom_to_valley`) принимают флаг `--nuclear-config` и
  используют `analysis.nuclear_cli.apply_nuclear_config_if_provided`.
- Добавлены профили ядра `configs/nuclear_shell_baseline.yaml`,
  `configs/nuclear_shell_soft.yaml`, `configs/nuclear_shell_strong.yaml`.

**Статус.**
- При отсутствии `--nuclear-config` все ядерные скрипты воспроизводят
  прежнее поведение baseline v5.0.
- Режим ядра теперь задаётся только конфигами; захардкоженные λ/σ/a
  убраны из analysis-скриптов.

## [CORE-CLI-1] Вынос ядерного CLI из ядра

**Решение.**
- Из `core/nuclear_island.py` удалён разбор `argv` и флаг `--magic`.
  Ядро больше не зависит от `argparse`.
- Введён чистый интерфейс `core.nuclear_island.set_magic_mode(mode: str)`,
  управляющий выбором neutron magic чисел (`legacy` vs `ws`).
- Добавлен CLI-скрипт `analysis/nuclear_magic_cli.py`, который объединяет
  выбор профиля ядра (`--nuclear-config`) и режима magic (`--magic`).
- Скрипт `analysis/tune_ws_magic.py` переведён на новый ядерный слой:
  использует `apply_nuclear_config_if_provided` и `set_magic_mode`
  вместо ручной настройки ядра.

**Статус.**
- Все изменения совместимы с существующим пайплайном; при запуске
  без новых флагов поведение остаётся в границах v5.0.
- Настройка ядра и magic-режима теперь сосредоточена в analysis-уровне
  и не протекает в core.

## [ANALYSIS-IO-1] Введение слоя IO для analysis-скриптов

**Дата:** 2025-12-12  
**Статус:** реализовано  

**Решение.**  
Введён модуль `analysis/io_utils.py` с единым слоем доступа к:
- каталогу `data/` (`PROJECT_ROOT/data`),
- каталогу `results/` (`PROJECT_ROOT/results`),
- вспомогательным функциям:
  - `read_data_csv(...)` с валидацией схемы,
  - `write_result_csv(...)`,
  - `write_text_result(...)`,
  - `data_path(...)`, `results_path(...)`.

Ключевые analysis-скрипты переведены на IO-слой:
- `analyze_complexity_correlations.py`,
- `analyze_geom_nuclear_complexity.py`,
- `analyze_fdm_vs_nucleus.py`,
- `analyze_d_block_plateau.py`,
- `analyze_plateaus_with_dblock.py`,
- `analyze_isotope_bands.py`,
- `analyze_crossing_proxy.py`,
- `analyze_cycle_stats.py`,
- `analyze_loopy_modes.py`,
- `calibrate_f_levels_fdm_1d.py`,
- `compare_f_levels_fdm_variants_1d.py`,
- `explore_loopy_cross_beta.py`,
- `explore_toy_chi_spec_1d.py`,
- `fit_tree_capacity.py`,
- `scan_cycles_vs_params.py`,
- `plot_DA_with_dblock.py`,
- `plot_geom_periodic_table.py`,
- `check_heavy_sp_pauling.py`,
- `extend_d_block_from_pauling.py`,
- `export_atoms_db.py`,
- `analyze_geom_table.py`.

**Инварианты.**
- Формат существующих CSV в `data/` не меняется.
- Путь по умолчанию всегда `PROJECT_ROOT/data/...` и `PROJECT_ROOT/results/...`.
- При падении по отсутствию файла или колонок поднимается `MissingDataError`
  с понятным сообщением, а не молчаливый NaN.

## [GROWTH-CLEAN-1] Единый reporting/RNG для growth-скриптов

**Решение.**
- Введён `analysis/growth/reporting.py` с хелпером `write_growth_txt(name, lines, header=None)`,
  завязанным на `analysis.io_utils.results_path` и `write_text_result`.
- Введён `analysis/growth/rng.py` с хелпером `make_rng(label, offset=0)` для детерминированного RNG.
- Growth-скрипты переведены на эти хелперы: `analyze_cycle_stats.py`,
  `analyze_loopy_modes.py`, `analyze_loopy_fdm_penalty.py`, `analyze_crossing_proxy.py`,
  `scan_cycles_vs_params.py`, `scan_temperature_effects.py`, `fit_tree_capacity.py`,
  `explore_loopy_cross_beta.py`.

**Инварианты.**
- Все growth-скрипты пишут текстовые отчёты только через `write_growth_txt`.
- RNG внутри growth-стендов создаётся только через `make_rng(label)`;
  ручные `np.random.default_rng(1234)` и глобальные `np.random.seed` не используются.

## [NUC-CLEAN-1] Магические числа и N-коридоры

**Решение.**
- Введён `core/nuclear_magic.py` с неизменяемыми наборами magic-чисел `LEGACY_MAGIC` и `WS_MAGIC`
  и единым интерфейсом `set_magic_mode(mode: str)` / `get_magic_numbers()`.
- `core/nuclear_island.py` переведён на `core.nuclear_magic`: `shell_penalty` читает magic-числа
  только через `get_magic_numbers()`, дефолтный режим задаётся `set_magic_mode("legacy")`.
- В `core/nuclear_bands.py` добавлен хелпер `make_default_corridor(Z, factor=1.7, min_width=1)`
  для стандартного диапазона `N` при поиске долины стабильности.
- Nuclear-скрипты переведены на новые хелперы: `scan_valley.py`, `map_geom_to_valley.py`,
  `geom_vs_ws_magic.py`, `geom_band_vs_ws_magic.py`, `nuclear_magic_cli.py`, `tune_ws_magic.py`.

**Инварианты.**
- Во всём проекте используется единый интерфейс `core.nuclear_magic.set_magic_mode` /
  `get_magic_numbers` для выбора набора magic-чисел (legacy vs WS).
- Любой поиск `N_best` по nuclear_functional использует helper `make_default_corridor(...)`
  вместо захардкоженных формул `N_min = Z`, `N_max ≈ factor * Z`.

## [TEST-2] Базовые smoke-тесты по доменам

**Решение.**
- Добавлены быстрые sanity-тесты, покрывающие ключевые домены:
  - `tests/test_geom_schema.py` — проверка схемы `data/element_indices_v4.csv`
    (обязательные колонки, отсутствие NaN в `Z/El/D_index/A_index`, базовая проверка `role`).
  - `tests/test_scan_isotope_band_smoke.py` — смоук по `analysis.nuclear.scan_isotope_band`
    на узком диапазоне Z, проверка структуры `data/geom_isotope_bands.csv`.
  - `tests/test_growth_scans_small.py` (`@pytest.mark.slow`) — малые прогоны
    `scan_cycles_vs_params` и `scan_temperature_effects` с проверкой наличия
    соответствующих CSV в `results/`.

**Инварианты.**
- Тяжёлые R&D-прогоны помечаются `@pytest.mark.slow` и не должны входить в
  дефолтный CI-набор.
- Быстрые smoke-тесты обязаны проверять существование ключевых артефактов
  (`geom_isotope_bands.csv`, `cycle_param_scan.csv`, `temperature_scan_growth.csv`)
  и базовые свойства данных.

## [DOC-1] Структура analysis/ и навигация

**Решение.**
- Добавлен документ `docs/analysis_structure.md` с обзором подпакетов
  `analysis/growth/`, `analysis/nuclear/`, `analysis/geom/`, `analysis/dblock/`,
  `analysis/spectral1d/` и перечислением ключевых скриптов и их артефактов.
- Зафиксирован набор утилит верхнего уровня: `analysis/io_utils.py`,
  `analysis/seeds.py`, `analysis/growth_cli.py`, `analysis/nuclear_cli.py`,
  `analysis/cli_common.py`.

**Инварианты.**
- Любой новый analysis-скрипт должен быть вписан в структуру `analysis/`
  (либо в существующий подпакет, либо с явным описанием в docs).
- Основные точки входа для ростовых и ядерных стендов документируются
  с указанием CLI-аргументов и выходных файлов.

## [NUC-TUNE-1] Нормализация тюнинга ядерных magic-чисел

**Решение.**
- Введён модуль `analysis/nuclear/tune_metrics.py` с общим набором целевых magic-чисел
  `TARGET_MAGIC = [2, 8, 20, 28, 50, 82]` и метрикой `cost_magic_l2(...)` для оценки совпадения
  toy magic с эталоном.
- Скрипт `analysis/nuclear/tune_ws_magic.py`:
  - переведён на общий `TARGET_MAGIC` и `cost_magic_l2(...)`;
  - принимает опциональный `--ws-config` (YAML) для задания сеток по параметрам WS-потенциала;
  - пишет полный результат скана в `results/ws_magic_tuning_results.csv` и краткий summary
    в `results/ws_magic_tuning_summary.txt`.
- Скрипт `analysis/nuclear/tune_nuclear_spectrum.py` использует тот же `TARGET_MAGIC` из
  `tune_metrics`, сохраняя собственную относительную метрику `magic_cost(...)`.

**Инварианты.**
- Эталонный набор magic-чисел для тюнинга хранится в одном месте (`tune_metrics.TARGET_MAGIC`)
  и используется в обоих tune-скриптах.
- Формат вывода WS-тюнинга стандартизован: CSV с параметрами и списком magic_N плюс текстовый
  summary с топ-строками по минимальному cost.

Дополнение (R&D-цепочка).
- Добавлен скрипт `analysis/nuclear/export_tuned_magic.py`, который читает
  `results/ws_magic_tuning_results.csv` и экспортирует лучший набор `magic_N`
  в YAML-конфиг `configs/nuclear_magic_ws_tuned.yaml` в формате `MagicSet` (Z/N).
- CLI `analysis/nuclear_magic_cli.py` получил режим `--magic=custom` и флаг
  `--magic-config=...`, позволяющие загрузить кастомный `MagicSet` из YAML и
  активировать его для всех ядерных расчётов (через `core.nuclear_magic.set_magic_numbers`).

## [THERMO-1] Введение термодинамического слоя (ThermoConfig)

**Дата:** 2025-12-13  
**Статус:** инфраструктура реализована  

**Решение.**
- Введён `core/thermo_config.py` с dataclass `ThermoConfig`, глобальным состоянием
  `get_current_thermo_config` / `set_current_thermo_config` и загрузкой из YAML/JSON
  через `load_thermo_config(path)`.
- Поддержаны два режима конфигурирования:
  1) отдельный thermo-конфиг (YAML/JSON);
  2) секция `thermo:` внутри общего экспериментального YAML (growth/nuclear).
- Добавлен модуль `analysis/thermo_cli.py` с helper-ами `apply_thermo_config_if_provided`
  и `apply_thermo_from_args`, реализующими порядок разрешения:
  defaults -> YAML -> CLI overrides.
- Для обратной совместимости загрузчик ростового конфига (`core/growth_config.py`)
  игнорирует top-level секцию `thermo:`, чтобы ростовые YAML могли содержать
  термодинамику без поломки парсинга.

**Инварианты.**
- На этапе THERMO-1 термодинамический слой является инфраструктурой: поведение v5/v6
  не меняется, couplings по умолчанию равны 0.0, а доменные модули ещё не используют
  `ThermoConfig` в расчётах.

## [THERMO-2A] delta_F как функция температуры (через coupling)

**Контекст.**
- В `analysis/nuclear/scan_isotope_band.py` ширина изотопной полосы задавалась
  магической константой/параметром `delta_F` (по умолчанию 5.0), которая не была
  связана с термодинамикой среды.

**Решение.**
- В `core/nuclear_config.py` добавлено поле `delta_F_base` (default 5.0) в
  `NuclearShellConfig` как источник legacy-поведения для ширины полосы.
- В `scan_isotope_band.py` введён расчёт эффективной ширины `delta_F_eff` через helper
  `compute_delta_F(args_deltaF, delta_F_base, coupling_delta_F, temperature)`:
  - если задан `--deltaF`, используется он (явный override);
  - иначе `delta_F_eff = delta_F_base` при `ThermoConfig.coupling_delta_F == 0.0`
    (legacy);
  - иначе `delta_F_eff = delta_F_base * ThermoConfig.temperature` при
    `ThermoConfig.coupling_delta_F > 0.0`.
- В CLI `scan_isotope_band` добавлены термо-аргументы (через `analysis.thermo_cli`)
  и опциональный override `--deltaF` для сохранения старого интерфейса.

**Инварианты.**
- При `coupling_delta_F == 0.0` поведение полностью совпадает с v5/v6 (delta_F_eff
  равен 5.0, если `delta_F_base` не переопределён в конфиге).
- Включение температурного coupling'а (`coupling_delta_F > 0.0` или явный thermo-конфиг)
  происходит только по явному запросу и не влияет на существующие скрипты/tests.

## [THERMO-2B] W_COMPLEXITY как функция температуры (энтропийный член)

**Решение.**
- В геометрическом функционале `F_geom` вес сложности `W_COMPLEXITY` переведён
  в управляемый термодинамикой параметр через helper
  `compute_W_complexity_eff(W_base, coupling, temperature)`:
  - при `coupling_complexity = 0.0` используется чистый `W_base` (legacy v5/v6);
  - при `coupling_complexity > 0.0` линейно смешиваются `W_base` и `W_base * T_thermo`.
- `AtomGraph.F_geom(...)` читает текущий `ThermoConfig` через
  `get_current_thermo_config()` и использует `W_eff` вместо фиксированного
  `W_COMPLEXITY` при добавлении `C_complex`.

**Инварианты.**
- При `ThermoConfig.coupling_complexity = 0.0` поведение идентично v5/v6, так как
  `W_eff == W_COMPLEXITY`.
- Изменение локализовано в `core/geom_atoms.py` и не требует правок R&D-скриптов,
  которые используют `F_geom`/`F_mol` как «чёрный ящик».

## [THERMO-2C] Spectral/period softness при включённом coupling_softness

**Решение.**
- Введён метод `AtomGraph.effective_softness(thermo)` с логикой:
  - при `coupling_softness = 0.0` возвращается softness из atoms_db (legacy);
  - при `coupling_softness > 0.0` подмешивается «физическая» мягкость,
    зависящая от периода элемента и грубой оценки жёсткости графа `epsilon_spec`.
- В grower (`core/grower.py`) seed-softness теперь вычисляется через
  `root_atom_data.effective_softness(get_current_thermo_config())`, а
  дальнейшее применение демпфера `p_continue *= (1 - seed_softness)` остаётся прежним.

**Инварианты.**
- При `ThermoConfig.coupling_softness = 0.0` рост деревьев воспроизводит v5/v6,
  так как seed-softness берётся напрямую из atoms_db.
- При включённом coupling_softness R&D-режим получает более интерпретируемую
  связь между спектральной/периодной структурой атома и глобальным демпфром
  ветвления, не ломая интерфейс grower'а.

## [DENSITY-1] Замена toy beta(Z) на физически мотивированную шкалу плотности (через coupling)

**Контекст.**
- В FDM-плотности (`estimate_atom_energy_fdm`) использовалась линейная toy-модель
  `beta(Z) = 0.5 + 0.05 * Z`, слабо связанная с физическим масштабом радиуса/плотности
  ядра/облака.

**Решение.**
- Введён модуль `core/density_models.py` с:
  - `beta_legacy(Z)` — старая toy-модель;
  - `beta_hydrogenic(Z)` / `beta_tf(Z)` — физически мотивированные шкалы
    (грубые hydrogenic и Thomas–Fermi);
  - `beta_effective(Z, coupling, model="tf")` — смешивание legacy и физической шкалы
    по coupling∈[0,1].
- В `ThermoConfig` добавлен параметр `coupling_density`, а в `analysis/thermo_cli`
  появился CLI-override `--coupling-density`.
- В `core/geom_atoms.py` функция `estimate_atom_energy_fdm` вместо захардкоженного
  `beta = 0.5 + 0.05 * Z` теперь использует:

      from core.thermo_config import get_current_thermo_config
      from core.density_models import beta_effective

      thermo = get_current_thermo_config()
      beta = beta_effective(Z, thermo.coupling_density, model="tf")

**Инварианты.**
- При `coupling_density = 0.0` поведение совпадает с v5/v6, так как
  `beta_effective(Z, 0) == beta_legacy(Z)`.
- При `coupling_density > 0.0` экспериментальные режимы получают более физичную
  зависимость масштаба beta(Z) от Z, оставаясь при этом в рамках Gaussian proxy
  для плотности (`exp(-beta*r^2)`).

## [DATA-CLEAN-ROOT-1] Канонические CSV не должны жить в корне репозитория

Решение:
- `periodic_after.csv` (и аналогичные канонические таблицы) перемещаются в `data/` и читаются через `analysis.io_utils.data_path(...)`.
- `geom_nuclear_map.csv` рассматривается как канонический датасет для последующего анализа:
  - либо хранится в `data/geom_nuclear_map.csv` (и может быть сгенерирован пайплайном при отсутствии),
  - либо явно объявляется “генерируемым артефактом” и тогда не коммитится, но всегда создаётся шагом пайплайна в `data/`.

Инвариант:
- Корень репозитория не содержит сиротских CSV/PNG/TXT-артефактов: данные → `data/`, результаты → `results/`.

## [DEV-TESTS-2] Pytest bootstrap: добавляем корень проекта в sys.path через tests/conftest.py

Контекст:
- Тесты используют абсолютные импорты `from core...` и `from analysis...`.
- В разных окружениях pytest это может падать как `ModuleNotFoundError`, если корень репозитория не попал в `sys.path`.

Решение:
- Добавлен `tests/conftest.py`, который перед импортом тестов гарантированно вставляет корень проекта в `sys.path`.
- Фикс локален к тестовому контуру и не влияет на “зелёную зону”: runtime-логика `core/` и `analysis/` не меняется.

## [DEPS-1] PyYAML как обязательная runtime-зависимость

Контекст:
- Ядерные модули ядра (`core.nuclear_*`) читают YAML-конфиги и импортируют `yaml`.
- Без PyYAML ломаются импорты `core.*` и запуск analysis-скриптов с `--nuclear-config`.

Решение:
- Добавлен `PyYAML>=6.0` в runtime-зависимости проекта (`requirements.txt`).
- `requirements-dev.txt` теперь включает runtime-зависимости через `-r requirements.txt` и добавляет dev-инструменты (`pytest`).

## [DEV-TESTS-3] Синхронизация тестов роста с текущим API и CLI

Контекст:
- `pytest` падал из-за рассинхронизации интерфейсов роста:
  - тесты ожидали у `Molecule` атрибуты `nodes` и `depth`,
  - growth-CLI вызывался с флагом `--num-runs`, которого не было в исходном скрипте.

Решение:
- В `core/geom_atoms.Molecule` добавлены совместимые read-only свойства `nodes` и `depth`, определяемые через `atoms` и граф связей `bonds` (BFS от корня), без изменения внутренней структуры данных.
- Тесты в `tests/test_growth_scans_small.py` упрощены до использования только `--config=...`; количество прогонов берётся из дефолта скриптов сканирования параметров.

## [DOC-FIX-1] Устранение merge-конфликтов в decision log

Решение:
- Удалены маркеры merge-конфликтов (<<<<<<< / ======= / >>>>>>>) из `docs/05_decision_log.md`.
- Содержимое конфликтующих секций слито вручную без потери смысловых пунктов.
- Это считается обязательным инвариантом: в docs не должно быть конфликт-маркеров.
## [SPECTRAL-DENSITY-1] WS radial spectrum -> 3D density hook for FDM

Дата: 2025-12-13

Решение:
- Добавлен альтернативный источник плотности `density_source = ws_radial`, который строит 3D плотность ρ(r) из радиального спектра WS (`core/nuclear_spectrum_ws.py`).
- В `estimate_atom_energy_fdm` реализовано смешивание gaussian proxy и WS-плотности через `ThermoConfig.coupling_density` (0..1) и `density_blend`.
- Для устойчивости масштаба WS-плотность масштабируется к аналитическому интегралу гауссианы при текущем `beta`.
- Инвариант: при `density_source=gaussian` (default) поведение идентично legacy; включение WS требует явного выбора источника и ненулевого coupling.

## [SPECTRAL-DENSITY-1 | Empirical sanity-check (Z=1,6,14,26)]

Запуск: `python -m analysis.spectral_density.compare_density_sources` (WS params: R_max=12, R_well=5, V0=40, N_grid=220, ell=0).

Наблюдения (coupling_density_shape=1, density_source=ws_radial):
- Z=1: ratio_E = E_ws/E_gauss ≈ 1.026
- Z=6: ratio_E ≈ 1.000
- Z=14: ratio_E ≈ 1.000
- Z=26: ratio_E ≈ 1.000
- mass_ratio ≈ 1 на кубе интегрирования (R=4): масштабирование WS-плотности к I_target воспроизводит массу без заметных потерь.

Вывод: мост WS ρ(r) → FDM по масштабу устойчив. Для «видимого» физического эффекта требуется shape-sensitive метрика (moments/grad-terms), т.к. текущий E_fdm почти инвариантен к форме при фиксированной массе.

## [SPECTRAL-GEOM-1 | Empirical sanity-check (B,C,N,O,Si,P,S)]

Запуск: `python -m analysis.geom.compare_port_geometry_sources` (ws_geom_* defaults: R_max=25, R_well=6, V0=45, N_grid=800, gap_scale=1.0, gap_ref=1.0).

Факт:
- В текущем профиле WS s-p разрыв даёт практически одинаковый gap для малого набора элементов; h определяется главным образом через port-логику (триг/пирамидал/бент) и порог 0.5.
- При исходных настройках совпадение legacy vs inferred примерно 6/7; после центрирования hybrid_strength на gap_ref=1.0 часть элементов (B) исправляется, но другие начинают дрейфовать (N,O,P,S).
- Картинка по C («tetra») визуально корректна: канонические портовые векторы дают тетраэдрическую конфигурацию.

Вывод: механизм портовых векторов работает и пригоден как вход в 3D-укладчик, но spectral-признак `ws_sp_gap` в текущем виде остаётся почти константой по Z. Для фаз SPECTRAL-GEOM-2/3 потребуется либо Z-зависимый WS-потенциал, либо дополнительная калибровка gap_ref/scale на расширенном наборе элементов.

## [SPECTRAL-WS-Z-1] Z-coupling для WS-радиальной задачи (инфраструктурно, без поломки legacy)

Решение:
- Введено управляемое поле `ThermoConfig.coupling_ws_Z` (0..1), а также параметры `ws_Z_ref` и `ws_Z_alpha`.
- При `coupling_ws_Z=0` сохраняется legacy-поведение: WS-спектр и плотность не зависят от Z (как в baseline).
- При `coupling_ws_Z>0` применяется радиусное масштабирование WS-параметров: `R_eff = (1-c)*R + c*(R*(Z_ref/Z)^alpha)` для `R_max` и `R_well`.
- Масштабирование используется как в `make_ws_rho3d_interpolator` (ρ_ws(r;Z)), так и в `ws_sp_gap(Z, ...)`, делая оба моста чувствительными к Z.
Инвариант зелёной зоны: по умолчанию `coupling_ws_Z=0`, поведение v5/v6 не меняется.

## [SPECTRAL-REPORT-CLI-1] ThermoConfig-driven reproducibility for SD/SG analysis scripts

Дата: 2025-12-13

Решение:
- R&D-скрипты `analysis.spectral_density.compare_density_sources` и
  `analysis.geom.compare_port_geometry_sources` приведены к единому CLI-паттерну ThermoConfig:
  `add_thermo_args()` + `apply_thermo_from_args()`.
- Скрипты печатают Effective ThermoConfig в консоль, чтобы режим (baseline vs wsZ-on) был
  однозначно воспроизводим из логов.
- В `analysis.geom.scan_ws_gap_vs_Z` добавлен параметр `--out`, чтобы разные режимы не перетирали
  один и тот же CSV.

Инвариант зелёной зоны:
- При отсутствии thermo-параметров поведение скриптов соответствует прежнему (legacy запуск).

## [SPECTRAL-DENSITY-WSZ-CAL-1] Box-aware scaling и диагностика формы WS-плотности

Дата: 2025-12-13

Решение:
- Введена box-aware нормировка WS-ветки: вместо бесконечного интеграла I_target используется
  интеграл гауссианы в том же кубе, что и FDM-интегратор: `I_box(R,beta) = (sqrt(pi/beta)*erf(sqrt(beta)*R))^3`.
- Для WS-плотности введён адаптивный радиус `R_eff = max(R_default, 1.2 * r_99)`, где `r_99` берётся из CDF
  по радиальной функции u(r) (URDF из WS-оператора); это уменьшает потери массы для лёгких Z.
- В модуле `core/spectral_density_ws` добавлены диагностические данные (`WSRadialDiagnostics`), включая
  `r_mean`, `r_rms` и `r_99`, которые могут использоваться в анализ-скриптах.
- В `estimate_atom_energy_fdm` WS-плотность масштабируется так, чтобы масса в кубе совпадала с I_box, после чего
  смешивается с гауссианой по прежним правилам (linear/log) на уровне ρ(r).

Инварианты:
- Gaussian-режим (`density_source="gaussian"`) остаётся неизменным при любых настройках ws Z-coupling.
- При `coupling_ws_Z=0` поведение WS-ветки соответствует legacy-профилю без Z-зависимости радиусного масштаба.

## [TOPO-PREFILTER-0/0.1] Prefilter для деревьев в fdm_entanglement + benchmark

Дата: 2025-12-15

Решение:
- В `ThermoConfig` добавлен флаг `topo3d_prefilter_tree: bool = False` и CLI-переключатель `--topo3d-prefilter-tree`.
- В backend `fdm_entanglement` (`compute_complexity_features_v2`) введён ранний выход: при `topo3d_prefilter_tree=True` и `cyclomatic==0` возвращается чистый FDM (`total_entangled == total_fdm`) без вызова 3D layout/entanglement.
- Добавлены тесты `tests/test_topo_prefilter_tree.py`, доказывающие:
  - для цепочки (tree) при включённом prefilter layout не вызывается и total совпадает с FDM;
  - для K4 prefilter не срабатывает, layout вызывается (через monkeypatch на `force_directed_layout_3d`).
- Добавлен R&D-скрипт `analysis/topo3d/benchmark_prefilter_tree.py`, который бенчит `fdm_entanglement` на 200 случайных деревьях (n=20) с prefilter off/on.

Цифры:
- По summary `results/topo3d_prefilter_bench_summary.txt`:
  - `n_graphs=200`, `n_tree(cyclomatic==0)=200`;
  - медианный speedup по времени `t_off/t_on` на деревьях ≈ **8.2×10²**, максимум ≈ **1.0×10³**.
- Это “идеальный” сценарий (100% деревья): на реальных наборах с циклами выигрыш будет ниже, но prefilter существенно уменьшает стоимость entanglement-backend на acyclic-графах.

Инварианты:
- При `topo3d_prefilter_tree=False` поведение backend `fdm_entanglement` полностью совпадает с предыдущим (зелёная зона не меняется).

## [FAST-SPECTRUM-1] Векторизация WS-FDM и выбор (base*, depth*)

Дата: 2025-12-15

Решение:
- В `core/shape_observables.get_shape_observables` FDM-ветка для `kurt_ws` векторизована:
  - вместо Python-цикла по FDM-точкам используется batch-вычисление: `samples = fdm.sample(...)`, `vals = moments_func(samples)`, `m_vec = R_max * vals.mean(axis=0)`.
- Добавлен свип-скрипт `analysis/ws/sweep_ws_fdm_depth.py`, который:
  - для Z = [1, 6, 8, 14, 26] считает baseline `kurt_ws` и `t_trapz` для trapz@4096;
  - для сетки (base ∈ {2,3,4}, depth ∈ {5..10}) считает FDM-значения `kurt_fdm`, времена `t_fdm` и speedup `t_trapz / t_fdm`;
  - записывает результаты в `results/ws_fdm_sweep.csv` и агрегаты в `results/ws_fdm_sweep_summary.txt`.
- По агрегатам выбирается пара `(ws_fdm_base*, ws_fdm_depth*)`, удовлетворя строгим критериям:
  - `max_abs_err(kurt_ws) = max_Z |kurt_fdm(Z) - kurt_trapz(Z)| ≤ 0.05`;
  - `median_speedup = median_Z (t_trapz(Z)/t_fdm(Z)) ≥ 1.8`.
- Выбор зафиксирован тестом `tests/test_ws_integrator_fdm_vs_trapz.py`, который проверяет DoD.1 для выбранных `(base*, depth*)`.

Цифры (из `results/ws_fdm_sweep_summary.txt` для выбранных параметров):
- Свип по (base, depth) показал:
  - `base=2, depth=5`: `max_abs_err_over_Z ≈ 0.00487`, `median_speedup_over_Z ≈ 2.63×`;
  - `base=2, depth=6`: `max_abs_err_over_Z ≈ 2.8e-4`, `median_speedup_over_Z ≈ 2.43×`;
  - `base=3, depth=5`: `max_abs_err_over_Z ≈ 1e-5`, `median_speedup_over_Z ≈ 2.00×`;
  - для более глубоких depth speedup падает (<1.8×), хотя ошибка становится ещё меньше.
- Выбрано `(ws_fdm_base*, ws_fdm_depth*) = (2, 5)` как компромисс:
  - `max_abs_err(kurt_ws) ≈ 0.00487` на Z=[1, 6, 8, 14, 26] (≈2% от типичного значения ~-0.227);
  - `median_speedup ≈ 2.63×` по времени `trapz/FDM`, что удовлетворяет целевому порогу ≥1.8×.
- Для всех Z в [1,6,8,14,26] ошибка по kurtosis на (2,5) почти одинакова (~0.00487), speedup варьируется в пределах ~2–5×.

Инварианты:
- Trapz-baseline (4096 точек) сохранён как эталон; FDM-параметры хранятся в ThermoConfig и могут быть переопределены для R&D.
- Публичный API `get_shape_observables` не менялся (возвращается тот же ShapeObs).

## [RESULTS-1] Политика хранения артефактов в `results/`

Дата: 2025-12-15

Решение:
- Введено правило: папка `results/` — это регенерируемые артефакты (CSV/TXT/PNG) из R&D-скриптов; в git по умолчанию храним только код, тесты и decision log.
- Для ключевых “golden baselines”, которые являются частью зафиксированных экспериментов, допускается хранение в git:
  - `results/ws_fdm_sweep.csv`
  - `results/ws_fdm_sweep_summary.txt`
  - `results/topo3d_prefilter_bench.csv`
  - `results/topo3d_prefilter_bench_summary.txt`
  - `results/topo3d_entanglement_penalty.csv`
  - `results/topo3d_entanglement_penalty_summary.txt`
- Все новые R&D-выгрузки (например, будущие `results/fast_spectrum_2_bench*.csv/txt`) игнорируются git’ом и должны регенерироваться при необходимости соответствующим скриптом.

Техническая реализация:
- В `.gitignore` добавлены строки:

  ```gitignore
  # R&D results: ignore by default
  /results/*.csv
  /results/*.txt

  # Keep golden baselines already tracked
  !results/ws_fdm_sweep.csv
  !results/ws_fdm_sweep_summary.txt
  !results/topo3d_prefilter_bench.csv
  !results/topo3d_prefilter_bench_summary.txt
  !results/topo3d_entanglement_penalty.csv
  !results/topo3d_entanglement_penalty_summary.txt
  ```

Инварианты:
- Уже закоммиченные golden-baseline файлы остаются tracked и используются как эталон для принятия решений.
- Новые экспериментальные выгрузки не добавляются в историю, чтобы не раздувать репозиторий и не смешивать код с данными.

## [FAST-SPECTRUM-2] Shape observables FDM single-pass и бенч по get_shape_observables

Дата: 2025-12-16

Решение:
- В `core/shape_observables.ShapeObs` добавлены новые WS-наблюдаемые: `effective_volume_ws`, `softness_integral_ws`, `density_overlap_ws`.
- В `get_shape_observables` параметр `beta = beta_effective(...)` вычисляется один раз и используется согласованно для trapz/FDM и гауссовой ссылки.
- FDM-ветка переведена в однопроходный режим: один набор FDM-сэмплов и многоканальный интегранд для моментов, softness и overlap без дополнительных циклов.
- В `thermo_fingerprint_for_shape` к кортежу ключа кэша добавлены `coupling_density`, `density_model`, `density_blend`, `density_Z_ref`, чтобы shape-кэш отражал выбор плотности.
- Добавлен тест согласия trapz/FDM по новым observables на Z ∈ {1, 6, 8, 14} с порогами `rel_err(volume) ≤ 10%`, `rel_err(softness) ≤ 5%`; для overlap проверяется только `>0` и конечность.
- Реализован бенчмарк `analysis/ws/fast_spectrum_2_bench.py`, измеряющий время полного вызова `get_shape_observables()` при trapz/FDM с (base=2, depth=5) и принудительным `cache_clear()` перед замером.

Цифры (из `results/fast_spectrum_2_bench_summary.txt`):
- Z = [1, 6, 8, 14, 26], `ws_fdm_base=2`, `ws_fdm_depth=5`.
- Для всех Z: `kurt_trapz ≈ -0.2271`, `kurt_fdm ≈ -0.2222`, `max_abs_err_kurt = 0.0049` (как в FAST-SPECTRUM-1).
- Времена (сек): 
  - Z=1: `t_trapz ≈ 6.51e-4`, `t_fdm ≈ 1.04e-4`, `speedup ≈ 6.27×`;
  - Z=6: `t_trapz ≈ 2.92e-4`, `t_fdm ≈ 1.36e-4`, `speedup ≈ 2.15×`;
  - Z=8: `t_trapz ≈ 2.92e-4`, `t_fdm ≈ 1.06e-4`, `speedup ≈ 2.76×`;
  - Z=14: `t_trapz ≈ 9.13e-4`, `t_fdm ≈ 3.29e-4`, `speedup ≈ 2.78×`;
  - Z=26: `t_trapz ≈ 6.33e-4`, `t_fdm ≈ 9.8e-5`, `speedup ≈ 6.49×`.
- Медиана по Z: `median_speedup ≈ 2.78×`, `median_abs_err_kurt = 0.0049`.

Инварианты:
- WS-trapz (4096 точек) остаётся эталонным baseline по форме; FDM-параметры по-прежнему хранятся в ThermoConfig и могут переопределяться в R&D.
- Файлы `results/fast_spectrum_2_bench*.csv/txt` игнорируются git согласно [RESULTS-1]; в истории остаются только код, тесты и данный decision log.

## [SMART-GROWTH-1] End-to-end проверка turbo-atom в MH-росте

Дата: 2025-12-16

Решение:
- Добавлен бенч-скрипт `analysis/growth/smart_growth_1_bench.py`, который сравнивает рост деревьев в двух режимах:
  - `ws_integrator="trapz"` (baseline),
  - `ws_integrator="fdm"` с параметрами `(ws_fdm_base=2, ws_fdm_depth=5)` (turbo-atom из FAST-SPECTRUM-2).
- В обоих режимах используются одинаковые термо-параметры, кроме интегратора:
  - `coupling_shape_softness=1.0`, `coupling_shape_chi=1.0` (shape-ветка принудительно включена),
  - `grower_use_mh=True`, `deltaG_backend="fdm_entanglement"`, `temperature_T=1.0`.
- Для каждого интегратора и Z из списка `[6, 8, 14, 26]` (C, O, Si, Fe) grower запускается с одними и теми же `GrowthParams`:
  - `max_depth=4`, `max_atoms=25`,
  - `n_trees_per_Z=50` деревьев на каждую пару (режим, Z).
- Для каждого дерева собираются:
  - время роста одной молекулы (по Python-таймеру вокруг `grow_molecule_christmas_tree`),
  - размер дерева (число атомов),
  - FDM-сложность дерева по `compute_complexity_features_v2(..., backend="fdm")`.
- По результатам по (режим, Z) считаются агрегаты:
  - `runtime_total_sec`, `runtime_per_tree_sec_mean/median`,
  - `size_mean/median`,
  - `complexity_fdm_mean`, `complexity_fdm_max`.
- Текстовый summary пишется в `results/smart_growth_1_bench.txt`, CSV — в `results/smart_growth_1_bench.csv` (оба игнорируются git согласно [RESULTS-1]).

Цифры (из `results/smart_growth_1_bench.txt`, n_trees_per_Z=50, max_depth=4, max_atoms=25):
- Z=6 (C): `trapz_mean≈6.29e-4s`, `fdm_mean≈7.75e-4s`, `speedup≈0.81×`, `size_mean_trapz≈2.06`, `size_mean_fdm≈2.12`, `Cfdm_mean_trapz≈0.99`, `Cfdm_mean_fdm≈0.82`.
- Z=8 (O): `trapz_mean≈2.30e-4s`, `fdm_mean≈2.78e-4s`, `speedup≈0.83×`, `size_mean_trapz≈1.20`, `size_mean_fdm≈1.40`, `Cfdm_mean_trapz≈0.34`, `Cfdm_mean_fdm≈0.49`.
- Z=14 (Si): `trapz_mean≈5.02e-4s`, `fdm_mean≈3.54e-4s`, `speedup≈1.42×`, `size_mean_trapz≈1.84`, `size_mean_fdm≈1.42`, `Cfdm_mean_trapz≈0.75`, `Cfdm_mean_fdm≈0.44`.
- Z=26 (Fe): `trapz_mean≈1.02e-3s`, `fdm_mean≈1.32e-3s`, `speedup≈0.77×`, `size_mean_trapz≈2.78`, `size_mean_fdm≈3.68`, `Cfdm_mean_trapz≈1.31`, `Cfdm_mean_fdm≈1.67`.

Интерпретация:
- На текущем малом стенде MH-роста (max_depth=4, max_atoms=25) выигрыш turbo-atom по времени на дерево не монотонен по Z:
  - FDM быстрее trapz только для Si (Z=14, speedup≈1.42×),
  - для C, O и Fe наблюдается лёгкое замедление (speedup<1×) при включённой shape-ветке и FH-энергетике.
- Средние размеры деревьев и FDM-комплексности отличаются между режимами, что отражает чувствительность MH-роста к форме потенциала и shape-наблюдаемым; при этом порядок величин размеров и сложностей сопоставим между trapz и FDM.
- Для более жёсткой end-to-end оценки требуется либо увеличить глубину/размер деревьев, либо оптимизировать ещё и остальные bottle-neck-и MH-роста (энергетика, proposal-логика); текущий SMART-GROWTH-1 фиксирует базовую картину и служит R&D-стендом для дальнейших SMART-GROWTH-* задач.

## [SMART-GROWTH-2] Профилирование MH-роста и честное сравнение trapz vs FDM

Дата: 2025-12-16

Решение:
- Добавлен профилировочный бенч `analysis/growth/smart_growth_2_bench.py`, который сравнивает режимы `ws_integrator="trapz"` и `ws_integrator="fdm"` при одинаковых сценариях роста, фиксированных seed’ах и включённой shape-ветке:
  - для обоих режимов ThermoConfig: `ws_fdm_base=2`, `ws_fdm_depth=5`, `coupling_shape_softness=1.0`, `coupling_shape_chi=1.0`, `grower_use_mh=True`, `deltaG_backend="fdm_entanglement"`, `temperature_T=1.0`;
  - список Z: `[6, 8, 14, 26]` (C, O, Si, Fe);
  - для каждого профиля заранее генерируется таблица seed’ов `seeds[(Z, tree_idx)]` одним базовым RNG (`make_rng("smart_growth_2_<profile>")`), и trapz/FDM используют одни и те же seed’ы.
- Введены два режима нагрузки:
  - SMALL: `max_depth=4`, `max_atoms=25`, `n_trees_per_Z=50`;
  - HEAVY: `max_depth=8`, `max_atoms=80`, `n_trees_per_Z=100`.
- Для каждого профиля/режима/элемента собираются:
  - общие тайминги роста (per-tree и total),
  - профили `t_shape_total` (через обёртку вокруг `get_shape_observables`) и `t_complexity_total` (обёртка вокруг `compute_complexity_features_v2`),
  - счётчики `n_shape_calls`, `shape_cache_hits/misses` по LRU-кэшу ShapeObs,
  - MH-статистика по деревьям (`mh_proposals`, `mh_accepted`, `mh_rejected`, `mh_accept_rate`).
- Результаты сохраняются в `results/smart_growth_2_bench.csv` и `results/smart_growth_2_bench.txt` (игнорируются git согласно [RESULTS-1]); summary включает speedup по total и по shape-компоненте, доли времени shape/complexity и accept rate.

Цифры (из `results/smart_growth_2_bench.txt`):
- SMALL (max_depth=4, max_atoms=25, n_trees_per_Z=50):
  - Z=6: `speedup_total≈0.87×`, `speedup_shape≈1.00×`, `shape_frac_trapz≈0.00`, `shape_frac_fdm≈0.00`, `complexity_frac_trapz≈3.84`, `complexity_frac_fdm≈1.32`, `mh_accept_rate_trapz≈0.41`, `mh_accept_rate_fdm≈0.41`, `shape_hits_trapz=320`, `shape_hits_fdm=320`, `shape_misses_trapz=40`, `shape_misses_fdm=40`.
  - Z=8: `speedup_total≈0.99×`, `speedup_shape≈1.00×`, `shape_frac_trapz≈0.00`, `shape_frac_fdm≈0.00`, `complexity_frac_trapz≈15.60`, `complexity_frac_fdm≈6.08`, `mh_accept_rate_trapz≈0.22`, `mh_accept_rate_fdm≈0.22`, shape-кэш: те же hits/misses.
  - Z=14: `speedup_total≈0.98×`, `speedup_shape≈1.00×`, `shape_frac_trapz≈0.00`, `shape_frac_fdm≈0.00`, `complexity_frac_trapz≈10.45`, `complexity_frac_fdm≈4.04`, `mh_accept_rate_trapz≈0.22`, `mh_accept_rate_fdm≈0.23`.
  - Z=26: `speedup_total≈0.93×`, `speedup_shape≈1.00×`, `shape_frac_trapz≈0.00`, `shape_frac_fdm≈0.00`, `complexity_frac_trapz≈3.07`, `complexity_frac_fdm≈1.12`, `mh_accept_rate_trapz≈0.46`, `mh_accept_rate_fdm≈0.46`.
- HEAVY (max_depth=8, max_atoms=80, n_trees_per_Z=100):
  - Z=6: `speedup_total≈1.04×`, `speedup_shape≈1.00×`, `shape_frac_trapz≈0.00`, `shape_frac_fdm≈0.00`, `complexity_frac_trapz≈1.88`, `complexity_frac_fdm≈2.02`, `mh_accept_rate_trapz≈0.32`, `mh_accept_rate_fdm≈0.32`, `shape_hits_trapz=520`, `shape_hits_fdm=520`, `shape_misses_trapz=40`, `shape_misses_fdm=40`.
  - Z=8: `speedup_total≈0.99×`, `speedup_shape≈1.00×`, `shape_frac_trapz≈0.00`, `shape_frac_fdm≈0.00`, `complexity_frac_trapz≈3.94`, `complexity_frac_fdm≈4.05`, `mh_accept_rate_trapz≈0.37`, `mh_accept_rate_fdm≈0.37`.
  - Z=14: `speedup_total≈1.01×`, `speedup_shape≈1.00×`, `shape_frac_trapz≈0.00`, `shape_frac_fdm≈0.00`, `complexity_frac_trapz≈4.08`, `complexity_frac_fdm≈4.27`, `mh_accept_rate_trapz≈0.27`, `mh_accept_rate_fdm≈0.27`.
  - Z=26: `speedup_total≈0.91×`, `speedup_shape≈1.00×`, `shape_frac_trapz≈0.00`, `shape_frac_fdm≈0.00`, `complexity_frac_trapz≈0.98`, `complexity_frac_fdm≈0.92`, `mh_accept_rate_trapz≈0.44`, `mh_accept_rate_fdm≈0.44`.

Интерпретация:
- RNG-детерминизм обеспечен: trapz и FDM видят идентичные сценарии роста (одинаковые seed’ы на каждый Z и индекс дерева), различия в метриках обусловлены только выбором интегратора и связанными изменениями в shape/энергетике.
- Для обоих профилей SMALL/HEAVY доля времени, уходящая на `get_shape_observables`, на текущем стенде близка к нулю (shape_frac≈0.0), а основное время уходит в блок сложности/энергетики (`compute_complexity_features_v2`), причём FDM-реализация там даёт сопоставимую или меньшую долю времени.
- Speedup_total колеблется вокруг 1× (от ≈0.87× до ≈1.04×) и не демонстрирует устойчивого выигрыша FDM над trapz на уровне полного MH-роста при заданных параметрах; turbo-atom остаётся локальной оптимизацией shape-слоя, а глобальный bottleneck лежит в complexity/энергетике и других частях пайплайна роста.

## [FAST-COMPLEXITY-1] Prefilter для fdm_entanglement и ускорение complexity-слоя

Дата: 2025-12-16

Решение:
- В `core.thermo_config.ThermoConfig` добавлен параметр `topo3d_prefilter_min_n` (по умолчанию 0), управляющий size-prefilter для 3D entanglement.
- В `core.complexity.compute_complexity_features_v2` (backend `fdm_entanglement`) реализован prefilter:
  - если `topo3d_prefilter_tree=True` и граф дерево (`cyclomatic==0`), возвращается чистый FDM (`fdm` backend) без вызова 3D layout/entanglement;
  - если `topo3d_prefilter_min_n>0` и `n < topo3d_prefilter_min_n`, также используется чистый FDM без 3D entanglement.
- Добавлен микробенч `analysis/complexity/fast_complexity_1_bench.py`, который:
  - генерирует случайные графы шести классов: `small|medium|large` × `tree|cyclic`;
  - сравнивает время `compute_complexity_features_v2(..., backend="fdm_entanglement")` в режимах `baseline` (без size-prefilter) и `optimized` (с `topo3d_prefilter_tree=True`, `topo3d_prefilter_min_n=N_MIN_PREFILTER`);
  - сохраняет результаты в `results/fast_complexity_1_bench.csv|txt` (игнорируются git согласно [RESULTS-1]).
- Для N_MIN_PREFILTER=10 микробенч даёт следующие speedup’ы (из `results/fast_complexity_1_bench.txt`, p50/p90/p99):
  - `small_tree`: `speedup_p50≈538×`, `speedup_p90≈607×`, `speedup_p99≈678×`;
  - `small_cyclic`: `speedup_p50≈413×`, `speedup_p90≈1.20×`, `speedup_p99≈1.40×`;
  - `medium_tree`: `speedup_p50≈1414×`, `speedup_p90≈1254×`, `speedup_p99≈973×`;
  - `medium_cyclic`: `speedup_p50≈1.24×`, `speedup_p90≈1.45×`, `speedup_p99≈1.07×`;
  - `large_tree`: `speedup_p50≈4300×`, `speedup_p90≈3898×`, `speedup_p99≈2989×`;
  - `large_cyclic`: `speedup_p50≈0.73×`, `speedup_p90≈0.79×`, `speedup_p99≈0.85×` (циклические крупные графы почти всегда проходят полный entanglement).
- В `analysis/growth/smart_growth_2_bench.py` запущен HEAVY-профиль с учётом нового prefilter (та же конфигурация, coupling_topo_3d активен в ThermoConfig для R&D-сценариев по entanglement):
  - HEAVY: `speedup_total` по Z ∈ {6,8,14,26} остаётся близким к 1× (≈0.91–1.01×), `speedup_shape≈1.00×`, `complexity_frac_trapz` и `complexity_frac_fdm` остаются сопоставимыми (в районе 1–4× total), что подтверждает: даже сильное ускорение entanglement-на части графов не становится доминирующим bottleneck’ом в общем MH-росте.

Интерпретация:
- Prefilter FAST-COMPLEXITY-1 радикально ускоряет `fdm_entanglement` на деревьях и малых графах (до 10³–10⁴× для tree-классов в микробенче) и даёт умеренный выигрыш на средних циклических графах, практически не ухудшая поведение на крупных циклических графах.
- В end-to-end MH-росте (SMART-GROWTH-2 HEAVY) вклад entanglement в общее время по-прежнему невелик, поэтому общие speedup’ы остаются около 1×; это подтверждает, что главный bottleneck complexity-слоя в текущей конфигурации связан не столько с 3D entanglement, сколько с остальной частью FDM-complexity/энергетики и, возможно, с MH/proposal-логикой.

## [FAST-COMPLEXITY-2] Paired microbench + HEAVY baseline/optimized + layout profiling

Дата: 2025-12-16

Решение:
- Добавлен микробенч `analysis/complexity/fast_complexity_2_bench.py`, который фиксирует набор случайных графов шести классов (`small|medium|large` × `tree|cyclic`) и запускает `compute_complexity_features_v2(..., backend="fdm_entanglement")` в режимах `baseline` (`topo3d_prefilter_tree=False`, `topo3d_prefilter_min_n=0`) и `optimized` (`topo3d_prefilter_tree=True`, `topo3d_prefilter_min_n=10`) на одном и том же (paired) наборе графов.
- По результатам `results/fast_complexity_2_bench.txt` (n_graphs_per_class=50, N_MIN_PREFILTER=10) получены следующие p50/p90/p99 speedup’ы:
  - `small_tree`: `speedup_p50≈289×`, `speedup_p90≈412×`, `speedup_p99≈171×`, paired `p50≈294×`, `p90≈430×`, `p99≈522×`;
  - `small_cyclic`: `speedup_p50≈335×`, `speedup_p90≈0.99×`, `speedup_p99≈1.07×`, paired `p50≈242×`, `p90≈352×`, `p99≈435×`;
  - `medium_tree`: `speedup_p50≈1400×`, `speedup_p90≈1931×`, `speedup_p99≈1283×`, paired `p50≈1415×`, `p90≈2021×`, `p99≈2343×`;
  - `medium_cyclic`: `speedup_p50≈1.02×`, `speedup_p90≈1.02×`, `speedup_p99≈1.03×`, paired `p50≈1.01×`, `p90≈1.04×`, `p99≈1.05×`;
  - `large_tree`: `speedup_p50≈4991×`, `speedup_p90≈5007×`, `speedup_p99≈3214×`, paired `p50≈4575×`, `p90≈5851×`, `p99≈6179×`;
  - `large_cyclic`: `speedup_p50≈1.01×`, `speedup_p90≈1.01×`, `speedup_p99≈1.00×`, paired `p50≈0.99×`, `p90≈1.02×`, `p99≈1.04×`.
- В `analysis/growth/smart_growth_2_bench.py` entanglement-конфиг расширен параметрами `coupling_topo_3d`, `topo_3d_beta`, `topo3d_prefilter_tree`, `topo3d_prefilter_min_n`, введён новый профилировщик `profile_shape_complexity_layout`, который monkeypatch’ит:
  - `core.shape_observables.get_shape_observables`,
  - `core.complexity.compute_complexity_features_v2`,
  - `core.energy_model.compute_complexity_features_v2`,
  - `core.layout_3d.force_directed_layout_3d`,
  и собирает `t_shape_total_sec`, `t_complexity_total_sec`, `t_layout_total_sec`, `n_layout_calls` отдельно для каждого Z.
- Для профиля `SMALL` сохранено прежнее сравнение `trapz vs fdm`; для профиля `HEAVY` реализованы два режима:
  - `baseline`: `topo3d_prefilter_tree=False`, `topo3d_prefilter_min_n=0`;
  - `optimized`: `topo3d_prefilter_tree=True`, `topo3d_prefilter_min_n=10`;
  при этом в обоих случаях используется `coupling_topo_3d=1.0`, `topo_3d_beta=1.0`, `deltaG_backend="fdm_entanglement"`.
- По HEAVY-профилю SMART-GROWTH-2 (из `results/smart_growth_2_bench.csv|txt`) для Z ∈ {6,8,14,26} получены следующие оценки:
  - Z=6: `speedup_total≈144×`, `speedup_complexity≈107×`, layout в optimized-режиме фактически устранён (`layout_eliminated`, `n_layout_calls_baseline=507`, `optimized=0`);
  - Z=8: `speedup_total≈176×`, `speedup_complexity≈134×`, `layout_eliminated`, `n_layout_calls_baseline=262`, `optimized=0`;
  - Z=14: `speedup_total≈108×`, `speedup_complexity≈83×`, `layout_eliminated`, `n_layout_calls_baseline=303`, `optimized=0`;
  - Z=26: `speedup_total≈426×`, `speedup_complexity≈307×`, `layout_eliminated`, `n_layout_calls_baseline=843`, `optimized=0`.

Интерпретация:
- Paired-микробенч FAST-COMPLEXITY-2 подтверждает, что size/tree-prefilter для backend`a `fdm_entanglement` даёт колоссальные выигрыши на деревьях (до 10³–10⁴×) и умеренные/нейтральные эффекты на циклических графах, при этом результаты устойчивы при переходе к paired-сравнению baseline/optimized на одном и том же наборе графов.
- В HEAVY-профиле SMART-GROWTH-2 при активации entanglement-слоя с layout-ом prefilter фактически полностью выключает 3D layout/entanglement в optimized-режиме для Z ∈ {6,8,14,26}, что приводит к очень большим speedup’ам по complexity/layout и заметному ускорению полного MH-роста. Это фиксирует FAST-COMPLEXITY-2 как «heavy» baseline/optimized стенд для entanglement-слоя с явным учётом layout-профилирования.

## [CHEMVAL-BURNIN-1] CHEM-VALIDATION-5: burnin_frac вынесен в конфиг/CLI (C15/C16)

Дата: 2025-12-23

Решение:
- В CHEM-VALIDATION-5 (Mode A) параметр burn-in больше не зашит как `0.1 * steps_per_chain`, а задаётся как `burnin_frac * steps_per_chain`.
- Для `analysis.chem.chem_validation_5_hexadecane` и `analysis.chem.chem_validation_5_pentadecane` добавлен CLI-параметр `--burnin_frac` и прокидывается в `EqCfg`.

Мотивация:
- Калибровка/пороговая устойчивость eqdist-Wave5 для N=16 достигалась при `burnin_frac=0.30`; фикс 0.10 повышает риск стоппера по `KL_SPLIT_MAX` на финальном прогоне при том же бюджете.

Результат:
- CHEM-VALIDATION-5 C16 (hexadecane) закрыт как финальный DoD-прогон в режиме equilibrium-first, Mode A, fixed budget:
  - `STEPS_TOTAL=320,000,000` (starts×chains = 2×5 ⇒ 32,000,000 per-chain),
  - `burnin_frac=0.30`,
  - строгая coverage: `n_unique_eq=10359/10359`,
  - пороги по KL/Rhat/ESS выполнены, `FAIL` отсутствует.

## [RESULTS-POLICY-1] Трекинг рабочих отчётов в `results/`

Дата: 2025-12-26

Решение:
- `results/` снова включён в git: текстовые/CSV/json-отчёты (acid, EQ-target, ростовые сводки) считаются “истиной эксперимента” и остаются в истории. Для устойчивых эталонов заведена папка `results/golden/…` (например, `results/golden/hetero/*`).
- Игнорируются только тяжёлые/машино-специфичные артефакты (`*.pkl`, cache-файлы, submissions/), чтобы не плодить мусор и зависимость от хоста.

Мотивация:
- Архитекторы и продуктовая команда читают отчёты прямо из репозитория; когда `results/` был в ignore, отчёты терялись и нельзя было воспроизвести состояние эксперимента.
- Exact-baseline и unit/property тесты важны, но человекочитаемый отчёт по прогону (пороговые метрики, coverage, telemetry) нужен в истории — иначе через неделю непонятно, что именно было запуском “PASS”.

## [HETERO-1A P0.3.1] Functional Separation Gate & score tables

Дата: 2025-12-26

Решение:
- Для C2H6O/C3H8O/C2H7N введён жёсткий классификатор функциональных классов (alcohol vs ether, primary vs secondary amine) на основе `deg(O)`/`deg(N)` и валентностей.
- `analysis.chem.hetero_score_utils` и `hetero_validation_suite` формируют две таблицы:
  - per-state `states_<formula>.csv` (energy, fingerprint, класс, P_exact/P_emp),
  - score `hetero_validation_suite.csv` с корректными метриками (взвешенные mean/std, delta_abs, effect_size только при `n>=2`, ROC AUC + флаг `*_auc_is_trivial`, collision_rates).
- Golden отчёты и state-таблицы размещаются в `results/golden/hetero/*`.

DoD:
- coverage=1.0, `KL < 0.02` на малых формулах,
- energy/fingerprint различают классы (`delta_abs > eps`, effect_size ≥ 1 при `n>=2`, collision_rate=0).

## [HETERO-1A P0.4] Расширенный acid-suite + calibration loop

Дата: 2025-12-27

Решение:
- SCORE v2 “обезвреживает” trivial-кейсы: при `n_a<2`/`n_b<2` или `pooled_std<1e-12` метрики жёстко фиксируются (`auc_raw=auc_best=0.5`, `effect_size=NaN`). В `hetero_validation_suite.csv` добавлены явные поля (`E_is_trivial`, `E_auc_raw/best`, `fp_best_*`, `n_other`, `pair_is_exhaustive`, `energy_collision_eps`), а тесты на искусственных данных ловят regression’ы (singleton, zero-variance, perfect-separation, tertiary-count).
- Acid-suite расширен до C4H10O/C4H11N и fallback C5H12O/C5H13N (tree-only). Для O- и N-кейсов получены нетривиальные пары (`fp_best_is_trivial=False`), а tertiary амин учитывается через `n_other`/`pair_is_exhaustive`.
- CLI `analysis.chem.hetero_calibration_loop` запускает сетку по (`beta`, `alpha_H`), вызывает suite в per-trial подкаталогах, фильтрует строки `fp_best_is_trivial=False`, считает objective `sum(fp_best_auc_best - λ_other·other_frac - λ_coll·energy_collision_rate)` и фиксирует `calib_trials.csv`, `calib_best.json`, `calib_best_suite.csv`. Fail-fast: если нет ≥2 нетривиальных строк или coverage/KL за пределами DoD, trial считается `fail`.
- Golden отчёты обновлены: `results/golden/hetero/states/states_<formula>.csv`, `acid_*.txt/csv`, `hetero_validation_suite.csv` (6 формул). Калибровочный запуск (`beta∈{0.9,1.0}`, `alpha_H∈{0.45,0.50}`, формулы C4/C5) дал `best score≈3.43` при `(beta=1.0, alpha_H=0.45)`; артефакты — `results/hetero_calibration/*`.

DoD:
- suite содержит ≥1 O-кейс и ≥1 N-кейс с `fp_best_is_trivial=False`;
- `calib_trials.csv` отражает все попытки, `calib_best.json` и `calib_best_suite.csv` присутствуют/обновляются; при отсутствии PASS — явный `status=fail`.

## [HETERO-1A P0.5.3] GO: Reproducible θ_best + collision logs + FP policy

Дата: 2025-12-27

Решение:
- Зафиксирован рабочий θ_best для HETERO-1A: `beta=1.0`, `alpha_H=0.45` (из `results/hetero_calibration/calib_best.json`).
- Политика FP для suite и calibration: `fp_policy_used=exclude_energy_like` (energy-like признаки исключаются по умолчанию; для отладки есть allow-режим).
- Коллизии энергии: `collision_eps=1e-9`, `energy_key_scheme=absdiff`.
- Артефакты коллизий пишутся внутри `out_dir/collisions/<formula>_cross_collisions.csv` и создаются всегда (header-only при `coll_cross_pairs=0`), чтобы исключить staled-файлы.
- Наблюдение: на θ_default cross-коллизии на O-формах существуют, но соответствуют дегенерациям энергии (`max_abs_delta_cross ~ 1e-15`/`0.0`); на θ_best `coll_cross_pairs=0`.

DoD:
- Suite воспроизводимо запускается на θ из JSON (θ_default vs θ_best отличается только θ), и cross-коллизии исчезают на θ_best.
- Calibration loop применяет trial-level гейты (any-formula-fails ⇒ trial fails), и trial’ы с `coll_cross_pairs>0` отбрасываются.

## [HETERO-1A P0.7] Release STOP criteria for HETERO-1B

Дата: 2025-12-27

Решение:
- Введён релизный STOP → HETERO-1B operator redesign при любом из условий (на train или holdout):
  - `coll_cross_pairs_strict > 0` на выбранном θ_best,
  - `coverage_unique_eq < 1.0`,
  - `kl_exact_emp > 0.05`,
  - одновременно `fp_best_auc_best < 0.85` и `fp_best_auc_gap < 0.02` (FP не даёт разделения).

## [HETERO-1A P0.8.1] Harden neg-controls policy (stable gating)

Дата: 2025-12-28

Решение:
- Neg-controls больше не одношотные: для каждой формулы AUC по neg-controls считается как квантиль `q=0.95` по `K=50` повторениям (для `perm_labels` и `rand_fp` отдельно), чтобы убрать дискретную “магическую границу” PASS/FAIL на малых N.
- В suite (`analysis.chem.hetero_validation_suite`) добавлены параметры `--neg_control_reps`, `--neg_control_quantile` и расширенные поля в CSV: `fp_neg_auc_*_mean`, `fp_neg_auc_*_q`, а `fp_neg_auc_best_*` трактуется как выбранный квантиль (gate-метрика).
- В calibration (`analysis.chem.hetero_calibration_loop`) при `--use_neg_controls` в `calib_trials.csv` логируются `neg_control_seed`, `neg_control_reps`, `neg_control_quantile`, `neg_auc_max_gate`, `neg_control_sources`, а `max_neg_auc_any` заполняется и на FAIL-trial.

DoD:
- Любой trial, где `max_neg_auc_any > neg_auc_max_gate`, обязан получать `gate_failed_any=True` и `gate_reason_any=neg_control`.

## [HETERO] Product hypothesis (рекомендации, не приказ)

Дата: 2026-01-02

Решение:
- Концепт “HETERO Screening Engine / MVP-1 Decoy & Dataset Audit Engine” фиксируется как **рекомендованный план развития** (Product Hypothesis), а не как обязательная “политика репозитория”.
- Цель гипотезы: ultra-fast 2D screening на базе текущих метрик/гейтов + контроль отрицательных проверок (decoys/neg-controls) + аудит датасета/коллизий; это не заявляется как замена QM/docking.

Почему это выбранный путь:
- В контуре HETERO есть воспроизводимый “suite → calibration → gates” пайплайн и тестовая база, куда можно встраивать проверки устойчивости (tie-aware AUC, neg-controls, seed-robustness).
- На малых N отрицательные контроли требуют аккуратной квантильной/комбинаторной калибровки, иначе “магические пороги” теряют смысл.

Верификация:
- Любое продвижение гипотезы делается только через: эксперимент в `analysis/` → артефакт в `results/` → тест в `tests/` → запись в `docs/90_lineage.md`.

## [HETERO-2 P0] Cycles are required for WOW hard negatives

Дата: 2026-01-03

Обоснование:
- В текущем контуре большинство генерируемых молекул — деревья (cyclomatic=0).
- Деревья легко распознаются простыми признаками, поэтому негативы слабые.
- Для продуктового WOW нужны hard negatives с кольцами (cycle space), иначе отчет не дает реальной проверки.

Решение:
- Запустить параллельный трек HETERO-2 (cycles + spectral fingerprint), не ломая HETERO-1A.

## [HETERO2-PHI-EXP-0] Ray Harmony — эксперимент (пока не в продукт)

Дата: 2026-01-04

Обоснование:
- PHI (Ray Harmony) потенциально даёт дополнительный маркер “hardness”, но не доказано на данных.
- Риск: внедрять без проверки = “касса-казино”, ломает доверие.

Решение:
- Реализовать PHI в экспериментальном модуле (analysis/experimental), не трогая core/report.
- Критерий принятия: на ≥3 кольцевых молекулах устойчиво `median(phi_decoys) > phi_original` при разных seed; иначе — остаётся R&D.

### PHI-EXP-1 (bench 5 ring molecules, k≈30)

Результаты (scale=300, seed=0):
- aspirin: phi_orig=14898.59, phi_decoys_median=14827.60 (count=2, delta=-71)
- acetaminophen: decoys=0 (sanitization/rewire не прошли)
- ibuprofen: phi_orig=20301.70, phi_decoys_median=18824.56 (count=9, delta=-1477)
- naproxen: phi_orig=16970.21, phi_decoys_median=18224.90 (count=3, delta=+1254)
- salicylic_acid: decoys=0

Вывод: сигнала пока нет (медианы часто ниже или выборка отсутствует). PHI остаётся экспериментом; перед включением в продукт требуется стабильный сдвиг на ≥3 молекулах и при других seed.
### 2026-01-05 - HETERO-2 Sprint-4b: Guardrails before Scale

- **�������:** ����� ����� preflight `hetero2.guardrails.preflight_smiles` ����� ������� ������ pipeline v2.
- **�������� SKIP:** invalid SMILES, `n_heavy_atoms>200`, `n_components>1`; ������� ����������� ��� `skip.reason` � � `warnings` (`skip:invalid_smiles`, `skip:too_large:...`, `skip:disconnected:...`).
- **����������:** `run_pipeline_v2` ���������� �������� payload c neg-controls `verdict=SKIP` ��� ����������; `hetero2-batch` ������ ����� `summary.csv` �� ������� �� ������ ���� (`status=OK/SKIP/ERROR`, `reason` ��������������, `report_path` ���� ��� SKIP/ERROR`).
### 2026-01-05 - HETERO-2 Sprint-5: seed_strategy and stable hash

- **�������:** ��� per_row seed ����� ���������� ��� id: `crc32(id.encode)` (Python `hash()` �������������� � �� ������� ��� ������������); `seed_used = seed XOR stable_hash(id)`.
- **CLI:** `hetero2-batch --seed_strategy {global,per_row}` (default=global), `--guardrails_max_atoms`, `--guardrails_require_connected`.
- **��������:** ��� `scores_input` default `score_mode=mock`; `external_scores` ��� ����� -> SKIP `missing_scores_input`.
