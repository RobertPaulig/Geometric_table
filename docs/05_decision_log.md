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
