# Структура analysis/

## analysis/growth/
- `scan_cycles_vs_params.py` — сканы по параметрам роста, пишет `results/cycle_param_scan.{csv,txt}`.
- `scan_temperature_effects.py` — температурные сканы, пишет `results/temperature_scan_growth.{csv,txt}`.
- `analyze_cycle_stats.py` — агрегированные статистики по циклам для базовых режимов.
- `analyze_loopy_modes.py` — режимы loopy-growth (CY1A/B), `results/loopy_modes.txt`.
- `analyze_loopy_fdm_penalty.py` — сравнение FDM / FDM_loopy, CSV + summary TXT.
- `analyze_crossing_proxy.py` — proxy crossing-number vs цикломатика/loopy-нагрузка.
- `explore_loopy_cross_beta.py` — сканы по β_cross для штрафа crossing-aware.

## analysis/nuclear/
- `scan_isotope_band.py` — генерация `data/geom_isotope_bands.csv` по geom-таблице.
- `analyze_isotope_bands.py` — анализ полос, роли, living hubs/donors по `geom_isotope_bands.csv`.
- `map_geom_to_valley.py` — сопоставление геометрии с ядерной долиной стабильности.
- `scan_valley.py` — линия valley of stability по nuclear_functional.
- `geom_vs_ws_magic.py` — сравнение geom N_best с текущими magic-числами.
- `geom_band_vs_ws_magic.py` — попадания magic N в полосы по `delta_F`.

## analysis/geom/
- `analyze_geom_table.py` — базовый анализ таблицы элементов, `data/element_indices_v4.csv`.
- `analyze_complexity_correlations.py` — корреляции сложности с geom-индексами.
- `analyze_geom_nuclear_complexity.py` — объединение geom, FDM и ядерных индексов.

## analysis/dblock/
- `extend_d_block_from_pauling.py` — расширение d-блока по Паулингу.
- `analyze_d_block_plateau.py` — анализ плато d-блока.
- `analyze_plateaus_with_dblock.py` — совместный анализ плато с учётом расширенного d-блока.

## analysis/spectral1d/
- `calibrate_f_levels_fdm_1d.py` — калибровка 1D FDM-уровней.
- `compare_f_levels_fdm_variants_1d.py` — сравнение вариантов FDM для 1D-потенциалов.
- `explore_toy_chi_spec_1d.py` — игрушечные спектры и χ-like характеристики.

## Утилиты верхнего уровня
- `analysis/io_utils.py` — доступ к `data/` и `results/`, CSV/TXT-IO.
- `analysis/seeds.py` — наборы базовых seed-элементов для роста.
- `analysis/growth_cli.py` — общий конструктор GrowthParams из YAML-конфигов.
- `analysis/nuclear_cli.py` — применение ядерных конфигов (`--nuclear-config`).
- `analysis/cli_common.py` — общие CLI-хелперы и логирование.
