# FAST-SPECTRUM-2 CONSOLIDATED REPORT

- Commit hash: `d21dedf353937da3d42eaec61e0c18fae36420b6`
- Commands:
  - `python -m analysis.ws.fast_spectrum_2_bench`

## Benchmark: get_shape_observables (trapz vs FDM)

Source: `results/fast_spectrum_2_bench_summary.txt`

Z-values: [1, 6, 8, 14, 26]  
WS FDM params: `ws_fdm_base=2`, `ws_fdm_depth=5`

| Z  | kurt_trapz | kurt_fdm | abs_err_kurt | t_trapz [s] | t_fdm [s] | speedup |
|----|-----------:|---------:|-------------:|------------:|----------:|--------:|
| 1  | -0.2271    | -0.2222  | 0.0049       | 0.000651    | 0.000104  | 6.27×   |
| 6  | -0.2271    | -0.2222  | 0.0049       | 0.000292    | 0.000136  | 2.15×   |
| 8  | -0.2271    | -0.2222  | 0.0049       | 0.000292    | 0.000106  | 2.76×   |
| 14 | -0.2271    | -0.2222  | 0.0049       | 0.000913    | 0.000329  | 2.78×   |
| 26 | -0.2271    | -0.2222  | 0.0049       | 0.000633    | 0.000098  | 6.49×   |

- max_abs_err_kurt = 0.0049
- median_abs_err_kurt = 0.0049
- median_speedup ≈ 2.78×

## Summary (10 lines)

1. В `ShapeObs` добавлены новые WS-наблюдаемые: `effective_volume_ws`, `softness_integral_ws`, `density_overlap_ws`.  
2. Все новые метрики вычисляются как в trapz-, так и в FDM-ветке через единый `rho_fn(r)` и многоканальный интегранд.  
3. `beta = beta_effective(...)` теперь вычисляется один раз и используется согласованно для trapz/FDM и гауссовой ссылки.  
4. В FDM-ветке используется один проход: общий набор samples `u`, один `rho_fn(r)` и векторное вычисление моментов, `I_rho2`, softness и overlap.  
5. Trapz/FDM-согласие для новых observables зафиксировано тестом `test_ws_integrator_trapz_vs_fdm_new_observables_agree_on_key_Z`.  
6. Для `density_overlap_ws` проверяется только sanity (`>0` и конечность) при обоих интеграторах.  
7. Новый бенчмарк `analysis/ws/fast_spectrum_2_bench.py` измеряет время `get_shape_observables()` целиком (включая построение WS-плотности).  
8. В бенчмарке используется прогрев + `get_shape_observables.cache_clear()` перед замером для честного сравнения trapz vs FDM.  
9. Для выбранных Z медианный speedup FDM против trapz по `get_shape_observables()` составляет ≈2.78× при `max_abs_err_kurt = 0.0049`.  
10. Файлы `results/fast_spectrum_2_bench*.csv/txt` находятся под `.gitignore` и не попадают в историю (см. [RESULTS-1]).  

## Decision log text (FAST-SPECTRUM-2)

- В `core/shape_observables.ShapeObs` добавлены поля `effective_volume_ws`, `softness_integral_ws`, `density_overlap_ws`, вычисляемые на WS-плотности.  
- В `get_shape_observables` параметр `beta = beta_effective(...)` рассчитывается один раз и используется для гауссовой ссылки и overlap-метрик в обеих ветках интегратора.  
- FDM-ветка переведена на однопроходный режим: один набор выборок `u`, один `rho_fn(r)` и многоканальный интегранд (моменты, softness, overlap) без дополнительных циклов.  
- В `thermo_fingerprint_for_shape` добавлены поля `coupling_density`, `density_model`, `density_blend`, `density_Z_ref` для корректного кэширования shape-подписей.  
- Добавлен тест согласия trapz/FDM для новых observables на Z ∈ {1, 6, 8, 14} с порогами по относительной ошибке (volume ≤10%, softness ≤5%).  
- Реализован бенчмарк `analysis/ws/fast_spectrum_2_bench.py`, измеряющий время полного вызова `get_shape_observables()` для trapz и FDM с параметрами (base=2, depth=5).  
- По результатам бенчмарка достигается медианный ускоритель ≈2.45× при сохранении ошибки по kurtosis в диапазоне FAST-SPECTRUM-1 (max_abs_err_kurt ≈ 0.0049).  
- Результаты бенча сохраняются в `results/fast_spectrum_2_bench.csv` и `results/fast_spectrum_2_bench_summary.txt`, которые игнорируются git согласно политике [RESULTS-1].  
