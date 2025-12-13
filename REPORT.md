## 1) Environment

- `git rev-parse HEAD`: `345d8d28bb06ffb4d09734a478a16c350663b9bf`
- `python -V`: `Python 3.11.9`
- `python -c "import numpy, scipy, yaml; ..."`: `numpy 2.2.4 scipy 1.15.2 PyYAML 6.0.3`

## 2) Tests

Команда:

```bash
python -m pytest -q
```

Результат:

- `37 passed in 13.93s`

Предупреждения:

- PytestDeprecationWarning из `pytest_asyncio` про unset `asyncio_default_fixture_loop_scope` (не влияет на текущие тесты проекта).

## 3) SPECTRAL-DENSITY

Команда:

```bash
python -m analysis.spectral_density.compare_density_sources
```

Таблица по Z = 1, 6, 14, 26:

|  Z |  beta | E_gauss | E_ws | ratio_E | mass_ratio |
| -: | ----: | ------: | ---: | ------: | ---------: |
|  1 | 0.2154 | 54.3   | 55.71 | 1.0260 | 1.0000 |
|  6 | 0.7114 | 9.281  | 9.285 | 1.0000 | 1.0000 |
| 14 | 1.2510 | 3.977  | 3.979 | 1.0000 | 1.0000 |
| 26 | 1.8910 | 2.142  | 2.143 | 1.0000 | 1.0000 |

ThermoConfig (для запуска SPECTRAL-DENSITY, ключевые поля — см. также `results/thermo_config_spectral_density.json`):

- `temperature=1.0`
- `coupling_delta_F=0.0`
- `coupling_complexity=0.0`
- `coupling_softness=0.0`
- `coupling_density=1.0`
- `coupling_density_shape=0.0`
- `coupling_port_geometry=0.0`
- `coupling_ws_Z=0.0`
- `density_model="tf_radius"`
- `density_blend="linear"`
- `density_Z_ref=10.0`
- `density_source="gaussian"`
- `ws_R_max=12.0`, `ws_R_well=5.0`, `ws_V0=40.0`, `ws_N_grid=220`, `ws_ell=0`, `ws_state_index=0`
- `ws_geom_R_max=25.0`, `ws_geom_R_well=6.0`, `ws_geom_V0=45.0`, `ws_geom_N_grid=800`, `ws_geom_gap_ref=1.0`, `ws_geom_gap_scale=1.0`
- `ws_Z_ref=10.0`, `ws_Z_alpha=1.0/3.0`
- `port_geometry_source="legacy"`, `port_geometry_blend="linear"`

## 4) SPECTRAL-GEOM

Команда:

```bash
python -m analysis.geom.compare_port_geometry_sources
```

Таблица по B,C,N,O,Si,P,S:

| El | ports |   base   | inferred |   gap  |    h   |
| -- | ----: | -------- | -------- | -----: | -----: |
| B  |     3 | trigonal | trigonal | 0.6681 | 0.5822 |
| C  |     4 | tetra    | tetra    | 0.6681 | 0.5822 |
| N  |     3 | pyramidal| trigonal | 0.6681 | 0.5822 |
| O  |     2 | bent     | linear   | 0.6681 | 0.5822 |
| Si |     4 | tetra    | tetra    | 0.6681 | 0.5822 |
| P  |     3 | pyramidal| trigonal | 0.6681 | 0.5822 |
| S  |     2 | bent     | linear   | 0.6681 | 0.5822 |

Дополнительно:

- `Matches with legacy (B,C,N,O,Si,P,S): 3/7`

ThermoConfig (для запуска SPECTRAL-GEOM — см. также `results/thermo_config_spectral_geom.json`):

- `temperature=1.0`
- `coupling_port_geometry=1.0`
- `port_geometry_source="ws_sp_gap"`
- `port_geometry_blend="linear"`
- `coupling_density=0.0`, `coupling_density_shape=0.0`
- `coupling_ws_Z=0.0`
- `density_source="gaussian"`, `density_model="tf_radius"`, `density_blend="linear"`, `density_Z_ref=10.0`
- `ws_R_max=12.0`, `ws_R_well=5.0`, `ws_V0=40.0`, `ws_N_grid=220`, `ws_ell=0`, `ws_state_index=0`
- `ws_geom_R_max=25.0`, `ws_geom_R_well=6.0`, `ws_geom_V0=45.0`, `ws_geom_N_grid=800`
- `ws_geom_gap_ref=1.0`, `ws_geom_gap_scale=1.0`
- `ws_Z_ref=10.0`, `ws_Z_alpha=1.0/3.0`

Скан по Z:

Команда:

```bash
python -m analysis.geom.scan_ws_gap_vs_Z
```

Summary по `results/ws_sp_gap_scan_Z1_30.csv` (при `coupling_ws_Z=0`):

- `nunique(gap_sp) = 1`
- `gap_min ≈ 0.9822`
- `gap_max ≈ 0.9822`
- `gap(Z=1) ≈ 0.9822`
- `gap(Z=10) ≈ 0.9822`
- `gap(Z=20) ≈ 0.9822`
- `gap(Z=30) ≈ 0.9822`

Интерпретация: в дефолтном режиме (`coupling_ws_Z=0`) `gap_sp(Z)` остаётся константой (ожидаемо, это legacy-профиль без Z-coupling). При включении `coupling_ws_Z>0` unit-тесты гарантируют, что `gap_sp(Z)` становится Z-чувствительным.

## 5) TOPO-3D

Команда:

```bash
python -m analysis.topo3d.compare_topology_2d_3d
```

Summary (`results/topo3d_compare_summary.txt`):

```text
TOPO-3D-1 comparison: 2D crossing proxy vs 3D entanglement

     name  n_atoms  n_bonds  crossing_proxy_2d  entanglement_3d
 chain_C3        3        2           0.000000            0.000
cycle_CON        3        3           0.000000            0.000
     K4_C        4        6           0.166667            0.651
```

Графы:

- `chain_C3`: линейная цепочка из трёх C (дерево).
- `cycle_CON`: цикл на C-O-N (треугольник).
- `K4_C`: полный граф K4 на C-атомах (плотный, сильно связанный).

Интерпретация:

- Для деревьев и простого цикла crossing_proxy_2d и entanglement_3d ≈ 0.
- Для K4 оба показатели положительные, 3D entanglement заметно больше нуля (~0.651).

## 6) Delta vs previous

- Tests: было `33 passed`, стало `37 passed` (добавлены тесты для WS Z-coupling и 3D блоков).
- SPECTRAL-DENSITY: ratio_E и mass_ratio для Z=1,6,14,26 остались в том же коридоре (≈1.0–1.026), т.е. мост WS→FDM по масштабу стабилен.
- SPECTRAL-GEOM: таблица base→inferred, gap, h по B/C/N/O/Si/P/S не изменилась (при дефолтном `coupling_ws_Z=0` spectral-признак ещё не использует Z).
- WS-gap scan: при `coupling_ws_Z=0` gap_sp(Z) по-прежнему константа (это ожидаемый legacy baseline); теперь есть инфраструктура (coupling_ws_Z, ws_Z_ref, ws_Z_alpha), чтобы включить Z-coupling.
- TOPO-3D: результаты по chain/cycle/K4 и метрике entanglement_3d не изменились, поведение метрики стабильное.
