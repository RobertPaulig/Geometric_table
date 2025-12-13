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

## 3) SPECTRAL-DENSITY (wsZ-on config, но WS-ветка по-прежнему по форме близка к baseline)

Команда:

```bash
python -m analysis.spectral_density.compare_density_sources
```

Таблица по Z = 1, 6, 14, 26 (численно совпадает с baseline, т.к. в текущем SD-скрипте ws-ветка не использует Z-coupling):

|  Z |  beta | E_gauss | E_ws | ratio_E | mass_ratio |
| -: | ----: | ------: | ---: | ------: | ---------: |
|  1 | 0.2154 | 54.3   | 55.71 | 1.0260 | 1.0000 |
|  6 | 0.7114 | 9.281  | 9.285 | 1.0000 | 1.0000 |
| 14 | 1.2510 | 3.977  | 3.979 | 1.0000 | 1.0000 |
| 26 | 1.8910 | 2.142  | 2.143 | 1.0000 | 1.0000 |

ThermoConfig (wsZ-on профиль для SD, фактически записан в `results/thermo_config_spectral_density_wsZ1.json`):

- как в baseline, но:
  - `coupling_ws_Z=1.0`
  - `ws_Z_ref=10.0`
  - `ws_Z_alpha=1.0/3.0`

## 4) SPECTRAL-GEOM (wsZ-on профиль; порты и h как в baseline, gap_sp теперь Z-зависим в скане)

Команда:

```bash
python -m analysis.geom.compare_port_geometry_sources
```

Таблица по B,C,N,O,Si,P,S (на текущем профиле ws_geom_* и пороге h=0.5 численно совпадает с baseline, т.к. ThermoConfig для SG ещё не включает coupling_ws_Z при этом прогоне):

| El | ports |   base   | inferred |   gap  |    h   |
| -- | ----: | -------- | -------- | -----: | -----: |
| B  |     3 | trigonal | trigonal | 0.6681 | 0.5822 |
| C  |     4 | tetra    | tetra    | 0.6681 | 0.5822 |
| N  |     3 | pyramidal| trigonal | 0.6681 | 0.5822 |
| O  |     2 | bent     | linear   | 0.6681 | 0.5822 |
| Si |     4 | tetra    | tetra    | 0.6681 | 0.5822 |
| P  |     3 | pyramidal| trigonal | 0.6681 | 0.5822 |
| S  |     2 | bent     | linear   | 0.6681 | 0.5822 |

ThermoConfig (wsZ-on профиль для SG, сохранён в `results/thermo_config_spectral_geom_wsZ1.json`):

- как в baseline SG, но:
  - `coupling_ws_Z=1.0`
  - `ws_Z_ref=10.0`
  - `ws_Z_alpha=1.0/3.0`

Скан по Z (wsZ-on, ключевой эффект SPECTRAL-WS-Z-1):

Команда:

```bash
python -m analysis.geom.scan_ws_gap_vs_Z --coupling-ws-z 1.0
```

Summary по `results/ws_sp_gap_scan_Z1_30.csv` (wsZ-on):

- `nunique(gap_sp) = 30`
- `gap_min ≈ 0.1531`
- `gap_max ≈ 2.0516`
- `gap(Z=1) ≈ 0.1531`
- `gap(Z=10) ≈ 0.9822`
- `gap(Z=20) ≈ 1.5947`
- `gap(Z=30) ≈ 2.0516`

Интерпретация: при `coupling_ws_Z=1.0` `gap_sp(Z)` становится монотонно возрастающей функцией Z (больше Z → более глубокие уровни, больший разрыв s–p). Это и есть основной эффект SPECTRAL-WS-Z-1: spectral-признак перестаёт быть константой.

## 5) TOPO-3D

Команда:

```bash
python -m analysis.topo3d.compare_topology_2d_3d
```

Summary (`results/topo3d_compare_summary.txt` — как в baseline, т.к. TOPO-3D не зависит от wsZ-профиля):

```text
TOPO-3D-1 comparison: 2D crossing proxy vs 3D entanglement

     name  n_atoms  n_bonds  crossing_proxy_2d  entanglement_3d
 chain_C3        3        2           0.000000            0.000
cycle_CON        3        3           0.000000            0.000
     K4_C        4        6           0.166667            0.651
```

## 6) Delta vs previous (baseline vs wsZ-on)

- Tests: число тестов увеличилось (37), все зелёные как в baseline; дополнительных падений нет.
- SPECTRAL-DENSITY: при текущем профиле SD-скрипта включение wsZ влияет только на потенциальную форму ρ_ws(r;Z), но не меняет агрегатную метрику E_ws/E_gauss (она остаётся в коридоре ≈1.0–1.026).
- SPECTRAL-GEOM: локальная таблица base→inferred, gap, h по B/C/N/O/Si/P/S пока совпадает с baseline (т.к. SG-скрипт ещё не использует coupling_ws_Z в своём ThermoConfig профиле).
- WS-gap scan: **главный эффект** — при wsZ-on `gap_sp(Z)` перестаёт быть константой и становится монотонной функцией Z; это делает spectral-признак геометрии потенциально информативным.
- TOPO-3D: метрики не зависят от wsZ и совпадают с baseline, как ожидается.

