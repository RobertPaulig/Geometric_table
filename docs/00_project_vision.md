## Геометрическая таблица (Geom-Mendeleev)

Цель: построить аналог таблицы Менделеева, где ячейки определяются не орбиталями \((n,l)\),
а геометрическими классами атомных графов (количество портов, симметрия, роль на D/A-плоскости).

Базовые элементы таблицы:
- строки (периоды): `period` из `geom_atoms` / `element_indices_with_dblock.csv`;
- столбцы (geom-классы): `s_donor`, `s_bridge`, `p_semihub`, `p_acceptor`, `inert`, `d_octa`,
  (в будущем — `f_hyperhub`);
- содержимое ячейки: \((Z, \mathrm{El})\), \((D, A, role)\), `E_port`, `geom_class`,
  а также производные параметры (сложность, \(N_{\text{best}}/Z\) и т.п.).

Geom-Mendeleev v1:
- реализован в `export_geom_periodic_table.py`;
- результаты: `results/geom_periodic_table_v1.csv` и `results/geom_periodic_law_stats.txt`;
- подтверждает базовый закон геометрической периодичности D/A для s/p-блоков и инертных элементов.

Визуальная форма Geom-таблицы реализована как рисунок `results/geom_periodic_table_v1.png`,
где по вертикали отложены периоды, а по горизонтали — geom-классы
(`s_donor`, `s_bridge`, `p_semihub`, `p_acceptor`, `inert`, `d_octa`, `other`).
