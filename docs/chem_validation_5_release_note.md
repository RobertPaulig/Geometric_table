# CHEM-VALIDATION-5 — product PASS (C15/C16, Mode A, fixed budget)

Дата: 2025-12-25

## Что доказано

Контур CHEM-VALIDATION-5 закрыт как продуктовый стенд для равновесных распределений по топологиям *tree-only* алканов при фиксированном числе атомов `N`:

- Равновесное распределение `P_eq` (equilibrium-first, Mode A) согласовано по mixing-диагностике и проходит DoD-пороги.
- Coverage строго 100% по ожидаемому числу непомеченных топологий:
  - C15: `n_unique_eq=4347/4347`
  - C16: `n_unique_eq=10359/10359`

## Golden recipe (финальные параметры)

Общее:

- Mode: `A` (FDM-only на деревьях).
- `start_specs`: `path`, `max_branch` (2 старта).
- `chains`: 5 (итого 10 цепей = starts×chains).
- `thin`: 10.
- `burnin_frac`: 0.30 (экспонирован как CLI).
- `--no-auto_escalate` (фиксированный бюджет; FAIL не “краснит” прогон).

Интерпретация бюджета:

- В CHEM-VALIDATION-5 `steps_init`/`max_steps` трактуются как **TOTAL steps** по всем starts×chains.
- `steps_per_chain = steps_total / (len(start_specs) * chains)`.

## Итоговые финальные прогоны

### C16 (hexadecane)

- Бюджет: `STEPS_TOTAL=320,000,000` ⇒ `32,000,000` шагов на цепь.
- PASS:
  - `KL_max_pairwise=0.004858`
  - `KL_split_max=0.009774`
  - `Rhat_energy_max≈1.000001`
  - `ESS_energy_min≈673,447`
- Артефакты: `results/chem_validation_5_hexadecane.txt`, `results/chem_validation_5_hexadecane.csv`
- Коммиты: артефакты `f1854c9`, фиксация в decision log `956055d`.

### C15 (pentadecane)

- Бюджет: `STEPS_TOTAL=160,000,000` ⇒ `16,000,000` шагов на цепь.
- PASS:
  - `KL_max_pairwise=0.004114`
  - `KL_split_max=0.008336`
  - `Rhat_energy_max=1.000000`
  - `ESS_energy_min≈378,992`
- Артефакты: `results/chem_validation_5_pentadecane.txt`, `results/chem_validation_5_pentadecane.csv`
- Коммит: артефакты `03f4937`.

## Изменения кода, критичные для воспроизводимости

- `burnin_frac` вынесен в `EqCfg` и CLI для C15/C16: commit `b339f94`.

## Следующий инженерный эпик (без изменения математики)

- End-to-end автоматизация: `eqdist_make → worker → aggregate → final chem_validation` под единым манифестом и автокоммитом артефактов.
- Формализация “статистического пола” для `KL_split` как части DoD (инженерное обоснование порога 0.01 при конечной выборке).

