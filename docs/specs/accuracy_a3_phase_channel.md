# ACCURACY-A3.0 — Фазовый / магнитный канал (SPEC + инвариантные тесты, без кода ядра)

**Roadmap-ID:** ACCURACY-A3.0  
**Статус:** SPEC + tests-only (нет production-кода ядра)  
**Цель:** зафиксировать контракт фазового канала (magnetic/phase Laplacian) и инвариантные тесты до реализации A3.x.

---

## 0) Executive summary (до 5 строк)

A2.x показали: даже при вариационно-стабильном SCF (A2.2) инверсии могут сохраняться.  
A3 вводит фазу на рёбрах (магнитный лапласиан) как новый канал для циклов/колец.  
В A3.0 мы **не** реализуем ядро и **не** меняем A2.x — только спека + тесты (opt-in).

---

## 1) Scope / Non-goals

**В scope (A3.0):**
- Этот документ (SoT спеки).
- Тесты-инварианты (skip/xfail по умолчанию).
- DOF-guard: фазовый канал имеет ≤ 2 глобальных параметра и применяется только к разрешённым рёбрам.

**Не делаем (A3.0):**
- Не добавляем production-код `H(A)` / `L_A` в `src/`.
- Не запускаем publish-run / release / registry / lineage.
- Не меняем truth-датасеты и A2.x runner’ы.

---

## 2) Definitions

### 2.1 Graph and base weights
Граф молекулы: `G = (V, E)` (2D graph-only).  
Базовые веса рёбер: `w_ij >= 0` (bond order + aromatic multiplier и т.п., как в A2).

### 2.2 Edge phase / gauge field
Вводим фазу на рёбрах:
- `A_ij = -A_ji` (антисимметрия)
- комплексный множитель ребра: `exp(i * A_ij)`

Смысл: `A` — дискретный gauge-потенциал; физически существенны только циркуляции по циклам.

---

## 3) Operator: magnetic adjacency / Laplacian (минимальная спека)

### 3.1 Complex weighted adjacency
Для `(i,j) in E`:
- `(W_A)_ij = w_ij * exp(i * A_ij)`
- иначе `(W_A)_ij = 0`

При антисимметрии `A_ji=-A_ij` матрица `W_A` является сопряжённо-симметричной.

### 3.2 Degree and Laplacian
Степени считаем по **реальным** весам:
- `d_i = sum_j w_ij`
- `D = diag(d)`

Магнитный лапласиан:
- `L_A = D - W_A`

**Требование:** `L_A` — эрмитова матрица: `L_A == L_A^H` (в пределах tol).

### 3.3 Future Hamiltonian (A3.x)
В будущем (не A3.0):
- `H_A = L_A + diag(V)` где `V` вещественный.

---

## 4) Degrees of freedom (anti-chaos guard)

### 4.1 Eligibility: где разрешена фаза
Фаза разрешена только на:
- ring-edges (ребро входит в какой-то цикл),
или (возможная под-ветка):
- aromatic ring-edges (только ароматические циклы).

### 4.2 Parameterization: ≤ 2 глобальных скаляра
`A_ij` **не** является свободным по ребру. Разрешены только варианты:

**P1 (1 параметр):**
- `A_ij = s_ring * sign_ij` для eligible edges

**P2 (2 параметра):**
- `A_ij = s_ring * sign_ij` на ring-edges
- `A_ij = s_arom * sign_ij` на aromatic ring-edges

`sign_ij` — детерминированное правило ориентации (например, по индексу узлов или BFS).

**Запрещено:** принимать “вектор фаз по ребрам” без явного override.

---

## 5) Gauge invariance (контракт)

Для любых `theta_i` (на узлах) определим:
- `A'_ij = A_ij + theta_i - theta_j`

**Требование:** спектр `L_A` инвариантен (в пределах tol), т.е. `eig(L_A) == eig(L_A')`.

---

## 6) Tests (A3.0, opt-in)

Все тесты A3.0 должны быть:
- под флагом `RUN_A3_TESTS=1` **или** помечены xfail/skip по умолчанию,
чтобы CI по умолчанию оставался зелёным.

### T1 — Hermiticity
Проверка `L_A ≈ L_A.conj().T` при `A_ji=-A_ij`.

### T2 — Gauge invariance (spectrum)
Случайный `theta` → строим `A'` → проверяем совпадение `eigvalsh(L_A)` и `eigvalsh(L_A')`.

### T3 — DOF guard
Проверяем, что ненулевые `A_ij` появляются только на eligible edges и что фаза строится из ≤ 2 глобальных параметров.

---

## 7) Definition of Done (A3.0)

A3.0 считается DONE, когда:
1) Этот документ существует: `docs/specs/accuracy_a3_phase_channel.md`.
2) Есть тесты T1–T3 (opt-in / skip by default).
3) Дефолтный CI зелёный.
4) Нет изменений production-кода A2 runner’ов / метрик / truth-chain.

