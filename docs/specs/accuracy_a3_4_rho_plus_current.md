# ACCURACY-A3.4 — Rho + Current (J) (Φ fixed; nested κ; rho-realness; phase-only-via-rho)

**Roadmap-ID:** ACCURACY-A3.4  
**Status:** CONTRACT (SoT) — must be read before any A3.4 code changes

## 0) Executive summary (≤ 8 lines)

We run A3.4 as **one hypothesis**: A2/A3.3 showed that diagonal density ρ (`diag f(H)`) is
information-insufficient for cyclic topology. We add a **minimal local current condensate** `J` and
mix it into an **effective density** `rho_eff`.

To avoid multiple DOF drift:
- `Phi` is **fixed** (`Phi_fixed = π/2`).
- The only new DOF is **one scalar** `kappa ∈ {0.0, 0.25, 0.5, 1.0}`, selected nested (train-only)
  inside each outer LOOCV fold.

## 1) STOP / GO (hard)

**STOP:** production A3.4 code is forbidden until this contract is:
1) added to the repo,  
2) referenced from `docs/99_index.md` as a REF,  
3) included in `ENTRYPOINT.md` read order,  
4) listed in `docs/ROADMAP.md`.

**GO:** after the “Memory Fix” PR is merged to `main` and CI is green, open a separate PR for the
actual A3.4 implementation.

## 2) One hypothesis (frozen)

Inversions persist because `ρ = diag f(H)` sees only “returns”, while the missing signal is cyclic
circulation/current. Adding a gauge-invariant local current condensate `J` and mixing it into an
effective density `rho_eff` must yield:

- **MVP gate:** `num_groups_spearman_negative_test == 0` on `functional_only` (LOOCV by `group_id`).

## 3) Invariants (violate → STOP)

1) Truth is unchanged (same truth files; truth copies inside evidence pack ZIP).  
2) Split is unchanged: LOOCV strictly by `group_id`.  
3) Graph-only (no 3D / conformers).  
4) SCF discipline is identical to A2.2: monotonicity by final score, backtracking,
   `phi -= mean(phi)`, safety floor before `-log`.  
5) Zero-leak: A2 runners/artifacts stay bit-for-bit identical (forensic proof required).  
6) Phase enters only through `rho` / `J` construction; energy/curvature/monotonicity stay on `L_base`.

## 4) Definitions (ρ and J)

### 4.1 Phase parameter (fixed)

- `Phi_fixed = π/2`

`Phi` is not a selection DOF in A3.4 (otherwise we introduce two new DOF and lose causality).

### 4.2 Heat kernel on the phase operator

Let `L_A(Phi_fixed)` be the magnetic/phase Laplacian from A3.* (SSSR flux-on-cycles), and define:

- `K = exp(-tau * L_A)` (computed via eigenpairs; Hermitian-safe).

### 4.3 ρ (as in A3.3)

- `rho_raw_i = Re(K_ii)`
- `rho := rho_raw / sum(rho_raw)` (hard normalization to a probability distribution)

**Realness sensor (mandatory):**
- `rho_imag_max = max(abs(Im(K_ii)))`
- Requirement: `rho_imag_max < 1e-12`, otherwise `rho_complex_violation=true` and verdict=FAIL.

### 4.4 J — local current condensate (edge → node)

Let `theta_ij` be the edge connection phase used to build `L_A` (same `theta` as in A3 operator code).

For each edge `(i,j)`:
- `q_ij := exp(-i * theta_ij) * K_ij`  (gauge-invariant)
- `j_edge(i,j) := abs(Im(q_ij))` (≥ 0)

Node current density:
- `j_i := sum_{j~i} j_edge(i,j)`

Normalization:
- If `sum(j) > 0`: `j_norm := j / sum(j)`
- Else: `j_norm := 0` (chains / no-cycles should have zero current signal)

## 5) The only new DOF: κ (mixing ρ and J)

Effective density:

- `rho_eff := (1 - kappa) * rho + kappa * j_norm`
- Optionally `rho_eff /= sum(rho_eff)` (numeric renorm); must be logged as a flag.

Then (frozen A2.2 discipline):
- `phi = -log(rho_eff + eps)` → safety floor → `phi -= mean(phi)` → SCF/update/energy as in A2.2.

### κ grid (fixed; 4 points)

- `kappa ∈ {0.0, 0.25, 0.5, 1.0}`

## 6) Nested selection (κ; train-only; inside each outer fold)

Outer: LOOCV test fold = one `group_id`.

Inner (train-only selection of κ):
1) minimize `num_negative_train_inner`  
2) then maximize `median_spearman_train_inner`  

Tie-breaker (only if both equal): prefer **larger κ** (stronger current channel).

Mandatory fields in `metrics.json` / `best_config.json`:
- `nested_selection=true`
- `phi_fixed=pi/2`
- `kappa_candidates=[...]`
- `selected_kappa_by_outer_fold`
- `search_space_size=4`

## 7) Minimal implementation requirements

1) Add a new opt-in runner, e.g. `scripts/accuracy_a1_isomers_a3_4_rho_plus_current.py`.  
2) No A3.4 imports in A2 runners (zero-leak).  
3) Evidence pack must include:
   - `rho_compare.csv` (extend with `j_sum`, `j_entropy`, `kappa_selected`, `rho_eff_entropy`)
   - `current_summary.csv` (ring stats + `j_sum`)
   - `search_results.csv` (nested selection results per κ per outer fold)
   - `metrics.json` (nested fields + KPI verdict)
4) Zero-leak forensic + bit-for-bit proof for A2.

## 8) KPI (MVP)

On LOOCV(test), `functional_only`:

- **must:** `num_groups_spearman_negative_test == 0`  
- then: `median_spearman_by_group_test >= 0.55`  
- then: `pairwise_order_accuracy_overall_test >= 0.60`  
- then: `top1_accuracy_mean_test >= 0.30`

## 9) What FAIL proves

If A3.4 fails while all invariants and this contract are satisfied, then “ρ+J (with one κ) is
insufficient”, and any next attempt must be a new Roadmap-ID with a new contract (no silent drift).

