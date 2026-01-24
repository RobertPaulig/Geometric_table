# ACCURACY-A3.3 — Phase → Rho Pivot (nested Φ, rho-realness, phase-only-via-rho)

**Roadmap-ID:** ACCURACY-A3.3  
**Status:** CONTRACT (SoT) — must be read before any A3.3 code changes

## 0) Executive summary (≤ 8 lines)

We run A3.3 as **one hypothesis**: A2 inversions persist because the real-only ρ channel is missing a
cycle/holonomy degree of freedom.

In A3.3 the phase/magnetic operator is allowed to affect the system **only via ρ** (causality lock),
so the comparison vs A2 remains valid (same E/SCF discipline).

The only new DOF is **one global** `Phi ∈ {0, π/6, π/3, π/2}`, chosen **nested** (train-only) inside
each outer LOOCV fold.

## 1) STOP / GO (hard)

**STOP:** production A3.3 code is forbidden until this contract is:

1) added to the repo,  
2) referenced from `docs/99_index.md` as a REF,  
3) included in `ENTRYPOINT.md` read order,  
4) listed in `docs/ROADMAP.md`.

**GO:** after the “Memory Fix” PR is merged to `main` and CI is green, open a separate PR for the
actual A3.3 implementation.

## 2) One hypothesis (frozen)

Inversions remain because the A2 real-only ρ channel is information-insufficient for cyclic topology.
Adding a phase/magnetic operator that affects **only ρ** must yield:

- **MVP gate:** `num_groups_spearman_negative_test == 0` on `functional_only` (LOOCV by `group_id`).

## 3) Invariants (violate → STOP)

1) Truth is unchanged (same truth files; truth copies inside evidence pack ZIP).  
2) Split is unchanged: LOOCV strictly by `group_id`.  
3) Graph-only (no 3D / conformers).  
4) SCF discipline is identical to A2.2: monotonicity by final score, backtracking, `phi -= mean(phi)`,
   safety floor before `-log`.  
5) Zero-leak: A2 runners/artifacts stay bit-for-bit identical (forensic proof required).  
6) Phase enters **only via ρ** (see Fixator #1).

## 4) FIXATOR #1 — Phase enters ONLY via ρ (law of A3.3)

- `L_base` (real) is used for everything related to E / curvature / SCF monotonicity.  
- `L_A` (phase/magnetic) is used **only** to compute `rho_phase`.  

Forbidden: using `L_A` inside energy/gradients/monotonicity checks.

## 5) FIXATOR #2 — ρ normalization + realness sensor

Definition (minimal):

- `rho_raw := diag(exp(-tau * L_A))` (or an accepted equivalent for a Hermitian operator)  
- `rho_imag_max := max(abs(Im(rho_raw)))`  
- Requirement: `rho_imag_max < 1e-12`, otherwise `rho_complex_violation=true` and verdict=FAIL  
- `rho := Re(rho_raw)`  
- `rho := rho / sum(rho)` (hard normalization)

Must be logged in `rho_compare.csv`:
`rho_sum`, `rho_imag_max`, `rho_renorm_applied`, `rho_floor_rate`.

## 6) FIXATOR #3 — Nested Φ selection (1 DOF)

The only new parameter is:

- `Phi ∈ {0, π/6, π/3, π/2}`

`Phi` is chosen **nested** inside each outer LOOCV fold using train-only data:

1) minimize `num_negative_train_inner`  
2) then maximize `median_spearman_train_inner`

Mandatory fields in `metrics.json` / `best_config.json`:
`nested_selection=true`, `phi_candidates=[...]`, `selected_phi_by_outer_fold`, `search_space_size=4`.

## 7) Minimal implementation requirements

1) Add a new opt-in runner, e.g. `scripts/accuracy_a1_isomers_a3_3_phase_rho_pivot.py`.  
2) Implement Fixator #1 (phase only affects ρ).  
3) Implement Fixator #2 (ρ sensors + normalization).  
4) Implement Fixator #3 (nested Φ selection).  
5) Evidence pack must include:
   - `phase_summary.csv` (rings info + selected Φ)
   - `rho_compare.csv` (ρ sensors)
   - `search_results.csv` (nested selection results)
   - `metrics.json` (nested fields)
6) Zero-leak forensic + bit-for-bit proof for A2.

## 8) KPI (MVP)

On LOOCV(test), `functional_only`:

- **must:** `num_groups_spearman_negative_test == 0`  
- then: `median_spearman_by_group_test >= 0.55`  
- then: `pairwise_order_accuracy_overall_test >= 0.60`  
- then: `top1_accuracy_mean_test >= 0.30`

## 9) What FAIL proves

If A3.3 fails while all invariants and fixators are satisfied, then “phase-through-ρ is insufficient”,
and any next attempt must be a new Roadmap-ID with a new contract (no silent drift).

