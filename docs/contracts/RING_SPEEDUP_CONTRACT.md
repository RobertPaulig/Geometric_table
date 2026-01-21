# RING SPEEDUP CONTRACT (P5.6.v1)

## 0. Purpose

This contract freezes how **ring-suite speedup at scale** is evaluated and prevents "polymer-only" scaling claims from being treated as chemistry-wide evidence.

## 1. Scope

Applies to P5 large-scale evidence packs that include:

- `topology_family = polymer` fixtures (polymer-suite)
- `topology_family = ring` fixtures (ring-suite)
- per-family speed and cost aggregates

## 2. Definitions

### 2.1 Scale set (ring)

Let:

- `gate_n_min = N_gate`
- `family ∈ {"polymer","ring"}`

Define the **ring scale set**:

`S_ring = { sample | sample.family == "ring" and sample.n_atoms >= gate_n_min }`

### 2.2 Per-sample metrics

- `speedup = baseline_walltime_ms / adaptive_walltime_ms`
- `eval_ratio = baseline_points / adaptive_evals_total` (larger is better)
- `correctness_pass` as defined by P5.1 tolerance rail

## 3. Required ring-at-scale metrics

Computed over `S_ring`:

- `ring_speedup_median_at_scale = median(speedup)`
- `ring_eval_ratio_median_at_scale = median(eval_ratio)`
- `ring_correctness_pass_rate_at_scale = mean(correctness_pass)`

## 4. Preconditions (correctness gates speed)

If `ring_correctness_pass_rate_at_scale < correctness_gate_rate`, then:

- `ring_speedup_verdict_at_scale = NOT_VALID_DUE_TO_CORRECTNESS`
- speedup is not treated as a KPI

## 5. Verdict logic (P5.6)

Let:

- `gate_speedup = scale_speedup_gate_break_even` (default 1.0)
- `polymer_speedup_median_at_scale = speedup_median_at_scale_polymer`

If correctness is NOT valid at ring-at-scale:

- `NOT_VALID_DUE_TO_CORRECTNESS`

Else if `|S_ring| < min_scale_samples`:

- `NO_SPEEDUP_YET` (insufficient evidence; not a failure claim)

Else:

- `PASS_RING_SPEEDUP_AT_SCALE` if `ring_speedup_median_at_scale >= gate_speedup`
- `FAIL_RING_SPEEDUP_AT_SCALE` if `ring_speedup_median_at_scale < gate_speedup` AND `polymer_speedup_median_at_scale >= gate_speedup`
- `NO_SPEEDUP_YET` otherwise

**Reason rail:** `ring_speedup_verdict_reason_at_scale` MUST be non-empty and MUST include:

- `gate_n_min`, `min_scale_samples`, `|S_ring|`
- `ring_correctness_pass_rate_at_scale`, `ring_speedup_median_at_scale`
- `polymer_speedup_median_at_scale`, `gate_speedup`
- `topology_ring_cost_gap_verdict_at_scale` (P5.5 link rail)

## 6. Required artifacts (canonical names)

Evidence pack MUST contain:

- `speedup_vs_n_by_family.csv`
- `timing_breakdown_by_family.csv`
- `summary_metadata.json`

## 7. Required metadata fields (`summary_metadata.json`)

Minimum required P5.6 fields:

- `ring_speedup_median_at_scale`
- `ring_eval_ratio_median_at_scale`
- `ring_correctness_pass_rate_at_scale`
- `ring_speedup_verdict_at_scale`
- `ring_speedup_verdict_reason_at_scale`
- `topology_ring_cost_gap_verdict_at_scale` (required linkage to P5.5)

Allowed values for `ring_speedup_verdict_at_scale`:

- `PASS_RING_SPEEDUP_AT_SCALE`
- `FAIL_RING_SPEEDUP_AT_SCALE`
- `NO_SPEEDUP_YET`
- `NOT_VALID_DUE_TO_CORRECTNESS`

## 8. Chain of truth

Truth exists only when the chain is closed:

PR → publish-run → release(zip+.sha256) → registry → lineage → main CI 3/3

## 9. Non-negotiables

- Speed without correctness is invalid.
- Ring results MUST be reported explicitly (no "combined" speedup claims).
- The contract is enforced by tests + publish gates; success must be readable from inside the evidence pack.
