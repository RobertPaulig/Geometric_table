# Spectral Calibration (Experimental)

Goal: sanity-check whether spectral_gap and spectral_entropy_norm show any consistent separation between originals and rewire-decoys. This is a diagnostics-only note, not a policy.

## Method

- Inputs: ~25 small/medium SMILES (mix of aromatics, aliphatics, and small bio-relevant molecules).
- For each original, generate 15 rewire decoys (same constraints as pipeline defaults).
- Compute: spectral_gap, spectral_entropy, spectral_entropy_norm from Laplacian eigenvalues (eps=1e-9, entropy normalized by log(K) where K = count of positive modes).
- Output CSV: `out_calib/spectral_calibration.csv` (not committed).

## Summary table (orig vs median decoys)

| id | orig_gap | decoy_gap_median | orig_entropy_norm | decoy_entropy_norm_median |
| --- | --- | --- | --- | --- |
| alanine | 0.438447 | 0.438447 | 0.818212 | 0.818212 |
| aspirin | 0.138129 | 0.374776 | 0.885300 | 0.912329 |
| cysteine | 0.321720 | 0.321720 | 0.834349 | 0.834349 |
| glucose | 0.262390 | 0.425380 | 0.875628 | 0.901962 |
| ibuprofen | 0.091060 | 0.239784 | 0.886552 | 0.907963 |
| lysine | 0.123574 | 0.438447 | 0.863322 | 0.902652 |
| naproxen | 0.105938 | 0.105938 | 0.898585 | 0.898585 |

## Observations (early, non-binding)

1. For several aromatics (aspirin, ibuprofen), decoy medians show higher gap and entropy_norm than the original.
2. Some small aliphatics (alanine, cysteine) show little to no separation with this decoy generator/seed.
3. No filtering decisions derived from these metrics yet.

## How to run

```bash
python scripts/spectral_calibration_smoke.py
```

Then review `out_calib/spectral_calibration.csv` and update the table above with medians.
