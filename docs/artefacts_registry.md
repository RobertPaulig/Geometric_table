# Artefacts Registry

## stress_10k (light) - 2026-01-05

- Commit: https://github.com/RobertPaulig/Geometric_table/commit/9c54b043047010f165549e42399bf265a9395198
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/stress-10k-2026-01-05/evidence_pack.zip
- SHA256(evidence_pack.zip): 458EF5315D52B17B7797F67F3B89E2A091A12058406C9876DCC6D9925E95C76B
- Command:
  hetero2-batch --input stress_10k.csv --out_dir out_stress_10k --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
- Pack contents (expected):
  summary.csv, metrics.json, index.md, manifest.json, checksums.sha256, evidence_pack.zip
- Outcome:
  OK=7200, SKIP=2800 (no_decoys_generated=1800, too_many_atoms=500, disconnected=250, invalid_smiles=208, missing_smiles=42), ERROR=0
- Determinism check: PASS (workers=6, seed_strategy=per_row, seed=0)
- Input: determinism_1k.csv (generated locally)
- Command: --seed_strategy per_row --seed 0 --workers 6 --k_decoys 2 --artifacts light --score_mode mock --zip_pack --timeout_s 60 --maxtasksperchild 100
- counts (run1=run2): {'ERROR':0,'OK':560,'SKIP':440}
- top_reasons (run1=run2): too_many_atoms=200, no_decoys_generated=140, disconnected=50, invalid_smiles=41, missing_smiles=9


## stress-10k-2026-01-07

- Source commit: d70394661474acd7c1b74d581471fa7fb10bb263
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/stress-10k-2026-01-07/evidence_pack.zip
- SHA256(evidence_pack.zip): DF8EF1412990461CD52FFE35019E8F8BA3A40A7BDEACBE3AB1EEF00191D3AC35
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack

## stress-50k-2026-01-08

- Source commit: 21de1e6765120d5752ed2f8bffb106c000324731
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/stress-50k-2026-01-08/evidence_pack.zip
- SHA256(evidence_pack.zip): 8D7A0106B66DB452439842F97B97B504A124E2A8B3E1D7EE752458B25E9A02C0
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 6 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack

## pilot-2026-01-08-r1

- Source commit: fb971cad4beb2a34dc11c4668557b75392b8c3a0
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/pilot-2026-01-08-r1/pilot_evidence_pack.zip
- SHA256(pilot_evidence_pack.zip): DA215F5F32C33122BE922E78F8CEBACC66BDE65CFFBC9377B25E3A4E0CC65F62
- Command:
  python scripts/pilot_generate_input.py --out_dir out_pilot --rows 1000 --k_decoys 2 --seed 0
  hetero2-batch --input out_pilot/input.csv --out_dir out_pilot --artifacts light --score_mode external_scores --scores_input out_pilot/scores.json --k_decoys 2 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack

## pilot-2026-01-08-r2

- Source commit: e8fe85dd41350d80924c86103bf7ac21ae085866
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/pilot-2026-01-08-r2/pilot_evidence_pack.zip
- SHA256(pilot_evidence_pack.zip): BB564070C2B02087B589A511FB12AD4DEDC13C1EE06A58BC793DD5CF51D3B2A8
- Command:
  python scripts/pilot_generate_input.py --out_dir out_pilot --rows 1000 --k_decoys 2 --seed 0
  hetero2-batch --input out_pilot/input.csv --out_dir out_pilot --artifacts light --score_mode external_scores --scores_input out_pilot/scores.json --k_decoys 2 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack

## value-ring-suite-2026-01-10

- Source commit: ba63273ea6f9b3f8c87cf0791b372fb7fc5d2871
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-ring-suite-2026-01-10/value_ring_suite_evidence_pack.zip
- SHA256(value_ring_suite_evidence_pack.zip): 912071F3927D733FF4F5EDA1AB5A3158F83D18EBA4E99B1C2CC174FD6EE08274
- Command:
  python scripts/pilot_generate_input.py --out_dir out_ring_suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  hetero2-batch --input out_ring_suite/input.csv --out_dir out_ring_suite --artifacts light --score_mode external_scores --scores_input out_ring_suite/scores.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
- Outcome (facts from summary.csv):
  - rows_total: 200
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons:
    - no_decoys_generated: 140
  - share_rows_with_n_decoys_gt_0: 0.300 (30.0%)

## value-known-bad-good-2026-01-11

- Source commit: 706aaaf32c52e2df9b79bc611421d57af3cbecb4

- Variant: BAD-constant
  - Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-11/value_known_bad_good_BAD-constant_evidence_pack.zip
  - SHA256(value_known_bad_good_BAD-constant_evidence_pack.zip): 043940CC6FE275D7393BD9F1AAB8A6CB8033890430F5B495782D226FB97CD5DF
  - Command:
    python scripts/pilot_generate_input.py --out_dir out_value_m2/suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
    (scores variant) BAD-constant: all scores equal
    hetero2-batch --input out_value_m2/suite/input.csv --out_dir out_value_m2/BAD-constant --artifacts light --score_mode external_scores --scores_input out_value_m2/score_variants/scores_BAD-constant.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
  - Outcome (facts from summary.csv):
  - rows_total: 200
  - rows_ok: 60
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons:
    - no_decoys_generated: 140
  - share_rows_with_n_decoys_gt_0: 0.300 (30.0%)
  - median_slack: 0.050000
  - pass_rate: 1.000000

- Variant: BAD-random
  - Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-11/value_known_bad_good_BAD-random_evidence_pack.zip
  - SHA256(value_known_bad_good_BAD-random_evidence_pack.zip): 38393053ABDF710D3AB4BAE68C7EA1A55547A8F984B0600E17411953B65294C1
  - Command:
    python scripts/pilot_generate_input.py --out_dir out_value_m2/suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
    (scores variant) BAD-random: random scores (seed=0)
    hetero2-batch --input out_value_m2/suite/input.csv --out_dir out_value_m2/BAD-random --artifacts light --score_mode external_scores --scores_input out_value_m2/score_variants/scores_BAD-random.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
  - Outcome (facts from summary.csv):
  - rows_total: 200
  - rows_ok: 60
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons:
    - no_decoys_generated: 140
  - share_rows_with_n_decoys_gt_0: 0.300 (30.0%)
  - median_slack: 0.050000
  - pass_rate: 1.000000

- Variant: GOOD-synthetic
  - Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-11/value_known_bad_good_GOOD-synthetic_evidence_pack.zip
  - SHA256(value_known_bad_good_GOOD-synthetic_evidence_pack.zip): DF27F9CA9CA4A74089EF1966D9591FEDDE7F9C452CD62BDE94F4D384F09F27B3
  - Command:
    python scripts/pilot_generate_input.py --out_dir out_value_m2/suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
    (scores variant) GOOD-synthetic: original=1.0, decoys=0.0
    hetero2-batch --input out_value_m2/suite/input.csv --out_dir out_value_m2/GOOD-synthetic --artifacts light --score_mode external_scores --scores_input out_value_m2/score_variants/scores_GOOD-synthetic.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
  - Outcome (facts from summary.csv):
  - rows_total: 200
  - rows_ok: 60
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons:
    - no_decoys_generated: 140
  - share_rows_with_n_decoys_gt_0: 0.300 (30.0%)
  - median_slack: 0.050000
  - pass_rate: 1.000000

- Separation facts (OK-only; no auto-threshold gating):
separation facts (computed on status==OK rows only):
- BAD-constant: rows_ok=60/200, median_slack=0.050000, pass_rate=1.000000
- BAD-random: rows_ok=60/200, median_slack=0.050000, pass_rate=1.000000
- GOOD-synthetic: rows_ok=60/200, median_slack=0.050000, pass_rate=1.000000

- Δ_median_slack(GOOD - BAD-constant): 0.000000
- Δ_PASS_rate(GOOD - BAD-constant): 0.000000
- Δ_median_slack(GOOD - BAD-random): 0.000000
- Δ_PASS_rate(GOOD - BAD-random): 0.000000

## value-known-bad-good-2026-01-11-r2

- Source commit: 8110a1b78f2a67d684d64343c996e64d218f99e4

- Variant: BAD-constant
  - Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-11-r2/value_known_bad_good_BAD-constant_evidence_pack.zip
  - SHA256(value_known_bad_good_BAD-constant_evidence_pack.zip): 5B117E204E9E98128EE4C6BEBC609D4282862DBF3BEB935FF432076809F0046A
  - Command:
    python scripts/pilot_generate_input.py --out_dir out_value_m2/suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
    (scores variant) BAD-constant: all scores equal
    hetero2-batch --input out_value_m2/suite/input.csv --out_dir out_value_m2/BAD-constant --artifacts light --score_mode external_scores --scores_input out_value_m2/score_variants/scores_BAD-constant.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
  - Outcome (facts from summary.csv):
  - rows_total: 200
  - rows_ok: 60
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons:
    - no_decoys_generated: 140
  - share_rows_with_n_decoys_gt_0: 0.300 (30.0%)
  - median_slack: -0.500000
  - pass_rate: 0.000000

- Variant: BAD-random
  - Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-11-r2/value_known_bad_good_BAD-random_evidence_pack.zip
  - SHA256(value_known_bad_good_BAD-random_evidence_pack.zip): E4255428FC9EEE082D54B04D6A01E7EE98F5F59717CBA259590D6457F1064916
  - Command:
    python scripts/pilot_generate_input.py --out_dir out_value_m2/suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
    (scores variant) BAD-random: random scores (seed=0)
    hetero2-batch --input out_value_m2/suite/input.csv --out_dir out_value_m2/BAD-random --artifacts light --score_mode external_scores --scores_input out_value_m2/score_variants/scores_BAD-random.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
  - Outcome (facts from summary.csv):
  - rows_total: 200
  - rows_ok: 60
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons:
    - no_decoys_generated: 140
  - share_rows_with_n_decoys_gt_0: 0.300 (30.0%)
  - median_slack: 0.000000
  - pass_rate: 0.666667

- Variant: GOOD-synthetic
  - Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-11-r2/value_known_bad_good_GOOD-synthetic_evidence_pack.zip
  - SHA256(value_known_bad_good_GOOD-synthetic_evidence_pack.zip): 228E5B0515316831DE5D208BEE624240973215BDAC85236C1583DEC1B7EA6B5C
  - Command:
    python scripts/pilot_generate_input.py --out_dir out_value_m2/suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
    (scores variant) GOOD-synthetic: original=1.0, decoys=0.0
    hetero2-batch --input out_value_m2/suite/input.csv --out_dir out_value_m2/GOOD-synthetic --artifacts light --score_mode external_scores --scores_input out_value_m2/score_variants/scores_GOOD-synthetic.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
  - Outcome (facts from summary.csv):
  - rows_total: 200
  - rows_ok: 60
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons:
    - no_decoys_generated: 140
  - share_rows_with_n_decoys_gt_0: 0.300 (30.0%)
  - median_slack: 0.000000
  - pass_rate: 1.000000

- Separation facts (OK-only; no auto-threshold gating):
separation facts (computed on status==OK rows only):
- BAD-constant: rows_ok=60/200, median_slack=-0.500000, pass_rate=0.000000
- BAD-random: rows_ok=60/200, median_slack=0.000000, pass_rate=0.666667
- GOOD-synthetic: rows_ok=60/200, median_slack=0.000000, pass_rate=1.000000

- Δ_median_slack(GOOD - BAD-constant): 0.500000
- Δ_PASS_rate(GOOD - BAD-constant): 1.000000
- Δ_median_slack(GOOD - BAD-random): 0.000000
- Δ_PASS_rate(GOOD - BAD-random): 0.333333
