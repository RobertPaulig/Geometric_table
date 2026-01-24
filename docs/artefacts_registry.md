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

## value-customer-proxy-2026-01-12

- Source commit: 6951804e7892b208a38b877e102df643a7d8e30d
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-customer-proxy-2026-01-12/value_customer_proxy_evidence_pack.zip
- SHA256(value_customer_proxy_evidence_pack.zip): FE7AA762DCC6D512771DA40E90EB58557B32D6A3641033C65546D0553C16B225
- Acceptance criteria (contract): docs/value_m3_acceptance_criteria.md
- Command:
  python scripts/pilot_generate_input.py --out_dir out_value_customer_proxy --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  (scores variant) proxy random (seed=0): out_value_customer_proxy/scores_proxy.json
  hetero2-batch --input out_value_customer_proxy/input.csv --out_dir out_value_customer_proxy --artifacts light --score_mode external_scores --scores_input out_value_customer_proxy/scores_proxy.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
- Outcome (facts from summary.csv):
  - rows_total: 200
  - rows_ok: 60
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons:
    - no_decoys_generated: 140
  - share_rows_with_n_decoys_gt_0: 0.300 (30.0%)
  - median_slack: 0.000000
  - pass_rate: 0.666667

## value-customer-proxy-2026-01-12-r2

- Source commit: 2bd92404e139804cc3cc088766ede94106962ead
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-customer-proxy-2026-01-12-r2/value_customer_proxy_evidence_pack.zip
- SHA256(value_customer_proxy_evidence_pack.zip): C2A8350EFA0D8BEB957E65DE42C0591080085F614B10E255430774B463F67029
- Acceptance criteria (contract): docs/value_m3_acceptance_criteria.md
- Command:
  python scripts/pilot_generate_input.py --out_dir out_value_customer_proxy --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  (scores variant) proxy random (seed=0): out_value_customer_proxy/scores_proxy.json
  hetero2-batch --input out_value_customer_proxy/input.csv --out_dir out_value_customer_proxy --artifacts light --score_mode external_scores --scores_input out_value_customer_proxy/scores_proxy.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
- Outcome (facts from summary.csv):
  - rows_total: 200
  - rows_ok: 60
  - scores_coverage.rows_missing_scores_input: 0
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons:
    - no_decoys_generated: 140
  - share_rows_with_n_decoys_gt_0: 0.300 (30.0%)
  - median_slack: 0.000000
  - pass_rate: 0.666667

## ci-auto-pr-verify-2026-01-13

- Source commit: 53a61417fb1b07d81506cdc539fd797151af805c
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/ci-auto-pr-verify-2026-01-13/pilot_evidence_pack.zip
- SHA256(pilot_evidence_pack.zip): CC0FEA7239BF7367036279758CCF256D9BFF5D6F0C3B7F77A82E77215A7B9D87
- Command:
  python scripts/pilot_generate_input.py --out_dir out_pilot --rows 1000 --k_decoys 2 --seed 0
  hetero2-batch --input out_pilot/input.csv --out_dir out_pilot --artifacts light --score_mode external_scores --scores_input out_pilot/scores.json --k_decoys 2 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack

## ci-auto-pr-verify-2026-01-13-r3

- Source commit: 2be5db15056a5149732a1f96a57a9937c6f0fc3d
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/ci-auto-pr-verify-2026-01-13-r3/pilot_evidence_pack.zip
- SHA256(pilot_evidence_pack.zip): 5C5CCAD4C1A4CCEC46BB87244D33152689124C15E13ECCF39725F592832B99D5
- Command:
  python scripts/pilot_generate_input.py --out_dir out_pilot --rows 1000 --k_decoys 2 --seed 0
  hetero2-batch --input out_pilot/input.csv --out_dir out_pilot --artifacts light --score_mode external_scores --scores_input out_pilot/scores.json --k_decoys 2 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack

## value-utility-proxy-2026-01-13

- Source commit: 97d1e2e24b31defded76bf74618409eb611d92bc
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-proxy-2026-01-13/value_utility_proxy_evidence_pack.zip
- SHA256(value_utility_proxy_evidence_pack.zip): C1AFC8992DDB88B3528030395D8D1E69DB395C7EE89AA5B902EC300A761A3FD4
- Truth contract: docs/contracts/customer_truth.v1.md
- Utility report contract: docs/contracts/cost_lift.v1.md
- Command:
  python scripts/pilot_generate_input.py --out_dir out_value_utility_proxy --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  (scores variant) proxy random (seed=0): out_value_utility_proxy/scores_proxy.json
  python scripts/generate_proxy_truth.py --input_csv out_value_utility_proxy/input.csv --out_csv out_value_utility_proxy/truth.csv
  hetero2-batch --input out_value_utility_proxy/input.csv --out_dir out_value_utility_proxy --artifacts light --score_mode external_scores --scores_input out_value_utility_proxy/scores_proxy.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --no_manifest
  python scripts/cost_lift.py --summary_csv out_value_utility_proxy/summary.csv --truth_csv out_value_utility_proxy/truth.csv --k 10000 --seed 0 --skip_policy unknown_bucket --out out_value_utility_proxy/cost_lift_report.json --bootstrap_n 500
- Outcome (facts from summary.csv + cost_lift_report.json):
  - rows_total: 200
  - rows_ok: 60
  - scores_coverage.rows_missing_scores_input: 0
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons:
    - no_decoys_generated: 140
  - coverage_ok_rate: 0.300000
  - share_rows_with_n_decoys_gt_0: 0.300000
  - utility (cost_lift.v1):
    - truth_source: proxy_rule_v1
    - truth_schema: customer_truth.v1
    - skip_policy: unknown_bucket
    - selection_K_requested: 10000
    - selection_K_effective: 60
    - baseline_random_hit_rate: 0.333333 (ci: 0.216667..0.450000)
    - baseline_score_only_hit_rate: 0.333333 (ci: 0.216667..0.433333)
    - filtered_hit_rate: 0.500000 (ci: 0.350000..0.675000)
    - uplift_vs_random: 0.166667
    - uplift_vs_score_only: 0.166667

## value-utility-realtruth-2026-01-14-r1

- Source commit: 72720901439cc5f3e2b559f5e606568a8d40bece
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-14-r1/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): 65A00E8879B9B03BF558F630C85EABFC0C285C1B8DF3635D231B4A90DD7D816B
- Truth contract: docs/contracts/customer_truth.v1.md
- Utility report contract: docs/contracts/cost_lift.v1.md
- Command:
  python scripts/pilot_generate_input.py --out_dir out_value_utility_realtruth --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  (scores variant) random proxy (seed=0): out_value_utility_realtruth/scores_proxy.json
  (external truth) truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  (external truth) truth_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  hetero2-batch --input out_value_utility_realtruth/input.csv --out_dir out_value_utility_realtruth --artifacts light --score_mode external_scores --scores_input out_value_utility_realtruth/scores_proxy.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --no_manifest
  python scripts/cost_lift.py --summary_csv out_value_utility_realtruth/summary.csv --truth_csv out_value_utility_realtruth/truth.csv --k 10000 --seed 0 --skip_policy unknown_bucket --out out_value_utility_realtruth/cost_lift_report.json --bootstrap_n 500
- Outcome (facts from summary.csv + cost_lift_report.json):
  - rows_total: 200
  - rows_ok: 60
  - scores_coverage.rows_missing_scores_input: 0
  - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  - truth_csv_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons:
    - no_decoys_generated: 140
  - coverage_ok_rate: 0.300000
  - share_rows_with_n_decoys_gt_0: 0.300000
  - utility (cost_lift.v1):
    - truth_source: external
    - truth_schema: customer_truth.v1
    - skip_policy: unknown_bucket
    - selection_K_requested: 10000
    - selection_K_effective: 60
    - baseline_random_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - baseline_score_only_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - filtered_hit_rate: 0.075000 (ci: 0.000000..0.150000)
    - uplift_vs_random: 0.008333
    - uplift_vs_score_only: 0.008333

## value-utility-realtruth-2026-01-14-r2

- Source commit: 4b89a5a464bdc5e547649dd610ee8af24b250368
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-14-r2/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): 18E54A8CDE6550DCE9E965711189331E8D0F05DA44C6A4CAB5A5A8FEED64D5B9
- Truth contract: docs/contracts/customer_truth.v1.md
- Utility report contract: docs/contracts/cost_lift.v1.md
- Command:
  python scripts/pilot_generate_input.py --out_dir out_value_utility_realtruth --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  (scores input) scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340466779
  (scores input) scores_sha256: 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  (scores input) scores_input_file: out_value_utility_realtruth/scores_external.json
  (external truth) truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  (external truth) truth_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  hetero2-batch --input out_value_utility_realtruth/input.csv --out_dir out_value_utility_realtruth --artifacts light --score_mode external_scores --scores_input out_value_utility_realtruth/scores_external.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --no_manifest
  python scripts/cost_lift.py --summary_csv out_value_utility_realtruth/summary.csv --truth_csv out_value_utility_realtruth/truth.csv --k 10000 --seed 0 --skip_policy unknown_bucket --out out_value_utility_realtruth/cost_lift_report.json --bootstrap_n 500
- Outcome (facts from summary.csv + cost_lift_report.json):
  - rows_total: 200
  - rows_ok: 60
  - scores_coverage.rows_missing_scores_input: 0
  - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  - truth_csv_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  - scores_source: external
  - scores_input_file: scores_external.json
  - scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340466779
  - scores_sha256_expected: 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  - scores_json_sha256: 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  - scores_schema_version: hetero_scores.v1
  - score_key: external_ci_rule_v1
  - status_counts: OK=60, SKIP=140, ERROR=0
  - top_skip_reasons:
    - no_decoys_generated: 140
  - coverage_ok_rate: 0.300000
  - share_rows_with_n_decoys_gt_0: 0.300000
  - utility (cost_lift.v1):
    - truth_source: external
    - truth_schema: customer_truth.v1
    - skip_policy: unknown_bucket
    - selection_K_requested: 10000
    - selection_K_effective: 60
    - eligibility (K_effective decomposition; from cost_lift_report.json):
      - rows_total: 200
      - rows_ok: 200
      - rows_truth_known: 200
      - rows_scores_present: 200
      - rows_verdict_pass_fail: 60
      - rows_eligible_for_cost_lift: 60
      - K_effective_reason_top:
        - verdict_not_pass_fail: 140 (share_total: 0.700000)
    - baseline_random_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - baseline_score_only_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - filtered_hit_rate: 0.066667 (ci: 0.016667..0.150000)
    - uplift_vs_random: 0.000000
    - uplift_vs_score_only: 0.000000
  - comparison_vs_value-utility-realtruth-2026-01-17-r1 (from r1 summary.csv + r2 cost_lift_report.json):
    - selection_K_requested: 10000 -> 10000
    - selection_K_effective: 60 -> 60
    - limiting_reason (r1/r2): verdict_not_pass_fail=140 (OK rows with verdict=SKIP => gate/slack empty)
  - comparison_vs_value-utility-realtruth-2026-01-14-r2 (same pinned truth+scores; pre-coverage):
    - status_counts: OK=60, SKIP=140, ERROR=0 -> OK=200, SKIP=0, ERROR=0
    - top_skip_reasons: no_decoys_generated: 140 -> (none)
    - coverage_ok_rate: 0.300000 -> 1.000000
    - share_rows_with_n_decoys_gt_0: 0.300000 -> 1.000000
    - selection_K_effective: 60 -> 60

## value-utility-realtruth-2026-01-14-r3

- Source commit: cdc081b4d18b3e3f1d63d7f6ac335e4fb0f8d437
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-14-r3/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): 704B1F82933799E65F5F33D982A0D3EEBC34DA06BE8001670500B869CE3C5A00
- Truth contract: docs/contracts/customer_truth.v1.md
- Utility report contract: docs/contracts/cost_lift.v1.md
- Command:
  python scripts/pilot_generate_input.py --out_dir out_value_utility_realtruth --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  (scores input) scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340539600
  (scores input) scores_sha256: D22A19B51EAEBDE1A778B2FE69E10F9E78BA726F64CBF9A643ADD235D167D157
  (scores input) scores_input_file: out_value_utility_realtruth/scores_external.json
  (external truth) truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  (external truth) truth_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  hetero2-batch --input out_value_utility_realtruth/input.csv --out_dir out_value_utility_realtruth --artifacts light --score_mode external_scores --scores_input out_value_utility_realtruth/scores_external.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --no_manifest
  python scripts/cost_lift.py --summary_csv out_value_utility_realtruth/summary.csv --truth_csv out_value_utility_realtruth/truth.csv --k 10000 --seed 0 --skip_policy unknown_bucket --out out_value_utility_realtruth/cost_lift_report.json --bootstrap_n 500
- Outcome (facts from summary.csv + cost_lift_report.json):
  - rows_total: 200
  - rows_ok: 200
  - scores_coverage.rows_missing_scores_input: 0
  - request-pack (missing decoy scores):
    - file: missing_decoy_scores.csv (decoy_hash, decoy_smiles, count_rows_affected)
    - unique_missing_decoy_hashes: 32
    - missing_decoy_hashes_top10 (count_rows_affected):
      - 16838e384a1ad07ba566eb1ba546b792f81ef008cadf8ca5bd4a150eb4278e04: 20
      - 261f06aef3a079146fbf5c640ed6255634398d2403c8f8a28cc1dceedf829701: 20
      - 27d534891fdf401b06112f6782b5ccb19d330981222d665690788615e6ea60ff: 20
      - 2aaa81cdce6a13e8fcb6c93aee12980032c0631a0aa903cc28fcde1b3bcd2620: 20
      - 2e792df3a5423e877b28ed8151dfa7ec236cb31dbfa40c9fa87bd7bd0907a56d: 20
      - 2e893b8a311f1d397c0ad45bfccaea754b3fa4f6e28408117b2f911590c7d1e7: 20
      - 38b6ff5412a0dbf382f7d62c25f39295b686be99ccc2e454316a32258e2fd8f3: 20
      - 39b7b2271fc96bba370e79b12892eec6a454be9ab1092def75846dc16f0c7da6: 20
      - 41e029556ec97be0b90edd439ae810406f0b79c4657087ccfee27be8f69876b5: 20
      - 4da1a135c98d550c0d25c7247de70a24a4516ae4a2fdb1b49941c63f54b94bdb: 20
  - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  - truth_csv_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  - scores_source: external
  - scores_input_file: scores_external.json
  - scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340539600
  - scores_sha256_expected: D22A19B51EAEBDE1A778B2FE69E10F9E78BA726F64CBF9A643ADD235D167D157
  - scores_json_sha256: D22A19B51EAEBDE1A778B2FE69E10F9E78BA726F64CBF9A643ADD235D167D157
  - scores_schema_version: hetero_scores.v1
  - score_key: external_ci_rule_v2
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - coverage_ok_rate: 1.000000
  - share_rows_with_n_decoys_gt_0: 1.000000
  - scores_coverage (from cost_lift_report.json; summary.csv n_decoys_scored/n_decoys_generated):
    - rows_total: 200
    - rows_with_decoys_scored_gt0: 60
    - rows_with_decoys_scored_eq0: 140
    - decoys_scored_total: 240
    - decoys_missing_total: 640
  - comparison vs value-utility-realtruth-2026-01-17-r2:
    - rows_with_decoys_scored_eq0: 140 -> 140
    - selection_K_effective: 60 -> 60
  - utility (cost_lift.v1):
    - truth_source: external
    - truth_schema: customer_truth.v1
    - skip_policy: unknown_bucket
    - selection_K_requested: 10000
    - selection_K_effective: 200
    - baseline_random_hit_rate: 0.055000 (ci: 0.025000..0.085000)
    - baseline_score_only_hit_rate: 0.055000 (ci: 0.025000..0.085000)
    - filtered_hit_rate: 0.055000 (ci: 0.025000..0.085000)
    - uplift_vs_random: 0.000000
    - uplift_vs_score_only: 0.000000
  - scores_coverage (from cost_lift_report.json):
    - rows_with_decoys_scored_eq0: 0
    - rows_with_decoys_scored_gt0: 200
    - decoys_scored_total: 880
    - decoys_missing_total: 0
  - comparison_vs_value-utility-realtruth-2026-01-17-r4 (same pinned truth; refreshed scores):
    - scores_coverage.unique_missing_decoy_hashes: 32 -> 0
    - rows_with_decoys_scored_eq0: 140 -> 0
    - decoys_missing_total: 640 -> 0
    - selection_K_effective: 60 -> 200

## value-ring-suite-2026-01-17

- Source commit: 7a7085e77faa8459295c625d9b413529a185e360
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-ring-suite-2026-01-17/value_ring_suite_evidence_pack.zip
- SHA256(value_ring_suite_evidence_pack.zip): 3F117FC323E2D727E83874EA97A6F3011181AEEDC2D041C9D0CA9B9EF2FE4B69
- Command:
  python scripts/pilot_generate_input.py --out_dir out_ring_suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  hetero2-batch --input out_ring_suite/input.csv --out_dir out_ring_suite --artifacts light --score_mode external_scores --scores_input out_ring_suite/scores.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
- Outcome (facts from summary.csv):
  - rows_total: 200
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - share_rows_with_n_decoys_gt_0: 1.000 (100.0%)
  - decoy_realism (hardness curve, Morgan Tanimoto bins):
    - tanimoto_median: 0.119643
    - pairs_total: 880 (easy=760, medium=120, hard=0)
    - share_pairs: easy=0.863636, medium=0.136364, hard=0.000000
    - pairs_scored: 380 (easy=280, medium=100, hard=0)
    - auc_tie_aware: easy=1.000000, medium=1.000000, hard=N/A
    - auc_interpretation: INCONCLUSIVE (decoys_too_easy)
    - evidence_pack_files: hardness_curve.csv, hardness_curve.md, operator_features.csv
  - comparison_vs_value-ring-suite-2026-01-10:
    - status_counts: OK=60, SKIP=140, ERROR=0 -> OK=200, SKIP=0, ERROR=0
    - top_skip_reasons: no_decoys_generated: 140 -> (none)
    - share_rows_with_n_decoys_gt_0: 0.300 (30.0%) -> 1.000 (100.0%)

## stress-10k-2026-01-17

- Source commit: 7a7085e77faa8459295c625d9b413529a185e360
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/stress-10k-2026-01-17/evidence_pack.zip
- SHA256(evidence_pack.zip): A41ED16B2910A2BF58EEBDE5B9637778DA319202EBB71BBD9CBF663A98CED602
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
- Outcome (facts from summary.csv):
  - rows_total: 10000
  - status_counts: OK=1512, SKIP=8488, ERROR=0
  - top_skip_reasons:
    - no_decoys_generated: 7488
    - too_many_atoms: 500
    - disconnected: 250
    - invalid_smiles: 208
    - missing_smiles: 42
  - ok_rate: 0.1512
  - comparison_vs_stress-10k-2026-01-07 (computed from summary.csv inside evidence_pack.zip):
    - status_counts: OK=0, SKIP=10000, ERROR=0 -> OK=1512, SKIP=8488, ERROR=0
    - no_decoys_generated: 9000 -> 7488
    - ok_rate: 0.0000 -> 0.1512

## value-utility-realtruth-2026-01-17-r1

- Source commit: 0673c7d44192fd591a5910c7352d1c37aa1718d4
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-17-r1/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): 43B489FBFBF813AB0FE5E62FC9038649A9DD3C5A75D9D038241DE6800FACFF1F
- Truth contract: docs/contracts/customer_truth.v1.md
- Utility report contract: docs/contracts/cost_lift.v1.md
- Command:
  python scripts/pilot_generate_input.py --out_dir out_value_utility_realtruth --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  (scores input) scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340466779
  (scores input) scores_sha256: 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  (scores input) scores_input_file: out_value_utility_realtruth/scores_external.json
  (external truth) truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  (external truth) truth_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  hetero2-batch --input out_value_utility_realtruth/input.csv --out_dir out_value_utility_realtruth --artifacts light --score_mode external_scores --scores_input out_value_utility_realtruth/scores_external.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --no_manifest
  python scripts/cost_lift.py --summary_csv out_value_utility_realtruth/summary.csv --truth_csv out_value_utility_realtruth/truth.csv --k 10000 --seed 0 --skip_policy unknown_bucket --out out_value_utility_realtruth/cost_lift_report.json --bootstrap_n 500
- Outcome (facts from summary.csv + cost_lift_report.json):
  - rows_total: 200
  - rows_ok: 200
  - scores_coverage.rows_missing_scores_input: 0
  - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  - truth_csv_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  - scores_source: external
  - scores_input_file: scores_external.json
  - scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340466779
  - scores_sha256_expected: 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  - scores_json_sha256: 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  - scores_schema_version: hetero_scores.v1
  - score_key: external_ci_rule_v1
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - coverage_ok_rate: 1.000000
  - share_rows_with_n_decoys_gt_0: 1.000000
  - utility (cost_lift.v1):
    - truth_source: external
    - truth_schema: customer_truth.v1
    - skip_policy: unknown_bucket
    - selection_K_requested: 10000
    - selection_K_effective: 60
    - baseline_random_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - baseline_score_only_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - filtered_hit_rate: 0.066667 (ci: 0.016667..0.150000)
    - uplift_vs_random: 0.000000
    - uplift_vs_score_only: 0.000000

## value-utility-realtruth-2026-01-17-r2

- Source commit: 2a37d0aab5c3f7f66ac340bfa816966d377b45d3
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-17-r2/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): 1AE60548048E02B321FDE969B8540A88BE1B8D3B34C66CF23A125946E0D60785
- Truth contract: docs/contracts/customer_truth.v1.md
- Utility report contract: docs/contracts/cost_lift.v1.md
- Command:
  python scripts/pilot_generate_input.py --out_dir out_value_utility_realtruth --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  (scores input) scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340466779
  (scores input) scores_sha256: 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  (scores input) scores_input_file: out_value_utility_realtruth/scores_external.json
  (external truth) truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  (external truth) truth_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  hetero2-batch --input out_value_utility_realtruth/input.csv --out_dir out_value_utility_realtruth --artifacts light --score_mode external_scores --scores_input out_value_utility_realtruth/scores_external.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --no_manifest
  python scripts/cost_lift.py --summary_csv out_value_utility_realtruth/summary.csv --truth_csv out_value_utility_realtruth/truth.csv --k 10000 --seed 0 --skip_policy unknown_bucket --out out_value_utility_realtruth/cost_lift_report.json --bootstrap_n 500
- Outcome (facts from summary.csv + cost_lift_report.json):
  - rows_total: 200
  - rows_ok: 200
  - scores_coverage.rows_missing_scores_input: 0
  - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  - truth_csv_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  - scores_source: external
  - scores_input_file: scores_external.json
  - scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340466779
  - scores_sha256_expected: 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  - scores_json_sha256: 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  - scores_schema_version: hetero_scores.v1
  - score_key: external_ci_rule_v1
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - coverage_ok_rate: 1.000000
  - share_rows_with_n_decoys_gt_0: 1.000000
  - utility (cost_lift.v1):
    - truth_source: external
    - truth_schema: customer_truth.v1
    - skip_policy: unknown_bucket
    - selection_K_requested: 10000
    - selection_K_effective: 60
    - baseline_random_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - baseline_score_only_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - filtered_hit_rate: 0.066667 (ci: 0.016667..0.150000)
    - uplift_vs_random: 0.000000
    - uplift_vs_score_only: 0.000000

## value-utility-realtruth-2026-01-17-r3

- Source commit: dd0af1f9c36297a196ac5df3472c07dcd6c7df6a
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-17-r3/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): 815A9E3E1D8BBBE6BB16801A3BBC27C2CCD10E93D40168D10DD4A882C84B5236
- Truth contract: docs/contracts/customer_truth.v1.md
- Utility report contract: docs/contracts/cost_lift.v1.md
- Command:
  python scripts/pilot_generate_input.py --out_dir out_value_utility_realtruth --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  (scores input) scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340466779
  (scores input) scores_sha256: 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  (scores input) scores_input_file: out_value_utility_realtruth/scores_external.json
  (external truth) truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  (external truth) truth_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  hetero2-batch --input out_value_utility_realtruth/input.csv --out_dir out_value_utility_realtruth --artifacts light --score_mode external_scores --scores_input out_value_utility_realtruth/scores_external.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --no_manifest
  python scripts/cost_lift.py --summary_csv out_value_utility_realtruth/summary.csv --truth_csv out_value_utility_realtruth/truth.csv --k 10000 --seed 0 --skip_policy unknown_bucket --out out_value_utility_realtruth/cost_lift_report.json --bootstrap_n 500
- Outcome (facts from summary.csv + cost_lift_report.json):
  - rows_total: 200
  - rows_ok: 200
  - scores_coverage.rows_missing_scores_input: 0
  - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  - truth_csv_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  - scores_source: external
  - scores_input_file: scores_external.json
  - scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340466779
  - scores_sha256_expected: 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  - scores_json_sha256: 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  - scores_schema_version: hetero_scores.v1
  - score_key: external_ci_rule_v1
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - coverage_ok_rate: 1.000000
  - share_rows_with_n_decoys_gt_0: 1.000000
  - utility (cost_lift.v1):
    - truth_source: external
    - truth_schema: customer_truth.v1
    - skip_policy: unknown_bucket
    - selection_K_requested: 10000
    - selection_K_effective: 60
    - baseline_random_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - baseline_score_only_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - filtered_hit_rate: 0.066667 (ci: 0.016667..0.150000)
    - uplift_vs_random: 0.000000
    - uplift_vs_score_only: 0.000000

## value-utility-realtruth-2026-01-17-r4

- Source commit: 8d5a38c2dcc71a25dbcab6c9e00929d679e2018a
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-17-r4/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): CEA2E0599355CC5A31CA4B2318EC707AF85BE4298196E2AEB672F32C9A9C29AA
- Truth contract: docs/contracts/customer_truth.v1.md
- Utility report contract: docs/contracts/cost_lift.v1.md
- Command:
  python scripts/pilot_generate_input.py --out_dir out_value_utility_realtruth --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  (scores input) scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340466779
  (scores input) scores_sha256: 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  (scores input) scores_input_file: out_value_utility_realtruth/scores_external.json
  (external truth) truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  (external truth) truth_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  hetero2-batch --input out_value_utility_realtruth/input.csv --out_dir out_value_utility_realtruth --artifacts light --score_mode external_scores --scores_input out_value_utility_realtruth/scores_external.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --no_manifest
  python scripts/cost_lift.py --summary_csv out_value_utility_realtruth/summary.csv --truth_csv out_value_utility_realtruth/truth.csv --k 10000 --seed 0 --skip_policy unknown_bucket --out out_value_utility_realtruth/cost_lift_report.json --bootstrap_n 500
- Outcome (facts from summary.csv + cost_lift_report.json):
  - rows_total: 200
  - rows_ok: 200
  - scores_coverage.rows_missing_scores_input: 0
  - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  - truth_csv_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  - scores_source: external
  - scores_input_file: scores_external.json
  - scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340466779
  - scores_sha256_expected: 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  - scores_json_sha256: 19F08F234C438515A37B6CB0B95040C74191BC2C383EAFD6CF6EFF9B26A3F168
  - scores_schema_version: hetero_scores.v1
  - score_key: external_ci_rule_v1
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - coverage_ok_rate: 1.000000
  - share_rows_with_n_decoys_gt_0: 1.000000
  - utility (cost_lift.v1):
    - truth_source: external
    - truth_schema: customer_truth.v1
    - skip_policy: unknown_bucket
    - selection_K_requested: 10000
    - selection_K_effective: 60
    - baseline_random_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - baseline_score_only_hit_rate: 0.066667 (ci: 0.016667..0.133333)
    - filtered_hit_rate: 0.066667 (ci: 0.016667..0.150000)
    - uplift_vs_random: 0.000000
    - uplift_vs_score_only: 0.000000

## value-utility-realtruth-2026-01-17-r5

- Source commit: d74e46ee23e98d49fbc8a37bcae32fdacbcf49ec
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-utility-realtruth-2026-01-17-r5/value_utility_realtruth_evidence_pack.zip
- SHA256(value_utility_realtruth_evidence_pack.zip): A637058199BBD523B69C95680BAF0D7D768293CBCE1FEAC7237F6478F1304BB1
- Truth contract: docs/contracts/customer_truth.v1.md
- Utility report contract: docs/contracts/cost_lift.v1.md
- Command:
  python scripts/pilot_generate_input.py --out_dir out_value_utility_realtruth --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  (scores input) scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/342035099
  (scores input) scores_sha256: E3A00B22B7419D87DE059E66045C8466F8871FBE8380D7A4EC4F3F6B4CCA87C0
  (scores input) scores_input_file: out_value_utility_realtruth/scores_external.json
  (external truth) truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  (external truth) truth_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  hetero2-batch --input out_value_utility_realtruth/input.csv --out_dir out_value_utility_realtruth --artifacts light --score_mode external_scores --scores_input out_value_utility_realtruth/scores_external.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --no_manifest
  python scripts/cost_lift.py --summary_csv out_value_utility_realtruth/summary.csv --truth_csv out_value_utility_realtruth/truth.csv --k 10000 --seed 0 --skip_policy unknown_bucket --out out_value_utility_realtruth/cost_lift_report.json --bootstrap_n 500
- Outcome (facts from summary.csv + cost_lift_report.json):
  - rows_total: 200
  - rows_ok: 200
  - scores_coverage.rows_missing_scores_input: 0
  - truth_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/340297819
  - truth_csv_sha256: 1403593FC0497E19CA2A8DD78B5BC6DEE88790F809AED0FA47F6F0744198C2A2
  - scores_source: external
  - scores_input_file: scores_external.json
  - scores_url: https://api.github.com/repos/RobertPaulig/Geometric_table/releases/assets/342035099
  - scores_sha256_expected: E3A00B22B7419D87DE059E66045C8466F8871FBE8380D7A4EC4F3F6B4CCA87C0
  - scores_json_sha256: E3A00B22B7419D87DE059E66045C8466F8871FBE8380D7A4EC4F3F6B4CCA87C0
  - scores_schema_version: hetero_scores.v1
  - score_key: external_ci_rule_v1
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - coverage_ok_rate: 1.000000
  - share_rows_with_n_decoys_gt_0: 1.000000
  - utility (cost_lift.v1):
    - truth_source: external
    - truth_schema: customer_truth.v1
    - skip_policy: unknown_bucket
    - selection_K_requested: 10000
    - selection_K_effective: 200
    - baseline_random_hit_rate: 0.055000 (ci: 0.025000..0.085000)
    - baseline_score_only_hit_rate: 0.055000 (ci: 0.025000..0.085000)
    - filtered_hit_rate: 0.055000 (ci: 0.025000..0.085000)
    - uplift_vs_random: 0.000000
    - uplift_vs_score_only: 0.000000

## value-ring-suite-2026-01-18

- Source commit: 64ab419118e85229760a0935a43ec2f05ac4a839
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-ring-suite-2026-01-18/value_ring_suite_evidence_pack.zip
- SHA256(value_ring_suite_evidence_pack.zip): 3BFB1865AE6C6A0163F8F729E7B9BBFAF61B96D8099BC7E9F8B35C0A6B3D0030
- Command:
  python scripts/pilot_generate_input.py --out_dir out_ring_suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  hetero2-batch --input out_ring_suite/input.csv --out_dir out_ring_suite --artifacts light --score_mode external_scores --scores_input out_ring_suite/scores.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
- Outcome (facts from summary.csv):
  - rows_total: 200
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - share_rows_with_n_decoys_gt_0: 1.000 (100.0%)

## physics-operator-rails-2026-01-18-r1

- Source commit: 2a9bd703abee100d9ee0fdafe3e89acc42c1316f
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-rails-2026-01-18-r1/evidence_pack.zip
- SHA256(evidence_pack.zip): 20980360782EBE926F1F4E448369D8F692431059EE3863489EE3AD27805773D1
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack

## physics-operator-weights-2026-01-18-r1

- Source commit: 80fbe9895609b8c43a758c301bd933c4e87ef38f
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-weights-2026-01-18-r1/evidence_pack.zip
- SHA256(evidence_pack.zip): 388FA597A852B0CC881136B0A45FA089CE2797E2E0E6BACDB7B3FA47D9158F4F
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --physics_mode both --edge_weight_mode bond_order_delta_chi --zip_pack

## physics-operator-ldos-2026-01-18-r1

- Source commit: 8a5a1fe2aecfee07942493bd815a4a65a7252f8e
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-ldos-2026-01-18-r1/evidence_pack.zip
- SHA256(evidence_pack.zip): 6589FA29A9ABC3BA3CD65446EA45ABA2033C663F6AF61AC38B01BFEEAC00C652
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --physics_mode both --edge_weight_mode unweighted --zip_pack
- DOS/LDOS artifacts (from summary_metadata.json in evidence_pack.zip):
  - dos_ldos_schema: hetero2_dos_ldos.v1
  - dos_grid_n: 128
  - dos_eta: 0.05
  - dos_energy_min: -0.15
  - dos_energy_max: 4.15
  - evidence_pack_files: dos_curve.csv, ldos_summary.csv, summary_metadata.json

## physics-operator-scf-2026-01-19-r1

- Source commit: 8e9ab74dc1a44c91c09d082c03c314a448ed9a02
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-scf-2026-01-19-r1/evidence_pack.zip
- SHA256(evidence_pack.zip): ADFFF664035F103661E011FE5EF8FB490D4A48449BCC8BA101B9D71BB17061A4
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --physics_mode both --edge_weight_mode bond_order_delta_chi --potential_mode both --scf_max_iter 50 --scf_tol 1e-6 --scf_damping 0.5 --scf_occ_k 5 --scf_tau 1.0 --zip_pack
- SCF artifacts (from summary_metadata.json in evidence_pack.zip):
  - scf_schema: hetero2_scf.v1
  - potential_mode: both
  - scf_converged: true
  - scf_rows_total: 1512
  - scf_rows_converged: 1512
  - scf_iters: 1
  - scf_residual_final: 3.469446951953614e-17
  - evidence_pack_files: scf_trace.csv, potential_vectors.csv, summary_metadata.json

## physics-operator-units-gamma-2026-01-19-r1

- Source commit: 587fd8f816d4e66cc2569815f4210ca972bc3525
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-units-gamma-2026-01-19-r1/evidence_pack.zip
- SHA256(evidence_pack.zip): 6529B5356345576F858D333F71737332D049EB745A6191AD6AD10775A93FA4BD
- potential_unit_model: dimensionless
- potential_scale_gamma: 1.0
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --physics_mode both --edge_weight_mode unweighted --potential_mode both --scf_max_iter 50 --scf_tol 1e-6 --scf_damping 0.5 --scf_occ_k 5 --scf_tau 1.0 --zip_pack

## physics-operator-scf-2026-01-19-r2

- Source commit: dfc5b72301bfdaea4e5d5c7834c8e0202c1dfddf
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-scf-2026-01-19-r2/evidence_pack.zip
- SHA256(evidence_pack.zip): E89E6E99972840A1900C776529065C2009EF87D5A386545289DA15C71F020179
- potential_unit_model: dimensionless
- potential_scale_gamma: 1.0
- scf_enabled: true
- scf_status: CONVERGED
- scf_rows_total: 1512
- scf_rows_converged: 1512
- scf_iters_max: 1
- scf_residual_final_max: 3.469446951953614e-17
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --physics_mode both --edge_weight_mode bond_order_delta_chi --potential_mode both --scf_max_iter 50 --scf_tol 1e-6 --scf_damping 0.5 --scf_occ_k 5 --scf_tau 1.0 --zip_pack

## physics-operator-scf-audit-2026-01-19-r1

- Source commit: 69ac40d21eaa7bc22fc77d9d99ec139da4283ca7
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-scf-audit-2026-01-19-r1/evidence_pack.zip
- SHA256(evidence_pack.zip): 01E630207A05DE92DB405DDB3248061F634691E9592C15AA7DDE42A22D158B21
- Input CSV asset (asym fixture set): https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-scf-audit-2026-01-19-r1/scf_audit.csv
- Command:
  hetero2-batch --input scf_audit.csv --out_dir out_scf_audit --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --physics_mode both --edge_weight_mode bond_order_delta_chi --potential_mode both --potential_scale_gamma 1.0 --scf_max_iter 50 --scf_tol 1e-6 --scf_damping 0.5 --scf_occ_k 5 --scf_tau 1.0 --zip_pack
- SCF audit (from summary_metadata.json / scf_summary.json in evidence_pack.zip):
  - scf_audit_verdict: SUCCESS
  - scf_audit_reason: nontrivial_on_asym_fixture
  - stats_asym_fixture:
    - rows_with_scf: 12
    - iters_median: 15.0
    - iters_mean: 14.75
    - iters_max: 15
    - converged_rate: 1.0
    - delta_V_p95: 0.0880012216377013
    - residual_init_mean: 0.037387575359043745
    - residual_final_mean: 7.321790598292799e-07
  - evidence_pack_files: scf_trace.csv, scf_summary.json, potential_vectors.csv, summary_metadata.json

## physics-operator-scf-audit-2026-01-19-r2

- Source commit: 220f00773a6e87a3af642ebb3c6e4eb35ebd0042
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-scf-audit-2026-01-19-r2/evidence_pack.zip
- SHA256(evidence_pack.zip): 3409C1137495CE450D22D6D23C4EB174FDF696EE317712124CB5248F5C18BD7E
- Input CSV asset (asym fixture set): https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-scf-audit-2026-01-19-r2/scf_audit.csv
- Command:
  hetero2-batch --input scf_audit.csv --out_dir out_scf_audit --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --physics_mode both --edge_weight_mode bond_order_delta_chi --potential_mode both --potential_scale_gamma 1.0 --scf_max_iter 50 --scf_tol 1e-6 --scf_damping 0.5 --scf_occ_k 5 --scf_tau 1.0 --zip_pack
- SCF audit (from summary_metadata.json in evidence_pack.zip):
  - scf_audit_verdict: SUCCESS
  - scf_audit_reason: nontrivial_rate_ge_0p50_and_converged_rate_ge_0p95
  - scf_converged_rate: 1.0
  - scf_nontrivial_rate: 1.0
  - scf_iters_mean/p95/max: 14.75 / 15.0 / 15
  - deltaV_max_max: 0.1567902269999999
  - residual_final_p95: 9.73477578228299e-07
  - evidence_pack_files: scf_audit_metrics.csv, scf_trace.csv, scf_summary.json, potential_vectors.csv, summary_metadata.json

## physics-operator-integration-baseline-2026-01-19-r1

- Source commit: eb0fa851d685f79d561f92f439fac7c3000cd1c9
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-integration-baseline-2026-01-19-r1/evidence_pack.zip
- SHA256(evidence_pack.zip): C07DC02484C1EB75751A6CEE3BE83C82664E51980DAC9192DA25BEEC95F6140B
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --physics_mode both --edge_weight_mode unweighted --potential_mode static --scf_max_iter 50 --scf_tol 1e-6 --scf_damping 0.5 --scf_occ_k 5 --scf_tau 1.0 --zip_pack
- Integration benchmark (from summary_metadata.json in evidence_pack.zip):
  - integrator_mode: baseline
  - energy_range: [-0.15, 4.15]
  - energy_points: 128
  - eta: 0.05
  - integrator_eps: 1e-06
  - integration_walltime_ms_median: 2.3876529999995455
  - evidence_pack_files: integration_benchmark.csv, integration_benchmark.md, summary_metadata.json

## physics-operator-integration-adaptive-2026-01-19-r1

- Source commit: 19b21d980e968525783e635b87689aa854403128
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-integration-adaptive-2026-01-19-r1/evidence_pack.zip
- SHA256(evidence_pack.zip): 6C2F3ED358220DB60F725AF202B310051CD9F901FC454420F3B7A9FD08464C0C
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --physics_mode topological --edge_weight_mode unweighted --potential_mode static --scf_max_iter 50 --scf_tol 1e-6 --scf_damping 0.5 --scf_occ_k 5 --scf_tau 1.0 --integrator_mode both --integrator_eps_abs 1e-6 --integrator_eps_rel 1e-4 --integrator_subdomains_max 64 --integrator_poly_degree_max 8 --integrator_quad_order_max 16 --integrator_eval_budget_max 4096 --integrator_split_criterion max_abs_error --zip_pack
- Adaptive integration audit (from summary_metadata.json + adaptive_integration_summary.json in evidence_pack.zip):
  - integrator_mode: both
  - integrator_verdict: FAIL_CORRECTNESS
  - integrator_correctness_pass_rate: 0.666667
  - integrator_speedup_target: 1.5
  - integrator_speedup_median: 0.322449
  - integrator_speedup_verdict: N/A_CORRECTNESS_FAILED
  - dos_L_segments_used: 39
  - dos_L_evals_total: 975
  - dos_L_error_est_total: 2.5392529243
  - energy_range: [-0.15, 4.15]
  - energy_points: 128
  - eta: 0.05
  - integrator_eps_abs: 1e-06
  - integrator_eps_rel: 1e-04
  - evidence_pack_files: integration_benchmark.csv, integration_benchmark.md, adaptive_integration_trace.csv, adaptive_integration_summary.json, integration_compare.csv

## physics-operator-integration-adaptive-2026-01-19-r2

- Source commit: 5600faa7debce8647e574ee2f858b3ddd534a9c3
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-integration-adaptive-2026-01-19-r2/evidence_pack.zip
- SHA256(evidence_pack.zip): C76F584884E28CA04EF792D19D5BE2A0B13F3B67FAD526EEA36DD21CED09028C
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --physics_mode topological --edge_weight_mode unweighted --potential_mode static --scf_max_iter 50 --scf_tol 1e-6 --scf_damping 0.5 --scf_occ_k 5 --scf_tau 1.0 --integrator_mode both --integrator_eps_abs 1e-6 --integrator_eps_rel 1e-4 --integrator_subdomains_max 64 --integrator_poly_degree_max 8 --integrator_quad_order_max 16 --integrator_eval_budget_max 4096 --integrator_split_criterion max_abs_error --zip_pack
- Adaptive integration audit (from summary_metadata.json + adaptive_integration_summary.json in evidence_pack.zip):
  - integrator_mode: both
  - integrator_verdict: SUCCESS
  - integrator_correctness_pass_rate: 1.000000
  - integrator_speedup_target: 1.5
  - integrator_speedup_median: 0.231419
  - integrator_speedup_verdict: NO_SPEEDUP_YET
  - dos_L_segments_used: 20
  - dos_L_evals_total: 975
  - dos_L_error_est_total: 0.0000946613
  - energy_range: [-0.15, 4.15]
  - energy_points: 128
  - eta: 0.05
  - integrator_eps_abs: 1e-06
  - integrator_eps_rel: 1e-04
  - evidence_pack_files: integration_benchmark.csv, integration_benchmark.md, adaptive_integration_trace.csv, adaptive_integration_summary.json, integration_compare.csv

## physics-operator-integration-adaptive-speedup-2026-01-19-r1

- Source commit: f969fbd7801353e1a5295a0cf747e4a53efe3790
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-integration-adaptive-speedup-2026-01-19-r1/evidence_pack.zip
- SHA256(evidence_pack.zip): 10157D81032ACBB137D26F916B77860E7331AC41FB50577F128CF70B3E18E8E4
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --physics_mode topological --edge_weight_mode unweighted --potential_mode static --scf_max_iter 50 --scf_tol 1e-6 --scf_damping 0.5 --scf_occ_k 5 --scf_tau 1.0 --integrator_mode both --integrator_eps_abs 1e-6 --integrator_eps_rel 1e-4 --integrator_subdomains_max 64 --integrator_poly_degree_max 32 --integrator_quad_order_max 16 --integrator_eval_budget_max 4096 --integrator_split_criterion max_abs_error --zip_pack

## physics-operator-integration-adaptive-speedup-2026-01-19-r2

- Source commit: 31babd0f08fd62f731bb2af3de2750c2ae8b9e57
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-integration-adaptive-speedup-2026-01-19-r2/evidence_pack.zip
- SHA256(evidence_pack.zip): 46E9F6088A06002D767D36E39F585F5763230A1CB5FD41976DE647D84327C1B4
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --physics_mode topological --edge_weight_mode unweighted --potential_mode static --scf_max_iter 50 --scf_tol 1e-6 --scf_damping 0.5 --scf_occ_k 5 --scf_tau 1.0 --integrator_mode both --integrator_eps_abs 1e-6 --integrator_eps_rel 1e-4 --integrator_subdomains_max 64 --integrator_poly_degree_max 8 --integrator_quad_order_max 16 --integrator_eval_budget_max 4096 --integrator_split_criterion max_abs_error --zip_pack

## physics-operator-integration-adaptive-speedup-2026-01-19-r3

- Source commit: 72ea11636462ac29f39c31ec44ab8723f56df788
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-integration-adaptive-speedup-2026-01-19-r3/evidence_pack.zip
- SHA256(evidence_pack.zip): B0A669402074032C910C8C32B973146DDA710B6172EAF1A3F5DBF69BCF951D61
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --physics_mode both --edge_weight_mode unweighted --potential_mode static --scf_max_iter 50 --scf_tol 1e-6 --scf_damping 0.5 --scf_occ_k 5 --scf_tau 1.0 --integrator_mode both --integrator_eps_abs 1e-6 --integrator_eps_rel 1e-4 --integrator_subdomains_max 64 --integrator_poly_degree_max 8 --integrator_quad_order_max 16 --integrator_eval_budget_max 4096 --integrator_split_criterion max_abs_error --zip_pack

## physics-operator-integration-select-truth-2026-01-20-r1

- Source commit: f2950c205da43a09e95bd0a4de9ebd0c994b6817
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-integration-select-truth-2026-01-20-r1/evidence_pack.zip
- SHA256(evidence_pack.zip): AC47AD2A3BC3D8EDE69CF804D8B3A2B7F5664127E6F8EA69F538D135B9A9AFAA
- Command:
  hetero2-batch --input stress.csv --out_dir out_stress --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --physics_mode both --edge_weight_mode unweighted --potential_mode static --scf_max_iter 50 --scf_tol 1e-6 --scf_damping 0.5 --scf_occ_k 5 --scf_tau 1.0 --integrator_mode both --integrator_eps_abs 1e-6 --integrator_eps_rel 1e-4 --integrator_subdomains_max 64 --integrator_poly_degree_max 8 --integrator_quad_order_max 16 --integrator_eval_budget_max 4096 --integrator_split_criterion max_abs_error --zip_pack

## physics-operator-large-scale-2026-01-20-r1

- Source commit: 77f98b9c5fd892a27c2c8cc929132824b36a4f77
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r1/physics_large_scale_evidence_pack.zip
- SHA256(physics_large_scale_evidence_pack.zip): C651C583D893C37A91E25AFC5D3FD049933E7A8ABA3D6E5AE47E6DB57FFF6653
- Command:
  python scripts/build_p5_large_scale_pack.py --out_dir out_physics_large_scale --n_atoms_bins "20,50,100,200,400,800" --samples_per_bin 10 --seed 0 --curve_id "dos_H" --energy_points 128 --dos_eta 0.2 --potential_scale_gamma 1.0 --edge_weight_mode bond_order_delta_chi --integrator_eps_abs 1e-6 --integrator_eps_rel 1e-4 --integrator_subdomains_max 32 --integrator_poly_degree_max 16 --integrator_quad_order_max 32 --integrator_eval_budget_max 256 --integrator_split_criterion "curvature" --overhead_region_n_max 100 --gate_n_min 200 --speedup_gate_break_even 1.0 --speedup_gate_strong 2.0
- Outcome (facts from summary_metadata.json):
  - scale (P5):
    - scale_n_atoms_min: 20
    - scale_n_atoms_max: 800
    - scale_overhead_region_n_max: 100
    - scale_gate_n_min: 200
    - scale_break_even_n_estimate: None
    - scale_speedup_median_at_maxN: 0.201066376990021
    - scale_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - bins:
      - n_atoms=20 n_samples=10 median_speedup=0.00866924220751798 median_eval_ratio=1.390625 pass_rate=1.0
      - n_atoms=50 n_samples=10 median_speedup=0.015964776349234544 median_eval_ratio=0.8359375 pass_rate=1.0
      - n_atoms=100 n_samples=10 median_speedup=0.02442258179220877 median_eval_ratio=0.8359375 pass_rate=1.0
      - n_atoms=200 n_samples=10 median_speedup=0.03720015333905859 median_eval_ratio=0.8359375 pass_rate=1.0
      - n_atoms=400 n_samples=10 median_speedup=0.06403126872207464 median_eval_ratio=0.8359375 pass_rate=1.0
      - n_atoms=800 n_samples=10 median_speedup=0.201066376990021 median_eval_ratio=0.8359375 pass_rate=1.0

## physics-operator-large-scale-2026-01-20-r2

- Source commit: e7f576d7bfdbea31f8229ae90c5806ff7508331d
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r2/physics_large_scale_evidence_pack.zip
- SHA256(physics_large_scale_evidence_pack.zip): BB8A54751BFA98D8A68C719A19B7B8A0284977BED591459CF0D029878654F999
- Command:
  python scripts/build_p5_large_scale_pack.py --out_dir out_physics_large_scale --n_atoms_bins "20,50,100,200,400,800" --samples_per_bin 10 --seed 0 --curve_id "dos_H" --energy_points 128 --dos_eta 0.2 --potential_scale_gamma 1.0 --edge_weight_mode bond_order_delta_chi --integrator_eps_abs 1e-6 --integrator_eps_rel 1e-4 --integrator_subdomains_max 32 --integrator_poly_degree_max 16 --integrator_quad_order_max 32 --integrator_eval_budget_max 256 --integrator_split_criterion "curvature" --overhead_region_n_max 100 --gate_n_min 200 --speedup_gate_break_even 1.0 --speedup_gate_strong 2.0
- Outcome (facts from summary_metadata.json):
  - law_ref:
    - contract_path: docs/contracts/INTEGRATION_SCALE_CONTRACT.md
    - contract_commit: e7f576d7bfdbea31f8229ae90c5806ff7508331d
    - contract_version: p5.1.v1
  - integrator (P5.1):
    - gate_n_min: 200
    - correctness_gate_rate: 1.0
    - min_scale_samples: 5
    - integrator_correctness_pass_rate_at_scale: 1.0
    - integrator_speedup_median_at_scale: 0.11588066997158922
    - integrator_eval_ratio_median_at_scale: 1.1962616822429906
    - integrator_correctness_verdict: PASS_CORRECTNESS_AT_SCALE
    - integrator_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - integrator_verdict_reason: FAIL: correctness ok at scale; speedup_median_at_scale=0.11588066997158922 < 1.0
  - scale (P5):
    - scale_n_atoms_min: 20
    - scale_n_atoms_max: 800
    - scale_overhead_region_n_max: 100
    - scale_gate_n_min: 200
    - scale_break_even_n_estimate: None
    - scale_speedup_median_at_maxN: 0.19773954628146345
    - scale_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - bins:
      - n_atoms=20 n_samples=10 median_speedup=0.008302636278321243 median_eval_ratio=0.7218578715584704 pass_rate=1.0
      - n_atoms=50 n_samples=10 median_speedup=0.015548566986817034 median_eval_ratio=1.1962616822429906 pass_rate=1.0
      - n_atoms=100 n_samples=10 median_speedup=0.02435279487229796 median_eval_ratio=1.1962616822429906 pass_rate=1.0
      - n_atoms=200 n_samples=10 median_speedup=0.03995939025453221 median_eval_ratio=1.1962616822429906 pass_rate=1.0
      - n_atoms=400 n_samples=10 median_speedup=0.11588066997158922 median_eval_ratio=1.1962616822429906 pass_rate=1.0
      - n_atoms=800 n_samples=10 median_speedup=0.19773954628146345 median_eval_ratio=1.1962616822429906 pass_rate=1.0

## physics-operator-large-scale-2026-01-20-r3

- Source commit: f465e9be203e39ac0d6c98c91cad080322fe487c
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r3/physics_large_scale_evidence_pack.zip
- SHA256(physics_large_scale_evidence_pack.zip): 8A1999E0DB0E03A59B6AB1318698B002A6594FE842E71CCED250FEF1947E84CE
- Command:
  python scripts/build_p5_large_scale_pack.py --out_dir out_physics_large_scale --n_atoms_bins "20,50,100,200,400,800" --samples_per_bin 10 --seed 0 --curve_id "dos_H" --energy_points 128 --dos_eta 0.2 --potential_scale_gamma 1.0 --edge_weight_mode bond_order_delta_chi --integrator_eps_abs 1e-6 --integrator_eps_rel 1e-4 --integrator_subdomains_max 32 --integrator_poly_degree_max 16 --integrator_quad_order_max 32 --integrator_eval_budget_max 256 --integrator_split_criterion "curvature" --overhead_region_n_max 100 --gate_n_min 200 --speedup_gate_break_even 1.0 --speedup_gate_strong 2.0
- Outcome (facts from summary_metadata.json):
  - law_ref:
    - contract_path: docs/contracts/INTEGRATION_SCALE_CONTRACT.md
    - contract_commit: f465e9be203e39ac0d6c98c91cad080322fe487c
    - contract_version: p5.1.v1
  - integrator (P5.1):
    - gate_n_min: 200
    - correctness_gate_rate: 1.0
    - min_scale_samples: 5
    - integrator_correctness_pass_rate_at_scale: 1.0
    - integrator_speedup_median_at_scale: 0.11754749203803679
    - integrator_eval_ratio_median_at_scale: 1.1962616822429906
    - integrator_correctness_verdict: PASS_CORRECTNESS_AT_SCALE
    - integrator_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - integrator_verdict_reason: FAIL: correctness ok at scale; speedup_median_at_scale=0.11754749203803679 < 1.0
  - cost (P5.2):
    - pack contains timing_breakdown.csv
    - cost_bottleneck_verdict_at_scale: BOTTLENECK_IS_INTEGRATOR
    - cost_median_dos_ldos_eval_ms_at_scale: 2.487639000023023
    - cost_median_integration_logic_ms_at_scale: 4.544935500007341
  - scale (P5):
    - scale_n_atoms_min: 20
    - scale_n_atoms_max: 800
    - scale_overhead_region_n_max: 100
    - scale_gate_n_min: 200
    - scale_break_even_n_estimate: None
    - scale_speedup_median_at_maxN: 0.19517237158308018
    - scale_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - bins:
      - n_atoms=20 n_samples=10 median_speedup=0.008269236864943375 median_eval_ratio=0.7218578715584704 pass_rate=1.0
      - n_atoms=50 n_samples=10 median_speedup=0.015597202120446593 median_eval_ratio=1.1962616822429906 pass_rate=1.0
      - n_atoms=100 n_samples=10 median_speedup=0.024094265079354193 median_eval_ratio=1.1962616822429906 pass_rate=1.0
      - n_atoms=200 n_samples=10 median_speedup=0.039599457975171976 median_eval_ratio=1.1962616822429906 pass_rate=1.0
      - n_atoms=400 n_samples=10 median_speedup=0.11754749203803679 median_eval_ratio=1.1962616822429906 pass_rate=1.0
      - n_atoms=800 n_samples=10 median_speedup=0.19517237158308018 median_eval_ratio=1.1962616822429906 pass_rate=1.0

## physics-operator-large-scale-2026-01-20-r4

- Source commit: 540a46511a1275fa8358aac00cd2e85cb36092a5
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r4/physics_large_scale_evidence_pack.zip
- SHA256(physics_large_scale_evidence_pack.zip): 94DE3D0A457F8A2129DBE2A577291862B60486F5959AC1D03FCF4D239CFD75A9
- Command:
  python scripts/build_p5_large_scale_pack.py --out_dir out_physics_large_scale --n_atoms_bins "20,50,100,200,400,800" --samples_per_bin 10 --seed 0 --curve_id "dos_H" --energy_points 128 --dos_eta 0.2 --potential_scale_gamma 1.0 --edge_weight_mode bond_order_delta_chi --integrator_eps_abs 1e-6 --integrator_eps_rel 1e-4 --integrator_subdomains_max 32 --integrator_poly_degree_max 16 --integrator_quad_order_max 32 --integrator_eval_budget_max 256 --integrator_split_criterion "curvature" --overhead_region_n_max 100 --gate_n_min 200 --speedup_gate_break_even 1.0 --speedup_gate_strong 2.0
- Outcome (facts from summary_metadata.json):
  - law_ref:
    - contract_path: docs/contracts/INTEGRATION_SCALE_CONTRACT.md
    - contract_commit: 540a46511a1275fa8358aac00cd2e85cb36092a5
    - contract_version: p5.1.v1
  - integrator (P5.1):
    - gate_n_min: 200
    - correctness_gate_rate: 1.0
    - min_scale_samples: 5
    - integrator_correctness_pass_rate_at_scale: 1.0
    - integrator_speedup_median_at_scale: 0.21174035241485198
    - integrator_eval_ratio_median_at_scale: 1.1531531531531531
    - integrator_correctness_verdict: PASS_CORRECTNESS_AT_SCALE
    - integrator_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - integrator_verdict_reason: FAIL: correctness ok at scale; speedup_median_at_scale=0.21174035241485198 < 1.0
  - cost (P5.2/P5.3):
    - cost_bottleneck_verdict_at_scale: MIXED
    - cost_median_dos_ldos_eval_ms_at_scale: 2.5924949999875935
    - cost_median_integration_logic_ms_at_scale: 2.218445499991617
    - cost_median_integration_logic_ms_at_scale_before: 4.544935500007341
    - cost_median_integration_logic_ms_at_scale_after: 2.218445499991617
    - cost_integration_logic_speedup_at_scale: 2.048702796631477
    - cost_integration_logic_opt_verdict_at_scale: PASS
    - cost_integration_logic_speedup_gate_break_even: 1.0
    - cost_integration_logic_speedup_gate_strong: 2.0
  - scale (P5):
    - scale_n_atoms_min: 20
    - scale_n_atoms_max: 800
    - scale_overhead_region_n_max: 100
    - scale_gate_n_min: 200
    - scale_break_even_n_estimate: None
    - scale_speedup_median_at_maxN: 0.3310290128123675
    - scale_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - bins:
      - n_atoms=20 n_samples=10 median_speedup=0.012134435725288602 median_eval_ratio=0.6998391037482867 pass_rate=1.0
      - n_atoms=50 n_samples=10 median_speedup=0.03219614241353548 median_eval_ratio=1.1378002528445006 pass_rate=1.0
      - n_atoms=100 n_samples=10 median_speedup=0.04760314136707567 median_eval_ratio=1.1636363636363636 pass_rate=1.0
      - n_atoms=200 n_samples=10 median_speedup=0.07580819744790274 median_eval_ratio=1.148005148005148 pass_rate=1.0
      - n_atoms=400 n_samples=10 median_speedup=0.21174035241485198 median_eval_ratio=1.1583947583947585 pass_rate=1.0
      - n_atoms=800 n_samples=10 median_speedup=0.3310290128123675 median_eval_ratio=1.148005148005148 pass_rate=1.0

## physics-operator-large-scale-2026-01-20-r5

- Source commit: 300c8657ef2c808462662cbc05f1c4245f8fe71b
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r5/physics_large_scale_evidence_pack.zip
- SHA256(physics_large_scale_evidence_pack.zip): 2414FBBFEC48A920E061FA3037BA0626FBFDD28F905C3DE874BBF1CCCFE8AF48
- Command:
  python scripts/build_p5_large_scale_pack.py --out_dir out_physics_large_scale --n_atoms_bins "20,50,100,200,400,800" --samples_per_bin 10 --seed 0 --curve_id "dos_H" --energy_points 128 --dos_eta 0.2 --potential_scale_gamma 1.0 --edge_weight_mode bond_order_delta_chi --integrator_eps_abs 1e-6 --integrator_eps_rel 1e-4 --integrator_subdomains_max 32 --integrator_poly_degree_max 16 --integrator_quad_order_max 32 --integrator_eval_budget_max 256 --integrator_split_criterion "curvature" --overhead_region_n_max 100 --gate_n_min 200 --speedup_gate_break_even 1.0 --speedup_gate_strong 2.0
- Outcome (facts from summary_metadata.json):
  - law_ref:
    - contract_path: docs/contracts/INTEGRATION_SCALE_CONTRACT.md
    - contract_commit: 300c8657ef2c808462662cbc05f1c4245f8fe71b
    - contract_version: p5.1.v1
  - integrator (P5.1):
    - gate_n_min: 200
    - correctness_gate_rate: 1.0
    - min_scale_samples: 5
    - integrator_correctness_pass_rate_at_scale: 1.0
    - integrator_speedup_median_at_scale: 0.0851339642569639
    - integrator_eval_ratio_median_at_scale: 1.1327433628318584
    - integrator_correctness_verdict: PASS_CORRECTNESS_AT_SCALE
    - integrator_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - integrator_verdict_reason: FAIL: correctness ok at scale; speedup_median_at_scale=0.0851339642569639 < 1.0
  - cost (P5.2/P5.3):
    - cost_bottleneck_verdict_at_scale: MIXED
    - cost_median_dos_ldos_eval_ms_at_scale: 1.866809499944111
    - cost_median_integration_logic_ms_at_scale: 2.0868710000740975
    - cost_median_integration_logic_ms_at_scale_before: 4.544935500007341
    - cost_median_integration_logic_ms_at_scale_after: 2.0868710000740975
    - cost_integration_logic_speedup_at_scale: 2.1778708410083643
    - cost_integration_logic_opt_verdict_at_scale: PASS
  - topology (P5.4):
    - topology_families: ['polymer', 'ring']
    - topology_gate_n_min: 200
    - pack contains fixtures_ring_scale.csv
    - pack contains speedup_vs_n_by_family.csv
    - speedup_median_at_scale_polymer: 0.0879949862363698
    - speedup_verdict_at_scale_polymer: FAIL_SPEEDUP_AT_SCALE
    - speedup_median_at_scale_ring: 0.07868455761041604
    - speedup_verdict_at_scale_ring: FAIL_SPEEDUP_AT_SCALE
    - topology_hardness_verdict: NO_SPEEDUP_YET
    - topology_hardness_reason: polymer(verdict=FAIL_SPEEDUP_AT_SCALE, median=0.0879949862363698) ring(verdict=FAIL_SPEEDUP_AT_SCALE, median=0.07868455761041604) gate_n_min=200
  - scale (P5):
    - scale_n_atoms_min: 20
    - scale_n_atoms_max: 800
    - scale_overhead_region_n_max: 100
    - scale_gate_n_min: 200
    - scale_break_even_n_estimate: None
    - scale_speedup_median_at_maxN: 0.2545261166843933
    - scale_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - bins:
      - n_atoms=20 n_samples=20 median_speedup=0.012754543778999802 median_eval_ratio=0.8889317578972751 pass_rate=1.0
      - n_atoms=50 n_samples=20 median_speedup=0.021665670378961815 median_eval_ratio=1.1428571428571428 pass_rate=1.0
      - n_atoms=100 n_samples=20 median_speedup=0.036416296425919446 median_eval_ratio=1.1428571428571428 pass_rate=1.0
      - n_atoms=200 n_samples=20 median_speedup=0.059633706605321975 median_eval_ratio=0.9143323638961172 pass_rate=1.0
      - n_atoms=400 n_samples=20 median_speedup=0.0851339642569639 median_eval_ratio=1.148005148005148 pass_rate=1.0
      - n_atoms=800 n_samples=20 median_speedup=0.2545261166843933 median_eval_ratio=1.1378002528445006 pass_rate=1.0

## physics-operator-large-scale-2026-01-20-r6

- Source commit: 93145ec561ce6565b149c4e1b5536f7e778731db
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r6/physics_large_scale_evidence_pack.zip
- SHA256(physics_large_scale_evidence_pack.zip): 7A933610708BEEF60FC69DD37BC6A200679E9BD3E92BBBF4CA981C4C7CBED530
- Command:
  python scripts/build_p5_large_scale_pack.py --out_dir out_physics_large_scale --n_atoms_bins "20,50,100,200,400,800" --samples_per_bin 10 --seed 0 --curve_id "dos_H" --energy_points 128 --dos_eta 0.2 --potential_scale_gamma 1.0 --edge_weight_mode bond_order_delta_chi --integrator_eps_abs 1e-6 --integrator_eps_rel 1e-4 --integrator_subdomains_max 32 --integrator_poly_degree_max 16 --integrator_quad_order_max 32 --integrator_eval_budget_max 256 --integrator_split_criterion "curvature" --overhead_region_n_max 100 --gate_n_min 200 --speedup_gate_break_even 1.0 --speedup_gate_strong 2.0
- Outcome (facts from summary_metadata.json):
  - law_ref:
    - contract_path: docs/contracts/INTEGRATION_SCALE_CONTRACT.md
    - contract_commit: 93145ec561ce6565b149c4e1b5536f7e778731db
    - contract_version: p5.1.v1
  - integrator (P5.1):
    - gate_n_min: 200
    - correctness_gate_rate: 1.0
    - min_scale_samples: 5
    - integrator_correctness_pass_rate_at_scale: 1.0
    - integrator_speedup_median_at_scale: 0.12003553852247553
    - integrator_eval_ratio_median_at_scale: 1.1327433628318584
    - integrator_correctness_verdict: PASS_CORRECTNESS_AT_SCALE
    - integrator_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - integrator_verdict_reason: FAIL: correctness ok at scale; speedup_median_at_scale=0.12003553852247553 < 1.0
  - cost (P5.2/P5.3):
    - cost_bottleneck_verdict_at_scale: MIXED
    - cost_median_dos_ldos_eval_ms_at_scale: 2.491177499997832
    - cost_median_integration_logic_ms_at_scale: 2.168626999981882
    - cost_median_integration_logic_ms_at_scale_before: 4.544935500007341
    - cost_median_integration_logic_ms_at_scale_after: 2.168626999981882
    - cost_integration_logic_speedup_at_scale: 2.0957663535708595
    - cost_integration_logic_opt_verdict_at_scale: PASS
  - topology (P5.4):
    - topology_families: ['polymer', 'ring']
    - topology_gate_n_min: 200
    - speedup_median_at_scale_polymer: 0.19196896218051646
    - speedup_verdict_at_scale_polymer: FAIL_SPEEDUP_AT_SCALE
    - speedup_median_at_scale_ring: 0.09027446617232507
    - speedup_verdict_at_scale_ring: FAIL_SPEEDUP_AT_SCALE
    - topology_hardness_verdict: NO_SPEEDUP_YET
    - topology_hardness_reason: polymer(verdict=FAIL_SPEEDUP_AT_SCALE, median=0.19196896218051646) ring(verdict=FAIL_SPEEDUP_AT_SCALE, median=0.09027446617232507) gate_n_min=200
  - topology cost (P5.5):
    - pack contains timing_breakdown_by_family.csv
    - cost_median_total_ms_at_scale_polymer_estimate: 7.468396366668155
    - cost_median_total_ms_at_scale_ring_estimate: 8.9061203666598
    - cost_ratio_ring_vs_polymer_total_ms_at_scale_estimate: 1.192507725809557
    - topology_ring_cost_gap_verdict_at_scale: RING_SLOWER_DUE_TO_BUILD_OPERATOR
    - topology_ring_cost_gap_reason_at_scale: scale_n_min=200 polymer_n=30 ring_n=30 ratio_total=1.192507725809557 ratio_build=1.305719370594242 ratio_dos=1.030257037793385 ratio_integration_logic=1.252262148098095 ratio_io=1.0
  - scale (P5):
    - scale_n_atoms_min: 20
    - scale_n_atoms_max: 800
    - scale_overhead_region_n_max: 100
    - scale_gate_n_min: 200
    - scale_break_even_n_estimate: None
    - scale_speedup_median_at_maxN: 0.2977215731095672
    - scale_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - bins:
      - n_atoms=20 n_samples=20 median_speedup=0.014669519575302988 median_eval_ratio=0.8889317578972751 pass_rate=1.0
      - n_atoms=50 n_samples=20 median_speedup=0.029286180498108615 median_eval_ratio=1.1428571428571428 pass_rate=1.0
      - n_atoms=100 n_samples=20 median_speedup=0.04391544879978222 median_eval_ratio=1.1428571428571428 pass_rate=1.0
      - n_atoms=200 n_samples=20 median_speedup=0.056289664882653453 median_eval_ratio=0.9143323638961172 pass_rate=1.0
      - n_atoms=400 n_samples=20 median_speedup=0.12003553852247553 median_eval_ratio=1.148005148005148 pass_rate=1.0
      - n_atoms=800 n_samples=20 median_speedup=0.2977215731095672 median_eval_ratio=1.1378002528445006 pass_rate=1.0

## physics-operator-large-scale-2026-01-20-r7

- Source commit: 4b4c33eb82673b214ac9ad8e50f5f0d64791dec0
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r7/physics_large_scale_evidence_pack.zip
- SHA256(physics_large_scale_evidence_pack.zip): D9DF8097C8C6EA639400ACBCC80F32694E02550C68E3DDBF07911C8683F12666
- Command:
  python scripts/build_p5_large_scale_pack.py --out_dir out_physics_large_scale --n_atoms_bins "20,50,100,200,400,800" --samples_per_bin 10 --seed 0 --curve_id "dos_H" --energy_points 128 --dos_eta 0.2 --potential_scale_gamma 1.0 --edge_weight_mode bond_order_delta_chi --integrator_eps_abs 1e-6 --integrator_eps_rel 1e-4 --integrator_subdomains_max 32 --integrator_poly_degree_max 16 --integrator_quad_order_max 32 --integrator_eval_budget_max 256 --integrator_split_criterion "curvature" --overhead_region_n_max 100 --gate_n_min 200 --speedup_gate_break_even 1.0 --speedup_gate_strong 2.0
- Outcome (facts from summary_metadata.json):
  - law_ref:
    - contract_path: docs/contracts/INTEGRATION_SCALE_CONTRACT.md
    - contract_commit: f01d32149faaac2d8d70872895e28ca7273be55d
    - contract_version: p5.1.v1
  - integrator (P5.1):
    - gate_n_min: 200
    - correctness_gate_rate: 1.0
    - min_scale_samples: 5
    - integrator_correctness_pass_rate_at_scale: 1.0
    - integrator_speedup_median_at_scale: 0.11537244558037008
    - integrator_eval_ratio_median_at_scale: 1.1327433628318584
    - integrator_correctness_verdict: PASS_CORRECTNESS_AT_SCALE
    - integrator_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - integrator_verdict_reason: FAIL: correctness ok at scale; speedup_median_at_scale=0.11537244558037008 < 1.0
  - cost (P5.2/P5.3):
    - cost_bottleneck_verdict_at_scale: MIXED
    - cost_median_dos_ldos_eval_ms_at_scale: 2.4019754999855536
    - cost_median_integration_logic_ms_at_scale: 2.5467599999871027
    - cost_median_integration_logic_ms_at_scale_before: 4.544935500007341
    - cost_median_integration_logic_ms_at_scale_after: 2.5467599999871027
    - cost_integration_logic_speedup_at_scale: 1.784595132651038
    - cost_integration_logic_opt_verdict_at_scale: PASS
  - topology (P5.4):
    - topology_families: ['polymer', 'ring']
    - topology_gate_n_min: 200
    - speedup_median_at_scale_polymer: 0.11620346812770953
    - speedup_verdict_at_scale_polymer: FAIL_SPEEDUP_AT_SCALE
    - speedup_median_at_scale_ring: 0.09713153126539123
    - speedup_verdict_at_scale_ring: FAIL_SPEEDUP_AT_SCALE
    - topology_hardness_verdict: NO_SPEEDUP_YET
    - topology_hardness_reason: polymer(verdict=FAIL_SPEEDUP_AT_SCALE, median=0.11620346812770953) ring(verdict=FAIL_SPEEDUP_AT_SCALE, median=0.09713153126539123) gate_n_min=200
  - topology cost (P5.5):
    - pack contains timing_breakdown_by_family.csv
    - cost_median_total_ms_at_scale_polymer_estimate: 7.252107708332645
    - cost_median_total_ms_at_scale_ring_estimate: 8.966695208329512
    - cost_ratio_ring_vs_polymer_total_ms_at_scale_estimate: 1.2364260941721552
    - topology_ring_cost_gap_verdict_at_scale: RING_SLOWER_DUE_TO_BUILD_OPERATOR
    - topology_ring_cost_gap_reason_at_scale: scale_n_min=200 polymer_n=30 ring_n=30 ratio_total=1.2364260941721552 ratio_build=1.3099192608151964 ratio_dos=1.2034641142253626 ratio_integration_logic=1.2464374252037669 ratio_io=1.0
  - ring speedup law (P5.6):
    - ring_speedup_median_at_scale: 0.09713153126539123
    - ring_eval_ratio_median_at_scale: 0.8951048951048951
    - ring_correctness_pass_rate_at_scale: 1.0
    - ring_speedup_verdict_at_scale: NO_SPEEDUP_YET
    - ring_speedup_verdict_reason_at_scale: gate_n_min=200 min_scale_samples=5 ring_n=30 correctness_gate_rate=1.0 ring_pass_rate=1.0 ring_median_speedup=0.09713153126539123 polymer_median_speedup=0.11620346812770953 gate_speedup=1.0 ring_cost_gap_verdict=RING_SLOWER_DUE_TO_BUILD_OPERATOR
    - topology_ring_cost_gap_verdict_at_scale: RING_SLOWER_DUE_TO_BUILD_OPERATOR
  - scale (P5):
    - scale_n_atoms_min: 20
    - scale_n_atoms_max: 800
    - scale_overhead_region_n_max: 100
    - scale_gate_n_min: 200
    - scale_break_even_n_estimate: None
    - scale_speedup_median_at_maxN: 0.3091472189713125
    - scale_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - bins:
      - n_atoms=20 n_samples=20 median_speedup=0.014916035064062005 median_eval_ratio=0.8889317578972751 pass_rate=1.0
      - n_atoms=50 n_samples=20 median_speedup=0.030044075099708974 median_eval_ratio=1.1428571428571428 pass_rate=1.0
      - n_atoms=100 n_samples=20 median_speedup=0.04378252841631866 median_eval_ratio=1.1428571428571428 pass_rate=1.0
      - n_atoms=200 n_samples=20 median_speedup=0.06457677123977995 median_eval_ratio=0.9143323638961172 pass_rate=1.0
      - n_atoms=400 n_samples=20 median_speedup=0.11537244558037008 median_eval_ratio=1.148005148005148 pass_rate=1.0
      - n_atoms=800 n_samples=20 median_speedup=0.3091472189713125 median_eval_ratio=1.1378002528445006 pass_rate=1.0

## physics-operator-large-scale-2026-01-20-r8

- Source commit: 8689913741dffd5aee094af591c6339fc1605f26
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r8/physics_large_scale_evidence_pack.zip
- SHA256(physics_large_scale_evidence_pack.zip): 441D58EB7D389E81BFD2434777B8FC33CB1698B99C9494B27FE66ABDD18665EC
- Command:
  python scripts/build_p5_large_scale_pack.py --out_dir out_physics_large_scale --n_atoms_bins "20,50,100,200,400,800" --samples_per_bin 10 --seed 0 --curve_id "dos_H" --energy_points 128 --dos_eta 0.2 --potential_scale_gamma 1.0 --edge_weight_mode bond_order_delta_chi --integrator_eps_abs 1e-6 --integrator_eps_rel 1e-4 --integrator_subdomains_max 32 --integrator_poly_degree_max 16 --integrator_quad_order_max 32 --integrator_eval_budget_max 256 --integrator_split_criterion "curvature" --overhead_region_n_max 100 --gate_n_min 200 --speedup_gate_break_even 1.0 --speedup_gate_strong 2.0
- Outcome (facts from summary_metadata.json):
  - law_ref:
    - contract_path: docs/contracts/INTEGRATION_SCALE_CONTRACT.md
    - contract_commit: 8689913741dffd5aee094af591c6339fc1605f26
    - contract_version: p5.1.v1
  - integrator (P5.1):
    - gate_n_min: 200
    - correctness_gate_rate: 1.0
    - min_scale_samples: 5
    - integrator_correctness_pass_rate_at_scale: 1.0
    - integrator_speedup_median_at_scale: 0.12719779730605427
    - integrator_eval_ratio_median_at_scale: 1.1327433628318584
    - integrator_correctness_verdict: PASS_CORRECTNESS_AT_SCALE
    - integrator_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - integrator_verdict_reason: FAIL: correctness ok at scale; speedup_median_at_scale=0.12719779730605427 < 1.0
  - cost (P5.2/P5.3):
    - cost_bottleneck_verdict_at_scale: MIXED
    - cost_median_dos_ldos_eval_ms_at_scale: 2.5613634999643864
    - cost_median_integration_logic_ms_at_scale: 2.1947799999679773
    - cost_median_integration_logic_ms_at_scale_before: 4.544935500007341
    - cost_median_integration_logic_ms_at_scale_after: 2.1947799999679773
    - cost_integration_logic_speedup_at_scale: 2.070793200263194
    - cost_integration_logic_opt_verdict_at_scale: PASS
  - topology (P5.4):
    - topology_families: ['polymer', 'ring']
    - topology_gate_n_min: 200
    - speedup_median_at_scale_polymer: 0.20073707150821735
    - speedup_verdict_at_scale_polymer: FAIL_SPEEDUP_AT_SCALE
    - speedup_median_at_scale_ring: 0.09606983162926894
    - speedup_verdict_at_scale_ring: FAIL_SPEEDUP_AT_SCALE
    - topology_hardness_verdict: NO_SPEEDUP_YET
    - topology_hardness_reason: polymer(verdict=FAIL_SPEEDUP_AT_SCALE, median=0.20073707150821735) ring(verdict=FAIL_SPEEDUP_AT_SCALE, median=0.09606983162926894) gate_n_min=200
  - topology cost (P5.5):
    - pack contains timing_breakdown_by_family.csv
    - cost_median_total_ms_at_scale_polymer_estimate: 7.581216233350811
    - cost_median_total_ms_at_scale_ring_estimate: 8.502138233314108
    - cost_ratio_ring_vs_polymer_total_ms_at_scale_estimate: 1.1214741766515028
    - topology_ring_cost_gap_verdict_at_scale: RING_SLOWER_DUE_TO_INTEGRATION_LOGIC
    - topology_ring_cost_gap_reason_at_scale: scale_n_min=200 polymer_n=30 ring_n=30 ratio_total=1.1214741766515028 ratio_build=1.1281262332542437 ratio_dos=1.0230059844235242 ratio_integration_logic=1.2374918832703354 ratio_io=1.0
  - ring speedup law (P5.6):
    - ring_speedup_median_at_scale: 0.09606983162926894
    - ring_eval_ratio_median_at_scale: 0.8951048951048951
    - ring_correctness_pass_rate_at_scale: 1.0
    - ring_speedup_verdict_at_scale: NO_SPEEDUP_YET
    - ring_speedup_verdict_reason_at_scale: gate_n_min=200 min_scale_samples=5 ring_n=30 correctness_gate_rate=1.0 ring_pass_rate=1.0 ring_median_speedup=0.09606983162926894 polymer_median_speedup=0.20073707150821735 gate_speedup=1.0 ring_cost_gap_verdict=RING_SLOWER_DUE_TO_INTEGRATION_LOGIC
    - topology_ring_cost_gap_verdict_at_scale: RING_SLOWER_DUE_TO_INTEGRATION_LOGIC
  - scale (P5):
    - scale_n_atoms_min: 20
    - scale_n_atoms_max: 800
    - scale_overhead_region_n_max: 100
    - scale_gate_n_min: 200
    - scale_break_even_n_estimate: None
    - scale_speedup_median_at_maxN: 0.3147446816387783
    - scale_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - bins:
      - n_atoms=20 n_samples=20 median_speedup=0.015368996179573642 median_eval_ratio=0.8889317578972751 pass_rate=1.0
      - n_atoms=50 n_samples=20 median_speedup=0.030548609568748383 median_eval_ratio=1.1428571428571428 pass_rate=1.0
      - n_atoms=100 n_samples=20 median_speedup=0.045945863838540324 median_eval_ratio=1.1428571428571428 pass_rate=1.0
      - n_atoms=200 n_samples=20 median_speedup=0.06266050225755272 median_eval_ratio=0.9143323638961172 pass_rate=1.0
      - n_atoms=400 n_samples=20 median_speedup=0.12719779730605427 median_eval_ratio=1.148005148005148 pass_rate=1.0
      - n_atoms=800 n_samples=20 median_speedup=0.3147446816387783 median_eval_ratio=1.1378002528445006 pass_rate=1.0

## physics-operator-large-scale-2026-01-20-r9

- Source commit: d5be05323c4de4954d4570785d0cf4a2fd61fa37
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/physics-operator-large-scale-2026-01-20-r9/physics_large_scale_evidence_pack.zip
- SHA256(physics_large_scale_evidence_pack.zip): E1381DC4DB2D099C69973370BE4C53CC6390FAE23CF960217DC500B599DC713B
- Command:
  python scripts/build_p5_large_scale_pack.py --out_dir out_physics_large_scale --n_atoms_bins "20,50,100,200,400,800" --samples_per_bin 10 --seed 0 --curve_id "dos_H" --energy_points 128 --dos_eta 0.2 --potential_scale_gamma 1.0 --edge_weight_mode bond_order_delta_chi --integrator_eps_abs 1e-6 --integrator_eps_rel 1e-4 --integrator_subdomains_max 32 --integrator_poly_degree_max 16 --integrator_quad_order_max 32 --integrator_eval_budget_max 256 --integrator_split_criterion "curvature" --overhead_region_n_max 100 --gate_n_min 200 --speedup_gate_break_even 1.0 --speedup_gate_strong 2.0
- Outcome (facts from summary_metadata.json):
  - law_ref:
    - contract_path: docs/contracts/INTEGRATION_SCALE_CONTRACT.md
    - contract_commit: d5be05323c4de4954d4570785d0cf4a2fd61fa37
    - contract_version: p5.1.v1
  - integrator (P5.1):
    - gate_n_min: 200
    - correctness_gate_rate: 1.0
    - min_scale_samples: 5
    - integrator_correctness_pass_rate_at_scale: 1.0
    - integrator_speedup_median_at_scale: 0.10289085018839059
    - integrator_eval_ratio_median_at_scale: 1.1327433628318584
    - integrator_correctness_verdict: PASS_CORRECTNESS_AT_SCALE
    - integrator_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - integrator_verdict_reason: FAIL: correctness ok at scale; speedup_median_at_scale=0.10289085018839059 < 1.0
  - cost (P5.2/P5.3):
    - cost_bottleneck_verdict_at_scale: MIXED
    - cost_median_dos_ldos_eval_ms_at_scale: 2.368375000010303
    - cost_median_integration_logic_ms_at_scale: 2.4908904999989545
    - cost_median_integration_logic_ms_at_scale_before: 4.544935500007341
    - cost_median_integration_logic_ms_at_scale_after: 2.4908904999989545
    - cost_integration_logic_speedup_at_scale: 1.824622760418111
    - cost_integration_logic_opt_verdict_at_scale: PASS
  - topology (P5.4):
    - topology_families: ['polymer', 'ring']
    - topology_gate_n_min: 200
    - speedup_median_at_scale_polymer: 0.10562706901479679
    - speedup_verdict_at_scale_polymer: FAIL_SPEEDUP_AT_SCALE
    - speedup_median_at_scale_ring: 0.08512662027971973
    - speedup_verdict_at_scale_ring: FAIL_SPEEDUP_AT_SCALE
    - topology_hardness_verdict: NO_SPEEDUP_YET
    - topology_hardness_reason: polymer(verdict=FAIL_SPEEDUP_AT_SCALE, median=0.10562706901479679) ring(verdict=FAIL_SPEEDUP_AT_SCALE, median=0.08512662027971973) gate_n_min=200
  - topology cost (P5.5):
    - pack contains timing_breakdown_by_family.csv
    - cost_median_total_ms_at_scale_polymer_estimate: 7.4300517416702405
    - cost_median_total_ms_at_scale_ring_estimate: 8.892115741664952
    - cost_ratio_ring_vs_polymer_total_ms_at_scale_estimate: 1.1967770953457784
    - topology_ring_cost_gap_verdict_at_scale: RING_SLOWER_DUE_TO_INTEGRATION_LOGIC
    - topology_ring_cost_gap_reason_at_scale: scale_n_min=200 polymer_n=30 ring_n=30 ratio_total=1.1967770953457784 ratio_build=1.130292975349101 ratio_dos=1.221658910025633 ratio_integration_logic=1.2641339636812614 ratio_io=1.0
  - integration_logic (ring) KPI (P5.8):
    - cost_ratio_ring_vs_polymer_integration_logic_ms_at_scale: 1.2641339636812614
    - cost_median_integration_logic_ms_at_scale_ring_before: 2.6423465000817714
    - cost_median_integration_logic_ms_at_scale_ring_after: 3.0608639999840648
    - cost_integration_logic_speedup_at_scale_ring: 0.8632681818256308
    - cost_integration_logic_opt_verdict_at_scale_ring: FAIL
    - topology_ring_cost_gap_verdict_at_scale: RING_SLOWER_DUE_TO_INTEGRATION_LOGIC
  - ring speedup law (P5.6):
    - ring_speedup_median_at_scale: 0.08512662027971973
    - ring_eval_ratio_median_at_scale: 0.8951048951048951
    - ring_correctness_pass_rate_at_scale: 1.0
    - ring_speedup_verdict_at_scale: NO_SPEEDUP_YET
    - ring_speedup_verdict_reason_at_scale: gate_n_min=200 min_scale_samples=5 ring_n=30 correctness_gate_rate=1.0 ring_pass_rate=1.0 ring_median_speedup=0.08512662027971973 polymer_median_speedup=0.10562706901479679 gate_speedup=1.0 ring_cost_gap_verdict=RING_SLOWER_DUE_TO_INTEGRATION_LOGIC
    - topology_ring_cost_gap_verdict_at_scale: RING_SLOWER_DUE_TO_INTEGRATION_LOGIC
  - scale (P5):
    - scale_n_atoms_min: 20
    - scale_n_atoms_max: 800
    - scale_overhead_region_n_max: 100
    - scale_gate_n_min: 200
    - scale_break_even_n_estimate: None
    - scale_speedup_median_at_maxN: 0.2855233217418385
    - scale_speedup_verdict: FAIL_SPEEDUP_AT_SCALE
    - bins:
      - n_atoms=20 n_samples=20 median_speedup=0.01400078103187933 median_eval_ratio=0.8889317578972751 pass_rate=1.0
      - n_atoms=50 n_samples=20 median_speedup=0.02627283859063287 median_eval_ratio=1.1428571428571428 pass_rate=1.0
      - n_atoms=100 n_samples=20 median_speedup=0.0397502243125315 median_eval_ratio=1.1428571428571428 pass_rate=1.0
      - n_atoms=200 n_samples=20 median_speedup=0.05476953125070314 median_eval_ratio=0.9143323638961172 pass_rate=1.0
      - n_atoms=400 n_samples=20 median_speedup=0.10289085018839059 median_eval_ratio=1.148005148005148 pass_rate=1.0
      - n_atoms=800 n_samples=20 median_speedup=0.2855233217418385 median_eval_ratio=1.1378002528445006 pass_rate=1.0

## value-ring-suite-2026-01-21

- Source commit: 632d5f1b231288ae1308a338cf52f9299eec70db
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-ring-suite-2026-01-21/value_ring_suite_evidence_pack.zip
- SHA256(value_ring_suite_evidence_pack.zip): 97A3181F897A89A0559E0099A40AF7537D8A7E08A5B3D7D6377514E042E27509
- Command:
  python scripts/pilot_generate_input.py --out_dir out_ring_suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  hetero2-batch --input out_ring_suite/input.csv --out_dir out_ring_suite --artifacts light --score_mode external_scores --scores_input out_ring_suite/scores.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
- Outcome (facts from summary.csv):
  - rows_total: 200
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - share_rows_with_n_decoys_gt_0: 1.000 (100.0%)

## value-ring-suite-2026-01-21-m1_1

- Source commit: 3ee006204c4fb5635c779c54254384c4132d98ed
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-ring-suite-2026-01-21-m1_1/value_ring_suite_evidence_pack.zip
- SHA256(value_ring_suite_evidence_pack.zip): A250164C1D47EDF650846C0D8E1F9B043D3530C6E962C7309255F0B50F629E96
- Command:
  python scripts/pilot_generate_input.py --out_dir out_ring_suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
  hetero2-batch --input out_ring_suite/input.csv --out_dir out_ring_suite --artifacts light --score_mode external_scores --scores_input out_ring_suite/scores.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
- Outcome (facts from summary.csv):
  - rows_total: 200
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - share_rows_with_n_decoys_gt_0: 1.000 (100.0%)
  - decoy_strategy_used_distribution_ok:
    - rewire_fallback_aromatic_as_single_v1: 100 (50.0%)
    - rewire_strict_v1: 60 (30.0%)
    - rewire_relax_a_v1: 40 (20.0%)

## value-known-bad-good-2026-01-21

- Source commit: cd8113bbe269caa3d171df19d5fe417b125ba92a

- Variant: BAD-constant
  - Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-21/value_known_bad_good_BAD-constant_evidence_pack.zip
  - SHA256(value_known_bad_good_BAD-constant_evidence_pack.zip): 6D4C12D4523AADC35CB65EAB5A0FB8E8E2EE01626769E74AD0E62B4D7BF182BF
  - Command:
    python scripts/pilot_generate_input.py --out_dir out_value_m2/suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
    (scores variant) BAD-constant: all scores equal
    hetero2-batch --input out_value_m2/suite/input.csv --out_dir out_value_m2/BAD-constant --artifacts light --score_mode external_scores --scores_input out_value_m2/score_variants/scores_BAD-constant.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
  - Outcome (facts from summary.csv):
  - rows_total: 200
  - rows_ok: 200
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - share_rows_with_n_decoys_gt_0: 1.000 (100.0%)
  - median_slack: -0.500000
  - pass_rate: 0.000000

- Variant: BAD-random
  - Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-21/value_known_bad_good_BAD-random_evidence_pack.zip
  - SHA256(value_known_bad_good_BAD-random_evidence_pack.zip): DC723348B495F0E6AC29ABF749D8858F023762FA947DBF75066BCB05D62B3046
  - Command:
    python scripts/pilot_generate_input.py --out_dir out_value_m2/suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
    (scores variant) BAD-random: random scores (seed=0)
    hetero2-batch --input out_value_m2/suite/input.csv --out_dir out_value_m2/BAD-random --artifacts light --score_mode external_scores --scores_input out_value_m2/score_variants/scores_BAD-random.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
  - Outcome (facts from summary.csv):
  - rows_total: 200
  - rows_ok: 200
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - share_rows_with_n_decoys_gt_0: 1.000 (100.0%)
  - median_slack: 0.000000
  - pass_rate: 0.600000

- Variant: GOOD-synthetic
  - Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-21/value_known_bad_good_GOOD-synthetic_evidence_pack.zip
  - SHA256(value_known_bad_good_GOOD-synthetic_evidence_pack.zip): 4FC460FEE5712FC3349CD44B8EF3D6ACF43BD4D98EDBDBD7DD7F01DC5C74AB25
  - Command:
    python scripts/pilot_generate_input.py --out_dir out_value_m2/suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
    (scores variant) GOOD-synthetic: original=1.0, decoys=0.0
    hetero2-batch --input out_value_m2/suite/input.csv --out_dir out_value_m2/GOOD-synthetic --artifacts light --score_mode external_scores --scores_input out_value_m2/score_variants/scores_GOOD-synthetic.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
  - Outcome (facts from summary.csv):
  - rows_total: 200
  - rows_ok: 200
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - share_rows_with_n_decoys_gt_0: 1.000 (100.0%)
  - median_slack: 0.000000
  - pass_rate: 1.000000

- Separation facts (OK-only; no auto-threshold gating):
separation facts (computed on status==OK rows only):
- BAD-constant: rows_ok=200/200, median_slack=-0.500000, pass_rate=0.000000
- BAD-random: rows_ok=200/200, median_slack=0.000000, pass_rate=0.600000
- GOOD-synthetic: rows_ok=200/200, median_slack=0.000000, pass_rate=1.000000

- Δ_median_slack(GOOD - BAD-constant): 0.500000
- Δ_PASS_rate(GOOD - BAD-constant): 1.000000
- Δ_median_slack(GOOD - BAD-random): 0.000000
- Δ_PASS_rate(GOOD - BAD-random): 0.400000

## value-known-bad-good-2026-01-21-r2

- Source commit: 6cbcc31cbd20749cecc0f62d4b68986801c801e8

- Variant: BAD-constant
  - Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-21-r2/value_known_bad_good_BAD-constant_evidence_pack.zip
  - SHA256(value_known_bad_good_BAD-constant_evidence_pack.zip): AC12456914248E6D1D0A44AD1827E367532B2A9452B77CA22D26A8A77BB84EE8
  - Command:
    python scripts/pilot_generate_input.py --out_dir out_value_m2/suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
    (scores variant) BAD-constant: all scores equal
    hetero2-batch --input out_value_m2/suite/input.csv --out_dir out_value_m2/BAD-constant --artifacts light --score_mode external_scores --scores_input out_value_m2/score_variants/scores_BAD-constant.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
  - Outcome (facts from summary.csv):
  - rows_total: 200
  - rows_ok: 200
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - share_rows_with_n_decoys_gt_0: 1.000 (100.0%)
  - mean_slack: -0.500000
  - p25_slack: -0.500000
  - median_slack: -0.500000
  - p75_slack: -0.500000
  - pass_rate: 0.000000

- Variant: BAD-random
  - Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-21-r2/value_known_bad_good_BAD-random_evidence_pack.zip
  - SHA256(value_known_bad_good_BAD-random_evidence_pack.zip): CE1DC0C1CD03DDA63F0078ECC5F4B71F3CE74B1BE4A9643790B3EE9A552676B5
  - Command:
    python scripts/pilot_generate_input.py --out_dir out_value_m2/suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
    (scores variant) BAD-random: random scores (seed=0)
    hetero2-batch --input out_value_m2/suite/input.csv --out_dir out_value_m2/BAD-random --artifacts light --score_mode external_scores --scores_input out_value_m2/score_variants/scores_BAD-random.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
  - Outcome (facts from summary.csv):
  - rows_total: 200
  - rows_ok: 200
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - share_rows_with_n_decoys_gt_0: 1.000 (100.0%)
  - mean_slack: -0.311111
  - p25_slack: -1.000000
  - median_slack: 0.000000
  - p75_slack: 0.000000
  - pass_rate: 0.600000

- Variant: GOOD-synthetic
  - Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/value-known-bad-good-2026-01-21-r2/value_known_bad_good_GOOD-synthetic_evidence_pack.zip
  - SHA256(value_known_bad_good_GOOD-synthetic_evidence_pack.zip): 84F5215A583D73481DB7612FA92D90BD54AB5BF2D310ECE54BE6667E5FFECEE9
  - Command:
    python scripts/pilot_generate_input.py --out_dir out_value_m2/suite --rows 200 --k_decoys 20 --seed 0 --full_cover_count 3
    (scores variant) GOOD-synthetic: original=1.0, decoys=0.0
    hetero2-batch --input out_value_m2/suite/input.csv --out_dir out_value_m2/GOOD-synthetic --artifacts light --score_mode external_scores --scores_input out_value_m2/score_variants/scores_GOOD-synthetic.json --k_decoys 20 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
  - Outcome (facts from summary.csv):
  - rows_total: 200
  - rows_ok: 200
  - status_counts: OK=200, SKIP=0, ERROR=0
  - top_skip_reasons: (none)
  - share_rows_with_n_decoys_gt_0: 1.000 (100.0%)
  - mean_slack: 0.000000
  - p25_slack: 0.000000
  - median_slack: 0.000000
  - p75_slack: 0.000000
  - pass_rate: 1.000000

- Separation facts (OK-only; no auto-threshold gating):
separation facts (computed on status==OK rows only):
- BAD-constant: rows_ok=200/200, mean_slack=-0.500000, p25_slack=-0.500000, median_slack=-0.500000, p75_slack=-0.500000, pass_rate=0.000000
- BAD-random: rows_ok=200/200, mean_slack=-0.311111, p25_slack=-1.000000, median_slack=0.000000, p75_slack=0.000000, pass_rate=0.600000
- GOOD-synthetic: rows_ok=200/200, mean_slack=0.000000, p25_slack=0.000000, median_slack=0.000000, p75_slack=0.000000, pass_rate=1.000000

- Δ_mean_slack(GOOD - BAD-constant): 0.500000
- Δ_p25_slack(GOOD - BAD-constant): 0.500000
- Δ_median_slack(GOOD - BAD-constant): 0.500000
- Δ_p75_slack(GOOD - BAD-constant): 0.500000
- Δ_PASS_rate(GOOD - BAD-constant): 1.000000

- Δ_mean_slack(GOOD - BAD-random): 0.311111
- Δ_p25_slack(GOOD - BAD-random): 1.000000
- Δ_median_slack(GOOD - BAD-random): 0.000000
- Δ_p75_slack(GOOD - BAD-random): 0.000000
- Δ_PASS_rate(GOOD - BAD-random): 0.400000

## accuracy-a1-isomers-2026-01-22-r1

- Source commit: 0efe6602c636bc587745f6acfa95c6295dc12f0c
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-22-r1/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): A44624D1B705CA000FEC2DB92EE871E29ACD052E51F30B1C06F7468CF8258A89
- Command:
  python scripts/build_isomer_truth_v1.py
  python scripts/accuracy_a1_isomers_run.py --input_csv data/accuracy/isomer_truth.v1.csv --out_dir out_accuracy_a1_isomers --potential_scale_gamma 1.0
- Outcome (facts from metrics.json):
  - rows_total: 35
  - groups_total: 10
  - mean_spearman_pred_vs_truth: 0.0533647488893285
  - pairwise_order_accuracy_overall: 0.5
  - top1_accuracy_mean: 0.2

## accuracy-a1-isomers-2026-01-22-r2

- Source commit: 13c2ee2d66bad98a811962181b4198c5f271a9d8
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-22-r2/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): E04117E5AB26B7248507AEA21159F98512274B8051ADFFD30D0ADA98F2D4A0D4
- Command:
  python scripts/build_isomer_truth_v1.py
  python scripts/accuracy_a1_isomers_sweep.py --input_csv data/accuracy/isomer_truth.v1.csv --out_dir out_accuracy_a1_isomers_sweep --edge_weight_modes unweighted,bond_order,bond_order_delta_chi --potential_modes static,self_consistent --gammas 0.0,0.25,0.5,1.0 --predictors free_energy_beta,heat_trace_beta,logdet_shifted_eps --betas 0.5,1.0,2.0 --baseline_gamma 1.0
- Outcome (facts from metrics.json):
  - rows_total: 35
  - groups_total: 10
  - verdict: SIGNAL_OK (mean_spearman_pred_vs_truth >= 0.2)
  - baseline (A1.1; H_trace):
    - mean_spearman_pred_vs_truth: -0.022816061641181683
    - pairwise_order_accuracy_overall: 0.5434782608695652
    - top1_accuracy_mean: 0.3
  - best (A1.2; sweep):
    - mean_spearman_pred_vs_truth: 0.3799999999999999
    - pairwise_order_accuracy_overall: 0.6956521739130435
    - top1_accuracy_mean: 0.5
  - best_config:
    - predictor: logdet_shifted_eps
    - edge_weight_mode: unweighted
    - potential_mode: static
    - potential_scale_gamma: 0.25
    - beta: None

## accuracy-a1-isomers-2026-01-22-a1_3-r1

- Source commit: 14970cf76d9f44d8f18e3a1df503e454353717e9
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-22-a1_3-r1/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): D5DE1211A7C6ADF78419A6AA9ADCB8F530E5B6C68985363F70715CAE159361A5
- Command:
  python scripts/build_isomer_truth_v1.py
  python scripts/accuracy_a1_isomers_sweep.py --experiment_id ACCURACY-A1.3 --input_csv data/accuracy/isomer_truth.v1.csv --out_dir out_accuracy_a1_isomers_a1_3 --edge_weight_modes unweighted --potential_modes static --gammas 0.15,0.18,0.20,0.22,0.24,0.25,0.26,0.28,0.30,0.32,0.35 --predictors logdet_shifted_eps --eps_values 1e-6,1e-5,1e-4,1e-3 --shift_values 0.0,1e-4,1e-3,1e-2 --baseline_gamma 1.0 --kpi_mean_spearman_by_group_min 0.55 --kpi_median_spearman_by_group_min 0.50
- Outcome (facts from metrics.json):
  - rows_total: 35
  - groups_total: 10
  - verdict: SIGNAL_OK (mean_spearman_pred_vs_truth >= 0.2)
  - kpi_verdict: FAIL (mean=0.3899999999999999, median=0.5499999999999998)
  - baseline (A1.1; H_trace):
    - mean_spearman_pred_vs_truth: 0.0533647488893285
    - median_spearman_by_group: 0.3054092553389459
    - pairwise_order_accuracy_overall: 0.5
    - pairwise_order_accuracy_by_group_mean: 0.42000000000000004
    - top1_accuracy_mean: 0.2
  - best (A1.3; narrow calibration):
    - mean_spearman_pred_vs_truth: 0.3899999999999999
    - median_spearman_by_group: 0.5499999999999998
    - pairwise_order_accuracy_overall: 0.6739130434782609
    - pairwise_order_accuracy_by_group_mean: 0.6633333333333333
    - top1_accuracy_mean: 0.5
  - best_config:
    - predictor: logdet_shifted_eps
    - edge_weight_mode: unweighted
    - potential_mode: static
    - potential_scale_gamma: 0.28
    - eps: 1e-06
    - shift: 0.0

## accuracy-a1-isomers-2026-01-22-a1_4-r1

- Source commit: 04eefabd05c262646d29ce93830951514646ab52
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-22-a1_4-r1/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): 785DD76FCD254EB46E447693BF10FC4C97BD33468BF3AE7FF850D6201DED864B
- Command:
  python scripts/build_isomer_truth_v1.py
  python scripts/accuracy_a1_isomers_feature_upgrade.py --experiment_id ACCURACY-A1.4 --input_csv data/accuracy/isomer_truth.v1.csv --out_dir out_accuracy_a1_isomers_a1_4 --seed 0 --n_train_groups 7 --gamma 0.28 --potential_variant epsilon_z_plus_features --v_deg_coeff 0.10 --v_arom_coeff 0.20 --v_charge_coeff 0.50 --logdet_eps 1e-6 --logdet_shift 0.0 --heat_betas 0.5,1.0,2.0 --entropy_beta 1.0 --ridge_lambda 1e-3 --kpi_mean_spearman_by_group_test_min 0.55 --kpi_median_spearman_by_group_test_min 0.55 --kpi_pairwise_order_accuracy_overall_test_min 0.65 --kpi_top1_accuracy_mean_test_min 0.40
- Outcome (facts from metrics.json):
  - rows_total: 35
  - groups_total: 10
  - split: train_groups=7, test_groups=3
  - kpi_verdict: FAIL (mean_spearman_by_group_test=0.5999999999999999, median_spearman_by_group_test=0.4999999999999999, pairwise_order_accuracy_overall_test=0.75, top1_accuracy_mean_test=0.3333333333333333)
  - test_metrics:
    - mean_spearman_by_group: 0.5999999999999999
    - median_spearman_by_group: 0.4999999999999999
    - pairwise_order_accuracy_overall: 0.75
    - top1_accuracy_mean: 0.3333333333333333
  - operator:
    - edge_weight_mode: bond_order
    - potential_variant: epsilon_z_plus_features
    - gamma: 0.28

## accuracy-a1-isomers-2026-01-22-a1_5-r2

- Source commit: 9f61b4e55dda142e6fed8668fe074532c4c53d10
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-22-a1_5-r2/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): 101D6FED30C26B3A2B049203C270E51E118E9EB4164F39A579E3FEDF9FBFD7A1
- Command:
  python scripts/build_isomer_truth_v1.py
  python scripts/accuracy_a1_isomers_pairwise_rank.py --experiment_id ACCURACY-A1.5 --input_csv data/accuracy/isomer_truth.v1.csv --out_dir out_accuracy_a1_isomers_a1_5 --seed 0 --gamma 0.28 --model_type pairwise_logistic_l2 --model_l2_lambda 1e-3 --model_lr 0.1 --model_max_iter 2000 --model_tol 1e-6 --kpi_mean_spearman_by_group_test_min 0.55 --kpi_median_spearman_by_group_test_min 0.55 --kpi_pairwise_order_accuracy_overall_test_min 0.70 --kpi_top1_accuracy_mean_test_min 0.40
- Outcome (facts from metrics.json):
  - rows_total: 35
  - groups_total: 10
  - cv_method: LOOCV_GROUP_ID
  - model_type: pairwise_logistic_l2
  - kpi_verdict: FAIL (mean_spearman_by_group_test=0.36999999999999994, median_spearman_by_group_test=0.5499999999999998, pairwise_order_accuracy_overall_test=0.6956521739130435, top1_accuracy_mean_test=0.5)
  - test_metrics:
    - mean_spearman_by_group: 0.36999999999999994
    - median_spearman_by_group: 0.5499999999999998
    - pairwise_order_accuracy_overall: 0.6956521739130435
    - top1_accuracy_mean: 0.5
  - operator:
    - edge_weight_mode: bond_order
    - potential_variant: epsilon_z_plus_features_v2
    - gamma: 0.28
  - worst_groups:
    - C11H21B1N2O4: spearman=-0.4999999999999999, top1=0.0, pairwise_acc=0.3333333333333333
    - C15H24O1: spearman=-0.4999999999999999, top1=0.0, pairwise_acc=0.3333333333333333
    - C21H23N3O3: spearman=-0.4999999999999999, top1=0.0, pairwise_acc=0.3333333333333333

## accuracy-a1-isomers-2026-01-23-a2-r1

- Source commit: 48e530faffd70a67905be03de0ac6a2d85cc0c55
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-23-a2-r1/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): D407C9FD85DFE87130D092E571B49F42B115836801BB80EF0B0C3843DB6E7A72
- Command:
  python scripts/build_isomer_truth_v1.py
  python scripts/accuracy_a1_isomers_a2_self_consistent.py --experiment_id ACCURACY-A2 --input_csv data/accuracy/isomer_truth.v1.csv --out_dir out_accuracy_a1_isomers_a2 --seed 0 --gamma 0.28 --potential_variant epsilon_z --edge_weight_mode bond_order_delta_chi --sc_max_iter 5 --eta_a 0.1 --eta_phi 0.0 --update_clip 0.5 --model_type pairwise_rank_ridge --model_ridge_lambda 1e-3 --kpi_mean_spearman_by_group_test_min 0.55 --kpi_median_spearman_by_group_test_min 0.55 --kpi_pairwise_order_accuracy_overall_test_min 0.70 --kpi_top1_accuracy_mean_test_min 0.40
- Outcome (facts from metrics.json):
  - rows_total: 35
  - groups_total: 10
  - kpi.verdict: FAIL
  - loocv_test:
    - mean_spearman_by_group: 0.15999999999999998
    - median_spearman_by_group: 0.3999999999999999
    - pairwise_order_accuracy_overall: 0.5869565217391305 (27/46)
    - top1_accuracy_mean: 0.4
  - worst_groups:
    - C13H20O1: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C15H24O1: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C22H22N4O2: spearman=-0.4999999999999999, top1=0.0, pairwise_acc=0.3333333333333333

## accuracy-a1-isomers-2026-01-23-a2_1-r1

- Source commit: 089cd1ed09003ed2df0feea038992c0ca107ed34
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-23-a2_1-r1/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): 99109A00B060F39C2C83028EB0D57CD2BC1CB227E74B7741B4000E732D41D2AC
- Command:
  python scripts/build_isomer_truth_v1.py
  python scripts/accuracy_a1_isomers_a2_self_consistent.py --a2_variant full_functional_v1 --experiment_id ACCURACY-A2.1 --input_csv data/accuracy/isomer_truth.v1.csv --out_dir out_accuracy_a1_isomers_a2 --seed 0 --gamma 0.28 --potential_variant epsilon_z --edge_weight_mode bond_order_delta_chi --calibrator_ridge_lambda 1e-3 --kpi_mean_spearman_by_group_test_min 0.55 --kpi_median_spearman_by_group_test_min 0.55 --kpi_pairwise_order_accuracy_overall_test_min 0.70 --kpi_top1_accuracy_mean_test_min 0.40
- Outcome (facts from metrics.json):
  - rows_total: 35
  - groups_total: 10
  - kpi.verdict: FAIL
  - loocv_test_functional_only:
    - mean_spearman_by_group: -0.019999999999999997
    - median_spearman_by_group: 0.24999999999999994
    - pairwise_order_accuracy_overall: 0.4782608695652174 (22/46)
    - top1_accuracy_mean: 0.1
    - num_groups_spearman_negative: 4
  - loocv_test_calibrated_linear:
    - mean_spearman_by_group: -0.059999999999999984
    - median_spearman_by_group: 0.24999999999999994
    - pairwise_order_accuracy_overall: 0.5217391304347826 (24/46)
    - top1_accuracy_mean: 0.2
    - num_groups_spearman_negative: 4
  - worst_groups:
    - C15H24O1: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C22H22N4O2: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C20H25N3O2: spearman=-0.7999999999999998, top1=0.0, pairwise_acc=0.16666666666666666

## accuracy-a1-isomers-2026-01-23-a2_2-r1

- Source commit: 8ee3c424dbe39fd1daab842d3ce2f2786704e80c
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-23-a2_2-r1/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): 24504681610E2C0382C08B2310D93AEFB9C6E54C4059CD5D0CC6ACF2DBDA257A
- Command:
  python scripts/build_isomer_truth_v1.py
  python scripts/accuracy_a1_isomers_a2_self_consistent.py --a2_variant full_functional_v1_a2_2 --experiment_id ACCURACY-A2.2 --input_csv data/accuracy/isomer_truth.v1.csv --out_dir out_accuracy_a1_isomers_a2 --seed 0 --gamma 0.28 --potential_variant epsilon_z --edge_weight_mode bond_order_delta_chi --calibrator_ridge_lambda 1e-3 --kpi_mean_spearman_by_group_test_min -1.0 --kpi_median_spearman_by_group_test_min -1.0 --kpi_pairwise_order_accuracy_overall_test_min 0.60 --kpi_top1_accuracy_mean_test_min 0.30
- Outcome (facts from metrics.json):
  - rows_total: 35
  - groups_total: 10
  - kpi.verdict: FAIL
  - loocv_test_functional_only:
    - mean_spearman_by_group: -0.039999999999999994
    - median_spearman_by_group: 0.3499999999999999
    - pairwise_order_accuracy_overall: 0.5 (23/46)
    - top1_accuracy_mean: 0.2
    - num_groups_spearman_negative: 4
  - loocv_test_calibrated_linear:
    - mean_spearman_by_group: 0.009999999999999998
    - median_spearman_by_group: 0.14999999999999997
    - pairwise_order_accuracy_overall: 0.5217391304347826 (24/46)
    - top1_accuracy_mean: 0.1
    - num_groups_spearman_negative: 4
  - worst_groups:
    - C21H23N3O3: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C15H24O1: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C22H22N4O2: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0

## accuracy-a1-isomers-2026-01-23-a2_3-r1

- Source commit: cb30a724778fe841d577af8b5c6f4d02d0c9e124
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-23-a2_3-r1/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): 10C09C5AEE2D59EA58FA4AF4BEF1B2DEBA76E878F6785ABAC046A59A51AA6D11
- Command:
  python scripts/build_isomer_truth_v1.py
  python scripts/accuracy_a1_isomers_a2_self_consistent.py --a2_variant full_functional_v1_a2_3 --experiment_id ACCURACY-A2.3 --input_csv data/accuracy/isomer_truth.v1.csv --out_dir out_accuracy_a1_isomers_a2 --seed 0 --gamma 0.28 --potential_variant epsilon_z --edge_weight_mode bond_order_delta_chi --calibrator_ridge_lambda 1e-3 --kpi_mean_spearman_by_group_test_min -1.0 --kpi_median_spearman_by_group_test_min -1.0 --kpi_pairwise_order_accuracy_overall_test_min 0.60 --kpi_top1_accuracy_mean_test_min 0.30
- Outcome (facts from metrics.json):
  - rows_total: 35
  - groups_total: 10
  - kpi.verdict: FAIL
  - loocv_test_functional_only:
    - mean_spearman_by_group: -0.039999999999999994
    - median_spearman_by_group: 0.3499999999999999
    - pairwise_order_accuracy_overall: 0.5 (23/46)
    - top1_accuracy_mean: 0.2
    - num_groups_spearman_negative: 4
  - loocv_test_calibrated_linear:
    - mean_spearman_by_group: 0.009999999999999998
    - median_spearman_by_group: 0.14999999999999997
    - pairwise_order_accuracy_overall: 0.5217391304347826 (24/46)
    - top1_accuracy_mean: 0.1
    - num_groups_spearman_negative: 4
  - worst_groups:
    - C21H23N3O3: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C15H24O1: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C22H22N4O2: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0

## accuracy-a1-isomers-2026-01-24-a2_4-r1

- Source commit: f76626fd162a75ea656b14e90ccd56ed4cce3bc9
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/accuracy-a1-isomers-2026-01-24-a2_4-r1/accuracy_a1_isomers_evidence_pack.zip
- SHA256(accuracy_a1_isomers_evidence_pack.zip): 0AA2A16CAF049822BD4DF887F218641A33C4393C3C663036814F7D27D4F86E2B
- Command:
  python scripts/build_isomer_truth_v1.py
  python scripts/accuracy_a1_isomers_a2_self_consistent.py --a2_variant full_functional_v1_a2_4 --experiment_id ACCURACY-A2.4 --input_csv data/accuracy/isomer_truth.v1.csv --out_dir out_accuracy_a1_isomers_a2 --seed 0 --gamma 0.28 --potential_variant epsilon_z --edge_weight_mode bond_order_delta_chi --calibrator_ridge_lambda 1e-3 --kpi_mean_spearman_by_group_test_min -1.0 --kpi_median_spearman_by_group_test_min -1.0 --kpi_pairwise_order_accuracy_overall_test_min 0.60 --kpi_top1_accuracy_mean_test_min 0.30
- Outcome (facts from metrics.json):
  - rows_total: 35
  - groups_total: 10
  - kpi.verdict: FAIL
  - rho_mode_selected.distribution: {"soft_occupancy_ldos": 10}
  - loocv_test_functional_only:
    - mean_spearman_by_group: -0.039999999999999994
    - median_spearman_by_group: 0.09999999999999998
    - pairwise_order_accuracy_overall: 0.5217391304347826 (24/46)
    - top1_accuracy_mean: 0.1
    - num_groups_spearman_negative: 4
  - loocv_test_calibrated_linear:
    - mean_spearman_by_group: -0.11999999999999997
    - median_spearman_by_group: 0.04999999999999999
    - pairwise_order_accuracy_overall: 0.4782608695652174 (22/46)
    - top1_accuracy_mean: 0.3
    - num_groups_spearman_negative: 4
  - worst_groups:
    - C15H24O1: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C22H22N4O2: spearman=-0.9999999999999998, top1=0.0, pairwise_acc=0.0
    - C21H23N3O3: spearman=-0.4999999999999999, top1=0.0, pairwise_acc=0.3333333333333333
