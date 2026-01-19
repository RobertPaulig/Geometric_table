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
