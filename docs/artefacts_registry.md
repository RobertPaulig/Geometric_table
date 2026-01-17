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
