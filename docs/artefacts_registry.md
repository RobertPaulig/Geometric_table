# Artefacts Registry

## stress_10k (light) — 2026-01-05

- Commit: https://github.com/RobertPaulig/Geometric_table/commit/9c54b043047010f165549e42399bf265a9395198
- Release asset: https://github.com/RobertPaulig/Geometric_table/releases/download/stress-10k-2026-01-05/evidence_pack.zip
- SHA256(evidence_pack.zip): 458EF5315D52B17B7797F67F3B89E2A091A12058406C9876DCC6D9925E95C76B
- Command:
  hetero2-batch --input stress_10k.csv --out_dir out_stress_10k --artifacts light --score_mode mock --k_decoys 2 --workers 1 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack
- Pack contents (expected):
  summary.csv, metrics.json, index.md, manifest.json, checksums.sha256, evidence_pack.zip
- Outcome:
  OK=7200, SKIP=2800 (no_decoys_generated=1800, too_many_atoms=500, disconnected=250, invalid_smiles=208, missing_smiles=42), ERROR=0

