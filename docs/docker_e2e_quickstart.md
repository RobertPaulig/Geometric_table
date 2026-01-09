# Docker E2E Quickstart (pilot pack)

Goal: one command -> local evidence pack zip with manifest/checksums.

## Command (Linux/macOS)

```bash
docker build -t hetero2:pilot . && \
docker run --rm -v "$PWD/out_pilot_docker:/out" hetero2:pilot /bin/bash -lc "\
python scripts/pilot_generate_input.py --out_dir /out/pilot --rows 200 --k_decoys 2 --seed 0 --full_cover_count 3 && \
hetero2-batch --input /out/pilot/input.csv --out_dir /out/pilot --artifacts light --score_mode external_scores --scores_input /out/pilot/scores.json --k_decoys 2 --workers 2 --timeout_s 60 --maxtasksperchild 100 --seed_strategy per_row --seed 0 --zip_pack"
```

## Expected outputs

- `/out_pilot_docker/pilot/evidence_pack.zip`
- `/out_pilot_docker/pilot/metrics.json` (ERROR=0, scores_coverage > 0)
- `/out_pilot_docker/pilot/manifest.json` and `/out_pilot_docker/pilot/checksums.sha256`

## Reference artifact

- Release: https://github.com/RobertPaulig/Geometric_table/releases/tag/pilot-2026-01-08-r2
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/pilot-2026-01-08-r2/pilot_evidence_pack.zip
