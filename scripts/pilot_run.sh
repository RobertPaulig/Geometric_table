#!/usr/bin/env bash
set -euo pipefail

python scripts/pilot_generate_input.py --out_dir out_pilot --rows 500 --k_decoys 2 --seed 0

docker build -t hetero2:pilot .
docker run --rm -v "$(pwd)/out_pilot:/out" hetero2:pilot \
  hetero2-batch \
  --input /out/input.csv \
  --out_dir /out/out \
  --score_mode external_scores \
  --scores_input /out/scores.json \
  --artifacts light \
  --zip_pack \
  --k_decoys 2 \
  --workers 1 \
  --timeout_s 60

echo "Evidence pack: out_pilot/out/evidence_pack.zip"
