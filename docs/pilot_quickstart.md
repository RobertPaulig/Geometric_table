# Pilot Quickstart (2-minute run)

This is a minimal end-to-end run that generates an evidence pack with spectral columns and external score provenance.

## Prereqs

- Docker installed
- Python available (for generating the pilot inputs)

## Steps

### 1) Generate inputs + run batch in Docker (Linux/macOS)

```bash
scripts/pilot_run.sh
```

### 2) Generate inputs + run batch in Docker (Windows PowerShell)

```powershell
.\scripts\pilot_run.ps1
```

## Outputs

After the run:

- Evidence pack zip: `out_pilot/out/evidence_pack.zip`
- Summary CSV: `out_pilot/out/summary.csv` (includes `spectral_gap`, `spectral_entropy`, `spectral_entropy_norm`)
- Manifest: `out_pilot/out/manifest.json` (includes `scores_provenance`)
- Metrics: `out_pilot/out/metrics.json` (includes `scores_coverage`)

## Expected Output

- Some rows should be `OK` (scores cover decoys), some `SKIP` (no decoys), and `ERROR` should be 0.
- `scores_coverage` should show nonzero `rows_with_scores_input` and `decoys_scored`.

## Verify SHA256

```powershell
Get-FileHash out_pilot\out\evidence_pack.zip -Algorithm SHA256
```

## Notes

- The pilot scores fixture only covers a subset of decoy hashes to exercise warnings and coverage counts.
- This is diagnostics-only; no gating based on spectral metrics.
- Docker E2E: see docs/docker_e2e_quickstart.md.

## Pilot Demo сценарий (1-страница)

### 1) Сгенерировать input + scores

```bash
python scripts/pilot_generate_input.py --out_dir out_pilot --rows 1000 --k_decoys 2 --seed 0
```

### 2) Запустить batch

```bash
hetero2-batch \
  --input out_pilot/input.csv \
  --out_dir out_pilot \
  --artifacts light \
  --score_mode external_scores \
  --scores_input out_pilot/scores.json \
  --k_decoys 2 \
  --workers 2 \
  --timeout_s 60 \
  --maxtasksperchild 100 \
  --seed_strategy per_row \
  --seed 0 \
  --zip_pack
```

### 3) Ожидаемый результат

- В `metrics.json`: `ERROR=0`, `OK>0`, `scores_coverage.decoys_scored > 0`.
- В `summary.csv` есть `spectral_gap`, `spectral_entropy`, `spectral_entropy_norm`.

### 4) Эталонный артефакт

- Release: https://github.com/RobertPaulig/Geometric_table/releases/tag/pilot-2026-01-08-r2
- Asset: https://github.com/RobertPaulig/Geometric_table/releases/download/pilot-2026-01-08-r2/pilot_evidence_pack.zip
