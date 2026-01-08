from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from hetero2.decoys_rewire import generate_rewire_decoys


SMILES_LIST: List[Tuple[str, str]] = [
    ("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("caffeine", "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"),
    ("ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
    ("acetaminophen", "CC(=O)NC1=CC=C(O)C=C1O"),
    ("naproxen", "CC1=CC=C2C(=C1)C=C(C=C2)C(C)C(=O)O"),
    ("salicylic_acid", "O=C(O)C1=CC=CC=C1O"),
    ("benzene", "c1ccccc1"),
    ("toluene", "Cc1ccccc1"),
    ("aniline", "Nc1ccccc1"),
    ("phenol", "Oc1ccccc1"),
]


def _scores_for_smiles(smiles: str, *, k_decoys: int, seed: int, full_coverage: bool) -> Dict[str, Dict[str, float]]:
    decoys = generate_rewire_decoys(smiles, k=k_decoys, seed=seed, max_attempts=None, lock_aromatic=True, allow_ring_bonds=False)
    if not decoys.decoys:
        return {}
    if full_coverage:
        return {str(d["hash"]): {"score": 0.1, "weight": 1.0} for d in decoys.decoys}
    # Provide scores for only the first decoy to ensure partial coverage.
    first_hash = str(decoys.decoys[0]["hash"])
    return {first_hash: {"score": 0.1, "weight": 1.0}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate pilot CSV + scores fixture for external_scores.")
    ap.add_argument("--out_dir", default="out_pilot", help="Output directory for pilot inputs.")
    ap.add_argument("--rows", type=int, default=500, help="Number of rows to generate.")
    ap.add_argument("--k_decoys", type=int, default=2, help="Decoys per molecule.")
    ap.add_argument("--seed", type=int, default=0, help="Seed for decoy generation.")
    ap.add_argument(
        "--full_cover_count",
        type=int,
        default=3,
        help="Number of seed molecules to fully cover with decoy scores (ensures OK rows).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_csv = out_dir / "input.csv"
    scores_json = out_dir / "scores.json"

    rows = []
    for idx in range(int(args.rows)):
        name, smiles = SMILES_LIST[idx % len(SMILES_LIST)]
        rows.append(f"{name}_{idx},{smiles}")
    input_csv.write_text("id,smiles\n" + "\n".join(rows) + "\n", encoding="utf-8")

    decoy_scores: Dict[str, Dict[str, float]] = {}
    for idx, (_, smiles) in enumerate(SMILES_LIST):
        full_coverage = idx < int(args.full_cover_count)
        decoy_scores.update(
            _scores_for_smiles(
                smiles,
                k_decoys=int(args.k_decoys),
                seed=int(args.seed),
                full_coverage=full_coverage,
            )
        )

    payload = {
        "schema_version": "hetero_scores.v1",
        "original": {"score": 1.0, "weight": 1.0},
        "decoys": decoy_scores,
    }
    scores_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {input_csv} and {scores_json}")


if __name__ == "__main__":
    main()
