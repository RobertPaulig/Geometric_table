from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

from hetero2.chemgraph import ChemGraph
from hetero2.decoys_rewire import generate_rewire_decoys
from hetero2.spectral import compute_stability_metrics, laplacian_eigvals


SMILES_LIST = [
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
    ("pyridine", "n1ccccc1"),
    ("indole", "c1cc2ccccc2[nH]1"),
    ("nicotine", "CN1CCC[C@H]1c2cccnc2"),
    ("cysteine", "N[C@@H](CS)C(=O)O"),
    ("alanine", "CC(N)C(=O)O"),
    ("lysine", "NCCCC[C@H](N)C(=O)O"),
    ("glucose", "C(C1C(C(C(C(O1)O)O)O)O)O"),
    ("urea", "C(=O)(N)N"),
    ("ethanol", "CCO"),
    ("acetone", "CC(=O)C"),
    ("chloroform", "ClC(Cl)Cl"),
    ("acetamide", "CC(=O)N"),
    ("imidazole", "c1ncc[nH]1"),
    ("piperidine", "C1CCNCC1"),
]


def _metrics_for_smiles(smiles: str) -> Dict[str, float]:
    cg = ChemGraph(smiles)
    eigvals = laplacian_eigvals(cg.laplacian())
    return compute_stability_metrics(eigvals)


def main() -> Path:
    out_dir = Path("out_calib")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "spectral_calibration.csv"

    rows: List[Dict[str, object]] = []
    for name, smiles in SMILES_LIST:
        try:
            metrics = _metrics_for_smiles(smiles)
            rows.append(
                {
                    "id": name,
                    "kind": "orig",
                    "decoy_idx": "",
                    "status": "OK",
                    "reason": "",
                    "spectral_gap": metrics["spectral_gap"],
                    "spectral_entropy": metrics["spectral_entropy"],
                    "spectral_entropy_norm": metrics["spectral_entropy_norm"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "id": name,
                    "kind": "orig",
                    "decoy_idx": "",
                    "status": "ERROR",
                    "reason": repr(exc),
                    "spectral_gap": float("nan"),
                    "spectral_entropy": float("nan"),
                    "spectral_entropy_norm": float("nan"),
                }
            )
            continue

        decoys_result = generate_rewire_decoys(
            smiles,
            k=15,
            seed=0,
            max_attempts=None,
            lock_aromatic=True,
            allow_ring_bonds=False,
        )
        if not decoys_result.decoys:
            rows.append(
                {
                    "id": name,
                    "kind": "decoy",
                    "decoy_idx": "",
                    "status": "SKIP",
                    "reason": "no_decoys_generated",
                    "spectral_gap": float("nan"),
                    "spectral_entropy": float("nan"),
                    "spectral_entropy_norm": float("nan"),
                }
            )
            continue

        for idx, decoy in enumerate(decoys_result.decoys):
            smi = str(decoy.get("smiles", ""))
            try:
                metrics = _metrics_for_smiles(smi)
                gap = metrics["spectral_gap"]
                ent = metrics["spectral_entropy"]
                ent_norm = metrics["spectral_entropy_norm"]
                status = "OK"
                reason = ""
            except Exception as exc:
                gap = float("nan")
                ent = float("nan")
                ent_norm = float("nan")
                status = "ERROR"
                reason = repr(exc)
            rows.append(
                {
                    "id": name,
                    "kind": "decoy",
                    "decoy_idx": idx,
                    "status": status,
                    "reason": reason,
                    "spectral_gap": gap,
                    "spectral_entropy": ent,
                    "spectral_entropy_norm": ent_norm,
                }
            )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id",
            "kind",
            "decoy_idx",
            "status",
            "reason",
            "spectral_gap",
            "spectral_entropy",
            "spectral_entropy_norm",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


if __name__ == "__main__":
    path = main()
    print(f"Wrote {path.as_posix()}")
