from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

from analysis.experimental.ray_audit import phi_from_smiles
from hetero2.decoys_rewire import generate_rewire_decoys


MOLECULES = {
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "acetaminophen": "CC(=O)NC1=CC=C(O)C=C1O",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(O)=O",
    "naproxen": "COC1=CC=CC2=C1C=CC(C(O)=O)=C2CC",
    "salicylic_acid": "C1=CC=C(C(=C1)C(=O)O)O",
}


def compute_phi(smiles: str, k: int = 30, seed: int = 0) -> dict:
    decoys = generate_rewire_decoys(smiles, k=k, seed=seed, max_attempts=k * 200).decoys
    phi_orig = phi_from_smiles(smiles, scale=300)
    phi_decoys = [phi_from_smiles(d["smiles"], scale=300) for d in decoys]
    phi_decoys = [x for x in phi_decoys if x == x]
    return {
        "phi_original": phi_orig,
        "phi_decoys_median": median(phi_decoys) if phi_decoys else float("nan"),
        "phi_decoys_min": min(phi_decoys) if phi_decoys else float("nan"),
        "phi_decoys_max": max(phi_decoys) if phi_decoys else float("nan"),
        "count": len(phi_decoys),
    }


def main() -> int:
    out_dir = Path("results/phi")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "bench.csv"

    rows = []
    for name, smi in MOLECULES.items():
        stats = compute_phi(smi, k=40, seed=0)
        delta = stats["phi_decoys_median"] - stats["phi_original"] if stats["count"] > 0 else float("nan")
        ratio = stats["phi_decoys_median"] / stats["phi_original"] if stats["phi_original"] else float("nan")
        row = {
            "molecule": name,
            "phi_original": stats["phi_original"],
            "phi_decoys_median": stats["phi_decoys_median"],
            "phi_decoys_min": stats["phi_decoys_min"],
            "phi_decoys_max": stats["phi_decoys_max"],
            "count": stats["count"],
            "delta": delta,
            "ratio": ratio,
        }
        rows.append(row)
        print(row)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "molecule",
                "phi_original",
                "phi_decoys_median",
                "phi_decoys_min",
                "phi_decoys_max",
                "count",
                "delta",
                "ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv} (git-ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
