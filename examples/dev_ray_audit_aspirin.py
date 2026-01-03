from __future__ import annotations

import json
from pathlib import Path
from statistics import median

from analysis.experimental.ray_audit import phi_from_smiles
from hetero2.decoys_rewire import generate_rewire_decoys


def main() -> int:
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    decoys = generate_rewire_decoys(smiles, k=20, seed=0).decoys
    phi_orig = phi_from_smiles(smiles)
    phi_decoys = [phi_from_smiles(d["smiles"]) for d in decoys]
    phi_decoys = [x for x in phi_decoys if x == x]
    summary = {
        "phi_original": phi_orig,
        "phi_decoys_median": median(phi_decoys) if phi_decoys else float("nan"),
        "phi_decoys_min": min(phi_decoys) if phi_decoys else float("nan"),
        "phi_decoys_max": max(phi_decoys) if phi_decoys else float("nan"),
        "count": len(phi_decoys),
    }
    out_dir = Path("results/phi")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "aspirin_phi.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_file} (git-ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
