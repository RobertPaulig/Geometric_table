from __future__ import annotations

from dataclasses import dataclass
from typing import List


MAX_ATOMS_DEFAULT = 200


@dataclass
class PreflightResult:
    ok: bool
    canonical_smiles: str
    warnings: List[str]
    skip_reason: str | None = None


def _require_rdkit():
    try:
        from rdkit import Chem  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("RDKit is required for hetero2 guardrails. Install: pip install -e \".[dev,chem]\"") from exc
    return Chem


def preflight_smiles(smiles: str, *, max_atoms: int = MAX_ATOMS_DEFAULT, require_connected: bool = True) -> PreflightResult:
    """Lightweight validation before heavy pipeline steps."""
    Chem = _require_rdkit()
    warnings: List[str] = []
    skip_reason: str | None = None
    canonical_smiles = smiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        skip_reason = "invalid_smiles"
        warnings.append("skip:invalid_smiles")
        return PreflightResult(False, canonical_smiles, sorted(set(warnings)), skip_reason)

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    n_heavy_atoms = int(mol.GetNumHeavyAtoms())
    if max_atoms and n_heavy_atoms > max_atoms:
        skip_reason = "too_many_atoms"
        warnings.append(f"skip:too_large:n_heavy_atoms={n_heavy_atoms}:limit={max_atoms}")

    if require_connected:
        components = Chem.GetMolFrags(mol, sanitizeFrags=False)
        if len(components) > 1:
            skip_reason = skip_reason or "disconnected"
            warnings.append(f"skip:disconnected:n_components={len(components)}")

    ok = skip_reason is None
    return PreflightResult(ok, canonical_smiles, sorted(set(warnings)), skip_reason)
