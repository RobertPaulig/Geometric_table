from __future__ import annotations

from dataclasses import dataclass

from hetero2.chemgraph import _require_rdkit
from hetero2.decoys_rewire import DecoyResult, generate_rewire_decoys

DECOY_STRATEGY_SCHEMA = "decoy_strategy.v1"
DECOY_STRATEGY_STRICT = "rewire_strict_v1"
DECOY_STRATEGY_FALLBACK = "rewire_fallback_aromatic_as_single_v1"


@dataclass(frozen=True)
class DecoyStrategy:
    schema_version: str
    strategy_id: str


def _bond_candidates_count(mol, *, lock_aromatic: bool, allow_ring_bonds: bool) -> int:
    count = 0
    for bond in mol.GetBonds():
        if bond.GetBondType().name != "SINGLE":
            continue
        if lock_aromatic and (bond.GetIsAromatic() or bond.GetBeginAtom().GetIsAromatic() or bond.GetEndAtom().GetIsAromatic()):
            continue
        if not allow_ring_bonds and bond.IsInRing():
            continue
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        if u == v:
            continue
        count += 1
    return count


def _aromatic_bonds_to_single_smiles(smiles: str) -> tuple[str, bool]:
    Chem, _, _, _, _ = _require_rdkit()
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        raise ValueError("Invalid SMILES")

    has_aromatic = any(b.GetIsAromatic() for b in base.GetBonds()) or any(a.GetIsAromatic() for a in base.GetAtoms())
    if not has_aromatic:
        return Chem.MolToSmiles(base), False

    rw = Chem.RWMol(base)
    for atom in rw.GetAtoms():
        if atom.GetIsAromatic():
            atom.SetIsAromatic(False)
    for bond in rw.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondType().name == "AROMATIC":
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)
    mol2 = rw.GetMol()
    Chem.SanitizeMol(mol2)
    return Chem.MolToSmiles(mol2), True


def generate_decoys_v1(
    smiles: str,
    *,
    k: int,
    seed: int,
    max_attempts: int | None = None,
) -> tuple[DecoyResult, DecoyStrategy]:
    Chem, _, _, _, _ = _require_rdkit()
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        raise ValueError("Invalid SMILES")

    strict_candidates = _bond_candidates_count(base, lock_aromatic=True, allow_ring_bonds=False)
    if strict_candidates >= 2:
        strict = generate_rewire_decoys(
            smiles,
            k=int(k),
            seed=int(seed),
            max_attempts=max_attempts,
            lock_aromatic=True,
            allow_ring_bonds=False,
        )
        if strict.decoys:
            return strict, DecoyStrategy(schema_version=DECOY_STRATEGY_SCHEMA, strategy_id=DECOY_STRATEGY_STRICT)
        strict_warnings = list(strict.warnings)
        strict_stats = dict(strict.stats)
    else:
        strict_warnings = [f"strict_strategy_no_candidate_swap:{strict_candidates}"]
        strict_stats = {"attempts": 0, "sanitize_fail": 0, "duplicate": 0, "no_candidate_swap": 0}

    fallback_warnings: list[str] = []
    fallback_smiles = smiles
    try:
        fallback_smiles, converted = _aromatic_bonds_to_single_smiles(smiles)
        if converted:
            fallback_warnings.append("decoy_fallback_aromatic_as_single:1")
    except Exception as exc:
        fallback_warnings.append(f"decoy_fallback_aromatic_as_single_failed:{exc.__class__.__name__}")

    relaxed = generate_rewire_decoys(
        fallback_smiles,
        k=int(k),
        seed=int(seed),
        max_attempts=max_attempts,
        lock_aromatic=False,
        allow_ring_bonds=True,
    )
    if relaxed.decoys:
        merged_warnings = sorted(set(strict_warnings + fallback_warnings + relaxed.warnings + ["decoy_fallback_used:1"]))
        return (
            DecoyResult(decoys=relaxed.decoys, warnings=merged_warnings, stats=relaxed.stats),
            DecoyStrategy(schema_version=DECOY_STRATEGY_SCHEMA, strategy_id=DECOY_STRATEGY_FALLBACK),
        )

    merged_warnings = sorted(set(strict_warnings + fallback_warnings + relaxed.warnings))
    return (
        DecoyResult(decoys=[], warnings=merged_warnings, stats=strict_stats or relaxed.stats),
        DecoyStrategy(schema_version=DECOY_STRATEGY_SCHEMA, strategy_id=DECOY_STRATEGY_FALLBACK),
    )

