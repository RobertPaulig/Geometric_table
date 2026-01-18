from __future__ import annotations

from dataclasses import dataclass

from hetero2.chemgraph import _require_rdkit
from hetero2.decoys_rewire import DecoyResult, generate_rewire_decoys

DECOY_STRATEGY_SCHEMA = "decoy_strategy.v1"
DECOY_STRATEGY_STRICT = "rewire_strict_v1"
DECOY_STRATEGY_RELAX_A = "rewire_relax_a_v1"
DECOY_STRATEGY_RELAX_B = "rewire_relax_b_v1"
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
    hard_mode: bool = False,
    hard_tanimoto_min: float = 0.65,
    hard_tanimoto_max: float = 0.95,
) -> tuple[DecoyResult, DecoyStrategy]:
    Chem, _, _, _, _ = _require_rdkit()
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        raise ValueError("Invalid SMILES")
    canonical_ref = Chem.MolToSmiles(base)

    strict_candidates = _bond_candidates_count(base, lock_aromatic=True, allow_ring_bonds=False)
    relax_a_candidates = _bond_candidates_count(base, lock_aromatic=False, allow_ring_bonds=False)
    relax_b_candidates = _bond_candidates_count(base, lock_aromatic=True, allow_ring_bonds=True)
    coverage_markers = [
        f"candidates_strict:{strict_candidates}",
        f"candidates_relax_a:{relax_a_candidates}",
        f"candidates_relax_b:{relax_b_candidates}",
    ]

    strict_stats: dict[str, int] | None = None
    warnings_accum: list[str] = []

    if strict_candidates >= 2:
        strict = generate_rewire_decoys(
            smiles,
            k=int(k),
            seed=int(seed),
            max_attempts=max_attempts,
            lock_aromatic=True,
            allow_ring_bonds=False,
            hard_tanimoto_min=float(hard_tanimoto_min) if hard_mode else None,
            hard_tanimoto_max=float(hard_tanimoto_max) if hard_mode else None,
            hard_tanimoto_ref_smiles=canonical_ref if hard_mode else None,
        )
        if strict.decoys:
            strict_warnings = sorted(set(strict.warnings + coverage_markers + [f"decoy_strategy_used:{DECOY_STRATEGY_STRICT}"]))
            return (
                DecoyResult(decoys=strict.decoys, warnings=strict_warnings, stats=strict.stats),
                DecoyStrategy(schema_version=DECOY_STRATEGY_SCHEMA, strategy_id=DECOY_STRATEGY_STRICT),
            )
        warnings_accum.extend(strict.warnings)
        strict_stats = dict(strict.stats)
    else:
        warnings_accum.append(f"strict_strategy_no_candidate_swap:{strict_candidates}")
        strict_stats = {"attempts": 0, "sanitize_fail": 0, "duplicate": 0, "no_candidate_swap": 0}

    if relax_a_candidates >= 2:
        relax_a = generate_rewire_decoys(
            smiles,
            k=int(k),
            seed=int(seed),
            max_attempts=max_attempts,
            lock_aromatic=False,
            allow_ring_bonds=False,
            hard_tanimoto_min=float(hard_tanimoto_min) if hard_mode else None,
            hard_tanimoto_max=float(hard_tanimoto_max) if hard_mode else None,
            hard_tanimoto_ref_smiles=canonical_ref if hard_mode else None,
        )
        if relax_a.decoys:
            relax_a_warnings = sorted(set(relax_a.warnings + warnings_accum + coverage_markers + [f"decoy_strategy_used:{DECOY_STRATEGY_RELAX_A}"]))
            return (
                DecoyResult(decoys=relax_a.decoys, warnings=relax_a_warnings, stats=relax_a.stats),
                DecoyStrategy(schema_version=DECOY_STRATEGY_SCHEMA, strategy_id=DECOY_STRATEGY_RELAX_A),
            )
        warnings_accum.extend(relax_a.warnings)

    if relax_b_candidates >= 2:
        relax_b = generate_rewire_decoys(
            smiles,
            k=int(k),
            seed=int(seed),
            max_attempts=max_attempts,
            lock_aromatic=True,
            allow_ring_bonds=True,
            hard_tanimoto_min=float(hard_tanimoto_min) if hard_mode else None,
            hard_tanimoto_max=float(hard_tanimoto_max) if hard_mode else None,
            hard_tanimoto_ref_smiles=canonical_ref if hard_mode else None,
        )
        if relax_b.decoys:
            relax_b_warnings = sorted(set(relax_b.warnings + warnings_accum + coverage_markers + [f"decoy_strategy_used:{DECOY_STRATEGY_RELAX_B}"]))
            return (
                DecoyResult(decoys=relax_b.decoys, warnings=relax_b_warnings, stats=relax_b.stats),
                DecoyStrategy(schema_version=DECOY_STRATEGY_SCHEMA, strategy_id=DECOY_STRATEGY_RELAX_B),
            )
        warnings_accum.extend(relax_b.warnings)

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
        hard_tanimoto_min=float(hard_tanimoto_min) if hard_mode else None,
        hard_tanimoto_max=float(hard_tanimoto_max) if hard_mode else None,
        hard_tanimoto_ref_smiles=canonical_ref if hard_mode else None,
    )
    if relaxed.decoys:
        merged_warnings = sorted(
            set(
                warnings_accum
                + fallback_warnings
                + relaxed.warnings
                + coverage_markers
                + ["decoy_fallback_used:1", f"decoy_strategy_used:{DECOY_STRATEGY_FALLBACK}"]
            )
        )
        return (
            DecoyResult(decoys=relaxed.decoys, warnings=merged_warnings, stats=relaxed.stats),
            DecoyStrategy(schema_version=DECOY_STRATEGY_SCHEMA, strategy_id=DECOY_STRATEGY_FALLBACK),
        )

    merged_warnings = sorted(
        set(warnings_accum + fallback_warnings + relaxed.warnings + coverage_markers + [f"decoy_strategy_used:{DECOY_STRATEGY_FALLBACK}"])
    )
    return (
        DecoyResult(decoys=[], warnings=merged_warnings, stats=strict_stats or relaxed.stats),
        DecoyStrategy(schema_version=DECOY_STRATEGY_SCHEMA, strategy_id=DECOY_STRATEGY_FALLBACK),
    )

