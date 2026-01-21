import zlib

import pytest

pytest.importorskip("rdkit")

from hetero2.decoy_strategy import DECOY_STRATEGY_FALLBACK, generate_decoys_v1


def _stable_hash_id(text: str) -> int:
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


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
]


@pytest.mark.parametrize("j", [0, 1, 2])
@pytest.mark.parametrize(("name", "smiles"), SMILES_LIST)
def test_value_m1_1_ring_suite_decoys_coverage_smoke(name: str, smiles: str, j: int) -> None:
    idx = SMILES_LIST.index((name, smiles))
    mol_id = f"{name}_{idx + 10 * j}"
    seed_used = _stable_hash_id(mol_id) ^ 0

    decoys_result, strategy = generate_decoys_v1(smiles, k=1, seed=seed_used, max_attempts=None)
    assert len(decoys_result.decoys) >= 1, f"id={mol_id} seed={seed_used} warnings={decoys_result.warnings!r}"
    assert any(w.startswith("decoy_strategy_used:") for w in decoys_result.warnings), f"Missing marker in warnings={decoys_result.warnings!r}"

    if name == "benzene":
        assert strategy.strategy_id == DECOY_STRATEGY_FALLBACK

