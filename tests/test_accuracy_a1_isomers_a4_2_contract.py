import csv
import json
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest


RUN_A4 = os.environ.get("RUN_A4_TESTS") == "1"
pytestmark = pytest.mark.skipif(not RUN_A4, reason="A4.2 contract tests are opt-in; set RUN_A4_TESTS=1")


if RUN_A4:
    from rdkit import Chem

    from hetero2.phase_channel import magnetic_laplacian, phase_matrix_flux_on_cycles, sssr_cycles_from_mol


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _heat_kernel_matrix_shifted_trace_normalized_hermitian(H: np.ndarray, *, tau: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(np.asarray(H, dtype=np.complex128))
    eigvals = np.asarray(eigvals).reshape(-1)
    lambda_min = float(np.real(eigvals[0]))
    weights = np.exp(-float(tau) * (np.real(eigvals) - lambda_min))
    weights = weights / float(np.sum(weights))
    return (eigvecs * weights) @ np.conj(eigvecs).T


def test_a4_2_sssr_cycle_order_and_orientation_are_deterministic() -> None:
    mol = Chem.MolFromSmiles("c1cccc2c1cccc2")  # naphthalene
    assert mol is not None

    cycles = sssr_cycles_from_mol(mol)
    assert cycles == [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9]]

    # Orientation rule: start at min atom, choose lexicographically-min direction.
    for cyc in cycles:
        assert cyc[0] == min(cyc)
        alt = [cyc[0]] + list(reversed(cyc[1:]))
        assert tuple(cyc) <= tuple(alt)

    # Deterministic ordering across rings.
    assert cycles == sorted(cycles, key=lambda c: (len(c), tuple(c)))


def test_a4_2_r_edge_orientation_is_conjugate() -> None:
    # Toy 3-cycle with uniform weights.
    w = np.zeros((3, 3), dtype=float)
    w[0, 1] = w[1, 0] = 1.0
    w[1, 2] = w[2, 1] = 1.0
    w[2, 0] = w[0, 2] = 1.0

    phi = math.pi / 2.0
    A = phase_matrix_flux_on_cycles(n=3, cycles=[[0, 1, 2]], phi=phi)
    H = magnetic_laplacian(weights=w, A=A)
    K = _heat_kernel_matrix_shifted_trace_normalized_hermitian(H, tau=1.0)

    for i, j in [(0, 1), (1, 2), (2, 0)]:
        q_ij = np.exp(-1j * float(A[i, j])) * K[i, j]
        q_ji = np.exp(-1j * float(A[j, i])) * K[j, i]

        assert q_ji == pytest.approx(np.conj(q_ij), abs=1e-12)

        mag = float(abs(q_ij))
        r_ij = (q_ij / mag) if mag > 0.0 else (1.0 + 0.0j)
        r_ji = (q_ji / mag) if mag > 0.0 else (1.0 + 0.0j)
        assert r_ji == pytest.approx(np.conj(r_ij), abs=1e-12)


def test_accuracy_a1_isomers_a4_2_cycle_basis_sssr_contract(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    repo_root = _repo_root()

    input_csv = repo_root / "data/accuracy/isomer_truth.v1.csv"
    assert input_csv.exists()

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/accuracy_a1_isomers_a4_2_cycle_basis_sssr.py",
        "--experiment_id",
        "ACCURACY-A4.2",
        "--input_csv",
        input_csv.as_posix(),
        "--out_dir",
        out_dir.as_posix(),
        "--seed",
        "0",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    expected = [
        out_dir / "predictions.csv",
        out_dir / "summary.csv",
        out_dir / "group_metrics.csv",
        out_dir / "cycle_list.csv",
        out_dir / "cycle_phi.csv",
        out_dir / "isomorphism_ceiling.csv",
        out_dir / "best_config.json",
        out_dir / "metrics.json",
        out_dir / "index.md",
        out_dir / "provenance.json",
        out_dir / "manifest.json",
        out_dir / "checksums.sha256",
        out_dir / "evidence_pack.zip",
    ]
    for p in expected:
        assert p.exists()

    preds = list(csv.DictReader((out_dir / "predictions.csv").read_text(encoding="utf-8").splitlines()))
    assert len(preds) == 35
    assert {
        "id",
        "group_id",
        "smiles",
        "truth_rel_kcalmol",
        "pred_raw",
        "pred_rel",
        "phi_fixed",
        "heat_tau",
        "edge_weight_mode",
    }.issubset(set(preds[0].keys()))

    cycle_list = list(csv.DictReader((out_dir / "cycle_list.csv").read_text(encoding="utf-8").splitlines()))
    assert len(cycle_list) >= 1
    assert {"group_id", "id", "cycle_id", "cycle_len", "cycle_nodes"}.issubset(set(cycle_list[0].keys()))

    cycle_phi = list(csv.DictReader((out_dir / "cycle_phi.csv").read_text(encoding="utf-8").splitlines()))
    assert len(cycle_phi) >= 1
    assert {
        "group_id",
        "id",
        "cycle_id",
        "phi_C",
        "sin2_phi",
        "w_C",
        "contrib",
        "trace_heat",
        "lambda_min",
        "lambda_max",
    }.issubset(set(cycle_phi[0].keys()))
    assert all(math.isfinite(float(r["contrib"])) for r in cycle_phi)

    ceiling = list(csv.DictReader((out_dir / "isomorphism_ceiling.csv").read_text(encoding="utf-8").splitlines()))
    assert len(ceiling) >= 1
    assert {"group_id", "a_id", "b_id", "isomorphic_H", "delta_truth"}.issubset(set(ceiling[0].keys()))

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == "accuracy_a1_isomers_a4_2.v1"
    assert metrics["kpi"]["verdict"] in {"PASS", "FAIL"}
    assert metrics["config"]["variant"] == "sssr"
    assert metrics["best_config"]["zero_dof"] is True

    zip_path = out_dir / "evidence_pack.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for required in [
            "predictions.csv",
            "group_metrics.csv",
            "cycle_list.csv",
            "cycle_phi.csv",
            "isomorphism_ceiling.csv",
            "metrics.json",
            "index.md",
            "best_config.json",
            "provenance.json",
            "manifest.json",
            "checksums.sha256",
            "data/accuracy/isomer_truth.v1.csv",
            "docs/contracts/isomer_truth.v1.md",
            "data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv",
            "data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv.sha256",
            "data/atoms_db_v1.json",
            "docs/specs/accuracy_a4_2_cycle_basis_sssr.md",
        ]:
            assert required in names
