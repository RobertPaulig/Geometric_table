from __future__ import annotations

"""
ACCURACY-A4.2 - Cycle-basis / deterministic SSSR holonomy (opt-in, 0 DOF).

Contract SoT:
  docs/specs/accuracy_a4_2_cycle_basis_sssr.md
"""

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from hetero2.chemgraph import ChemGraph
from hetero2.phase_channel import (
    magnetic_laplacian,
    normalize_flux_phi,
    phase_matrix_flux_on_cycles,
    sssr_cycles_from_mol,
)
from hetero2.physics_operator import AtomsDbV1, load_atoms_db_v1


class AccuracyA42Error(ValueError):
    pass


DEFAULT_INPUT_CSV = Path("data/accuracy/isomer_truth.v1.csv")
DEFAULT_OUT_DIR = Path("out_accuracy_a1_isomers_a4_2")

# 0 DOF: fixed parameters (must not be tuned).
PHI_FIXED = float(normalize_flux_phi(math.pi / 2.0))
HEAT_TAU = 1.0
EDGE_WEIGHT_MODE = "bond_order_delta_chi"
EDGE_AROMATIC_MULT = 0.0
EDGE_DELTA_CHI_ALPHA = 1.0

# Scenario C ceiling-test groups (from A4.2 SoT).
CEILING_GROUPS = ("C11H21B1N2O4", "C15H24O1", "C20H22N2O2")


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_git_sha() -> str:
    sha = str(os.environ.get("GITHUB_SHA") or "").strip()
    if sha:
        return sha
    try:
        repo_root = Path(__file__).resolve().parents[1]
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL, text=True)
        sha = str(out or "").strip()
        if sha:
            return sha
    except Exception:
        pass
    return "UNKNOWN"


def _require_columns(fieldnames: list[str] | None, required: Iterable[str]) -> None:
    if not fieldnames:
        raise AccuracyA42Error("missing CSV header")
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise AccuracyA42Error(f"missing required columns: {missing}")


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted((float(v), i) for i, v in enumerate(values))
    ranks: list[float] = [0.0] * len(values)
    i = 0
    n = len(indexed)
    while i < n:
        j = i + 1
        while j < n and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = (float(i + 1) + float(j)) / 2.0
        for k in range(i, j):
            ranks[indexed[k][1]] = float(avg_rank)
        i = j
    return ranks


def _pearson_corr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx = float(sum(x) / len(x))
    my = float(sum(y) / len(y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return float(cov / (sx * sy))


def _spearman_corr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    return _pearson_corr(_rankdata(x), _rankdata(y))


def _pairwise_order_accuracy(truth: list[float], pred: list[float]) -> tuple[int, int, float]:
    if len(truth) != len(pred) or len(truth) < 2:
        return 0, 0, float("nan")
    correct = 0
    total = 0
    n = len(truth)
    for i in range(n):
        for j in range(i + 1, n):
            dt = float(truth[j]) - float(truth[i])
            if dt == 0.0:
                continue
            dp = float(pred[j]) - float(pred[i])
            total += 1
            if dp == 0.0:
                continue
            if dt * dp > 0.0:
                correct += 1
    acc = float(correct) / float(total) if total else float("nan")
    return correct, total, acc


@dataclass(frozen=True)
class _Row:
    mid: int
    gid: str
    smiles: str
    truth_rel: float
    types: tuple[int, ...]
    bonds: tuple[tuple[int, int, float, int], ...]
    cycles_heavy: tuple[tuple[int, ...], ...]


def _heavy_atom_mapping(mol) -> tuple[list[int], dict[int, int]]:
    heavy: list[int] = []
    mapping: dict[int, int] = {}
    for idx, atom in enumerate(mol.GetAtoms()):
        if int(atom.GetAtomicNum()) > 1:
            mapping[int(idx)] = len(heavy)
            heavy.append(int(idx))
    return heavy, mapping


def _heavy_bonds_with_attrs(mol, mapping: dict[int, int]) -> tuple[tuple[int, int, float, int], ...]:
    edges: set[tuple[int, int, float, int]] = set()
    for bond in mol.GetBonds():
        u = int(bond.GetBeginAtomIdx())
        v = int(bond.GetEndAtomIdx())
        if u in mapping and v in mapping:
            i = int(mapping[u])
            j = int(mapping[v])
            if i == j:
                continue
            if i > j:
                i, j = j, i
            bo = float(bond.GetBondTypeAsDouble())
            arom = 1 if bool(bond.GetIsAromatic()) else 0
            edges.add((i, j, bo, int(arom)))
    return tuple(sorted(edges))


def _cycles_heavy_from_mol_sssr(*, mol, heavy_mapping: dict[int, int]) -> tuple[tuple[int, ...], ...]:
    cycles_full = sssr_cycles_from_mol(mol)
    cycles_heavy: list[tuple[int, ...]] = []
    for cyc in cycles_full:
        mapped: list[int] = []
        for a in cyc:
            if int(a) not in heavy_mapping:
                raise AccuracyA42Error("unexpected non-heavy atom in SSSR cycle")
            mapped.append(int(heavy_mapping[int(a)]))
        cycles_heavy.append(tuple(mapped))
    return tuple(cycles_heavy)


def _build_weight_adjacency(
    bonds: Iterable[tuple[int, int, float, int]],
    *,
    n: int,
    atoms_db: AtomsDbV1,
    types: Sequence[int],
) -> np.ndarray:
    chi_by_z = atoms_db.chi_by_atomic_num
    z = [int(t) for t in types]
    missing = sorted({int(v) for v in z if int(v) not in chi_by_z})
    if missing:
        raise AccuracyA42Error(f"missing atoms_db chi for Z={missing}")
    chi = np.asarray([float(chi_by_z[int(v)]) for v in z], dtype=float)

    w_adj = np.zeros((int(n), int(n)), dtype=float)
    am = float(EDGE_AROMATIC_MULT)
    alpha = float(EDGE_DELTA_CHI_ALPHA)
    for i, j, bond_order, aromatic in bonds:
        a = int(i)
        b = int(j)
        if a == b:
            continue
        w = float(bond_order)
        if int(aromatic):
            w = w * (1.0 + am)
        w = w * (1.0 + alpha * float(abs(float(chi[a]) - float(chi[b]))))
        w_adj[a, b] = w
        w_adj[b, a] = w
    return w_adj


def _heat_kernel_matrix_shifted_trace_normalized_hermitian(
    H: np.ndarray, *, tau: float
) -> tuple[np.ndarray, float, float, float]:
    """
    K = exp(-tau * (H - lambda_min I)) / trace(exp(-tau * (H - lambda_min I))).
    Returns: (K, trace_heat, lambda_min, lambda_max).
    """
    tau_val = float(tau)
    if tau_val <= 0.0 or not math.isfinite(tau_val):
        raise AccuracyA42Error("heat_tau must be > 0 and finite")

    H0 = np.asarray(H)
    if H0.ndim != 2 or H0.shape[0] != H0.shape[1] or H0.shape[0] == 0:
        raise AccuracyA42Error("H must be non-empty square")

    eigvals, eigvecs = np.linalg.eigh(H0)
    eigvals = np.asarray(eigvals).reshape(-1)
    eigvecs = np.asarray(eigvecs)
    if eigvals.size == 0:
        raise AccuracyA42Error("H must be non-empty")

    lambda_min = float(np.real(eigvals[0]))
    lambda_max = float(np.real(eigvals[-1]))
    weights = np.exp(-tau_val * (np.real(eigvals) - float(lambda_min)))
    trace_heat = float(np.sum(weights))
    if not math.isfinite(trace_heat) or trace_heat <= 0.0:
        raise AccuracyA42Error("invalid trace_heat in K computation")
    w_norm = weights / float(trace_heat)

    K = (eigvecs * w_norm) @ np.conj(eigvecs).T
    return np.asarray(K, dtype=np.complex128), float(trace_heat), float(lambda_min), float(lambda_max)


def _holonomy_phi_for_cycle(
    *,
    cycle: Sequence[int],
    K: np.ndarray,
    theta: np.ndarray,
) -> float:
    atoms = [int(a) for a in cycle]
    m = len(atoms)
    if m < 3:
        return 0.0

    prod = 1.0 + 0.0j
    for k in range(m):
        i = atoms[k]
        j = atoms[(k + 1) % m]
        q = np.exp(-1j * float(theta[i, j])) * K[i, j]
        mag = float(abs(q))
        r = (q / mag) if mag > 0.0 else (1.0 + 0.0j)
        prod *= r
    return float(np.angle(prod))


def _cycle_weight(*, cycle: Sequence[int], weights: np.ndarray) -> float:
    atoms = [int(a) for a in cycle]
    m = len(atoms)
    if m < 3:
        return 0.0
    total = 0.0
    for k in range(m):
        i = atoms[k]
        j = atoms[(k + 1) % m]
        total += float(weights[i, j])
    return float(total)


def _attach_pred_rel(records: list[dict[str, object]]) -> None:
    by_group: dict[str, list[dict[str, object]]] = {}
    for r in records:
        by_group.setdefault(str(r.get("group_id") or ""), []).append(r)

    for gid, group in by_group.items():
        vals = [float(rr["pred_raw"]) for rr in group]
        mn = float(min(vals)) if vals else 0.0
        for rr in group:
            rr["pred_rel"] = float(float(rr["pred_raw"]) - mn)
            rr["group_id"] = str(gid)


def _compute_group_metrics(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    by_group: dict[str, list[dict[str, object]]] = {}
    for r in records:
        by_group.setdefault(str(r.get("group_id") or ""), []).append(r)

    out: dict[str, dict[str, object]] = {}
    for gid, group in by_group.items():
        truth = [float(r["truth_rel_kcalmol"]) for r in group]
        pred = [float(r["pred_rel"]) for r in group]
        sp = _spearman_corr(pred, truth)
        correct, total, pairwise = _pairwise_order_accuracy(truth, pred)
        best_truth_id = str(min(group, key=lambda r: float(r["truth_rel_kcalmol"]))["id"])
        best_pred_id = str(min(group, key=lambda r: float(r["pred_rel"]))["id"])
        top1 = 1.0 if best_truth_id == best_pred_id else 0.0
        out[str(gid)] = {
            "group_id": str(gid),
            "n": int(len(group)),
            "spearman_pred_vs_truth": float(sp),
            "pairwise_correct": int(correct),
            "pairwise_total": int(total),
            "pairwise_order_accuracy": float(pairwise),
            "top1_accuracy": float(top1),
            "truth_best_id": str(best_truth_id),
            "pred_best_id": str(best_pred_id),
        }
    return out


def _aggregate_metrics(group_metrics: dict[str, dict[str, object]]) -> dict[str, object]:
    spearmans: list[float] = []
    negatives: list[str] = []
    pairwise_correct = 0
    pairwise_total = 0
    top1_vals: list[float] = []
    for gid, gm in group_metrics.items():
        sp = float(gm.get("spearman_pred_vs_truth") or float("nan"))
        if math.isfinite(sp):
            spearmans.append(float(sp))
            if sp < 0.0:
                negatives.append(str(gid))
        pairwise_correct += int(gm.get("pairwise_correct") or 0)
        pairwise_total += int(gm.get("pairwise_total") or 0)
        top1_vals.append(float(gm.get("top1_accuracy") or 0.0))

    mean_s = float(sum(spearmans) / len(spearmans)) if spearmans else float("nan")
    median_s = float(np.median(np.asarray(spearmans, dtype=float))) if spearmans else float("nan")
    pairwise_acc = float(pairwise_correct) / float(pairwise_total) if pairwise_total else float("nan")
    top1_mean = float(sum(top1_vals) / len(top1_vals)) if top1_vals else float("nan")
    num_negative = int(len(negatives))
    return {
        "mean_spearman_by_group_test": float(mean_s),
        "median_spearman_by_group_test": float(median_s),
        "pairwise_order_accuracy_overall_test": float(pairwise_acc),
        "top1_accuracy_mean_test": float(top1_mean),
        "num_groups_spearman_negative_test": int(num_negative),
        "negative_spearman_groups_test": sorted(negatives),
        "pairwise_correct_overall_test": int(pairwise_correct),
        "pairwise_total_overall_test": int(pairwise_total),
    }


def _compute_file_infos(out_dir: Path, *, skip_names: set[str]) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_dir():
            continue
        if path.name in skip_names:
            continue
        rel = path.relative_to(out_dir).as_posix()
        infos.append({"path": f"./{rel}", "size_bytes": int(path.stat().st_size), "sha256": _sha256_file(path)})
    return infos


def _write_manifest(out_dir: Path, *, config: dict[str, object], files: list[dict[str, object]]) -> None:
    payload = {
        "schema_version": "manifest.v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "config": config,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _write_provenance(out_dir: Path, *, payload: dict[str, object]) -> None:
    (out_dir / "provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _write_checksums(out_dir: Path, file_infos: list[dict[str, object]]) -> None:
    lines: list[str] = []
    for info in file_infos:
        sha = str(info.get("sha256") or "")
        rel = str(info.get("path") or "").lstrip("./")
        if not sha or not rel:
            continue
        lines.append(f"{sha}  {rel}")
    (out_dir / "checksums.sha256").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_zip_pack(out_dir: Path, *, zip_name: str = "evidence_pack.zip") -> None:
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_dir():
                continue
            if path.name == zip_name:
                continue
            zf.write(path, path.relative_to(out_dir).as_posix())


def _write_truth_copies(out_dir: Path) -> None:
    copies = [
        Path("data/accuracy/isomer_truth.v1.csv"),
        Path("docs/contracts/isomer_truth.v1.md"),
        Path("data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv"),
        Path("data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv.sha256"),
        Path("data/atoms_db_v1.json"),
        Path("docs/specs/accuracy_a4_2_cycle_basis_sssr.md"),
    ]
    for src in copies:
        if not src.exists():
            raise AccuracyA42Error(f"missing required truth/spec file: {src.as_posix()}")
        dst = out_dir / src.as_posix()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _write_index_md(out_dir: Path, *, metrics: dict[str, object], kpi_verdict: str) -> None:
    lines = [
        "# ACCURACY-A4.2 — deterministic SSSR holonomy (0 DOF)",
        "",
        f"- `kpi.verdict`: `{kpi_verdict}`",
        f"- `num_groups_spearman_negative_test`: `{metrics.get('num_groups_spearman_negative_test')}`",
        f"- `negative_spearman_groups_test`: `{metrics.get('negative_spearman_groups_test')}`",
        f"- `median_spearman_by_group_test`: `{metrics.get('median_spearman_by_group_test')}`",
        f"- `pairwise_order_accuracy_overall_test`: `{metrics.get('pairwise_order_accuracy_overall_test')}`",
        f"- `top1_accuracy_mean_test`: `{metrics.get('top1_accuracy_mean_test')}`",
        "",
    ]
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def _heavy_canonical_perm(mol, *, heavy_atom_indices: list[int], heavy_mapping: dict[int, int]) -> list[int]:
    from rdkit import Chem

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    heavy_order = sorted((int(idx) for idx in heavy_atom_indices), key=lambda i: (int(ranks[int(i)]), int(i)))
    return [int(heavy_mapping[int(idx)]) for idx in heavy_order]


def _compute_isomorphism_ceiling_csv(
    *,
    out_dir: Path,
    rows: Sequence[_Row],
    H_by_id: dict[int, np.ndarray],
    tol: float = 1e-12,
    delta_truth_min: float = 1e-6,
) -> tuple[int, str]:
    out_path = out_dir / "isomorphism_ceiling.csv"
    any_ceiling = 0

    by_group: dict[str, list[_Row]] = {}
    for r in rows:
        if str(r.gid) in set(CEILING_GROUPS):
            by_group.setdefault(str(r.gid), []).append(r)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group_id", "a_id", "b_id", "isomorphic_H", "delta_truth"], lineterminator="\n")
        w.writeheader()
        for gid, group_rows in sorted(by_group.items()):
            group_rows_sorted = sorted(group_rows, key=lambda rr: int(rr.mid))
            for i in range(len(group_rows_sorted)):
                a = group_rows_sorted[i]
                Ha = H_by_id[int(a.mid)]
                for j in range(i + 1, len(group_rows_sorted)):
                    b = group_rows_sorted[j]
                    Hb = H_by_id[int(b.mid)]
                    diff = float(np.max(np.abs(Ha - Hb))) if Ha.size and Hb.size else 0.0
                    iso = bool(diff <= float(tol))
                    delta_truth = float(abs(float(a.truth_rel) - float(b.truth_rel)))
                    w.writerow(
                        {
                            "group_id": str(gid),
                            "a_id": int(a.mid),
                            "b_id": int(b.mid),
                            "isomorphic_H": "true" if iso else "false",
                            "delta_truth": float(delta_truth),
                        }
                    )
                    if iso and delta_truth > float(delta_truth_min):
                        any_ceiling += 1

    return int(any_ceiling), _sha256_file(out_path)


def run_a4_2(*, input_csv: Path, out_dir: Path, seed: int, experiment_id: str) -> None:
    atoms_db = load_atoms_db_v1()

    rows: list[_Row] = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        _require_columns(r.fieldnames, {"id", "group_id", "smiles", "energy_rel_kcalmol"})
        for row in r:
            mid = int(str(row.get("id") or "").strip())
            gid = str(row.get("group_id") or "").strip()
            smiles = str(row.get("smiles") or "").strip()
            truth_rel = float(str(row.get("energy_rel_kcalmol") or "").strip())

            g = ChemGraph(smiles=smiles)
            mol = g.mol
            heavy, mapping = _heavy_atom_mapping(mol)
            types = tuple(int(mol.GetAtomWithIdx(int(idx)).GetAtomicNum()) for idx in heavy)
            bonds = _heavy_bonds_with_attrs(mol, mapping)
            cycles_heavy = _cycles_heavy_from_mol_sssr(mol=mol, heavy_mapping=mapping)

            rows.append(
                _Row(
                    mid=int(mid),
                    gid=str(gid),
                    smiles=str(smiles),
                    truth_rel=float(truth_rel),
                    types=types,
                    bonds=bonds,
                    cycles_heavy=cycles_heavy,
                )
            )

    if not rows:
        raise AccuracyA42Error("no rows loaded")

    out_dir.mkdir(parents=True, exist_ok=True)

    pred_records: list[dict[str, object]] = []
    cycle_list_rows: list[dict[str, object]] = []
    cycle_phi_rows: list[dict[str, object]] = []

    # For ceiling-test csv: store canonicalized H per molecule id for the fixed groups.
    H_canon_by_id: dict[int, np.ndarray] = {}

    for row in sorted(rows, key=lambda rr: (str(rr.gid), int(rr.mid))):
        g = ChemGraph(smiles=row.smiles)
        mol = g.mol
        heavy, mapping = _heavy_atom_mapping(mol)
        n = int(len(heavy))

        weights = _build_weight_adjacency(row.bonds, n=n, atoms_db=atoms_db, types=row.types)
        A = phase_matrix_flux_on_cycles(n=n, cycles=row.cycles_heavy, phi=float(PHI_FIXED))
        H = np.asarray(magnetic_laplacian(weights=weights, A=A), dtype=np.complex128)

        K, trace_heat, lambda_min, lambda_max = _heat_kernel_matrix_shifted_trace_normalized_hermitian(H, tau=float(HEAT_TAU))

        # Save canonicalized H for ceiling-test groups only (cheap but keeps output small).
        if str(row.gid) in set(CEILING_GROUPS):
            perm = _heavy_canonical_perm(mol, heavy_atom_indices=heavy, heavy_mapping=mapping)
            Hc = np.asarray(H)[np.ix_(perm, perm)]
            H_canon_by_id[int(row.mid)] = np.asarray(Hc, dtype=np.complex128)

        S_sssr = 0.0
        for cycle_id, cycle in enumerate(row.cycles_heavy, start=1):
            phi_c = _holonomy_phi_for_cycle(cycle=cycle, K=K, theta=A)
            sin2 = float(math.sin(float(phi_c)) ** 2)
            w_c = _cycle_weight(cycle=cycle, weights=weights)
            contrib = float(w_c * sin2)
            S_sssr += float(contrib)

            atoms_str = ",".join(str(int(a)) for a in cycle)
            cycle_list_rows.append(
                {
                    "group_id": str(row.gid),
                    "id": int(row.mid),
                    "cycle_id": int(cycle_id),
                    "cycle_len": int(len(cycle)),
                    "cycle_nodes": atoms_str,
                }
            )
            cycle_phi_rows.append(
                {
                    "group_id": str(row.gid),
                    "id": int(row.mid),
                    "cycle_id": int(cycle_id),
                    "phi_C": float(phi_c),
                    "sin2_phi": float(sin2),
                    "w_C": float(w_c),
                    "contrib": float(contrib),
                    "trace_heat": float(trace_heat),
                    "lambda_min": float(lambda_min),
                    "lambda_max": float(lambda_max),
                }
            )

        pred_records.append(
            {
                "id": int(row.mid),
                "group_id": str(row.gid),
                "smiles": str(row.smiles),
                "truth_rel_kcalmol": float(row.truth_rel),
                "pred_raw": float(S_sssr),
                "pred_rel": "",
                "phi_fixed": float(PHI_FIXED),
                "heat_tau": float(HEAT_TAU),
                "edge_weight_mode": str(EDGE_WEIGHT_MODE),
            }
        )

    _attach_pred_rel(pred_records)

    # Write outputs.
    predictions_path = out_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id",
            "group_id",
            "smiles",
            "truth_rel_kcalmol",
            "pred_raw",
            "pred_rel",
            "phi_fixed",
            "heat_tau",
            "edge_weight_mode",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in sorted(pred_records, key=lambda rr: (str(rr["group_id"]), int(rr["id"]))):
            w.writerow({k: rec.get(k, "") for k in fieldnames})

    shutil.copyfile(predictions_path, out_dir / "summary.csv")

    group_metrics = _compute_group_metrics(pred_records)
    group_metrics_path = out_dir / "group_metrics.csv"
    with group_metrics_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "group_id",
            "n",
            "spearman_pred_vs_truth",
            "pairwise_order_accuracy",
            "pairwise_correct",
            "pairwise_total",
            "top1_accuracy",
            "truth_best_id",
            "pred_best_id",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for gid in sorted(group_metrics.keys()):
            w.writerow({k: group_metrics[gid].get(k, "") for k in fieldnames})

    metrics = _aggregate_metrics(group_metrics)
    num_negative = int(metrics["num_groups_spearman_negative_test"])
    kpi_verdict = "PASS" if num_negative == 0 else "FAIL"

    # Ceiling-test output (for traceability inside evidence pack).
    ceiling_pairs, ceiling_sha = _compute_isomorphism_ceiling_csv(out_dir=out_dir, rows=rows, H_by_id=H_canon_by_id)

    best_config = {
        "schema_version": "accuracy_a1_isomers_a4_2.best_config.v1",
        "variant": "sssr",
        "phi_fixed": float(PHI_FIXED),
        "heat_tau": float(HEAT_TAU),
        "edge_weight_mode": str(EDGE_WEIGHT_MODE),
        "zero_dof": True,
    }
    (out_dir / "best_config.json").write_text(
        json.dumps(best_config, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    dataset_payload = {
        "rows_total": int(len(pred_records)),
        "groups_total": int(len({str(r["group_id"]) for r in pred_records})),
    }

    metrics_payload: dict[str, object] = {
        "schema_version": "accuracy_a1_isomers_a4_2.v1",
        "experiment_id": str(experiment_id),
        "source_sha": _detect_git_sha(),
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "python": str(platform.python_version()),
        "platform": str(platform.platform()),
        "config": {
            "phi_fixed": float(PHI_FIXED),
            "heat_tau": float(HEAT_TAU),
            "edge_weight_mode": str(EDGE_WEIGHT_MODE),
            "edge_aromatic_multiplier": float(EDGE_AROMATIC_MULT),
            "edge_delta_chi_alpha": float(EDGE_DELTA_CHI_ALPHA),
            "seed": int(seed),
            "variant": "sssr",
        },
        "dataset": dict(dataset_payload),
        "best_config": dict(best_config),
        "kpi": {
            "verdict": str(kpi_verdict),
            "num_groups_spearman_negative_test": int(metrics["num_groups_spearman_negative_test"]),
        },
        "metrics_loocv_test_functional_only": dict(metrics),
        "files": {
            "predictions_csv": "predictions.csv",
            "group_metrics_csv": "group_metrics.csv",
            "cycle_list_csv": "cycle_list.csv",
            "cycle_phi_csv": "cycle_phi.csv",
            "isomorphism_ceiling_csv": "isomorphism_ceiling.csv",
            "best_config_json": "best_config.json",
        },
        "ceiling_test": {
            "groups": list(CEILING_GROUPS),
            "tol": 1e-12,
            "delta_truth_min": 1e-6,
            "ceiling_pairs": int(ceiling_pairs),
            "isomorphism_ceiling_csv_sha256": str(ceiling_sha).upper(),
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    cycle_list_path = out_dir / "cycle_list.csv"
    with cycle_list_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["group_id", "id", "cycle_id", "cycle_len", "cycle_nodes"]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rr in sorted(cycle_list_rows, key=lambda r0: (str(r0["group_id"]), int(r0["id"]), int(r0["cycle_id"]))):
            w.writerow({k: rr.get(k, "") for k in fieldnames})

    cycle_phi_path = out_dir / "cycle_phi.csv"
    with cycle_phi_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["group_id", "id", "cycle_id", "phi_C", "sin2_phi", "w_C", "contrib", "trace_heat", "lambda_min", "lambda_max"]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rr in sorted(cycle_phi_rows, key=lambda r0: (str(r0["group_id"]), int(r0["id"]), int(r0["cycle_id"]))):
            w.writerow({k: rr.get(k, "") for k in fieldnames})

    _write_truth_copies(out_dir)
    _write_index_md(out_dir, metrics=metrics, kpi_verdict=kpi_verdict)

    provenance = {
        "schema_version": "accuracy_a1_isomers_a4_2.provenance.v1",
        "source_sha_main": _detect_git_sha(),
        "experiment_id": str(experiment_id),
        "seed": int(seed),
        "command_line": " ".join([repr(a) if " " in a else a for a in os.sys.argv]),
        "truth_csv": str(input_csv.as_posix()),
    }
    _write_provenance(out_dir, payload=provenance)

    manifest_config = {
        "roadmap_id": "ACCURACY-A4.2",
        "soT": "docs/specs/accuracy_a4_2_cycle_basis_sssr.md",
    }
    file_infos = _compute_file_infos(out_dir, skip_names={"manifest.json", "checksums.sha256", "evidence_pack.zip"})
    manifest_files = list(file_infos)
    manifest_files.append({"path": "./manifest.json", "size_bytes": None, "sha256": None})
    _write_manifest(out_dir, config=manifest_config, files=manifest_files)

    file_infos_final = _compute_file_infos(out_dir, skip_names={"checksums.sha256", "evidence_pack.zip"})
    _write_checksums(out_dir, file_infos_final)
    _write_zip_pack(out_dir, zip_name="evidence_pack.zip")


def main() -> None:
    ap = argparse.ArgumentParser(description="ACCURACY-A4.2: deterministic SSSR holonomy (0 DOF).")
    ap.add_argument("--experiment_id", type=str, default="ACCURACY-A4.2")
    ap.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_a4_2(input_csv=Path(args.input_csv), out_dir=Path(args.out_dir), seed=int(args.seed), experiment_id=str(args.experiment_id))


if __name__ == "__main__":
    main()
