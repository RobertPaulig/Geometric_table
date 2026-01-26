from __future__ import annotations

"""
ACCURACY-A4.1 - Cycle-Flux / Holonomy Observable runner (opt-in).

Contract SoT:
  docs/specs/accuracy_a4_1_cycle_flux_holonomy.md
"""

import argparse
import csv
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from hetero2.chemgraph import ChemGraph
from hetero2.phase_channel import magnetic_laplacian, normalize_flux_phi, phase_matrix_flux_on_cycles, sssr_cycles_from_mol
from hetero2.physics_operator import AtomsDbV1, load_atoms_db_v1


class AccuracyA41Error(ValueError):
    pass


DEFAULT_INPUT_CSV = Path("data/accuracy/isomer_truth.v1.csv")
DEFAULT_OUT_DIR = Path("out_accuracy_a1_isomers_a4_1")

# 0 DOF: fixed constants (not tuned/selected).
PHI_FIXED = float(normalize_flux_phi(math.pi / 2.0))
HEAT_TAU = 1.0


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text_normalized(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = "\n".join(text.splitlines()) + ("\n" if text else "")
    return _sha256_bytes(normalized.encode("utf-8"))


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
        raise AccuracyA41Error("missing CSV header")
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise AccuracyA41Error(f"missing required columns: {missing}")


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
    mx = float(statistics.fmean(x))
    my = float(statistics.fmean(y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return float(cov / (sx * sy))


def _spearman_corr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson_corr(rx, ry)


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
                raise AccuracyA41Error("unexpected non-heavy atom in SSSR cycle")
            mapped.append(int(heavy_mapping[int(a)]))
        cycles_heavy.append(tuple(mapped))
    return tuple(cycles_heavy)


def _build_weight_adjacency(
    bonds: Sequence[Sequence[object]],
    *,
    n: int,
    atoms_db: AtomsDbV1,
    types: Sequence[int],
    mode: str,
    aromatic_multiplier: float,
    delta_chi_alpha: float,
) -> np.ndarray:
    if mode not in {"bond_order", "bond_order_delta_chi"}:
        raise AccuracyA41Error(f"invalid edge_weight_mode: {mode}")
    am = float(aromatic_multiplier)
    alpha = float(delta_chi_alpha)

    chi: np.ndarray | None = None
    if mode == "bond_order_delta_chi":
        missing = sorted({int(z) for z in types if int(z) not in atoms_db.chi_by_atomic_num})
        if missing:
            raise AccuracyA41Error(f"missing atoms_db chi for Z={missing}")
        chi = np.asarray([float(atoms_db.chi_by_atomic_num[int(z)]) for z in types], dtype=float)

    w_adj = np.zeros((int(n), int(n)), dtype=float)
    for i, j, bond_order, aromatic in bonds:
        a, b = int(i), int(j)
        if a == b:
            continue
        w = float(bond_order)
        if int(aromatic):
            w = w * (1.0 + am)
        if chi is not None:
            w = w * (1.0 + alpha * float(abs(float(chi[a]) - float(chi[b]))))
        w_adj[a, b] = w
        w_adj[b, a] = w
    return w_adj


def _phase_operator_for_row(
    *,
    weights: np.ndarray,
    cycles_heavy: Sequence[Sequence[int]],
    flux_phi: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(weights.shape[0])
    A = phase_matrix_flux_on_cycles(n=n, cycles=cycles_heavy, phi=float(flux_phi))
    L_phase = magnetic_laplacian(weights=weights, A=A)
    return np.asarray(L_phase, dtype=np.complex128), np.asarray(A, dtype=float)


def _heat_kernel_matrix_shifted_trace_normalized_hermitian(
    H: np.ndarray, *, tau: float
) -> tuple[np.ndarray, float, float, float]:
    """
    K = exp(-tau * (H - lambda_min I)) / trace(exp(-tau * (H - lambda_min I))).
    Returns: (K, trace_heat, lambda_min, lambda_max).
    """
    tau_val = float(tau)
    if tau_val <= 0.0 or not math.isfinite(tau_val):
        raise AccuracyA41Error("heat_tau must be > 0 and finite")

    H0 = np.asarray(H)
    if H0.ndim != 2 or H0.shape[0] != H0.shape[1] or H0.shape[0] == 0:
        raise AccuracyA41Error("H must be non-empty square")

    eigvals, eigvecs = np.linalg.eigh(H0)
    eigvals = np.asarray(eigvals).reshape(-1)
    eigvecs = np.asarray(eigvecs)
    if eigvals.size == 0:
        raise AccuracyA41Error("H must be non-empty")

    lambda_min = float(np.real(eigvals[0]))
    lambda_max = float(np.real(eigvals[-1]))
    weights = np.exp(-tau_val * (np.real(eigvals) - float(lambda_min)))
    trace_heat = float(np.sum(weights))
    if not math.isfinite(trace_heat) or trace_heat <= 0.0:
        raise AccuracyA41Error("invalid trace_heat in kernel")
    w_norm = weights / float(trace_heat)

    K = (eigvecs * w_norm) @ np.conj(eigvecs).T
    return np.asarray(K, dtype=np.complex128), float(trace_heat), float(lambda_min), float(lambda_max)


def _bfs_spanning_tree_parents(*, n: int, edges: Sequence[tuple[int, int]]) -> list[int]:
    neigh: list[list[int]] = [[] for _ in range(int(n))]
    for a, b in edges:
        u = int(a)
        v = int(b)
        if u == v:
            continue
        neigh[u].append(v)
        neigh[v].append(u)
    for i in range(int(n)):
        neigh[i].sort()

    root = 0
    parent = [-1] * int(n)
    seen = [False] * int(n)
    q: list[int] = [int(root)]
    seen[root] = True
    for u in q:
        for v in neigh[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = int(u)
                q.append(int(v))

    if not all(seen):
        raise AccuracyA41Error("graph must be connected (unexpected disconnected heavy-atom graph)")
    return parent


def _tree_edges_from_parents(parent: Sequence[int]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for v, p in enumerate(parent):
        if int(p) < 0:
            continue
        a = int(v)
        b = int(p)
        if a > b:
            a, b = b, a
        edges.add((a, b))
    return edges


def _tree_path_nodes(parent: Sequence[int], *, u: int, v: int) -> list[int]:
    uu = int(u)
    vv = int(v)
    path_u: list[int] = []
    x = uu
    while x >= 0:
        path_u.append(int(x))
        x = int(parent[x]) if int(parent[x]) >= 0 else -1
    idx_u = {node: i for i, node in enumerate(path_u)}

    path_v: list[int] = []
    y = vv
    lca = -1
    while y >= 0:
        path_v.append(int(y))
        if int(y) in idx_u:
            lca = int(y)
            break
        y = int(parent[y]) if int(parent[y]) >= 0 else -1

    if lca < 0:
        raise AccuracyA41Error("failed to find LCA in tree path")

    u_to_lca = path_u[: idx_u[lca] + 1]  # u ... lca
    v_to_lca = path_v  # v ... lca
    rev_v = list(reversed(v_to_lca))  # lca ... v
    nodes = list(u_to_lca) + list(rev_v[1:])
    if nodes[0] != uu or nodes[-1] != vv:
        raise AccuracyA41Error("invalid tree path endpoints")
    return [int(z) for z in nodes]


def _arg_unit_complex(z: complex) -> float:
    return float(np.angle(np.complex128(z)))


def _cycle_holonomy_angle(
    *,
    r_dir: dict[tuple[int, int], complex],
    cycle_edges: Sequence[tuple[int, int]],
) -> float:
    prod = np.complex128(1.0 + 0.0j)
    for a, b in cycle_edges:
        key = (int(a), int(b))
        if key not in r_dir:
            raise AccuracyA41Error(f"missing r_dir for edge {key}")
        prod *= np.complex128(r_dir[key])
    return _arg_unit_complex(complex(prod))


# PART 1 END. Remaining runner implementation below.


@dataclass(frozen=True)
class _Row:
    mid: str
    gid: str
    smiles: str
    truth_rel: float
    n_heavy_atoms: int
    types: tuple[int, ...]
    bonds: tuple[tuple[int, int, float, int], ...]
    cycles_heavy: tuple[tuple[int, ...], ...]


def _load_rows(input_csv: Path) -> list[_Row]:
    reader = csv.DictReader(input_csv.read_text(encoding="utf-8").splitlines())
    _require_columns(reader.fieldnames, required=["id", "group_id", "smiles", "energy_rel_kcalmol"])
    raw_rows = [dict(r) for r in reader]
    if not raw_rows:
        raise AccuracyA41Error("input_csv has no data rows")

    rows: list[_Row] = []
    for r in raw_rows:
        mid = str(r.get("id") or "").strip()
        gid = str(r.get("group_id") or "").strip()
        smiles = str(r.get("smiles") or "").strip()
        if not mid or not gid or not smiles:
            raise AccuracyA41Error(f"invalid row (id/group_id/smiles required): {r}")
        truth_rel = float(str(r.get("energy_rel_kcalmol") or "").strip())

        g = ChemGraph(smiles=smiles)
        mol = g.mol
        heavy, mapping = _heavy_atom_mapping(mol)
        types = tuple(int(mol.GetAtomWithIdx(int(idx)).GetAtomicNum()) for idx in heavy)
        bonds = _heavy_bonds_with_attrs(mol, mapping)
        cycles_heavy = _cycles_heavy_from_mol_sssr(mol=mol, heavy_mapping=mapping)

        rows.append(
            _Row(
                mid=mid,
                gid=gid,
                smiles=smiles,
                truth_rel=float(truth_rel),
                n_heavy_atoms=int(len(heavy)),
                types=types,
                bonds=bonds,
                cycles_heavy=cycles_heavy,
            )
        )
    return rows


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


def _attach_pred_rank(records: list[dict[str, object]]) -> None:
    by_group: dict[str, list[dict[str, object]]] = {}
    for r in records:
        by_group.setdefault(str(r.get("group_id") or ""), []).append(r)

    for gid, group in by_group.items():
        group_sorted = sorted(group, key=lambda rr: str(rr.get("id")))
        vals = [float(rr.get("pred_rel", rr.get("pred_raw", 0.0)) or 0.0) for rr in group_sorted]
        ranks = _rankdata(vals)
        for rr, rk in zip(group_sorted, ranks):
            rr["pred_rank"] = float(rk)
            rr["group_id"] = str(gid)


def _compute_group_metrics(records: list[dict[str, object]], *, pred_rel_key: str) -> dict[str, dict[str, object]]:
    by_group: dict[str, list[dict[str, object]]] = {}
    for r in records:
        gid = str(r.get("group_id") or "")
        by_group.setdefault(gid, []).append(r)

    out: dict[str, dict[str, object]] = {}
    for gid, group in by_group.items():
        group_sorted = sorted(group, key=lambda rr: str(rr.get("id")))
        truth = [float(rr["truth_rel_kcalmol"]) for rr in group_sorted]
        pred = [float(rr[pred_rel_key]) for rr in group_sorted]
        spearman = _spearman_corr(pred, truth)
        correct, total, acc = _pairwise_order_accuracy(truth, pred)

        best_pred_idx = int(np.argmin(np.asarray(pred, dtype=float)))
        truth_min = float(min(truth))
        truth_best = {str(group_sorted[i]["id"]) for i, t in enumerate(truth) if float(t) == truth_min}
        pred_best = str(group_sorted[best_pred_idx]["id"])
        top1 = 1.0 if pred_best in truth_best else 0.0

        out[gid] = {
            "group_id": str(gid),
            "n": int(len(group_sorted)),
            "spearman_pred_vs_truth": float(spearman),
            "pairwise_order_accuracy": float(acc),
            "pairwise_correct": int(correct),
            "pairwise_total": int(total),
            "top1_accuracy": float(top1),
            "pred_best_id": pred_best,
            "truth_best_ids": sorted(truth_best),
        }
    return out


def _aggregate_loocv_metrics(group_metrics: dict[str, dict[str, object]]) -> dict[str, object]:
    spearman_vals: list[float] = []
    top1_vals: list[float] = []
    negative_groups: list[str] = []
    pairwise_correct = 0
    pairwise_total = 0
    groups_total = int(len(group_metrics))

    for gid, gm in group_metrics.items():
        s = float(gm.get("spearman_pred_vs_truth", float("nan")))
        if math.isfinite(s):
            spearman_vals.append(float(s))
            if float(s) < 0.0:
                negative_groups.append(str(gid))
        top1_vals.append(float(gm.get("top1_accuracy", float("nan"))))
        pairwise_correct += int(gm.get("pairwise_correct", 0) or 0)
        pairwise_total += int(gm.get("pairwise_total", 0) or 0)

    mean_spearman = float(statistics.fmean(spearman_vals)) if spearman_vals else float("nan")
    median_spearman = float(statistics.median(spearman_vals)) if spearman_vals else float("nan")
    top1_mean = float(statistics.fmean(top1_vals)) if top1_vals else float("nan")
    pairwise_acc = float(pairwise_correct) / float(pairwise_total) if pairwise_total else float("nan")

    return {
        "groups_total": int(groups_total),
        "mean_spearman_by_group": float(mean_spearman),
        "median_spearman_by_group": float(median_spearman),
        "pairwise_order_accuracy_overall": float(pairwise_acc),
        "pairwise_correct": int(pairwise_correct),
        "pairwise_total": int(pairwise_total),
        "top1_accuracy_mean": float(top1_mean),
        "num_groups_spearman_negative": int(len(negative_groups)),
        "negative_spearman_groups": sorted(negative_groups),
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


def _cycle_score_for_row(
    row: _Row,
    *,
    atoms_db: AtomsDbV1,
) -> tuple[float, list[dict[str, object]], dict[tuple[int, int], complex]]:
    """
    Returns:
      (S_cycle, per_cycle_rows, r_dir)
    """
    n = int(row.n_heavy_atoms)
    if n <= 0:
        raise AccuracyA41Error("n_heavy_atoms must be > 0")

    w_adj = _build_weight_adjacency(
        row.bonds,
        n=n,
        atoms_db=atoms_db,
        types=row.types,
        mode="bond_order_delta_chi",
        aromatic_multiplier=0.0,
        delta_chi_alpha=1.0,
    )

    lap_phase, A = _phase_operator_for_row(weights=w_adj, cycles_heavy=row.cycles_heavy, flux_phi=float(PHI_FIXED))
    K, _trace_heat, _lambda_min, _lambda_max = _heat_kernel_matrix_shifted_trace_normalized_hermitian(
        lap_phase, tau=float(HEAT_TAU)
    )

    # Compute directed r_ij over edges.
    r_dir: dict[tuple[int, int], complex] = {}
    edges: list[tuple[int, int]] = []
    for i, j, *_rest in row.bonds:
        a = int(i)
        b = int(j)
        if a == b:
            continue
        edges.append((a, b))
        theta = float(A[a, b])
        q = np.exp(-1j * theta) * np.complex128(K[a, b])
        mag = float(abs(q))
        if mag > 0.0:
            r = complex(q / mag)
        else:
            r = 1.0 + 0.0j
        r_dir[(a, b)] = r
        r_dir[(b, a)] = complex(np.conj(np.complex128(r)))

    # Unique undirected edges.
    undirected_edges = sorted({(min(a, b), max(a, b)) for a, b in edges})

    parent = _bfs_spanning_tree_parents(n=n, edges=undirected_edges)
    tree_edges = _tree_edges_from_parents(parent)

    chord_edges = [(u, v) for (u, v) in undirected_edges if (u, v) not in tree_edges]
    per_cycle_rows: list[dict[str, object]] = []

    s_cycle = 0.0
    for idx, (u, v) in enumerate(chord_edges, start=1):
        nodes = _tree_path_nodes(parent, u=int(u), v=int(v))
        cycle_edges: list[tuple[int, int]] = [(int(nodes[k]), int(nodes[k + 1])) for k in range(len(nodes) - 1)]
        cycle_edges.append((int(v), int(u)))  # chord closes the cycle as (v->u)

        phi_uv = _cycle_holonomy_angle(r_dir=r_dir, cycle_edges=cycle_edges)
        sin2_phi = float(math.sin(float(phi_uv)) ** 2)
        w_uv = float(w_adj[int(u), int(v)])
        contrib = float(w_uv * sin2_phi)
        s_cycle += float(contrib)

        per_cycle_rows.append(
            {
                "cycle_id": int(idx),
                "u": int(u),
                "v": int(v),
                "phi_uv": float(phi_uv),
                "sin2_phi": float(sin2_phi),
                "w_uv": float(w_uv),
                "contrib": float(contrib),
                "num_cycle_edges": int(len(cycle_edges)),
            }
        )

    return float(s_cycle), per_cycle_rows, r_dir


def run_a4_1(*, input_csv: Path, out_dir: Path, seed: int, experiment_id: str) -> None:
    del seed  # 0 DOF, deterministic; keep arg for uniform CLI.
    atoms_db = load_atoms_db_v1()

    rows = _load_rows(input_csv)
    rows_sorted = sorted(rows, key=lambda r: (str(r.gid), str(r.mid)))
    group_ids = sorted({r.gid for r in rows_sorted})
    if len(group_ids) < 2:
        raise AccuracyA41Error("need at least 2 groups for LOOCV")

    fold_id_by_group = {gid: i for i, gid in enumerate(group_ids, start=1)}

    pred_records: list[dict[str, object]] = []
    cycle_rows: list[dict[str, object]] = []
    mol_rows: list[dict[str, object]] = []

    for row in rows_sorted:
        s_cycle, per_cycle, _r_dir = _cycle_score_for_row(row, atoms_db=atoms_db)
        fold_id = int(fold_id_by_group[str(row.gid)])

        pred_records.append(
            {
                "fold_id": int(fold_id),
                "id": str(row.mid),
                "group_id": str(row.gid),
                "smiles": str(row.smiles),
                "truth_rel_kcalmol": float(row.truth_rel),
                "pred_raw": float(s_cycle),
                "variant": "cycle_flux_holonomy",
                "phi_fixed": float(PHI_FIXED),
                "heat_tau": float(HEAT_TAU),
            }
        )

        for cyc in per_cycle:
            cycle_rows.append(
                {
                    "fold_id": int(fold_id),
                    "id": str(row.mid),
                    "group_id": str(row.gid),
                    **cyc,
                }
            )

        mol_rows.append(
            {
                "fold_id": int(fold_id),
                "id": str(row.mid),
                "group_id": str(row.gid),
                "smiles": str(row.smiles),
                "S_cycle": float(s_cycle),
                "num_cycles": int(len(per_cycle)),
                "sum_abs_cycle_contrib": float(sum(abs(float(c.get("contrib") or 0.0)) for c in per_cycle)),
            }
        )

    _attach_pred_rel(pred_records)
    _attach_pred_rank(pred_records)

    group_metrics = _compute_group_metrics(pred_records, pred_rel_key="pred_rel")
    metrics_test = _aggregate_loocv_metrics(group_metrics)
    num_negative = int(metrics_test.get("num_groups_spearman_negative", 0) or 0)
    kpi_verdict = "PASS" if num_negative == 0 else "FAIL"

    out_dir.mkdir(parents=True, exist_ok=True)

    # predictions.csv (and summary.csv copy)
    predictions_path = out_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold_id",
            "id",
            "group_id",
            "smiles",
            "truth_rel_kcalmol",
            "pred_raw",
            "pred_rel",
            "pred_rank",
            "variant",
            "phi_fixed",
            "heat_tau",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in sorted(pred_records, key=lambda rr: (int(rr.get("fold_id") or 0), str(rr.get("id") or ""))):
            w.writerow({k: rec.get(k, "") for k in fieldnames})

    shutil.copyfile(predictions_path, out_dir / "summary.csv")

    # cycle_flux_by_molecule.csv
    by_mol_path = out_dir / "cycle_flux_by_molecule.csv"
    with by_mol_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["fold_id", "id", "group_id", "smiles", "S_cycle", "num_cycles", "sum_abs_cycle_contrib"]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in sorted(mol_rows, key=lambda rr: (int(rr.get("fold_id") or 0), str(rr.get("id") or ""))):
            w.writerow({k: rec.get(k, "") for k in fieldnames})

    # cycle_flux_by_cycle.csv
    by_cycle_path = out_dir / "cycle_flux_by_cycle.csv"
    with by_cycle_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold_id",
            "id",
            "group_id",
            "cycle_id",
            "u",
            "v",
            "num_cycle_edges",
            "phi_uv",
            "sin2_phi",
            "w_uv",
            "contrib",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in sorted(
            cycle_rows, key=lambda rr: (int(rr.get("fold_id") or 0), str(rr.get("id") or ""), int(rr.get("cycle_id") or 0))
        ):
            w.writerow({k: rec.get(k, "") for k in fieldnames})

    # group_metrics.csv
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
            "pred_best_id",
            "truth_best_ids",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for gid, gm in sorted(group_metrics.items(), key=lambda x: str(x[0])):
            row = {k: gm.get(k, "") for k in fieldnames}
            row["truth_best_ids"] = json.dumps(list(gm.get("truth_best_ids") or []), ensure_ascii=False)
            w.writerow(row)

    # metrics.json + best_config.json
    best_cfg = {
        "schema_version": "accuracy_a1_isomers_a4_1.best_config.v1",
        "variant": "cycle_flux_holonomy",
        "phi_fixed": float(PHI_FIXED),
        "heat_tau": float(HEAT_TAU),
        "edge_weight_mode": "bond_order_delta_chi",
        "edge_aromatic_multiplier": 0.0,
        "edge_delta_chi_alpha": 1.0,
        "chosen_by_train_only": False,
        "search_space_size": 0,
    }
    (out_dir / "best_config.json").write_text(
        json.dumps(best_cfg, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    metrics_payload: dict[str, object] = {
        "schema_version": "accuracy_a1_isomers_a4_1.v1",
        "experiment_id": str(experiment_id),
        "dataset": {"rows_total": int(len(pred_records)), "groups_total": int(len(group_metrics))},
        "input_csv": input_csv.as_posix(),
        "input_sha256_normalized": _sha256_text_normalized(input_csv),
        "best_config": best_cfg,
        "kpi": {
            "verdict": str(kpi_verdict),
            "must_num_groups_spearman_negative_test": 0,
            "num_groups_spearman_negative_test": int(metrics_test.get("num_groups_spearman_negative") or 0),
            "negative_spearman_groups_test": list(metrics_test.get("negative_spearman_groups") or []),
        },
        "metrics_loocv_test_functional_only": metrics_test,
        "files": {
            "summary_csv": "summary.csv",
            "predictions_csv": "predictions.csv",
            "group_metrics_csv": "group_metrics.csv",
            "cycle_flux_by_molecule_csv": "cycle_flux_by_molecule.csv",
            "cycle_flux_by_cycle_csv": "cycle_flux_by_cycle.csv",
            "metrics_json": "metrics.json",
            "best_config_json": "best_config.json",
            "provenance_json": "provenance.json",
            "manifest_json": "manifest.json",
            "checksums_sha256": "checksums.sha256",
            "index_md": "index.md",
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    index_lines = [
        f"# {experiment_id} (Isomers) A4.1 cycle-flux / holonomy",
        "",
        "LOOCV (by group_id) metrics (functional_only):",
        f"- mean_spearman_by_group: {metrics_test.get('mean_spearman_by_group')}",
        f"- median_spearman_by_group: {metrics_test.get('median_spearman_by_group')}",
        f"- pairwise_order_accuracy_overall: {metrics_test.get('pairwise_order_accuracy_overall')} ({metrics_test.get('pairwise_correct')}/{metrics_test.get('pairwise_total')})",
        f"- top1_accuracy_mean: {metrics_test.get('top1_accuracy_mean')}",
        f"- num_groups_spearman_negative_test: {metrics_test.get('num_groups_spearman_negative')}",
        "",
        f"phi_fixed: {PHI_FIXED}",
        f"heat_tau: {HEAT_TAU}",
        "",
    ]
    (out_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    # Copy truth + SoT docs into pack (audit).
    repo_root = Path(__file__).resolve().parents[1]
    extra_files: list[tuple[Path, Path]] = [
        (Path("data/accuracy/isomer_truth.v1.csv"), input_csv),
        (Path("docs/contracts/isomer_truth.v1.md"), repo_root / "docs/contracts/isomer_truth.v1.md"),
        (
            Path("data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv"),
            repo_root / "data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv",
        ),
        (
            Path("data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv.sha256"),
            repo_root / "data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv.sha256",
        ),
        (Path("data/atoms_db_v1.json"), repo_root / "data/atoms_db_v1.json"),
        (
            Path("docs/specs/accuracy_a4_1_cycle_flux_holonomy.md"),
            repo_root / "docs/specs/accuracy_a4_1_cycle_flux_holonomy.md",
        ),
    ]
    for rel_dst, src in extra_files:
        if not src.exists():
            continue
        dst = out_dir / rel_dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    provenance: dict[str, object] = {
        "experiment_id": str(experiment_id),
        "git_sha": _detect_git_sha(),
        "source_sha_main": _detect_git_sha(),
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "input_csv": input_csv.as_posix(),
        "input_sha256_normalized": _sha256_text_normalized(input_csv),
        "best_config": best_cfg,
        "phi_fixed": float(PHI_FIXED),
        "heat_tau": float(HEAT_TAU),
    }
    _write_provenance(out_dir, payload=provenance)

    config_for_manifest = {
        "experiment_id": str(experiment_id),
        "input_csv": input_csv.as_posix(),
        "input_sha256_normalized": _sha256_text_normalized(input_csv),
        "best_config": best_cfg,
    }
    file_infos = _compute_file_infos(out_dir, skip_names={"manifest.json", "checksums.sha256", "evidence_pack.zip"})
    manifest_files = list(file_infos)
    manifest_files.append({"path": "./manifest.json", "size_bytes": None, "sha256": None})
    _write_manifest(out_dir, config=config_for_manifest, files=manifest_files)
    file_infos_final = _compute_file_infos(out_dir, skip_names={"checksums.sha256", "evidence_pack.zip"})
    _write_checksums(out_dir, file_infos_final)
    _write_zip_pack(out_dir, zip_name="evidence_pack.zip")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ACCURACY-A4.1 cycle-flux / holonomy runner (opt-in).")
    p.add_argument("--experiment_id", type=str, default="ACCURACY-A4.1", help="Experiment identifier for outputs.")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV, help="Canonical isomer truth CSV.")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    p.add_argument("--seed", type=int, default=0, help="Seed (unused; kept for compatibility).")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    run_a4_1(
        input_csv=Path(args.input_csv),
        out_dir=Path(args.out_dir),
        seed=int(args.seed),
        experiment_id=str(args.experiment_id),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
