from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

ATOMS_DB_V1_REL_PATH = Path("data") / "atoms_db_v1.json"
PHYSICS_PARAMS_SOURCE = "atoms_db_v1_json"
SPECTRAL_ENTROPY_BETA_DEFAULT = 1.0
DELTA_CHI_ALPHA_DEFAULT = 1.0
DOS_LDOS_SCHEMA = "hetero2_dos_ldos.v1"
DOS_GRID_N_DEFAULT = 128
DOS_ETA_DEFAULT = 0.05
DOS_ENERGY_MARGIN_SIGMAS_DEFAULT = 3.0
SCF_SCHEMA = "hetero2_scf.v1"
SCF_MAX_ITER_DEFAULT = 50
SCF_TOL_DEFAULT = 1e-6
SCF_DAMPING_DEFAULT = 0.5
SCF_OCC_K_DEFAULT = 5
SCF_TAU_DEFAULT = 1.0
SCF_GAMMA_DEFAULT = 0.5
POTENTIAL_UNIT_MODEL = "dimensionless"
POTENTIAL_SCALE_GAMMA_DEFAULT = 1.0


class MissingPhysicsParams(RuntimeError):
    def __init__(
        self,
        *,
        missing_atomic_numbers: Sequence[int],
        source_path: str,
        missing_key: str = "epsilon",
    ) -> None:
        missing = sorted({int(z) for z in missing_atomic_numbers})
        self.missing_atomic_numbers = tuple(missing)
        self.source_path = str(source_path)
        self.missing_key = str(missing_key)
        super().__init__(f"missing_physics_params: key={self.missing_key} atomic_numbers={missing} source={source_path}")


@dataclass(frozen=True)
class AtomsDbV1:
    source_path: str
    potential_by_atomic_num: Mapping[int, float]
    chi_by_atomic_num: Mapping[int, float]
    symbol_by_atomic_num: Mapping[int, str]


def _repo_root_fallback() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_atoms_db_v1_path() -> Path:
    cwd_path = Path.cwd() / ATOMS_DB_V1_REL_PATH
    if cwd_path.exists():
        return cwd_path
    repo_path = _repo_root_fallback() / ATOMS_DB_V1_REL_PATH
    return repo_path


@lru_cache(maxsize=1)
def load_atoms_db_v1() -> AtomsDbV1:
    path = resolve_atoms_db_v1_path()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"atoms_db_v1.json must be a list of objects (got {type(raw).__name__})")

    potential_by_z: dict[int, float] = {}
    chi_by_z: dict[int, float] = {}
    symbol_by_z: dict[int, str] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        if "Z" not in row or "name" not in row or "epsilon" not in row:
            continue
        try:
            z = int(row["Z"])
            sym = str(row["name"])
            eps = float(row["epsilon"])
        except Exception:
            continue
        potential_by_z[z] = eps
        symbol_by_z[z] = sym
        try:
            chi_val = row.get("chi", None)
            if chi_val is not None:
                chi_by_z[z] = float(chi_val)
        except Exception:
            pass

    if not potential_by_z:
        raise ValueError(f"atoms_db_v1.json has no usable rows with keys Z/name/epsilon: {path}")

    return AtomsDbV1(
        source_path=str(path.as_posix()),
        potential_by_atomic_num=potential_by_z,
        chi_by_atomic_num=chi_by_z,
        symbol_by_atomic_num=symbol_by_z,
    )


def laplacian_from_adjacency(adj: np.ndarray) -> np.ndarray:
    deg = np.diag(adj.sum(axis=1))
    return deg - adj


def eigvals_symmetric(m: np.ndarray) -> np.ndarray:
    vals = np.linalg.eigvalsh(m)
    return np.sort(vals)


def eigh_symmetric(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(m)
    idx = np.argsort(vals)
    return np.asarray(vals[idx], dtype=float), np.asarray(vecs[:, idx], dtype=float)


def spectral_gap(eigvals: Sequence[float]) -> float:
    vals = np.array(list(eigvals), dtype=float)
    if vals.size < 2:
        return float("nan")
    vals = np.sort(vals)
    return float(vals[1] - vals[0])


def spectral_trace(eigvals: Sequence[float]) -> float:
    vals = np.array(list(eigvals), dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.sum(vals))


def spectral_entropy_beta(eigvals: Iterable[float], *, beta: float, eps: float = 1e-12) -> float:
    vals = np.array(list(eigvals), dtype=float)
    if vals.size == 0:
        return float("nan")
    x = -float(beta) * vals
    x = x - float(np.max(x))
    w = np.exp(x)
    total = float(np.sum(w))
    if total <= 0.0 or not math.isfinite(total):
        return float("nan")
    p = w / total
    return float(-np.sum(p * np.log(p + float(eps))))


def _potentials_for_types(types: Sequence[int], atoms_db: AtomsDbV1) -> np.ndarray:
    missing = [int(z) for z in types if int(z) not in atoms_db.potential_by_atomic_num]
    if missing:
        raise MissingPhysicsParams(missing_atomic_numbers=missing, source_path=atoms_db.source_path, missing_key="epsilon")
    return np.array([float(atoms_db.potential_by_atomic_num[int(z)]) for z in types], dtype=float)


def _apply_potential_scale(v0_raw: np.ndarray, *, potential_scale_gamma: float) -> np.ndarray:
    gamma = float(potential_scale_gamma)
    if not math.isfinite(gamma):
        raise ValueError("potential_scale_gamma must be finite")
    return np.asarray(v0_raw, dtype=float) * gamma


def _chis_for_types(types: Sequence[int], atoms_db: AtomsDbV1) -> np.ndarray:
    missing = [int(z) for z in types if int(z) not in atoms_db.chi_by_atomic_num]
    if missing:
        raise MissingPhysicsParams(missing_atomic_numbers=missing, source_path=atoms_db.source_path, missing_key="chi")
    return np.array([float(atoms_db.chi_by_atomic_num[int(z)]) for z in types], dtype=float)


def _gaussian_kernel(x: np.ndarray, *, eta: float) -> np.ndarray:
    e = float(eta)
    if e <= 0.0:
        raise ValueError("dos_eta must be > 0")
    c = 1.0 / (e * math.sqrt(2.0 * math.pi))
    return c * np.exp(-0.5 * (x / e) ** 2)


def _auto_energy_grid(
    *,
    eigenvalues: Sequence[float],
    grid_n: int,
    eta: float,
    margin_sigmas: float = DOS_ENERGY_MARGIN_SIGMAS_DEFAULT,
) -> np.ndarray:
    vals = np.array(list(eigenvalues), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([], dtype=float)
    margin = float(margin_sigmas) * float(eta)
    e_min = float(np.min(vals)) - margin
    e_max = float(np.max(vals)) + margin
    if not math.isfinite(e_min) or not math.isfinite(e_max) or e_max <= e_min:
        return np.array([], dtype=float)
    n = int(grid_n)
    if n < 2:
        raise ValueError("dos_grid_n must be >= 2")
    return np.linspace(e_min, e_max, n, dtype=float)


def compute_dos_curve(
    *,
    eigenvalues: Sequence[float],
    energy_grid: np.ndarray,
    eta: float,
    normalize: bool = True,
    chunk_size: int = 10_000,
) -> np.ndarray:
    grid = np.asarray(energy_grid, dtype=float)
    if grid.size == 0:
        return np.array([], dtype=float)
    vals = np.array(list(eigenvalues), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(grid)

    dos = np.zeros_like(grid)
    step = int(chunk_size)
    if step < 1:
        step = vals.size
    for start in range(0, vals.size, step):
        chunk = vals[start : start + step]
        diff = grid[None, :] - chunk[:, None]
        dos += np.sum(_gaussian_kernel(diff, eta=float(eta)), axis=0)
    if normalize:
        dos = dos / float(vals.size)
    return dos


def compute_ldos_curve(
    *,
    eigenvalues: Sequence[float],
    weights: Sequence[float],
    energy_grid: np.ndarray,
    eta: float,
) -> np.ndarray:
    grid = np.asarray(energy_grid, dtype=float)
    vals = np.array(list(eigenvalues), dtype=float)
    w = np.array(list(weights), dtype=float)
    if grid.size == 0 or vals.size == 0:
        return np.array([], dtype=float)
    if vals.size != w.size:
        raise ValueError("eigenvalues and weights must have same length for LDOS")
    diff = grid[None, :] - vals[:, None]
    g = _gaussian_kernel(diff, eta=float(eta))
    return np.sum(w[:, None] * g, axis=0)


def _softmax(x: np.ndarray) -> np.ndarray:
    vals = np.asarray(x, dtype=float)
    if vals.size == 0:
        return np.array([], dtype=float)
    z = vals - float(np.max(vals))
    w = np.exp(z)
    s = float(np.sum(w))
    if s <= 0.0 or not math.isfinite(s):
        return np.zeros_like(vals)
    return w / s


def solve_self_consistent_potential(
    *,
    laplacian: np.ndarray,
    v0: np.ndarray,
    scf_max_iter: int,
    scf_tol: float,
    scf_damping: float,
    scf_occ_k: int,
    scf_tau: float,
    scf_gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]], bool, int, float]:
    max_iter = int(scf_max_iter)
    if max_iter < 1:
        raise ValueError("scf_max_iter must be >= 1")
    tol = float(scf_tol)
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("scf_tol must be > 0")
    damping = float(scf_damping)
    if not math.isfinite(damping) or not (0.0 < damping <= 1.0):
        raise ValueError("scf_damping must be in (0, 1]")
    occ_k = int(scf_occ_k)
    if occ_k < 1:
        raise ValueError("scf_occ_k must be >= 1")
    tau = float(scf_tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("scf_tau must be > 0")
    gamma = float(scf_gamma)
    if not math.isfinite(gamma):
        raise ValueError("scf_gamma must be finite")

    lap = np.asarray(laplacian, dtype=float)
    v0_vec = np.asarray(v0, dtype=float)
    if lap.ndim != 2 or lap.shape[0] != lap.shape[1]:
        raise ValueError("laplacian must be a square matrix")
    n = int(lap.shape[0])
    if v0_vec.ndim != 1 or v0_vec.size != n:
        raise ValueError("v0 must be a vector of length n")

    v = v0_vec.copy()
    trace: list[dict[str, object]] = []
    converged = False
    residual_final = float("nan")
    iters = 0
    rho = np.zeros((n,), dtype=float)

    for t in range(max_iter):
        iters = t + 1
        H = lap + np.diag(v)
        eigvals, eigvecs = eigh_symmetric(H)
        k = int(min(int(occ_k), int(eigvals.size)))
        vals_k = np.asarray(eigvals[:k], dtype=float)
        vecs_k = np.asarray(eigvecs[:, :k], dtype=float)

        weights = _softmax(-vals_k / float(tau))
        rho = (vecs_k**2) @ weights
        rho_tilde = rho - float(np.mean(rho))

        v_proposed = v0_vec + float(gamma) * rho_tilde
        v_next = (1.0 - float(damping)) * v + float(damping) * v_proposed
        residual = float(np.max(np.abs(v_next - v)))
        residual_final = residual

        trace.append(
            {
                "iter": int(t + 1),
                "residual_inf": residual,
                "damping": float(damping),
                "min_V": float(np.min(v_next)),
                "max_V": float(np.max(v_next)),
                "mean_V": float(np.mean(v_next)),
                "min_rho": float(np.min(rho)),
                "max_rho": float(np.max(rho)),
                "mean_rho": float(np.mean(rho)),
                "converged": bool(residual < float(tol)),
            }
        )

        v = v_next
        if residual < float(tol):
            converged = True
            break

    return v0_vec, v, rho, trace, converged, int(iters), float(residual_final)


def ldos_atom_record(
    *,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    types: Sequence[int],
) -> dict[str, object]:
    vals = np.asarray(eigenvalues, dtype=float)
    vecs = np.asarray(eigenvectors, dtype=float)
    if vecs.ndim != 2 or vals.ndim != 1 or vecs.shape[0] != vecs.shape[1] or vecs.shape[1] != vals.size:
        raise ValueError("Invalid eigen-decomposition shapes for LDOS record")
    weights_by_atom = vecs**2
    max_w_by_atom = np.max(weights_by_atom, axis=1)
    atom_idx = int(np.argmax(max_w_by_atom))
    atomic_number = int(types[atom_idx])
    weights = np.asarray(weights_by_atom[atom_idx], dtype=float)
    return {
        "atom_idx": atom_idx,
        "atomic_number": atomic_number,
        "eigvals": [float(x) for x in vals.tolist()],
        "weights": [float(x) for x in weights.tolist()],
    }


def weighted_adjacency_from_bonds(
    *,
    n: int,
    bonds: Sequence[tuple[int, int, float]],
    types: Sequence[int],
    edge_weight_mode: str,
    atoms_db: AtomsDbV1 | None = None,
    alpha: float = DELTA_CHI_ALPHA_DEFAULT,
) -> np.ndarray:
    mode = str(edge_weight_mode)
    if mode not in {"unweighted", "bond_order", "bond_order_delta_chi"}:
        raise ValueError(f"Invalid edge_weight_mode: {mode}")

    n_i = int(n)
    w_adj = np.zeros((n_i, n_i), dtype=float)
    if mode == "unweighted":
        return w_adj

    chi: np.ndarray | None = None
    if mode == "bond_order_delta_chi":
        atoms_db_obj = load_atoms_db_v1() if atoms_db is None else atoms_db
        chi = _chis_for_types(types, atoms_db_obj)

    for i, j, bond_order in bonds:
        a, b = int(i), int(j)
        if a == b:
            continue
        bo = float(bond_order)
        if mode == "bond_order":
            w = bo
        else:
            assert chi is not None
            w = bo * (1.0 + float(alpha) * float(abs(float(chi[a]) - float(chi[b]))))
        w_adj[a, b] = w
        w_adj[b, a] = w

    return w_adj


def compute_physics_features(
    *,
    adjacency: np.ndarray,
    bonds: Sequence[tuple[int, int, float]] | None = None,
    types: Sequence[int],
    physics_mode: str,
    edge_weight_mode: str = "unweighted",
    potential_scale_gamma: float = POTENTIAL_SCALE_GAMMA_DEFAULT,
    beta: float = SPECTRAL_ENTROPY_BETA_DEFAULT,
    atoms_db: AtomsDbV1 | None = None,
) -> dict[str, object]:
    mode = str(physics_mode)
    if mode not in {"topological", "hamiltonian", "both"}:
        raise ValueError(f"Invalid physics_mode: {mode}")

    lap = laplacian_from_adjacency(adjacency)

    out: dict[str, object] = {
        "physics_mode_used": mode,
        "physics_params_source": PHYSICS_PARAMS_SOURCE,
        "physics_entropy_beta": float(beta),
        "physics_missing_params_count": 0,
        "edge_weight_mode_used": str(edge_weight_mode),
        "L_gap": "",
        "L_trace": "",
        "L_entropy_beta": "",
        "H_gap": "",
        "H_trace": "",
        "H_entropy_beta": "",
    }

    if mode in {"topological", "both"}:
        vals_L = eigvals_symmetric(lap)
        out["L_gap"] = float(spectral_gap(vals_L))
        out["L_trace"] = float(spectral_trace(vals_L))
        out["L_entropy_beta"] = float(spectral_entropy_beta(vals_L, beta=float(beta)))

    atoms_db_obj = load_atoms_db_v1() if atoms_db is None else atoms_db
    potentials: np.ndarray | None = None
    if mode in {"hamiltonian", "both"}:
        potentials = _apply_potential_scale(
            _potentials_for_types(types, atoms_db_obj),
            potential_scale_gamma=float(potential_scale_gamma),
        )
        H = lap + np.diag(potentials)
        vals_H = eigvals_symmetric(H)
        out["H_gap"] = float(spectral_gap(vals_H))
        out["H_trace"] = float(spectral_trace(vals_H))
        out["H_entropy_beta"] = float(spectral_entropy_beta(vals_H, beta=float(beta)))

    ew_mode = str(edge_weight_mode)
    if ew_mode != "unweighted":
        if bonds is None:
            raise ValueError("edge_weight_mode requires bonds to be provided")
        out.update(
            {
                "W_gap": "",
                "W_entropy": "",
                "WH_gap": "",
                "WH_entropy": "",
            }
        )
        w_adj = weighted_adjacency_from_bonds(
            n=int(adjacency.shape[0]),
            bonds=bonds,
            types=types,
            edge_weight_mode=ew_mode,
            atoms_db=atoms_db_obj,
            alpha=float(DELTA_CHI_ALPHA_DEFAULT),
        )
        lap_w = laplacian_from_adjacency(w_adj)
        if mode in {"topological", "both"}:
            vals_W = eigvals_symmetric(lap_w)
            out["W_gap"] = float(spectral_gap(vals_W))
            out["W_entropy"] = float(spectral_entropy_beta(vals_W, beta=float(beta)))
        if mode in {"hamiltonian", "both"}:
            if potentials is None:
                potentials = _apply_potential_scale(
                    _potentials_for_types(types, atoms_db_obj),
                    potential_scale_gamma=float(potential_scale_gamma),
                )
            Hw = lap_w + np.diag(potentials)
            vals_WH = eigvals_symmetric(Hw)
            out["WH_gap"] = float(spectral_gap(vals_WH))
            out["WH_entropy"] = float(spectral_entropy_beta(vals_WH, beta=float(beta)))

    return out


def compute_spectra(
    *,
    adjacency: np.ndarray,
    types: Sequence[int],
    physics_mode: str,
    potential_scale_gamma: float = POTENTIAL_SCALE_GAMMA_DEFAULT,
    atoms_db: AtomsDbV1 | None = None,
) -> dict[str, np.ndarray]:
    mode = str(physics_mode)
    if mode not in {"topological", "hamiltonian", "both"}:
        raise ValueError(f"Invalid physics_mode: {mode}")

    lap = laplacian_from_adjacency(adjacency)
    out: dict[str, np.ndarray] = {}
    if mode in {"topological", "both"}:
        out["L"] = eigvals_symmetric(lap)
    if mode in {"hamiltonian", "both"}:
        atoms_db_obj = load_atoms_db_v1() if atoms_db is None else atoms_db
        v = _apply_potential_scale(
            _potentials_for_types(types, atoms_db_obj),
            potential_scale_gamma=float(potential_scale_gamma),
        )
        H = lap + np.diag(v)
        out["H"] = eigvals_symmetric(H)
    return out


def compute_dos_ldos_payload(
    *,
    adjacency: np.ndarray,
    bonds: Sequence[tuple[int, int, float]] | None,
    types: Sequence[int],
    physics_mode: str,
    edge_weight_mode: str,
    potentials_override: np.ndarray | None = None,
    potential_scale_gamma: float = POTENTIAL_SCALE_GAMMA_DEFAULT,
    atoms_db: AtomsDbV1 | None = None,
) -> dict[str, object]:
    mode = str(physics_mode)
    if mode not in {"topological", "hamiltonian", "both"}:
        raise ValueError(f"Invalid physics_mode: {mode}")

    ew_mode = str(edge_weight_mode)
    if ew_mode not in {"unweighted", "bond_order", "bond_order_delta_chi"}:
        raise ValueError(f"Invalid edge_weight_mode: {ew_mode}")

    lap = laplacian_from_adjacency(adjacency)
    atoms_db_obj = load_atoms_db_v1() if atoms_db is None else atoms_db
    potentials: np.ndarray | None = None

    payload: dict[str, object] = {
        "schema_version": DOS_LDOS_SCHEMA,
        "physics_mode_used": mode,
        "edge_weight_mode_used": ew_mode,
        "eigvals_L": [],
        "eigvals_H": [],
        "eigvals_WH": [],
        "ldos_H": None,
        "ldos_WH": None,
    }

    if mode in {"topological", "both"}:
        payload["eigvals_L"] = [float(x) for x in eigvals_symmetric(lap).tolist()]

    if mode in {"hamiltonian", "both"}:
        if potentials_override is not None:
            potentials = np.asarray(potentials_override, dtype=float)
        else:
            potentials = _apply_potential_scale(
                _potentials_for_types(types, atoms_db_obj),
                potential_scale_gamma=float(potential_scale_gamma),
            )
        H = lap + np.diag(potentials)
        vals_H, vecs_H = eigh_symmetric(H)
        payload["eigvals_H"] = [float(x) for x in vals_H.tolist()]
        payload["ldos_H"] = ldos_atom_record(eigenvalues=vals_H, eigenvectors=vecs_H, types=types)

    if ew_mode != "unweighted":
        if bonds is None:
            raise ValueError("edge_weight_mode requires bonds to be provided")
        w_adj = weighted_adjacency_from_bonds(
            n=int(adjacency.shape[0]),
            bonds=bonds,
            types=types,
            edge_weight_mode=ew_mode,
            atoms_db=atoms_db_obj,
            alpha=float(DELTA_CHI_ALPHA_DEFAULT),
        )
        lap_w = laplacian_from_adjacency(w_adj)
        if mode in {"hamiltonian", "both"}:
            if potentials is None:
                potentials = np.asarray(potentials_override, dtype=float) if potentials_override is not None else _potentials_for_types(types, atoms_db_obj)
            Hw = lap_w + np.diag(potentials)
            vals_WH, vecs_WH = eigh_symmetric(Hw)
            payload["eigvals_WH"] = [float(x) for x in vals_WH.tolist()]
            payload["ldos_WH"] = ldos_atom_record(eigenvalues=vals_WH, eigenvectors=vecs_WH, types=types)

    return payload


def compute_operator_payload(
    *,
    adjacency: np.ndarray,
    bonds: Sequence[tuple[int, int, float]] | None,
    types: Sequence[int],
    physics_mode: str,
    edge_weight_mode: str,
    potential_mode: str = "static",
    potential_scale_gamma: float = POTENTIAL_SCALE_GAMMA_DEFAULT,
    scf_max_iter: int = SCF_MAX_ITER_DEFAULT,
    scf_tol: float = SCF_TOL_DEFAULT,
    scf_damping: float = SCF_DAMPING_DEFAULT,
    scf_occ_k: int = SCF_OCC_K_DEFAULT,
    scf_tau: float = SCF_TAU_DEFAULT,
    scf_gamma: float = SCF_GAMMA_DEFAULT,
    beta: float = SPECTRAL_ENTROPY_BETA_DEFAULT,
    atoms_db: AtomsDbV1 | None = None,
) -> dict[str, object]:
    mode = str(physics_mode)
    if mode not in {"topological", "hamiltonian", "both"}:
        raise ValueError(f"Invalid physics_mode: {mode}")

    ew_mode = str(edge_weight_mode)
    if ew_mode not in {"unweighted", "bond_order", "bond_order_delta_chi"}:
        raise ValueError(f"Invalid edge_weight_mode: {ew_mode}")

    pot_mode = str(potential_mode)
    if pot_mode not in {"static", "self_consistent", "both"}:
        raise ValueError(f"Invalid potential_mode: {pot_mode}")

    lap = laplacian_from_adjacency(adjacency)
    atoms_db_obj = load_atoms_db_v1() if atoms_db is None else atoms_db
    v0_raw = _potentials_for_types(types, atoms_db_obj)
    v0_scaled = _apply_potential_scale(v0_raw, potential_scale_gamma=float(potential_scale_gamma))
    gamma_scale = float(potential_scale_gamma)

    lap_w: np.ndarray | None = None
    if ew_mode != "unweighted":
        if bonds is None:
            raise ValueError("edge_weight_mode requires bonds to be provided")
        w_adj = weighted_adjacency_from_bonds(
            n=int(adjacency.shape[0]),
            bonds=bonds,
            types=types,
            edge_weight_mode=ew_mode,
            atoms_db=atoms_db_obj,
            alpha=float(DELTA_CHI_ALPHA_DEFAULT),
        )
        lap_w = laplacian_from_adjacency(w_adj)

    scf_payload: dict[str, object] | None = None
    v_scf: np.ndarray | None = None
    rho_final: np.ndarray | None = None
    scf_converged = False
    scf_iters = 0
    scf_residual_final = float("nan")

    if pot_mode in {"self_consistent", "both"} and mode in {"hamiltonian", "both"}:
        base_lap = lap_w if lap_w is not None else lap
        _, v_scf, rho_final, trace, scf_converged, scf_iters, scf_residual_final = solve_self_consistent_potential(
            laplacian=base_lap,
            v0=v0_scaled,
            scf_max_iter=int(scf_max_iter),
            scf_tol=float(scf_tol),
            scf_damping=float(scf_damping),
            scf_occ_k=int(scf_occ_k),
            scf_tau=float(scf_tau),
            scf_gamma=float(scf_gamma),
        )
        scf_payload = {
            "schema_version": SCF_SCHEMA,
            "potential_mode_used": pot_mode,
            "potential_unit_model": POTENTIAL_UNIT_MODEL,
            "potential_scale_gamma": float(gamma_scale),
            "scf_max_iter": int(scf_max_iter),
            "scf_tol": float(scf_tol),
            "scf_damping": float(scf_damping),
            "scf_occ_k": int(scf_occ_k),
            "scf_tau": float(scf_tau),
            "scf_gamma": float(scf_gamma),
            "scf_converged": bool(scf_converged),
            "scf_iters": int(scf_iters),
            "scf_residual_final": float(scf_residual_final),
            "trace": trace,
            "vectors": [
                {
                    "node_index": int(i),
                    "atom_Z": int(types[i]),
                    "V0": float(v0_raw[i]),
                    "V_scaled": float(v0_scaled[i]),
                    "gamma": float(gamma_scale),
                    "V_scf": float(v_scf[i]),
                    "rho_final": float(rho_final[i]) if rho_final is not None else float("nan"),
                }
                for i in range(int(v0_scaled.size))
            ],
        }

    potentials_effective = v0_scaled
    if pot_mode in {"self_consistent", "both"} and v_scf is not None:
        potentials_effective = v_scf

    out: dict[str, object] = {
        "physics_mode_used": mode,
        "physics_params_source": PHYSICS_PARAMS_SOURCE,
        "physics_entropy_beta": float(beta),
        "physics_missing_params_count": 0,
        "edge_weight_mode_used": ew_mode,
        "potential_mode_used": pot_mode,
        "potential_unit_model": POTENTIAL_UNIT_MODEL,
        "potential_scale_gamma": float(gamma_scale),
        "L_gap": "",
        "L_trace": "",
        "L_entropy_beta": "",
        "H_gap": "",
        "H_trace": "",
        "H_entropy_beta": "",
    }

    if mode in {"topological", "both"}:
        vals_L = eigvals_symmetric(lap)
        out["L_gap"] = float(spectral_gap(vals_L))
        out["L_trace"] = float(spectral_trace(vals_L))
        out["L_entropy_beta"] = float(spectral_entropy_beta(vals_L, beta=float(beta)))

    if mode in {"hamiltonian", "both"}:
        H_static = lap + np.diag(v0_scaled)
        vals_H_static = eigvals_symmetric(H_static)
        out["H_gap"] = float(spectral_gap(vals_H_static))
        out["H_trace"] = float(spectral_trace(vals_H_static))
        out["H_entropy_beta"] = float(spectral_entropy_beta(vals_H_static, beta=float(beta)))

        if pot_mode in {"self_consistent", "both"} and v_scf is not None:
            H_scf = lap + np.diag(potentials_effective)
            vals_H_scf = eigvals_symmetric(H_scf)
            out["H_scf_gap"] = float(spectral_gap(vals_H_scf))
            out["H_scf_trace"] = float(spectral_trace(vals_H_scf))
            out["H_scf_entropy_beta"] = float(spectral_entropy_beta(vals_H_scf, beta=float(beta)))
            out["scf_converged"] = bool(scf_converged)
            out["scf_iters"] = int(scf_iters)
            out["scf_residual_final"] = float(scf_residual_final)

    if ew_mode != "unweighted" and lap_w is not None:
        out.update({"W_gap": "", "W_entropy": "", "WH_gap": "", "WH_entropy": ""})
        if mode in {"topological", "both"}:
            vals_W = eigvals_symmetric(lap_w)
            out["W_gap"] = float(spectral_gap(vals_W))
            out["W_entropy"] = float(spectral_entropy_beta(vals_W, beta=float(beta)))
        if mode in {"hamiltonian", "both"}:
            Hw_static = lap_w + np.diag(v0_scaled)
            vals_WH_static = eigvals_symmetric(Hw_static)
            out["WH_gap"] = float(spectral_gap(vals_WH_static))
            out["WH_entropy"] = float(spectral_entropy_beta(vals_WH_static, beta=float(beta)))
            if pot_mode in {"self_consistent", "both"} and v_scf is not None:
                Hw_scf = lap_w + np.diag(potentials_effective)
                vals_WH_scf = eigvals_symmetric(Hw_scf)
                out["WH_scf_gap"] = float(spectral_gap(vals_WH_scf))
                out["WH_scf_entropy"] = float(spectral_entropy_beta(vals_WH_scf, beta=float(beta)))

    out["dos_ldos"] = compute_dos_ldos_payload(
        adjacency=adjacency,
        bonds=bonds,
        types=types,
        physics_mode=mode,
        edge_weight_mode=ew_mode,
        potentials_override=potentials_effective if mode in {"hamiltonian", "both"} else None,
        potential_scale_gamma=float(potential_scale_gamma),
        atoms_db=atoms_db_obj,
    )

    if scf_payload is not None:
        out["scf"] = scf_payload

    return out

