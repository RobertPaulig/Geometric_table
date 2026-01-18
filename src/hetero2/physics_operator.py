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


class MissingPhysicsParams(RuntimeError):
    def __init__(self, *, missing_atomic_numbers: Sequence[int], source_path: str) -> None:
        missing = sorted({int(z) for z in missing_atomic_numbers})
        self.missing_atomic_numbers = tuple(missing)
        self.source_path = str(source_path)
        super().__init__(f"missing_physics_params: atomic_numbers={missing} source={source_path}")


@dataclass(frozen=True)
class AtomsDbV1:
    source_path: str
    potential_by_atomic_num: Mapping[int, float]
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

    if not potential_by_z:
        raise ValueError(f"atoms_db_v1.json has no usable rows with keys Z/name/epsilon: {path}")

    return AtomsDbV1(
        source_path=str(path.as_posix()),
        potential_by_atomic_num=potential_by_z,
        symbol_by_atomic_num=symbol_by_z,
    )


def laplacian_from_adjacency(adj: np.ndarray) -> np.ndarray:
    deg = np.diag(adj.sum(axis=1))
    return deg - adj


def eigvals_symmetric(m: np.ndarray) -> np.ndarray:
    vals = np.linalg.eigvalsh(m)
    return np.sort(vals)


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
        raise MissingPhysicsParams(missing_atomic_numbers=missing, source_path=atoms_db.source_path)
    return np.array([float(atoms_db.potential_by_atomic_num[int(z)]) for z in types], dtype=float)


def compute_physics_features(
    *,
    adjacency: np.ndarray,
    types: Sequence[int],
    physics_mode: str,
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
        atoms_db_obj = load_atoms_db_v1() if atoms_db is None else atoms_db
        v = _potentials_for_types(types, atoms_db_obj)
        H = lap + np.diag(v)
        vals_H = eigvals_symmetric(H)
        out["H_gap"] = float(spectral_gap(vals_H))
        out["H_trace"] = float(spectral_trace(vals_H))
        out["H_entropy_beta"] = float(spectral_entropy_beta(vals_H, beta=float(beta)))

    return out


def compute_spectra(
    *,
    adjacency: np.ndarray,
    types: Sequence[int],
    physics_mode: str,
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
        v = _potentials_for_types(types, atoms_db_obj)
        H = lap + np.diag(v)
        out["H"] = eigvals_symmetric(H)
    return out

