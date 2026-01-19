from __future__ import annotations

import csv
import json
import hashlib
import math
import os
import platform
import subprocess
import statistics
import time
import zlib
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import hetero1a
import numpy as np
from hetero2.decoy_realism import (
    AUC_INTERPRETATION_SCHEMA,
    AUC_HARD_MIN_PAIRS_DEFAULT,
    TANIMOTO_EASY_MAX_EXCLUSIVE,
    TANIMOTO_MEDIUM_MAX_EXCLUSIVE,
    auc_pair_contribution,
    interpret_auc,
    tanimoto_bin,
)
from hetero2.physics_operator import (
    DOS_ETA_DEFAULT,
    DOS_GRID_N_DEFAULT,
    DOS_LDOS_SCHEMA,
    POTENTIAL_SCALE_GAMMA_DEFAULT,
    POTENTIAL_UNIT_MODEL,
    SCF_DAMPING_DEFAULT,
    SCF_GAMMA_DEFAULT,
    SCF_MAX_ITER_DEFAULT,
    SCF_OCC_K_DEFAULT,
    SCF_SCHEMA,
    SCF_TAU_DEFAULT,
    SCF_TOL_DEFAULT,
    MissingPhysicsParams,
    SPECTRAL_ENTROPY_BETA_DEFAULT,
    compute_dos_curve,
    compute_ldos_curve,
)
from hetero2.pipeline import run_pipeline_v2
from hetero2.report import render_report_v2


def _read_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows


def _stable_hash_id(text: str) -> int:
    """Deterministic 32-bit hash for seeds (Python hash() is randomized)."""
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def _mock_score_from_hash(hash_text: str) -> float:
    val = int(hash_text[:12], 16)
    return float(val) / float(16**12 - 1)


def _require_rdkit_fps():
    try:
        from rdkit import Chem, DataStructs  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("RDKit is required for Morgan fingerprints. Install: pip install -e \".[dev,chem]\"") from exc
    return Chem, AllChem, DataStructs


def _median(values: List[float]) -> float:
    finite = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]
    if not finite:
        return float("nan")
    return float(statistics.median(finite))


def _write_index_md(
    out_dir: Path,
    summary_rows: List[Dict[str, object]],
    *,
    scores_coverage: Dict[str, object] | None = None,
) -> None:
    lines = ["# Evidence Index", ""]
    if scores_coverage:
        lines.extend(
            [
                "## Scores Coverage",
                "",
                f"- rows_total: {scores_coverage.get('rows_total', 0)}",
                f"- rows_with_scores_input: {scores_coverage.get('rows_with_scores_input', 0)}",
                f"- rows_missing_scores_input: {scores_coverage.get('rows_missing_scores_input', 0)}",
                f"- rows_with_missing_decoys: {scores_coverage.get('rows_with_missing_decoys', 0)}",
                f"- decoys_total: {scores_coverage.get('decoys_total', 0)}",
                f"- decoys_scored: {scores_coverage.get('decoys_scored', 0)}",
                f"- decoys_missing: {scores_coverage.get('decoys_missing', 0)}",
                "",
            ]
        )
    lines.extend(
        [
            "| id | status | verdict | gate | slack | warnings | seed_used | report | assets | pipeline |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in sorted(summary_rows, key=lambda r: str(r.get("id", ""))):
        mol_id = str(row.get("id", ""))
        report_path = Path(str(row.get("report_path", ""))) if row.get("report_path") else None
        report_link = f"./{report_path.name}" if report_path and report_path.exists() else ""
        assets_dir = out_dir / f"{mol_id}_assets"
        assets_link = f"./{assets_dir.name}/" if assets_dir.exists() else ""
        pipeline_path = out_dir / f"{mol_id}.pipeline.json"
        pipeline_link = f"./{pipeline_path.name}" if pipeline_path.exists() else ""
        lines.append(
            f"| {mol_id} | {row.get('status','')} | {row.get('verdict','')} | {row.get('gate','')} | "
            f"{row.get('slack','')} | {row.get('warnings_count','')} | {row.get('seed_used','')} | "
            f"{report_link} | {assets_link} | {pipeline_link} |"
        )
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_sha_fallback() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True)
        return res.stdout.strip()
    except Exception:
        return None


def _rdkit_version() -> str | None:
    try:
        import rdkit  # type: ignore

        return getattr(rdkit, "__version__", None) or None
    except Exception:
        return None


def _write_manifest(
    out_dir: Path,
    *,
    seed: int,
    seed_strategy: str,
    score_mode: str,
    scores_provenance: Dict[str, str] | None,
    guardrails_max_atoms: int,
    guardrails_require_connected: bool,
    potential_unit_model: str,
    potential_scale_gamma: float,
    files: List[Dict[str, object]],
) -> None:
    payload: Dict[str, object] = {
        "tool_version": getattr(hetero1a, "__version__", None),
        "git_sha": _git_sha_fallback(),
        "python_version": platform.python_version(),
        "rdkit_version": _rdkit_version(),
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "config": {
            "seed_strategy": seed_strategy,
            "seed": int(seed),
            "score_mode": score_mode,
            "scores_provenance": scores_provenance or {},
            "guardrails_max_atoms": int(guardrails_max_atoms),
            "guardrails_require_connected": bool(guardrails_require_connected),
            "potential_unit_model": str(potential_unit_model),
            "potential_scale_gamma": float(potential_scale_gamma),
        },
        "files": [],
    }
    seen = set()
    files_deduped: List[Dict[str, object]] = []
    for item in files:
        path = str(item.get("path", ""))
        if not path or path in seen:
            continue
        seen.add(path)
        files_deduped.append(item)
    payload["files"] = sorted(files_deduped, key=lambda x: str(x.get("path", "")))
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _compute_file_infos(out_dir: Path, *, skip_names: set[str] | None = None) -> List[Dict[str, object]]:
    skip = skip_names or set()
    infos: List[Dict[str, object]] = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(out_dir).as_posix()
        if path.name in skip:
            continue
        digest = _sha256_of_file(path)
        infos.append({"path": f"./{rel}", "size_bytes": path.stat().st_size, "sha256": digest})
    return infos


def _sha256_of_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _scores_provenance(scores_input: str | None) -> Dict[str, str]:
    if not scores_input:
        return {}
    path = Path(scores_input)
    if not path.exists():
        return {}
    schema_version = ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema_version = str(payload.get("schema_version", ""))
    except Exception:
        schema_version = ""
    return {
        "scores_input_id": path.name,
        "scores_input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scores_schema_version": schema_version,
    }


def _write_checksums(out_dir: Path, file_infos: List[Dict[str, object]]) -> None:
    lines: List[str] = []
    for info in file_infos:
        sha = str(info.get("sha256") or "")
        rel = str(info.get("path") or "").lstrip("./")
        if not sha or not rel:
            continue
        lines.append(f"{sha}  {rel}")
    checksums_path = out_dir / "checksums.sha256"
    checksums_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_zip_pack(out_dir: Path, *, zip_name: str = "evidence_pack.zip") -> None:
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_dir():
                continue
            if path.name == zip_name:
                continue
            rel = path.relative_to(out_dir)
            zf.write(path, rel.as_posix())


def _process_row(
    mol_id: str,
    smiles: str,
    *,
    scores_path: str,
    k_decoys: int,
    seed: int,
    timestamp: str,
    score_mode: str,
    guardrails_max_atoms: int,
    guardrails_require_connected: bool,
    decoy_hard_mode: bool,
    decoy_hard_tanimoto_min: float,
    decoy_hard_tanimoto_max: float,
    physics_mode: str,
    edge_weight_mode: str,
    potential_mode: str,
    potential_scale_gamma: float,
    scf_max_iter: int,
    scf_tol: float,
    scf_damping: float,
    scf_occ_k: int,
    scf_tau: float,
) -> Dict[str, object]:
    """Isolated worker to avoid RDKit leaks across tasks."""
    if not smiles:
        return {
            "id": mol_id,
            "scores_input": scores_path,
            "status": "SKIP",
            "reason": "missing_smiles",
            "pipeline": None,
            "warnings": [],
            "n_decoys": "",
            "neg": {},
            "seed_used": seed,
        }
    try:
        pipeline = run_pipeline_v2(
            smiles,
            k_decoys=int(k_decoys),
            seed=int(seed),
            timestamp=timestamp,
            score_mode=score_mode if score_mode in {"mock", "external_scores"} else "mock",
            scores_input=scores_path or None,
            guardrails_max_atoms=int(guardrails_max_atoms),
            guardrails_require_connected=bool(guardrails_require_connected),
            decoy_hard_mode=bool(decoy_hard_mode),
            decoy_hard_tanimoto_min=float(decoy_hard_tanimoto_min),
            decoy_hard_tanimoto_max=float(decoy_hard_tanimoto_max),
            physics_mode=str(physics_mode),
            edge_weight_mode=str(edge_weight_mode),
            potential_mode=str(potential_mode),
            potential_scale_gamma=float(potential_scale_gamma),
            scf_max_iter=int(scf_max_iter),
            scf_tol=float(scf_tol),
            scf_damping=float(scf_damping),
            scf_occ_k=int(scf_occ_k),
            scf_tau=float(scf_tau),
        )
        warnings = pipeline.get("warnings", []) if isinstance(pipeline, dict) else []
        if score_mode == "mock" and scores_path:
            warnings = list(warnings) + ["scores_input_ignored_in_mock_mode"]
        skip = pipeline.get("skip") if isinstance(pipeline, dict) else None
        status = "SKIP" if isinstance(skip, dict) else "OK"
        reason = str(skip.get("reason", "")) if isinstance(skip, dict) else ""
        neg = pipeline.get("audit", {}).get("neg_controls", {}) if isinstance(pipeline, dict) else {}
        n_decoys = len(pipeline.get("decoys", [])) if isinstance(pipeline, dict) else ""
        return {
            "id": mol_id,
            "scores_input": scores_path,
            "status": status,
            "reason": reason,
            "pipeline": pipeline,
            "warnings": warnings,
            "n_decoys": n_decoys,
            "neg": neg,
            "seed_used": seed,
        }
    except MissingPhysicsParams as exc:
        return {
            "id": mol_id,
            "scores_input": scores_path,
            "status": "ERROR",
            "reason": "missing_physics_params",
            "pipeline": None,
            "warnings": [
                f"missing_physics_params_key:{exc.missing_key}",
                f"missing_physics_params_atomic_numbers:{','.join(str(z) for z in exc.missing_atomic_numbers)}",
            ],
            "n_decoys": "",
            "neg": {},
            "seed_used": seed,
        }
    except Exception as exc:
        return {
            "id": mol_id,
            "scores_input": scores_path,
            "status": "ERROR",
            "reason": repr(exc),
            "pipeline": None,
            "warnings": [],
            "n_decoys": "",
            "neg": {},
            "seed_used": seed,
        }


def run_batch(
    *,
    input_csv: Path,
    out_dir: Path,
    artifacts: str = "full",
    seed: int = 0,
    timestamp: str = "",
    k_decoys: int = 20,
    score_mode: str = "mock",
    scores_input: str | None = None,
    guardrails_max_atoms: int = 200,
    guardrails_require_connected: bool = True,
    decoy_hard_mode: bool = False,
    decoy_hard_tanimoto_min: float = 0.65,
    decoy_hard_tanimoto_max: float = 0.95,
    physics_mode: str = "topological",
    edge_weight_mode: str = "unweighted",
    potential_mode: str = "static",
    potential_scale_gamma: float = POTENTIAL_SCALE_GAMMA_DEFAULT,
    scf_max_iter: int = SCF_MAX_ITER_DEFAULT,
    scf_tol: float = SCF_TOL_DEFAULT,
    scf_damping: float = SCF_DAMPING_DEFAULT,
    scf_occ_k: int = SCF_OCC_K_DEFAULT,
    scf_tau: float = SCF_TAU_DEFAULT,
    seed_strategy: str = "global",
    no_index: bool = False,
    no_manifest: bool = False,
    zip_pack: bool = False,
    workers: int = 1,
    timeout_s: float | None = None,
    resume: bool = False,
    overwrite: bool = False,
    maxtasksperchild: int = 100,
) -> Path:
    t_start = time.time()
    if artifacts not in {"full", "light"}:
        artifacts = "full"
    rows = _read_rows(input_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    fieldnames: List[str] = [
        "id",
        "status",
        "reason",
        "skip_reason",
        "verdict",
        "gate",
        "slack",
        "margin",
        "n_decoys",
        "n_decoys_generated",
        "n_decoys_scored",
        "decoy_strategy_used",
        "warnings_count",
        "report_path",
        "seed_used",
        "spectral_gap",
        "spectral_entropy",
        "spectral_entropy_norm",
        "physics_mode_used",
        "physics_params_source",
        "physics_entropy_beta",
        "physics_missing_params_count",
        "L_gap",
        "L_trace",
        "L_entropy_beta",
        "H_gap",
        "H_trace",
        "H_entropy_beta",
    ]
    if str(edge_weight_mode) != "unweighted":
        fieldnames.extend(["W_gap", "W_entropy", "WH_gap", "WH_entropy"])
    fieldnames.extend(["outcome_verdict", "outcome_reason"])
    summary_rows: List[Dict[str, object]] = []
    scores_coverage = {
        "rows_total": 0,
        "rows_with_scores_input": 0,
        "rows_missing_scores_input": 0,
        "rows_with_missing_decoys": 0,
        "decoys_total": 0,
        "decoys_scored": 0,
        "decoys_missing": 0,
    }
    # Decoy realism (hardness curve): aggregated across all (row, decoy) pairs.
    tanimoto_values: List[float] = []
    pairs_total_by_bin: Counter[str] = Counter()
    pairs_scored_by_bin: Counter[str] = Counter()
    auc_numer_by_bin: Counter[str] = Counter()
    auc_denom_by_bin: Counter[str] = Counter()
    hardness_pairs_rows: List[Dict[str, object]] = []
    # Physics operator foundation: per-row feature rows.
    operator_rows: List[Dict[str, object]] = []
    # Self-consistent potential (SCF): per-iteration and per-node artifacts (may be empty if disabled).
    scf_trace_rows: List[Dict[str, object]] = []
    scf_row_summaries: List[Dict[str, object]] = []
    potential_vector_rows: List[Dict[str, object]] = []
    scf_rows_total = 0
    scf_rows_converged = 0
    scf_iters_max = 0
    scf_residual_final_max = float("nan")
    scf_trace_residuals: List[float] = []
    # Cache external scores payloads by resolved path.
    scores_cache: Dict[str, Dict[str, object]] = {}

    missing_decoy_hash_rows_affected: Counter[str] = Counter()
    missing_decoy_hash_to_smiles: Dict[str, str] = {}
    done_ids: set[str] = set()
    if summary_path.exists() and resume:
        existing = list(csv.DictReader(summary_path.read_text(encoding="utf-8").splitlines()))
        summary_rows.extend(existing)
        done_ids = {str(row.get("id", "")) for row in existing}
    summary_file = summary_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
    if summary_path.stat().st_size == 0:
        writer.writeheader()
        summary_file.flush()
        os.fsync(summary_file.fileno())

    def _write_summary_row(row: Dict[str, object]) -> None:
        writer.writerow(row)
        summary_file.flush()
        os.fsync(summary_file.fileno())

    tasks: List[Dict[str, object]] = []
    dos_eigs_L: list[list[float]] = []
    dos_eigs_H: list[list[float]] = []
    dos_eigs_WH: list[list[float]] = []
    ldos_inputs_by_id: dict[str, dict[str, dict[str, object]]] = {}
    for idx, row in enumerate(rows):
        mol_id = row.get("id") or f"mol_{idx}"
        if resume and not overwrite and mol_id in done_ids:
            continue
        smiles = row.get("smiles", "")
        scores_path = row.get("scores_input") or scores_input or ""
        derived_seed = int(seed)
        if seed_strategy == "per_row":
            derived_seed = int(seed) ^ _stable_hash_id(str(mol_id))
        tasks.append(
            {
                "mol_id": mol_id,
                "smiles": smiles,
                "scores_path": scores_path,
                "seed": derived_seed,
            }
        )

    def handle_result(res: Dict[str, object]) -> None:
        nonlocal scf_rows_total, scf_rows_converged, scf_iters_max, scf_residual_final_max
        pipeline = res.get("pipeline")
        mol_id = str(res.get("id", ""))
        status = str(res.get("status", ""))
        reason = str(res.get("reason", ""))
        warnings = res.get("warnings", []) if isinstance(res.get("warnings"), list) else []
        rep_path: Path | None = None
        if isinstance(pipeline, dict):
            warnings = sorted(set(warnings))
            pipeline["warnings"] = warnings
            if artifacts == "full":
                pipeline_to_write = dict(pipeline)
                op_raw = pipeline.get("operator", {})
                if isinstance(op_raw, dict):
                    op_to_write = dict(op_raw)
                    scf_raw = op_raw.get("scf", {})
                    if isinstance(scf_raw, dict) and str(scf_raw.get("schema_version", "")) == SCF_SCHEMA:
                        scf_to_write = dict(scf_raw)
                        scf_to_write.pop("trace", None)
                        scf_to_write.pop("vectors", None)
                        op_to_write["scf"] = scf_to_write
                    pipeline_to_write["operator"] = op_to_write
                pipe_path = out_dir / f"{mol_id}.pipeline.json"
                pipe_path.write_text(
                    json.dumps(pipeline_to_write, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
                rep_path = out_dir / f"{mol_id}.report.md"
                render_report_v2(pipeline, out_path=str(rep_path), assets_dir=out_dir / f"{mol_id}_assets")
        if score_mode == "external_scores":
            scores_input = str(res.get("scores_input", ""))
            scores_coverage["rows_total"] += 1
            if scores_input:
                scores_coverage["rows_with_scores_input"] += 1
            else:
                scores_coverage["rows_missing_scores_input"] += 1
            if isinstance(pipeline, dict):
                scov = pipeline.get("scores_coverage", {}) if isinstance(pipeline.get("scores_coverage"), dict) else {}
                decoys_total = int(scov.get("decoys_total", 0) or 0)
                decoys_scored = int(scov.get("decoys_scored", 0) or 0)
                decoys_missing = int(scov.get("decoys_missing", 0) or 0)
                scores_coverage["decoys_total"] += decoys_total
                scores_coverage["decoys_scored"] += decoys_scored
                scores_coverage["decoys_missing"] += decoys_missing
                if decoys_missing > 0:
                    scores_coverage["rows_with_missing_decoys"] += 1

                missing_hashes_raw = scov.get("missing_decoy_hashes", [])
                if isinstance(missing_hashes_raw, list):
                    missing_hashes = {str(h).strip() for h in missing_hashes_raw if str(h).strip()}
                else:
                    missing_hashes = set()

                if missing_hashes:
                    for h in missing_hashes:
                        missing_decoy_hash_rows_affected[h] += 1

                    decoys = pipeline.get("decoys", [])
                    if isinstance(decoys, list):
                        for d in decoys:
                            if not isinstance(d, dict):
                                continue
                            h = str(d.get("hash", "")).strip()
                            if not h or h not in missing_hashes or h in missing_decoy_hash_to_smiles:
                                continue
                            smi = str(d.get("smiles", "")).strip()
                            if smi:
                                missing_decoy_hash_to_smiles[h] = smi

        # Collect physics-operator features and decoy-realism stats for OK rows.
        if isinstance(pipeline, dict) and status == "OK":
            op = pipeline.get("operator", {}) if isinstance(pipeline.get("operator"), dict) else {}
            operator_rows.append(
                {"id": mol_id, **{k: op.get(k, "") for k in fieldnames if k.startswith(("physics_", "L_", "H_", "W_", "WH_"))}}
            )

            dos_ldos = op.get("dos_ldos", {}) if isinstance(op.get("dos_ldos"), dict) else {}
            if isinstance(dos_ldos, dict) and str(dos_ldos.get("schema_version", "")) == DOS_LDOS_SCHEMA:
                eigvals_L = dos_ldos.get("eigvals_L", [])
                eigvals_H = dos_ldos.get("eigvals_H", [])
                eigvals_WH = dos_ldos.get("eigvals_WH", [])
                if isinstance(eigvals_L, list) and eigvals_L:
                    dos_eigs_L.append([float(x) for x in eigvals_L])
                if isinstance(eigvals_H, list) and eigvals_H:
                    dos_eigs_H.append([float(x) for x in eigvals_H])
                if isinstance(eigvals_WH, list) and eigvals_WH:
                    dos_eigs_WH.append([float(x) for x in eigvals_WH])

                ldos_H = dos_ldos.get("ldos_H", None)
                if isinstance(ldos_H, dict):
                    ldos_inputs_by_id.setdefault(mol_id, {})["H"] = ldos_H
                ldos_WH = dos_ldos.get("ldos_WH", None)
                if isinstance(ldos_WH, dict):
                    ldos_inputs_by_id.setdefault(mol_id, {})["WH"] = ldos_WH

            scf = op.get("scf", {}) if isinstance(op.get("scf"), dict) else {}
            if isinstance(scf, dict) and str(scf.get("schema_version", "")) == SCF_SCHEMA:
                scf_rows_total += 1
                scf_converged = bool(scf.get("scf_converged", False))
                if scf_converged:
                    scf_rows_converged += 1
                scf_iters = int(scf.get("scf_iters", 0) or 0)
                scf_iters_max = max(scf_iters_max, scf_iters)
                scf_residual = float(scf.get("scf_residual_final", float("nan")))
                if math.isfinite(float(scf_residual)):
                    if not math.isfinite(float(scf_residual_final_max)) or float(scf_residual) > float(scf_residual_final_max):
                        scf_residual_final_max = float(scf_residual)

                scf_gamma = float(scf.get("scf_gamma", float("nan")))
                smiles_orig = str(pipeline.get("smiles", ""))
                hardness_bin = "na"
                is_asym_fixture = bool(str(mol_id).startswith("asym_"))

                row_residual_init = float("nan")
                row_residual_final = float("nan")
                row_delta_v_inf_max = float("nan")
                row_delta_v_inf_last = float("nan")
                row_stop_reason = ""

                trace = scf.get("trace", [])
                if isinstance(trace, list):
                    delta_v_series: List[float] = []
                    residual_series: List[float] = []
                    stop_reason_series: List[str] = []
                    for t in trace:
                        if not isinstance(t, dict):
                            continue
                        residual = float(t.get("residual", t.get("residual_inf", float("nan"))))
                        residual_inf = float(t.get("residual_inf", residual))
                        residual_mean = float(t.get("residual_mean", float("nan")))
                        delta_rho_l1 = float(t.get("delta_rho_l1", float("nan")))
                        delta_v_inf = float(t.get("delta_V_inf", residual_inf))
                        gamma_used = float(t.get("gamma", scf_gamma))
                        min_v = float(t.get("min_V", float("nan")))
                        max_v = float(t.get("max_V", float("nan")))
                        mean_v = float(t.get("mean_V", float("nan")))
                        min_rho = float(t.get("min_rho", float("nan")))
                        max_rho = float(t.get("max_rho", float("nan")))
                        mean_rho = float(t.get("mean_rho", float("nan")))
                        trace_status = str(t.get("status", "") or "")
                        if not trace_status:
                            trace_status = "CONVERGED" if bool(t.get("converged", False)) else "ITERATING"
                        stop_reason = str(t.get("stop_reason", "") or "")
                        if not stop_reason and trace_status == "CONVERGED":
                            stop_reason = "converged"
                        stop_reason_series.append(stop_reason)
                        if math.isfinite(float(residual)):
                            residual_series.append(float(residual))
                            scf_trace_residuals.append(float(residual))
                        if math.isfinite(float(delta_v_inf)):
                            delta_v_series.append(float(delta_v_inf))
                        scf_trace_rows.append(
                            {
                                "row_id": mol_id,
                                "mol_id": mol_id,
                                "smiles": smiles_orig,
                                "is_decoy": 0,
                                "hardness_bin": hardness_bin,
                                "iter": int(t.get("iter", 0) or 0),
                                "residual": "" if not math.isfinite(float(residual)) else f"{float(residual):.10g}",
                                "residual_inf": "" if not math.isfinite(float(residual_inf)) else f"{float(residual_inf):.10g}",
                                "residual_mean": "" if not math.isfinite(float(residual_mean)) else f"{float(residual_mean):.10g}",
                                "delta_rho_l1": "" if not math.isfinite(float(delta_rho_l1)) else f"{float(delta_rho_l1):.10g}",
                                "delta_V_inf": "" if not math.isfinite(float(delta_v_inf)) else f"{float(delta_v_inf):.10g}",
                                "gamma": "" if not math.isfinite(float(gamma_used)) else f"{float(gamma_used):.10g}",
                                "damping": "" if not math.isfinite(float(t.get("damping", float('nan')))) else f"{float(t.get('damping')):.10g}",
                                "min_V": "" if not math.isfinite(float(min_v)) else f"{float(min_v):.10g}",
                                "max_V": "" if not math.isfinite(float(max_v)) else f"{float(max_v):.10g}",
                                "mean_V": "" if not math.isfinite(float(mean_v)) else f"{float(mean_v):.10g}",
                                "min_rho": "" if not math.isfinite(float(min_rho)) else f"{float(min_rho):.10g}",
                                "max_rho": "" if not math.isfinite(float(max_rho)) else f"{float(max_rho):.10g}",
                                "mean_rho": "" if not math.isfinite(float(mean_rho)) else f"{float(mean_rho):.10g}",
                                "converged": bool(t.get("converged", False)),
                                "status": trace_status,
                                "stop_reason": stop_reason,
                            }
                        )

                    if residual_series:
                        row_residual_init = float(residual_series[0])
                        row_residual_final = float(residual_series[-1])
                    if delta_v_series:
                        row_delta_v_inf_max = float(max(delta_v_series))
                        row_delta_v_inf_last = float(delta_v_series[-1])
                    if stop_reason_series:
                        row_stop_reason = str(stop_reason_series[-1])
                    if not row_stop_reason:
                        row_stop_reason = "converged" if scf_converged else "max_iters"

                scf_row_summaries.append(
                    {
                        "row_id": mol_id,
                        "mol_id": mol_id,
                        "smiles": smiles_orig,
                        "is_decoy": 0,
                        "hardness_bin": hardness_bin,
                        "is_asym_fixture": bool(is_asym_fixture),
                        "scf_converged": bool(scf_converged),
                        "scf_iters": int(scf_iters),
                        "residual_init": float(row_residual_init),
                        "residual_final": float(row_residual_final),
                        "delta_V_inf_max": float(row_delta_v_inf_max),
                        "delta_V_inf_last": float(row_delta_v_inf_last),
                        "gamma": float(scf_gamma),
                        "stop_reason": str(row_stop_reason),
                    }
                )

                vectors = scf.get("vectors", [])
                if isinstance(vectors, list):
                    for v in vectors:
                        if not isinstance(v, dict):
                            continue
                        v0_val = float(v.get("V0", float("nan")))
                        v_scaled_val = float(v.get("V_scaled", float("nan")))
                        gamma_val = float(v.get("gamma", float("nan")))
                        v_scf_val = float(v.get("V_scf", float("nan")))
                        rho_val = float(v.get("rho_final", float("nan")))
                        potential_vector_rows.append(
                            {
                                "id": mol_id,
                                "node_index": int(v.get("node_index", -1)),
                                "atom_Z": int(v.get("atom_Z", 0)),
                                "V0": "" if not math.isfinite(float(v0_val)) else f"{float(v0_val):.10g}",
                                "V_scaled": "" if not math.isfinite(float(v_scaled_val)) else f"{float(v_scaled_val):.10g}",
                                "gamma": "" if not math.isfinite(float(gamma_val)) else f"{float(gamma_val):.10g}",
                                "V_scf": "" if not math.isfinite(float(v_scf_val)) else f"{float(v_scf_val):.10g}",
                                "rho_final": "" if not math.isfinite(float(rho_val)) else f"{float(rho_val):.10g}",
                            }
                        )

            decoys = pipeline.get("decoys", [])
            smiles_orig = str(pipeline.get("smiles", ""))
            physics_mode_used = str(op.get("physics_mode_used", ""))
            if isinstance(decoys, list) and decoys and smiles_orig:
                try:
                    Chem, AllChem, DataStructs = _require_rdkit_fps()
                    mol_orig = Chem.MolFromSmiles(smiles_orig)
                    if mol_orig is not None:
                        fp_orig = AllChem.GetMorganFingerprintAsBitVect(mol_orig, 2, nBits=2048)
                    else:
                        fp_orig = None
                except Exception:
                    fp_orig = None

                # Resolve scoring payload (external) or use mock.
                scores_path_raw = str(res.get("scores_input", "")).strip()
                scores_payload = None
                if score_mode == "external_scores" and scores_path_raw:
                    try:
                        scores_path = str(Path(scores_path_raw).resolve())
                    except Exception:
                        scores_path = scores_path_raw
                    if scores_path in scores_cache:
                        scores_payload = scores_cache[scores_path]
                    else:
                        try:
                            scores_payload = json.loads(Path(scores_path).read_text(encoding="utf-8"))
                            if isinstance(scores_payload, dict):
                                scores_cache[scores_path] = scores_payload
                        except Exception:
                            scores_payload = None

                orig_score = 1.0
                score_used = "mock"
                if isinstance(scores_payload, dict):
                    orig_score = float((scores_payload.get("original") or {}).get("score", 1.0))
                    score_used = str(scores_payload.get("score_key") or "external_scores").strip() or "external_scores"

                decoy_scores = (scores_payload.get("decoys") or {}) if isinstance(scores_payload, dict) else {}

                for d in decoys:
                    if not isinstance(d, dict):
                        continue
                    decoy_smiles = str(d.get("smiles", "")).strip()
                    decoy_hash = str(d.get("hash", "")).strip()
                    if not decoy_smiles or not decoy_hash:
                        continue
                    sim = float("nan")
                    if fp_orig is not None:
                        try:
                            mol_d = Chem.MolFromSmiles(decoy_smiles)
                            if mol_d is not None:
                                fp_d = AllChem.GetMorganFingerprintAsBitVect(mol_d, 2, nBits=2048)
                                sim = float(DataStructs.FingerprintSimilarity(fp_orig, fp_d))
                        except Exception:
                            sim = float("nan")
                    tanimoto_values.append(float(sim))
                    bin_id = tanimoto_bin(float(sim))
                    pairs_total_by_bin[bin_id] += 1

                    if score_mode == "external_scores":
                        entry = decoy_scores.get(decoy_hash)
                        if not isinstance(entry, dict):
                            hardness_pairs_rows.append(
                                {
                                    "molecule_id": mol_id,
                                    "decoy_id": decoy_hash,
                                    "tanimoto": "" if not math.isfinite(float(sim)) else f"{float(sim):.6f}",
                                    "hardness_bin": bin_id,
                                    "is_original_label": "",
                                    "score_used": score_used,
                                    "physics_mode_used": physics_mode_used,
                                    "pos_score": "" if not math.isfinite(float(orig_score)) else f"{float(orig_score):.6f}",
                                    "neg_score": "",
                                    "weight": "",
                                    "decoy_scored": 0,
                                }
                            )
                            continue
                        neg_score = float(entry.get("score", 0.0))
                        weight = float(entry.get("weight", 1.0))
                    else:
                        neg_score = float(_mock_score_from_hash(decoy_hash))
                        weight = 1.0

                    pairs_scored_by_bin[bin_id] += 1
                    auc_numer_by_bin[bin_id] += float(weight) * float(auc_pair_contribution(orig_score, neg_score))
                    auc_denom_by_bin[bin_id] += float(weight)
                    hardness_pairs_rows.append(
                        {
                            "molecule_id": mol_id,
                            "decoy_id": decoy_hash,
                            "tanimoto": "" if not math.isfinite(float(sim)) else f"{float(sim):.6f}",
                            "hardness_bin": bin_id,
                            "is_original_label": "",
                            "score_used": score_used,
                            "physics_mode_used": physics_mode_used,
                            "pos_score": "" if not math.isfinite(float(orig_score)) else f"{float(orig_score):.6f}",
                            "neg_score": "" if not math.isfinite(float(neg_score)) else f"{float(neg_score):.6f}",
                            "weight": "" if not math.isfinite(float(weight)) else f"{float(weight):.6f}",
                            "decoy_scored": 1,
                        }
                    )
        neg = res.get("neg", {}) if isinstance(res.get("neg"), dict) else {}
        spectral = {}
        if isinstance(pipeline, dict):
            spectral = pipeline.get("spectral", {}) if isinstance(pipeline.get("spectral"), dict) else {}
        decoy_strategy_used = ""
        n_decoys_generated = res.get("n_decoys", "")
        n_decoys_scored = ""
        if isinstance(pipeline, dict):
            ds = pipeline.get("decoy_strategy", {}) if isinstance(pipeline.get("decoy_strategy"), dict) else {}
            decoy_strategy_used = str(ds.get("strategy_id", ""))
            n_decoys_generated = len(pipeline.get("decoys", [])) if isinstance(pipeline.get("decoys"), list) else ""
            if pipeline.get("score_mode") == "external_scores":
                scov = pipeline.get("scores_coverage", {}) if isinstance(pipeline.get("scores_coverage"), dict) else {}
                n_decoys_scored = scov.get("decoys_scored", "")
            elif pipeline.get("score_mode") == "mock":
                n_decoys_scored = n_decoys_generated

        op = pipeline.get("operator", {}) if isinstance(pipeline, dict) and isinstance(pipeline.get("operator"), dict) else {}
        outcome_verdict = ""
        outcome_reason = ""
        if status == "ERROR" and reason == "missing_physics_params":
            outcome_verdict = "ERROR_MISSING_PHYSICS_PARAMS"
            outcome_reason = "missing_physics_params"
        elif status == "OK":
            scf = op.get("scf", {}) if isinstance(op.get("scf"), dict) else {}
            if isinstance(scf, dict) and str(scf.get("schema_version", "")) == SCF_SCHEMA and not bool(scf.get("scf_converged", True)):
                outcome_verdict = "INCONCLUSIVE_SCF_NOT_CONVERGED"
                outcome_reason = "scf_not_converged"
        summary_entry = {
            "id": mol_id,
            "status": status or "ERROR",
            "reason": reason,
            "skip_reason": reason if status == "SKIP" else "",
            "verdict": neg.get("verdict", ""),
            "gate": neg.get("gate", ""),
            "slack": neg.get("slack", ""),
            "margin": neg.get("margin", ""),
            "n_decoys": res.get("n_decoys", ""),
            "n_decoys_generated": n_decoys_generated,
            "n_decoys_scored": n_decoys_scored,
            "decoy_strategy_used": decoy_strategy_used,
            "warnings_count": len(set(warnings)) if isinstance(warnings, list) else 0,
            "report_path": str(rep_path) if status == "OK" and rep_path is not None else "",
            "seed_used": res.get("seed_used", ""),
            "spectral_gap": spectral.get("spectral_gap", ""),
            "spectral_entropy": spectral.get("spectral_entropy", ""),
            "spectral_entropy_norm": spectral.get("spectral_entropy_norm", ""),
            "physics_mode_used": op.get("physics_mode_used", ""),
            "physics_params_source": op.get("physics_params_source", ""),
            "physics_entropy_beta": op.get("physics_entropy_beta", ""),
            "physics_missing_params_count": op.get("physics_missing_params_count", ""),
            "L_gap": op.get("L_gap", ""),
            "L_trace": op.get("L_trace", ""),
            "L_entropy_beta": op.get("L_entropy_beta", ""),
            "H_gap": op.get("H_gap", ""),
            "H_trace": op.get("H_trace", ""),
            "H_entropy_beta": op.get("H_entropy_beta", ""),
            "outcome_verdict": outcome_verdict,
            "outcome_reason": outcome_reason,
        }
        if str(edge_weight_mode) != "unweighted":
            summary_entry.update(
                {
                    "W_gap": op.get("W_gap", ""),
                    "W_entropy": op.get("W_entropy", ""),
                    "WH_gap": op.get("WH_gap", ""),
                    "WH_entropy": op.get("WH_entropy", ""),
                }
            )
        summary_rows.append(summary_entry)
        _write_summary_row(summary_entry)

    if workers <= 1:
        for task in tasks:
            res = _process_row(
                task["mol_id"],
                task["smiles"],
                scores_path=task["scores_path"],
                k_decoys=int(k_decoys),
                seed=int(task["seed"]),
                timestamp=timestamp,
                score_mode="mock" if score_mode == "mock" else "external_scores",
                guardrails_max_atoms=int(guardrails_max_atoms),
                guardrails_require_connected=bool(guardrails_require_connected),
                decoy_hard_mode=bool(decoy_hard_mode),
                decoy_hard_tanimoto_min=float(decoy_hard_tanimoto_min),
                decoy_hard_tanimoto_max=float(decoy_hard_tanimoto_max),
                physics_mode=str(physics_mode),
                edge_weight_mode=str(edge_weight_mode),
                potential_mode=str(potential_mode),
                potential_scale_gamma=float(potential_scale_gamma),
                scf_max_iter=int(scf_max_iter),
                scf_tol=float(scf_tol),
                scf_damping=float(scf_damping),
                scf_occ_k=int(scf_occ_k),
                scf_tau=float(scf_tau),
            )
            handle_result(res)
    else:
        import multiprocessing

        with multiprocessing.Pool(processes=int(workers), maxtasksperchild=int(maxtasksperchild)) as pool:
            async_results = []
            for task in tasks:
                async_res = pool.apply_async(
                    _process_row,
                    (
                        task["mol_id"],
                        task["smiles"],
                    ),
                    dict(
                        scores_path=task["scores_path"],
                        k_decoys=int(k_decoys),
                        seed=int(task["seed"]),
                        timestamp=timestamp,
                        score_mode="mock" if score_mode == "mock" else "external_scores",
                        guardrails_max_atoms=int(guardrails_max_atoms),
                        guardrails_require_connected=bool(guardrails_require_connected),
                        decoy_hard_mode=bool(decoy_hard_mode),
                        decoy_hard_tanimoto_min=float(decoy_hard_tanimoto_min),
                        decoy_hard_tanimoto_max=float(decoy_hard_tanimoto_max),
                        physics_mode=str(physics_mode),
                        edge_weight_mode=str(edge_weight_mode),
                        potential_mode=str(potential_mode),
                        potential_scale_gamma=float(potential_scale_gamma),
                        scf_max_iter=int(scf_max_iter),
                        scf_tol=float(scf_tol),
                        scf_damping=float(scf_damping),
                        scf_occ_k=int(scf_occ_k),
                        scf_tau=float(scf_tau),
                    ),
                )
                async_results.append((task["mol_id"], async_res))
            for mol_id, ar in async_results:
                try:
                    res = ar.get(timeout=timeout_s) if timeout_s else ar.get()
                except Exception as exc:
                    res = {
                        "id": mol_id,
                        "status": "ERROR",
                        "reason": "timeout" if "Timeout" in exc.__class__.__name__ else repr(exc),
                        "pipeline": None,
                        "warnings": [],
                        "n_decoys": "",
                        "neg": {},
                        "seed_used": seed,
                    }
                handle_result(res)

    summary_file.close()

    # Write operator features table (always present; columns depend on physics_mode).
    operator_features_path = out_dir / "operator_features.csv"
    operator_features_path.write_text("", encoding="utf-8")
    with operator_features_path.open("w", encoding="utf-8", newline="") as f:
        operator_fieldnames = [
            "id",
            "physics_mode_used",
            "physics_params_source",
            "physics_entropy_beta",
            "physics_missing_params_count",
            "L_gap",
            "L_trace",
            "L_entropy_beta",
            "H_gap",
            "H_trace",
            "H_entropy_beta",
        ]
        if str(edge_weight_mode) != "unweighted":
            operator_fieldnames.extend(["W_gap", "W_entropy", "WH_gap", "WH_entropy"])
        w = csv.DictWriter(f, fieldnames=operator_fieldnames)
        w.writeheader()
        for row in operator_rows:
            w.writerow(row)

    # Write SCF artifacts (always present; may be empty if potential_mode does not compute SCF).
    scf_trace_csv = out_dir / "scf_trace.csv"
    potential_vectors_csv = out_dir / "potential_vectors.csv"
    scf_trace_csv.write_text("", encoding="utf-8")
    potential_vectors_csv.write_text("", encoding="utf-8")
    with scf_trace_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "row_id",
                "mol_id",
                "smiles",
                "is_decoy",
                "hardness_bin",
                "iter",
                "residual",
                "residual_inf",
                "residual_mean",
                "delta_rho_l1",
                "delta_V_inf",
                "gamma",
                "damping",
                "min_V",
                "max_V",
                "mean_V",
                "min_rho",
                "max_rho",
                "mean_rho",
                "converged",
                "status",
                "stop_reason",
            ],
        )
        w.writeheader()
        for row in scf_trace_rows:
            w.writerow(row)
    with potential_vectors_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "node_index",
                "atom_Z",
                "V0",
                "V_scaled",
                "gamma",
                "V_scf",
                "rho_final",
            ],
        )
        w.writeheader()
        for row in potential_vector_rows:
            w.writerow(row)

    # Write DOS/LDOS artifacts (always present; may be empty if too little data).
    dos_grid_n = int(DOS_GRID_N_DEFAULT)
    dos_eta = float(DOS_ETA_DEFAULT)
    dos_curve_csv = out_dir / "dos_curve.csv"
    ldos_summary_csv = out_dir / "ldos_summary.csv"

    dos_curve_csv.write_text("", encoding="utf-8")
    ldos_summary_csv.write_text("", encoding="utf-8")

    # Global energy grid for this run (single grid for all operator variants).
    all_eigs: list[float] = []
    for seq in dos_eigs_L:
        all_eigs.extend(seq)
    for seq in dos_eigs_H:
        all_eigs.extend(seq)
    for seq in dos_eigs_WH:
        all_eigs.extend(seq)

    energy_grid: np.ndarray
    dos_energy_min = float("nan")
    dos_energy_max = float("nan")
    if all_eigs:
        vals = np.array(all_eigs, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size >= 1:
            margin = 3.0 * dos_eta
            dos_energy_min = float(np.min(vals)) - margin
            dos_energy_max = float(np.max(vals)) + margin
            energy_grid = np.linspace(dos_energy_min, dos_energy_max, dos_grid_n, dtype=float)
        else:
            energy_grid = np.array([], dtype=float)
    else:
        energy_grid = np.array([], dtype=float)

    dos_L = compute_dos_curve(eigenvalues=[x for seq in dos_eigs_L for x in seq], energy_grid=energy_grid, eta=dos_eta) if dos_eigs_L else np.zeros_like(energy_grid)
    dos_H = compute_dos_curve(eigenvalues=[x for seq in dos_eigs_H for x in seq], energy_grid=energy_grid, eta=dos_eta) if dos_eigs_H else np.zeros_like(energy_grid)
    dos_WH = compute_dos_curve(eigenvalues=[x for seq in dos_eigs_WH for x in seq], energy_grid=energy_grid, eta=dos_eta) if dos_eigs_WH else np.zeros_like(energy_grid)

    with dos_curve_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["energy", "dos_L", "dos_H", "dos_WH"])
        w.writeheader()
        for idx in range(int(energy_grid.size)):
            w.writerow(
                {
                    "energy": f"{float(energy_grid[idx]):.10g}",
                    "dos_L": f"{float(dos_L[idx]):.10g}" if energy_grid.size else "",
                    "dos_H": f"{float(dos_H[idx]):.10g}" if energy_grid.size else "",
                    "dos_WH": f"{float(dos_WH[idx]):.10g}" if energy_grid.size else "",
                }
            )

    def _ldos_peak_entropy(eigvals: list[float], weights: list[float]) -> tuple[float, float, float]:
        if energy_grid.size == 0:
            return float("nan"), float("nan"), float("nan")
        curve = compute_ldos_curve(eigenvalues=eigvals, weights=weights, energy_grid=energy_grid, eta=dos_eta)
        if curve.size == 0:
            return float("nan"), float("nan"), float("nan")
        peak_idx = int(np.argmax(curve))
        peak_energy = float(energy_grid[peak_idx])
        peak_value = float(curve[peak_idx])
        total = float(np.sum(curve))
        if total <= 0.0 or not math.isfinite(total):
            ent = float("nan")
        else:
            p = curve / total
            ent = float(-np.sum(p * np.log(p + 1e-12)))
        return peak_energy, peak_value, ent

    ldos_summary_rows: list[dict[str, object]] = []
    for mol_id in sorted(ldos_inputs_by_id.keys()):
        payload = ldos_inputs_by_id[mol_id]
        row: dict[str, object] = {"id": mol_id}
        for op_id in ["H", "WH"]:
            rec = payload.get(op_id, None)
            if not isinstance(rec, dict):
                row.update(
                    {
                        f"{op_id}_atom_idx": "",
                        f"{op_id}_atomic_number": "",
                        f"{op_id}_ldos_peak_energy": "",
                        f"{op_id}_ldos_peak_value": "",
                        f"{op_id}_ldos_entropy": "",
                    }
                )
                continue
            atom_idx = rec.get("atom_idx", "")
            atomic_number = rec.get("atomic_number", "")
            eigvals = rec.get("eigvals", [])
            weights = rec.get("weights", [])
            if not isinstance(eigvals, list) or not isinstance(weights, list):
                peak_energy = peak_value = ent = float("nan")
            else:
                peak_energy, peak_value, ent = _ldos_peak_entropy([float(x) for x in eigvals], [float(x) for x in weights])
            row.update(
                {
                    f"{op_id}_atom_idx": int(atom_idx) if str(atom_idx).isdigit() else atom_idx,
                    f"{op_id}_atomic_number": int(atomic_number) if str(atomic_number).isdigit() else atomic_number,
                    f"{op_id}_ldos_peak_energy": "" if not math.isfinite(float(peak_energy)) else f"{float(peak_energy):.10g}",
                    f"{op_id}_ldos_peak_value": "" if not math.isfinite(float(peak_value)) else f"{float(peak_value):.10g}",
                    f"{op_id}_ldos_entropy": "" if not math.isfinite(float(ent)) else f"{float(ent):.10g}",
                }
            )
        ldos_summary_rows.append(row)

    with ldos_summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "H_atom_idx",
                "H_atomic_number",
                "H_ldos_peak_energy",
                "H_ldos_peak_value",
                "H_ldos_entropy",
                "WH_atom_idx",
                "WH_atomic_number",
                "WH_ldos_peak_energy",
                "WH_ldos_peak_value",
                "WH_ldos_entropy",
            ],
        )
        w.writeheader()
        for row in ldos_summary_rows:
            w.writerow(row)

    # Write hardness curve artifacts (always present; may be inconclusive if too few pairs).
    hardness_curve_csv = out_dir / "hardness_curve.csv"
    hardness_curve_md = out_dir / "hardness_curve.md"
    median_tanimoto = _median(tanimoto_values)
    bins = [
        ("easy", 0.0, TANIMOTO_EASY_MAX_EXCLUSIVE),
        ("medium", TANIMOTO_EASY_MAX_EXCLUSIVE, TANIMOTO_MEDIUM_MAX_EXCLUSIVE),
        ("hard", TANIMOTO_MEDIUM_MAX_EXCLUSIVE, 1.0),
        ("unknown", float("nan"), float("nan")),
    ]
    auc_by_bin: Dict[str, float] = {}
    for bin_id, _, _ in bins:
        denom = float(auc_denom_by_bin.get(bin_id, 0.0))
        numer = float(auc_numer_by_bin.get(bin_id, 0.0))
        auc_by_bin[bin_id] = float(numer / denom) if denom > 0 else float("nan")

    hard_pairs = int(pairs_total_by_bin.get("hard", 0))
    auc_label, auc_reason = interpret_auc(
        median_tanimoto=float(median_tanimoto),
        auc_easy=auc_by_bin.get("easy"),
        auc_hard=auc_by_bin.get("hard"),
        hard_pairs=hard_pairs,
        hard_pairs_min=int(AUC_HARD_MIN_PAIRS_DEFAULT),
    )

    hardness_curve_csv.write_text("", encoding="utf-8")
    with hardness_curve_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "molecule_id",
                "decoy_id",
                "tanimoto",
                "hardness_bin",
                "is_original_label",
                "score_used",
                "physics_mode_used",
                "pos_score",
                "neg_score",
                "weight",
                "decoy_scored",
            ],
        )
        w.writeheader()
        for row in hardness_pairs_rows:
            w.writerow(row)

    hardness_lines = [
        "# Decoy Hardness Curve (AUC vs Tanimoto)",
        "",
        "## Fingerprint config",
        "",
        "- morgan_radius: 2",
        "- morgan_nbits: 2048",
        "",
        "## Binning",
        "",
        f"- easy: tanimoto < {TANIMOTO_EASY_MAX_EXCLUSIVE:.2f}",
        f"- medium: {TANIMOTO_EASY_MAX_EXCLUSIVE:.2f} <= tanimoto < {TANIMOTO_MEDIUM_MAX_EXCLUSIVE:.2f}",
        f"- hard: tanimoto >= {TANIMOTO_MEDIUM_MAX_EXCLUSIVE:.2f}",
        "",
        f"- tanimoto_median: {median_tanimoto if math.isfinite(float(median_tanimoto)) else 'nan'}",
        f"- auc_interpretation_schema: {AUC_INTERPRETATION_SCHEMA}",
        f"- auc_interpretation: {auc_label}",
        f"- auc_interpretation_reason: {auc_reason}",
        f"- auc_hard_min_pairs: {int(AUC_HARD_MIN_PAIRS_DEFAULT)}",
        "",
        "## AUC by bin",
        "",
        "| bin | tanimoto range | pairs_total | pairs_scored | auc_tie_aware |",
        "| --- | --- | --- | --- | --- |",
    ]
    for bin_id, lo, hi in bins:
        if bin_id == "hard":
            rng = f"[{TANIMOTO_MEDIUM_MAX_EXCLUSIVE:.2f}, 1.00]"
        elif bin_id == "medium":
            rng = f"[{TANIMOTO_EASY_MAX_EXCLUSIVE:.2f}, {TANIMOTO_MEDIUM_MAX_EXCLUSIVE:.2f})"
        elif bin_id == "easy":
            rng = f"[0.00, {TANIMOTO_EASY_MAX_EXCLUSIVE:.2f})"
        else:
            rng = "n/a"
        auc_val = auc_by_bin.get(bin_id, float("nan"))
        hardness_lines.append(
            f"| {bin_id} | {rng} | {int(pairs_total_by_bin.get(bin_id, 0))} | {int(pairs_scored_by_bin.get(bin_id, 0))} | "
            f"{'' if not math.isfinite(float(auc_val)) else f'{auc_val:.6f}'} |"
        )
    hardness_lines.append("")
    hardness_curve_md.write_text("\n".join(hardness_lines) + "\n", encoding="utf-8")

    # Write summary_metadata.json for audit-grade consumption (no schema bump, additive artifact).
    scf_converged_all = bool(scf_rows_total > 0 and scf_rows_converged == scf_rows_total)
    scf_enabled = bool(str(potential_mode) in {"self_consistent", "both"} and str(physics_mode) in {"hamiltonian", "both"})
    error_rows_missing_params = sum(
        1 for r in summary_rows if r.get("status") == "ERROR" and r.get("reason") == "missing_physics_params"
    )
    error_rows_other = sum(
        1 for r in summary_rows if r.get("status") == "ERROR" and r.get("reason") != "missing_physics_params"
    )

    scf_status: str | None = None
    if scf_enabled:
        if error_rows_missing_params > 0:
            scf_status = "ERROR_MISSING_PARAMS"
        elif error_rows_other > 0:
            scf_status = "ERROR_NUMERICAL"
        elif scf_rows_total <= 0:
            scf_status = "ERROR_NUMERICAL"
        else:
            scf_status = "CONVERGED" if bool(scf_converged_all) else "MAX_ITER"

    # SCF audit-proof aggregates (P3.6 rails).
    SCF_ZERO_EPS = 1e-12
    SCF_NONTRIVIAL_EPS_V = 1e-6

    def _p95(values: List[float]) -> float:
        finite = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]
        if not finite:
            return float("nan")
        finite.sort()
        idx = int(math.ceil(0.95 * len(finite))) - 1
        idx = max(0, min(idx, len(finite) - 1))
        return float(finite[idx])

    def _mean(values: List[float]) -> float:
        finite = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]
        if not finite:
            return float("nan")
        return float(statistics.mean(finite))

    def _to_float_or_nan(value: object) -> float:
        try:
            v = float(value)  # type: ignore[arg-type]
        except Exception:
            return float("nan")
        return float(v) if math.isfinite(float(v)) else float("nan")

    def _abs_deltas_by_row_id() -> Dict[str, List[float]]:
        by_id: Dict[str, List[float]] = {}
        for r in potential_vector_rows:
            if not isinstance(r, dict):
                continue
            row_id = str(r.get("id", "")).strip()
            if not row_id:
                continue
            v_scaled = _to_float_or_nan(r.get("V_scaled", float("nan")))
            v_scf = _to_float_or_nan(r.get("V_scf", float("nan")))
            if not (math.isfinite(v_scaled) and math.isfinite(v_scf)):
                continue
            by_id.setdefault(row_id, []).append(abs(float(v_scf) - float(v_scaled)))
        return by_id

    def _scf_subset_stats(rows: List[Dict[str, object]]) -> Dict[str, object]:
        iters = [int(r.get("scf_iters", 0) or 0) for r in rows]
        residual_init = [float(r.get("residual_init", float("nan"))) for r in rows]
        residual_final = [float(r.get("residual_final", float("nan"))) for r in rows]
        delta_v = [float(r.get("delta_V_inf_max", float("nan"))) for r in rows]
        converged_flags = [bool(r.get("scf_converged", False)) for r in rows]

        out: Dict[str, object] = {
            "rows_total": int(len(rows)),
            "rows_with_scf": int(len(rows)),
            "iters_min": int(min(iters)) if iters else 0,
            "iters_median": float(statistics.median(iters)) if iters else float("nan"),
            "iters_mean": float(statistics.mean(iters)) if iters else float("nan"),
            "iters_max": int(max(iters)) if iters else 0,
            "converged_rate": (sum(1 for x in converged_flags if x) / len(converged_flags)) if converged_flags else float("nan"),
            "residual_init_mean": float(_mean(residual_init)),
            "residual_final_mean": float(_mean(residual_final)),
            "residual_final_max": float(max([x for x in residual_final if math.isfinite(float(x))], default=float("nan"))),
            "delta_V_max_overall": float(max([x for x in delta_v if math.isfinite(float(x))], default=float("nan"))),
            "delta_V_p95": float(_p95(delta_v)),
            "delta_V_median": float(_median(delta_v)),
        }
        return out

    scf_stats_all = _scf_subset_stats(scf_row_summaries)
    scf_rows_asym_fixture = [r for r in scf_row_summaries if bool(r.get("is_asym_fixture", False))]
    scf_stats_asym = _scf_subset_stats(scf_rows_asym_fixture)

    # Per-row SCF audit metrics (deltaV is defined as max|V_scf - V_scaled| over nodes).
    scf_abs_deltas_by_id = _abs_deltas_by_row_id()
    scf_audit_metrics_rows: List[Dict[str, object]] = []
    for r in scf_row_summaries:
        row_id = str(r.get("mol_id", r.get("row_id", ""))).strip()
        deltas = scf_abs_deltas_by_id.get(row_id, [])
        delta_v_max = float(max(deltas)) if deltas else float("nan")
        delta_v_p95 = float(_p95(deltas)) if deltas else float("nan")
        scf_audit_metrics_rows.append(
            {
                "mol_id": row_id,
                "scf_converged": bool(r.get("scf_converged", False)),
                "scf_iters": int(r.get("scf_iters", 0) or 0),
                "residual_init": float(r.get("residual_init", float("nan"))),
                "residual_final": float(r.get("residual_final", float("nan"))),
                "deltaV_max": float(delta_v_max),
                "deltaV_p95": float(delta_v_p95),
                "potential_gamma": float(potential_scale_gamma),
                "operator_mode": str(physics_mode),
            }
        )

    scf_iters_series = [int(r.get("scf_iters", 0) or 0) for r in scf_audit_metrics_rows]
    scf_residual_final_series = [float(r.get("residual_final", float("nan"))) for r in scf_audit_metrics_rows]
    scf_delta_v_max_series = [float(r.get("deltaV_max", float("nan"))) for r in scf_audit_metrics_rows]
    scf_delta_v_p95_series = [float(r.get("deltaV_p95", float("nan"))) for r in scf_audit_metrics_rows]
    scf_converged_flags = [bool(r.get("scf_converged", False)) for r in scf_audit_metrics_rows]
    scf_converged_rate = (
        (sum(1 for x in scf_converged_flags if x) / len(scf_converged_flags)) if scf_converged_flags else float("nan")
    )
    scf_iters_mean = float(_mean([float(x) for x in scf_iters_series]))
    scf_iters_p95 = float(_p95([float(x) for x in scf_iters_series]))
    scf_iters_max = int(max(scf_iters_series)) if scf_iters_series else 0
    scf_residual_final_p95 = float(_p95(scf_residual_final_series))
    scf_residual_final_max = float(max([x for x in scf_residual_final_series if math.isfinite(float(x))], default=float("nan")))
    scf_delta_v_max_max = float(max([x for x in scf_delta_v_max_series if math.isfinite(float(x))], default=float("nan")))
    scf_delta_v_p95_max = float(max([x for x in scf_delta_v_p95_series if math.isfinite(float(x))], default=float("nan")))

    scf_nontrivial_flags = [
        bool(r.get("scf_converged", False))
        and int(r.get("scf_iters", 0) or 0) >= 2
        and math.isfinite(float(r.get("deltaV_max", float("nan"))))
        and float(r.get("deltaV_max", float("nan"))) >= float(SCF_NONTRIVIAL_EPS_V)
        for r in scf_audit_metrics_rows
    ]
    scf_nontrivial_rate = (
        (sum(1 for x in scf_nontrivial_flags if x) / len(scf_nontrivial_flags)) if scf_nontrivial_flags else float("nan")
    )

    scf_triviality_flags = {
        "all_iters_eq_1": bool(scf_row_summaries) and all(int(r.get("scf_iters", 0) or 0) == 1 for r in scf_row_summaries),
        "all_delta_V_eq_0": bool(scf_row_summaries)
        and all(
            (not math.isfinite(float(r.get("delta_V_inf_max", float("nan")))))
            or abs(float(r.get("delta_V_inf_max", float("nan")))) <= SCF_ZERO_EPS
            for r in scf_row_summaries
        ),
        "residual_always_zero_or_nan": not any(
            math.isfinite(float(x)) and abs(float(x)) > SCF_ZERO_EPS for x in scf_trace_residuals
        ),
    }

    scf_audit_verdict = "INCONCLUSIVE_INSUFFICIENT_ASYM"
    scf_audit_reason = "scf_disabled"
    if bool(scf_enabled):
        if not scf_trace_rows:
            scf_audit_verdict = "INCONCLUSIVE_INSUFFICIENT_ASYM"
            scf_audit_reason = "scf_trace_empty"
        else:
            if math.isfinite(float(scf_converged_rate)) and float(scf_converged_rate) < 0.95:
                scf_audit_verdict = "NONCONVERGED"
                scf_audit_reason = "converged_rate_below_0p95"
            else:
                if int(scf_iters_max) <= 1 or (
                    math.isfinite(float(scf_delta_v_max_max)) and float(scf_delta_v_max_max) < float(SCF_NONTRIVIAL_EPS_V)
                ):
                    scf_audit_verdict = "TRIVIAL_FIXED_POINT"
                    scf_audit_reason = "iters_max_le_1_or_deltaV_below_eps"
                else:
                    n_asym = int(scf_stats_asym.get("rows_with_scf", 0))
                    if n_asym < 10:
                        scf_audit_verdict = "INCONCLUSIVE_INSUFFICIENT_ASYM"
                        scf_audit_reason = f"insufficient_asym_fixture_rows:{n_asym}"
                    else:
                        if (
                            math.isfinite(float(scf_nontrivial_rate))
                            and float(scf_nontrivial_rate) >= 0.50
                            and math.isfinite(float(scf_converged_rate))
                            and float(scf_converged_rate) >= 0.95
                        ):
                            scf_audit_verdict = "SUCCESS"
                            scf_audit_reason = "nontrivial_rate_ge_0p50_and_converged_rate_ge_0p95"
                        else:
                            scf_audit_verdict = "TRIVIAL_FIXED_POINT"
                            scf_audit_reason = "nontrivial_rate_below_0p50"

    summary_metadata = {
        "auc_interpretation_schema": AUC_INTERPRETATION_SCHEMA,
        "auc_interpretation": str(auc_label),
        "auc_interpretation_reason": str(auc_reason),
        "tanimoto_median": float(median_tanimoto) if math.isfinite(float(median_tanimoto)) else float("nan"),
        "pairs_total_by_bin": {k: int(v) for k, v in pairs_total_by_bin.items()},
        "pairs_scored_by_bin": {k: int(v) for k, v in pairs_scored_by_bin.items()},
        "auc_tie_aware_by_bin": {k: float(v) for k, v in auc_by_bin.items()},
        "physics_mode": str(physics_mode),
        "edge_weight_mode": str(edge_weight_mode),
        "potential_mode": str(potential_mode),
        "potential_unit_model": str(POTENTIAL_UNIT_MODEL),
        "potential_scale_gamma": float(potential_scale_gamma),
        "scf_schema": str(SCF_SCHEMA),
        "scf_enabled": bool(scf_enabled),
        "scf_status": scf_status,
        "scf_max_iter": int(scf_max_iter),
        "scf_tol": float(scf_tol),
        "scf_damping": float(scf_damping),
        "scf_occ_k": int(scf_occ_k),
        "scf_tau": float(scf_tau),
        "scf_gamma": float(SCF_GAMMA_DEFAULT),
        "scf_rows_total": int(scf_rows_total),
        "scf_rows_converged": int(scf_rows_converged),
        "scf_converged": bool(scf_converged_all),
        "scf_iters": int(scf_iters_max),
        "scf_residual_final": float(scf_residual_final_max),
        # SCF audit aggregates (P3.6).
        "scf_iters_mean": float(scf_iters_mean),
        "scf_iters_p95": float(scf_iters_p95),
        "scf_iters_max": int(scf_iters_max),
        "scf_converged_rate": float(scf_converged_rate),
        "residual_final_p95": float(scf_residual_final_p95),
        "residual_final_max": float(scf_residual_final_max),
        "deltaV_max_max": float(scf_delta_v_max_max),
        "deltaV_p95_max": float(scf_delta_v_p95_max),
        "scf_nontrivial_rate": float(scf_nontrivial_rate),
        "scf_audit_verdict": str(scf_audit_verdict),
        "scf_audit_reason": str(scf_audit_reason),
        "physics_entropy_beta": float(SPECTRAL_ENTROPY_BETA_DEFAULT),
        "dos_ldos_schema": str(DOS_LDOS_SCHEMA),
        "dos_grid_n": int(dos_grid_n),
        "dos_eta": float(dos_eta),
        "dos_energy_min": float(dos_energy_min),
        "dos_energy_max": float(dos_energy_max),
    }
    (out_dir / "summary_metadata.json").write_text(
        json.dumps(summary_metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    # SCF audit metrics table (P3.6): one row per molecule.
    scf_audit_metrics_csv = out_dir / "scf_audit_metrics.csv"
    scf_audit_metrics_csv.write_text("", encoding="utf-8")
    with scf_audit_metrics_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "mol_id",
                "scf_converged",
                "scf_iters",
                "residual_init",
                "residual_final",
                "deltaV_max",
                "deltaV_p95",
                "potential_gamma",
                "operator_mode",
            ],
        )
        w.writeheader()
        for r in scf_audit_metrics_rows:
            out_row = dict(r)
            # normalize bools to 0/1 and floats to stable strings
            out_row["scf_converged"] = "1" if bool(out_row.get("scf_converged", False)) else "0"
            for k in ["residual_init", "residual_final", "deltaV_max", "deltaV_p95", "potential_gamma"]:
                v = _to_float_or_nan(out_row.get(k, float("nan")))
                out_row[k] = "" if not math.isfinite(float(v)) else f"{float(v):.10g}"
            out_row["scf_iters"] = str(int(out_row.get("scf_iters", 0) or 0))
            w.writerow(out_row)

    scf_summary = {
        "schema_version": "hetero2_scf_summary.v1",
        "scf_schema": str(SCF_SCHEMA),
        "physics_mode": str(physics_mode),
        "edge_weight_mode": str(edge_weight_mode),
        "potential_mode": str(potential_mode),
        "potential_unit_model": str(POTENTIAL_UNIT_MODEL),
        "potential_scale_gamma": float(potential_scale_gamma),
        "scf_enabled": bool(scf_enabled),
        "scf_status": scf_status,
        "scf_max_iter": int(scf_max_iter),
        "scf_tol": float(scf_tol),
        "scf_damping": float(scf_damping),
        "scf_occ_k": int(scf_occ_k),
        "scf_tau": float(scf_tau),
        "scf_gamma": float(SCF_GAMMA_DEFAULT),
        "rows_total": int(scf_rows_total),
        "rows_converged": int(scf_rows_converged),
        "converged": bool(scf_converged_all),
        "iters_max": int(scf_iters_max),
        "residual_final_max": float(scf_residual_final_max),
        "audit_verdict": str(scf_audit_verdict),
        "audit_reason": str(scf_audit_reason),
        "triviality_flags": dict(scf_triviality_flags),
        "stats_all": dict(scf_stats_all),
        "stats_asym_fixture": dict(scf_stats_asym),
        "error_rows_missing_params": int(error_rows_missing_params),
        "error_rows_other": int(error_rows_other),
    }
    (out_dir / "scf_summary.json").write_text(
        json.dumps(scf_summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    if score_mode == "external_scores":
        missing_decoy_scores_path = out_dir / "missing_decoy_scores.csv"
        missing_rows_sorted = sorted(missing_decoy_hash_rows_affected.items(), key=lambda kv: (-kv[1], kv[0]))
        missing_decoy_scores_path.write_text("", encoding="utf-8")  # ensure file exists even if empty
        with missing_decoy_scores_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["decoy_hash", "decoy_smiles", "count_rows_affected"])
            w.writeheader()
            for h, count in missing_rows_sorted:
                w.writerow(
                    {
                        "decoy_hash": h,
                        "decoy_smiles": missing_decoy_hash_to_smiles.get(h, ""),
                        "count_rows_affected": int(count),
                    }
                )

        scores_coverage["unique_missing_decoy_hashes"] = len(missing_rows_sorted)
        scores_coverage["missing_decoy_hashes_top10"] = [
            {
                "decoy_hash": h,
                "decoy_smiles": missing_decoy_hash_to_smiles.get(h, ""),
                "count_rows_affected": int(count),
            }
            for h, count in missing_rows_sorted[:10]
        ]

    runtime_s = max(time.time() - t_start, 1e-9)
    metrics = {
        "counts": {
            "OK": sum(1 for r in summary_rows if r.get("status") == "OK"),
            "SKIP": sum(1 for r in summary_rows if r.get("status") == "SKIP"),
            "ERROR": sum(1 for r in summary_rows if r.get("status") == "ERROR"),
        },
        "top_reasons": sorted(
            (
                (reason, count)
                for reason, count in (
                    __import__("collections").Counter(r.get("reason", "") for r in summary_rows).items()
                )
                if reason
            ),
            key=lambda t: (-t[1], t[0]),
        ),
        "runtime_s_total": runtime_s,
        "throughput_rows_per_s": len(summary_rows) / runtime_s if summary_rows else 0.0,
        "config": {
            "workers": workers,
            "timeout_s": timeout_s,
            "resume": resume,
            "overwrite": overwrite,
            "maxtasksperchild": maxtasksperchild,
            "seed_strategy": seed_strategy,
            "seed": seed,
            "k_decoys": k_decoys,
            "decoy_hard_mode": bool(decoy_hard_mode),
            "decoy_hard_tanimoto_min": float(decoy_hard_tanimoto_min),
            "decoy_hard_tanimoto_max": float(decoy_hard_tanimoto_max),
            "physics_mode": str(physics_mode),
            "edge_weight_mode": str(edge_weight_mode),
            "potential_mode": str(potential_mode),
            "potential_unit_model": str(POTENTIAL_UNIT_MODEL),
            "potential_scale_gamma": float(potential_scale_gamma),
            "scf_max_iter": int(scf_max_iter),
            "scf_tol": float(scf_tol),
            "scf_damping": float(scf_damping),
            "scf_occ_k": int(scf_occ_k),
            "scf_tau": float(scf_tau),
            "scf_gamma": float(SCF_GAMMA_DEFAULT),
            "physics_entropy_beta": float(SPECTRAL_ENTROPY_BETA_DEFAULT),
            "guardrails_max_atoms": guardrails_max_atoms,
            "guardrails_require_connected": guardrails_require_connected,
            "score_mode": score_mode,
        },
    }
    if scf_rows_total > 0:
        metrics["scf"] = {
            "schema_version": str(SCF_SCHEMA),
            "rows_total": int(scf_rows_total),
            "rows_converged": int(scf_rows_converged),
            "converged": bool(scf_converged_all),
            "iters_max": int(scf_iters_max),
            "residual_final_max": float(scf_residual_final_max),
        }
    if score_mode == "external_scores":
        metrics["scores_coverage"] = dict(scores_coverage)
    metrics["decoy_realism"] = {
        "tanimoto_bins": {
            "easy_max_exclusive": float(TANIMOTO_EASY_MAX_EXCLUSIVE),
            "medium_max_exclusive": float(TANIMOTO_MEDIUM_MAX_EXCLUSIVE),
        },
        "tanimoto_median": float(median_tanimoto) if math.isfinite(float(median_tanimoto)) else float("nan"),
        "pairs_total_by_bin": {k: int(v) for k, v in pairs_total_by_bin.items()},
        "pairs_scored_by_bin": {k: int(v) for k, v in pairs_scored_by_bin.items()},
        "auc_tie_aware_by_bin": {k: float(v) for k, v in auc_by_bin.items()},
        "auc_interpretation_schema": AUC_INTERPRETATION_SCHEMA,
        "auc_interpretation": str(auc_label),
        "auc_interpretation_reason": str(auc_reason),
        "hard_pairs_min_n": int(AUC_HARD_MIN_PAIRS_DEFAULT),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    if not no_index:
        _write_index_md(out_dir, summary_rows, scores_coverage=scores_coverage if score_mode == "external_scores" else None)
    file_infos = _compute_file_infos(out_dir, skip_names={"manifest.json", "checksums.sha256", "evidence_pack.zip"})
    if not no_manifest:
        scores_prov = _scores_provenance(scores_input) if score_mode == "external_scores" else {}
        manifest_files = list(file_infos)
        manifest_files.append({"path": "./manifest.json", "size_bytes": None, "sha256": None})
        manifest_files.append({"path": "./metrics.json", "size_bytes": metrics_path.stat().st_size, "sha256": _sha256_of_file(metrics_path)})
        _write_manifest(
            out_dir,
            seed=seed,
            seed_strategy=seed_strategy,
            score_mode=score_mode,
            scores_provenance=scores_prov,
            guardrails_max_atoms=guardrails_max_atoms,
            guardrails_require_connected=guardrails_require_connected,
            potential_unit_model=str(POTENTIAL_UNIT_MODEL),
            potential_scale_gamma=float(potential_scale_gamma),
            files=manifest_files,
        )
    # recompute after manifest to include it in checksums
    file_infos_final = _compute_file_infos(out_dir, skip_names={"checksums.sha256", "evidence_pack.zip"})
    _write_checksums(out_dir, file_infos_final)
    if zip_pack:
        _write_zip_pack(out_dir)
    return summary_path
