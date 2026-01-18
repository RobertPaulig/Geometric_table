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
from hetero2.decoy_realism import (
    AUC_INTERPRETATION_SCHEMA,
    AUC_HARD_MIN_PAIRS_DEFAULT,
    TANIMOTO_EASY_MAX_EXCLUSIVE,
    TANIMOTO_MEDIUM_MAX_EXCLUSIVE,
    auc_pair_contribution,
    interpret_auc,
    tanimoto_bin,
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
    operator_mode: str,
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
            operator_mode=str(operator_mode),
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
    operator_mode: str = "laplacian",
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
    fieldnames: Sequence[str] = [
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
    ]
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
    # Operator foundation: per-row energies (laplacian + optional H).
    operator_rows: List[Dict[str, object]] = []
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
                pipe_path = out_dir / f"{mol_id}.pipeline.json"
                pipe_path.write_text(
                    json.dumps(pipeline, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
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

        # Collect operator energies and decoy-realism stats for OK rows.
        if isinstance(pipeline, dict) and status == "OK":
            op = pipeline.get("operator", {}) if isinstance(pipeline.get("operator"), dict) else {}
            operator_rows.append(
                {
                    "id": mol_id,
                    "operator_mode": str(op.get("mode", "")),
                    "laplacian_energy": op.get("laplacian_energy", ""),
                    "h_operator_energy": op.get("h_operator_energy", ""),
                }
            )

            decoys = pipeline.get("decoys", [])
            smiles_orig = str(pipeline.get("smiles", ""))
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
                if isinstance(scores_payload, dict):
                    orig_score = float((scores_payload.get("original") or {}).get("score", 1.0))

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
                            continue
                        neg_score = float(entry.get("score", 0.0))
                        weight = float(entry.get("weight", 1.0))
                    else:
                        neg_score = float(_mock_score_from_hash(decoy_hash))
                        weight = 1.0

                    pairs_scored_by_bin[bin_id] += 1
                    auc_numer_by_bin[bin_id] += float(weight) * float(auc_pair_contribution(orig_score, neg_score))
                    auc_denom_by_bin[bin_id] += float(weight)
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
        }
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
                operator_mode=str(operator_mode),
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
                        operator_mode=str(operator_mode),
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

    # Write operator energies table (always present; H operator values may be NaN depending on operator_mode).
    operator_features_path = out_dir / "operator_features.csv"
    operator_features_path.write_text("", encoding="utf-8")
    with operator_features_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "operator_mode", "laplacian_energy", "h_operator_energy"])
        w.writeheader()
        for row in operator_rows:
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
        auc_hard=auc_by_bin.get("hard"),
        hard_pairs=hard_pairs,
        hard_pairs_min=int(AUC_HARD_MIN_PAIRS_DEFAULT),
    )

    hardness_curve_csv.write_text("", encoding="utf-8")
    with hardness_curve_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "bin",
                "tanimoto_min_inclusive",
                "tanimoto_max_exclusive",
                "pairs_total",
                "pairs_scored",
                "auc_tie_aware",
            ],
        )
        w.writeheader()
        for bin_id, lo, hi in bins:
            hi_str = "" if bin_id == "hard" else ("" if not math.isfinite(hi) else f"{hi:.2f}")
            lo_str = "" if not math.isfinite(lo) else f"{lo:.2f}"
            w.writerow(
                {
                    "bin": bin_id,
                    "tanimoto_min_inclusive": lo_str,
                    "tanimoto_max_exclusive": hi_str,
                    "pairs_total": int(pairs_total_by_bin.get(bin_id, 0)),
                    "pairs_scored": int(pairs_scored_by_bin.get(bin_id, 0)),
                    "auc_tie_aware": "" if not math.isfinite(float(auc_by_bin.get(bin_id, float("nan")))) else f"{auc_by_bin[bin_id]:.6f}",
                }
            )

    hardness_lines = [
        "# Decoy Hardness Curve (AUC vs Tanimoto)",
        "",
        f"- tanimoto_median: {median_tanimoto if math.isfinite(float(median_tanimoto)) else 'nan'}",
        f"- auc_interpretation_schema: {AUC_INTERPRETATION_SCHEMA}",
        f"- auc_interpretation: {auc_label}",
        f"- auc_interpretation_reason: {auc_reason}",
        "",
        "## Bins",
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
            "operator_mode": str(operator_mode),
            "guardrails_max_atoms": guardrails_max_atoms,
            "guardrails_require_connected": guardrails_require_connected,
            "score_mode": score_mode,
        },
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
            files=manifest_files,
        )
    # recompute after manifest to include it in checksums
    file_infos_final = _compute_file_infos(out_dir, skip_names={"checksums.sha256", "evidence_pack.zip"})
    _write_checksums(out_dir, file_infos_final)
    if zip_pack:
        _write_zip_pack(out_dir)
    return summary_path
