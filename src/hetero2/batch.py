from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
import zlib
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import hetero1a
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


def _write_index_md(out_dir: Path, summary_rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Evidence Index",
        "",
        "| id | status | verdict | gate | slack | warnings | seed_used | report | assets | pipeline |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
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
) -> Dict[str, object]:
    """Isolated worker to avoid RDKit leaks across tasks."""
    if not smiles:
        return {
            "id": mol_id,
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
        "verdict",
        "gate",
        "slack",
        "margin",
        "n_decoys",
        "warnings_count",
        "report_path",
        "seed_used",
    ]
    summary_rows: List[Dict[str, object]] = []
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
        neg = res.get("neg", {}) if isinstance(res.get("neg"), dict) else {}
        summary_entry = {
            "id": mol_id,
            "status": status or "ERROR",
            "reason": reason,
            "verdict": neg.get("verdict", ""),
            "gate": neg.get("gate", ""),
            "slack": neg.get("slack", ""),
            "margin": neg.get("margin", ""),
            "n_decoys": res.get("n_decoys", ""),
            "warnings_count": len(set(warnings)) if isinstance(warnings, list) else 0,
            "report_path": str(rep_path) if status == "OK" and rep_path is not None else "",
            "seed_used": res.get("seed_used", ""),
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
            "guardrails_max_atoms": guardrails_max_atoms,
            "guardrails_require_connected": guardrails_require_connected,
            "score_mode": score_mode,
        },
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    if not no_index:
        _write_index_md(out_dir, summary_rows)
    file_infos = _compute_file_infos(out_dir, skip_names={"manifest.json", "checksums.sha256", "evidence_pack.zip"})
    if not no_manifest:
        manifest_files = list(file_infos)
        manifest_files.append({"path": "./manifest.json", "size_bytes": None, "sha256": None})
        manifest_files.append({"path": "./metrics.json", "size_bytes": metrics_path.stat().st_size, "sha256": _sha256_of_file(metrics_path)})
        _write_manifest(
            out_dir,
            seed=seed,
            seed_strategy=seed_strategy,
            score_mode=score_mode,
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
