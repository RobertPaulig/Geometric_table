from __future__ import annotations

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
from hetero2.physics_operator import POTENTIAL_SCALE_GAMMA_DEFAULT, compute_operator_payload, compute_physics_features


class AccuracyA1SweepError(ValueError):
    pass


DEFAULT_INPUT_CSV = Path("data/accuracy/isomer_truth.v1.csv")
DEFAULT_OUT_DIR = Path("out_accuracy_a1_isomers_sweep")


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
        raise AccuracyA1SweepError("missing CSV header")
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise AccuracyA1SweepError(f"missing required columns: {missing}")


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


def _write_manifest(out_dir: Path, *, config: dict[str, object], files: list[dict[str, object]]) -> None:
    payload: dict[str, object] = {
        "git_sha": _detect_git_sha(),
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "config": dict(config),
        "files": sorted(list(files), key=lambda x: str(x.get("path", ""))),
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _compute_file_infos(out_dir: Path, *, skip_names: set[str] | None = None) -> list[dict[str, object]]:
    skip = skip_names or set()
    infos: list[dict[str, object]] = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_dir():
            continue
        if path.name in skip:
            continue
        rel = path.relative_to(out_dir).as_posix()
        infos.append({"path": f"./{rel}", "size_bytes": int(path.stat().st_size), "sha256": _sha256_file(path)})
    return infos


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


def _logsumexp(values: np.ndarray) -> float:
    if values.size == 0:
        return float("-inf")
    vmax = float(np.max(values))
    return float(vmax + math.log(float(np.sum(np.exp(values - vmax)))))


def _predict_from_eigvals(
    eigvals: np.ndarray,
    *,
    predictor: str,
    beta: float | None,
    eps: float,
    occ_k: int,
) -> float:
    name = str(predictor)
    vals = np.asarray(eigvals, dtype=float).reshape(-1)
    if vals.size == 0:
        raise AccuracyA1SweepError("empty eigvals")

    if name in {"free_energy_beta", "heat_trace_beta"}:
        if beta is None:
            raise AccuracyA1SweepError(f"predictor={name} requires beta")
        b = float(beta)
        if b <= 0.0:
            raise AccuracyA1SweepError("beta must be > 0")
        x = -b * vals
        lse = _logsumexp(x)
        if name == "free_energy_beta":
            return float((-1.0 / b) * lse)
        return float(math.exp(lse))

    if name == "logdet_shifted_eps":
        e = float(eps)
        if e <= 0.0:
            raise AccuracyA1SweepError("eps must be > 0")
        shifted = vals - float(np.min(vals))
        return float(np.sum(np.log(shifted + e)))

    if name == "occupied_sum_k":
        k = int(occ_k)
        if k <= 0:
            raise AccuracyA1SweepError("occ_k must be > 0")
        k_eff = min(k, int(vals.size))
        return float(np.sum(np.sort(vals)[:k_eff]))

    raise AccuracyA1SweepError(f"unknown predictor: {predictor}")


@dataclass(frozen=True)
class _Row:
    mid: str
    gid: str
    smiles: str
    truth_rel: float
    n_heavy_atoms: int
    adjacency: np.ndarray
    types: tuple[int, ...]
    bonds: tuple[tuple[int, int, float], ...]


def _load_rows(input_csv: Path) -> list[_Row]:
    reader = csv.DictReader(input_csv.read_text(encoding="utf-8").splitlines())
    _require_columns(reader.fieldnames, required=["id", "group_id", "smiles", "energy_rel_kcalmol"])
    raw_rows = [dict(r) for r in reader]
    if not raw_rows:
        raise AccuracyA1SweepError("input_csv has no data rows")

    rows: list[_Row] = []
    for r in raw_rows:
        mid = str(r.get("id") or "").strip()
        gid = str(r.get("group_id") or "").strip()
        smiles = str(r.get("smiles") or "").strip()
        if not mid or not gid or not smiles:
            raise AccuracyA1SweepError(f"invalid row (id/group_id/smiles required): {r}")
        truth_rel = float(str(r.get("energy_rel_kcalmol") or "").strip())

        g = ChemGraph(smiles=smiles)
        adj = np.asarray(g.adjacency(), dtype=float)
        types = tuple(int(x) for x in g.heavy_atom_types())
        bonds = tuple(tuple(x) for x in g.heavy_bonds_with_order())
        rows.append(
            _Row(
                mid=mid,
                gid=gid,
                smiles=smiles,
                truth_rel=float(truth_rel),
                n_heavy_atoms=int(g.n_heavy_atoms()),
                adjacency=adj,
                types=types,
                bonds=bonds,
            )
        )
    return rows


def _compute_grouped_records(
    rows: Sequence[_Row], *, pred_raw_by_id: dict[str, float]
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    records: list[dict[str, object]] = []
    for row in rows:
        pred_raw = float(pred_raw_by_id[row.mid])
        if not math.isfinite(pred_raw):
            raise AccuracyA1SweepError(f"non-finite pred_raw for id={row.mid}")
        records.append(
            {
                "id": row.mid,
                "group_id": row.gid,
                "smiles": row.smiles,
                "truth_rel_kcalmol": float(row.truth_rel),
                "pred_raw": float(pred_raw),
                "pred_rel": float("nan"),
                "n_heavy_atoms": int(row.n_heavy_atoms),
            }
        )

    by_group: dict[str, list[dict[str, object]]] = {}
    for rec in records:
        by_group.setdefault(str(rec["group_id"]), []).append(rec)

    for gid, group in by_group.items():
        if len(group) < 2:
            raise AccuracyA1SweepError(f"group_id has <2 rows: {gid}")
        min_pred = min(float(r["pred_raw"]) for r in group)
        min_truth = min(float(r["truth_rel_kcalmol"]) for r in group)
        for r in group:
            r["pred_rel"] = float(float(r["pred_raw"]) - float(min_pred))
            r["truth_rel_kcalmol"] = float(float(r["truth_rel_kcalmol"]) - float(min_truth))

    return records, by_group


def _compute_metrics_from_grouped(by_group: dict[str, list[dict[str, object]]]) -> tuple[dict[str, float], list[dict[str, object]]]:
    per_group: list[dict[str, object]] = []
    spearmans: list[float] = []
    top1_hits = 0
    pairwise_correct = 0
    pairwise_total = 0

    for gid, group in sorted(by_group.items(), key=lambda x: x[0]):
        group_sorted = sorted(group, key=lambda r: str(r.get("id", "")))
        truth = [float(r["truth_rel_kcalmol"]) for r in group_sorted]
        pred = [float(r["pred_rel"]) for r in group_sorted]
        sp = _spearman_corr(pred, truth)
        if math.isfinite(sp):
            spearmans.append(float(sp))
        c, t, acc = _pairwise_order_accuracy(truth, pred)
        pairwise_correct += int(c)
        pairwise_total += int(t)

        truth_min = min(truth)
        truth_best_ids = {str(r["id"]) for r, tr in zip(group_sorted, truth) if tr == truth_min}
        pred_best_id = str(min(group_sorted, key=lambda r: float(r["pred_rel"]))["id"])
        top1_ok = int(pred_best_id in truth_best_ids)
        top1_hits += int(top1_ok)

        per_group.append(
            {
                "group_id": gid,
                "n": int(len(group_sorted)),
                "spearman_pred_vs_truth": float(sp),
                "pairwise_order_accuracy": float(acc),
                "pairwise_correct": int(c),
                "pairwise_total": int(t),
                "truth_best_ids": sorted(truth_best_ids),
                "pred_best_id": pred_best_id,
                "top1_accuracy": float(top1_ok),
            }
        )

    metrics = {
        "mean_spearman_pred_vs_truth": float(statistics.fmean(spearmans)) if spearmans else float("nan"),
        "pairwise_order_accuracy_overall": (float(pairwise_correct) / float(pairwise_total)) if pairwise_total else float("nan"),
        "top1_accuracy_mean": float(top1_hits) / float(len(by_group)) if by_group else float("nan"),
    }
    return metrics, per_group


def _baseline_pred_raw(rows: Sequence[_Row], *, gamma: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        feats = compute_physics_features(
            adjacency=row.adjacency,
            bonds=None,
            types=row.types,
            physics_mode="hamiltonian",
            edge_weight_mode="unweighted",
            potential_scale_gamma=float(gamma),
        )
        pred = float(feats.get("H_trace")) if feats.get("H_trace") != "" else float("nan")
        out[row.mid] = float(pred)
    return out


def _sweep_pred_raw(
    rows: Sequence[_Row],
    *,
    edge_weight_mode: str,
    potential_mode: str,
    gamma: float,
    predictor: str,
    beta: float | None,
    eps: float,
    occ_k: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    ew_mode = str(edge_weight_mode)
    pot_mode = str(potential_mode)
    for row in rows:
        payload = compute_operator_payload(
            adjacency=row.adjacency,
            bonds=None if ew_mode == "unweighted" else row.bonds,
            types=row.types,
            physics_mode="hamiltonian",
            edge_weight_mode=ew_mode,
            potential_mode=pot_mode,
            potential_scale_gamma=float(gamma),
        )
        dos = payload.get("dos_ldos")
        if not isinstance(dos, dict):
            raise AccuracyA1SweepError("compute_operator_payload missing dos_ldos")
        if ew_mode == "unweighted":
            eig = dos.get("eigvals_H") or []
        else:
            eig = dos.get("eigvals_WH") or dos.get("eigvals_H") or []
        eigvals = np.asarray(eig, dtype=float)
        pred = _predict_from_eigvals(eigvals, predictor=predictor, beta=beta, eps=float(eps), occ_k=int(occ_k))
        out[row.mid] = float(pred)
    return out


def run_accuracy_a1_isomers_sweep(
    *,
    input_csv: Path,
    out_dir: Path,
    edge_weight_modes: list[str],
    potential_modes: list[str],
    gammas: list[float],
    predictors: list[str],
    betas: list[float],
    eps: float,
    occ_k: int,
    baseline_gamma: float,
    signal_gate_spearman: float,
    signal_gate_pairwise: float,
) -> None:
    if not input_csv.exists():
        raise AccuracyA1SweepError(f"missing input_csv: {input_csv}")

    rows = _load_rows(input_csv)

    dataset_info = {
        "rows_total": int(len(rows)),
        "groups_total": int(len({r.gid for r in rows})),
    }

    baseline_config: dict[str, object] = {
        "input_csv": input_csv.as_posix(),
        "input_sha256_normalized": _sha256_text_normalized(input_csv),
        "physics_mode": "hamiltonian",
        "edge_weight_mode": "unweighted",
        "potential_mode": "static",
        "potential_scale_gamma": float(baseline_gamma),
        "pred_proxy": "H_trace",
    }
    baseline_pred = _baseline_pred_raw(rows, gamma=float(baseline_gamma))
    baseline_records, baseline_grouped = _compute_grouped_records(rows, pred_raw_by_id=baseline_pred)
    baseline_metrics, baseline_per_group = _compute_metrics_from_grouped(baseline_grouped)

    sweep_rows: list[dict[str, object]] = []
    best_key: tuple[float, float, float] | None = None
    best_cfg: dict[str, object] | None = None
    best_records: list[dict[str, object]] | None = None
    best_grouped: dict[str, list[dict[str, object]]] | None = None
    best_metrics: dict[str, float] | None = None
    best_per_group: list[dict[str, object]] | None = None

    for predictor in predictors:
        pred_name = str(predictor)
        needs_beta = pred_name in {"free_energy_beta", "heat_trace_beta"}
        beta_values: list[float | None] = [None]
        if needs_beta:
            beta_values = [float(b) for b in betas]

        for edge_weight_mode in edge_weight_modes:
            for potential_mode in potential_modes:
                for gamma in gammas:
                    for beta in beta_values:
                        pred = _sweep_pred_raw(
                            rows,
                            edge_weight_mode=str(edge_weight_mode),
                            potential_mode=str(potential_mode),
                            gamma=float(gamma),
                            predictor=pred_name,
                            beta=float(beta) if beta is not None else None,
                            eps=float(eps),
                            occ_k=int(occ_k),
                        )
                        records, grouped = _compute_grouped_records(rows, pred_raw_by_id=pred)
                        m, pg = _compute_metrics_from_grouped(grouped)
                        mean_sp = float(m.get("mean_spearman_pred_vs_truth", float("nan")))
                        pair_acc = float(m.get("pairwise_order_accuracy_overall", float("nan")))
                        top1 = float(m.get("top1_accuracy_mean", float("nan")))

                        row_out: dict[str, object] = {
                            "predictor": pred_name,
                            "edge_weight_mode": str(edge_weight_mode),
                            "potential_mode": str(potential_mode),
                            "gamma": float(gamma),
                            "beta": "" if beta is None else float(beta),
                            "eps": float(eps),
                            "occ_k": int(occ_k),
                            "mean_spearman_pred_vs_truth": mean_sp,
                            "pairwise_order_accuracy_overall": pair_acc,
                            "top1_accuracy_mean": top1,
                        }
                        sweep_rows.append(row_out)

                        score_sp = mean_sp if math.isfinite(mean_sp) else float("-inf")
                        score_pair = pair_acc if math.isfinite(pair_acc) else float("-inf")
                        score_top1 = top1 if math.isfinite(top1) else float("-inf")
                        key = (score_sp, score_pair, score_top1)
                        if best_key is None or key > best_key:
                            best_key = key
                            best_cfg = {
                                "predictor": pred_name,
                                "edge_weight_mode": str(edge_weight_mode),
                                "potential_mode": str(potential_mode),
                                "potential_scale_gamma": float(gamma),
                                "beta": None if beta is None else float(beta),
                                "eps": float(eps),
                                "occ_k": int(occ_k),
                            }
                            best_records = records
                            best_grouped = grouped
                            best_metrics = m
                            best_per_group = pg

    if best_cfg is None or best_records is None or best_grouped is None or best_metrics is None or best_per_group is None:
        raise AccuracyA1SweepError("no sweep configs produced a best result")

    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_results_path = out_dir / "sweep_results.csv"
    with sweep_results_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "predictor",
            "edge_weight_mode",
            "potential_mode",
            "gamma",
            "beta",
            "eps",
            "occ_k",
            "mean_spearman_pred_vs_truth",
            "pairwise_order_accuracy_overall",
            "top1_accuracy_mean",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in sweep_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    (out_dir / "best_config.json").write_text(json.dumps(best_cfg, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "group_id", "smiles", "n_heavy_atoms", "truth_rel_kcalmol", "pred_raw", "pred_rel"]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in sorted(best_records, key=lambda rr: (str(rr["group_id"]), str(rr["id"]))):
            w.writerow({k: r.get(k, "") for k in fieldnames})

    verdict = "NO_SIGNAL_YET"
    reason = ""
    best_sp = float(best_metrics.get("mean_spearman_pred_vs_truth", float("nan")))
    best_pair = float(best_metrics.get("pairwise_order_accuracy_overall", float("nan")))
    if math.isfinite(best_sp) and best_sp >= float(signal_gate_spearman):
        verdict = "SIGNAL_OK"
        reason = f"mean_spearman_pred_vs_truth >= {signal_gate_spearman}"
    elif math.isfinite(best_pair) and best_pair >= float(signal_gate_pairwise):
        verdict = "SIGNAL_OK"
        reason = f"pairwise_order_accuracy_overall >= {signal_gate_pairwise}"

    metrics_payload: dict[str, object] = {
        "config": {
            "sweep": {
                "edge_weight_modes": list(edge_weight_modes),
                "potential_modes": list(potential_modes),
                "gammas": [float(x) for x in gammas],
                "predictors": list(predictors),
                "betas": [float(x) for x in betas],
                "eps": float(eps),
                "occ_k": int(occ_k),
            },
            "signal_gates": {
                "mean_spearman_min": float(signal_gate_spearman),
                "pairwise_order_accuracy_min": float(signal_gate_pairwise),
            },
        },
        "dataset": dataset_info,
        "baseline": {"config": baseline_config, "metrics": baseline_metrics, "per_group": baseline_per_group},
        "best": {"config": best_cfg, "metrics": best_metrics, "per_group": best_per_group},
        "verdict": {"code": verdict, "reason": reason},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    index_lines = [
        "# ACCURACY-A1.2 (Isomers) signal repair sweep",
        "",
        f"- verdict: {verdict}" + (f" ({reason})" if reason else ""),
        "",
        "Baseline (A1.1; H_trace):",
        f"- mean_spearman_pred_vs_truth: {baseline_metrics.get('mean_spearman_pred_vs_truth')}",
        f"- pairwise_order_accuracy_overall: {baseline_metrics.get('pairwise_order_accuracy_overall')}",
        f"- top1_accuracy_mean: {baseline_metrics.get('top1_accuracy_mean')}",
        "",
        "Best (from sweep):",
        f"- mean_spearman_pred_vs_truth: {best_metrics.get('mean_spearman_pred_vs_truth')}",
        f"- pairwise_order_accuracy_overall: {best_metrics.get('pairwise_order_accuracy_overall')}",
        f"- top1_accuracy_mean: {best_metrics.get('top1_accuracy_mean')}",
        "",
        "Best config:",
        "```json",
        json.dumps(best_cfg, ensure_ascii=False, sort_keys=True, indent=2),
        "```",
        "",
    ]
    (out_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

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
    ]
    for rel_dst, src in extra_files:
        try:
            if not src.exists():
                continue
            dst = out_dir / rel_dst
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        except Exception as exc:
            raise AccuracyA1SweepError(f"failed to copy provenance file {src} -> {rel_dst}") from exc

    config_for_manifest = {
        "input_csv": input_csv.as_posix(),
        "input_sha256_normalized": _sha256_text_normalized(input_csv),
        "baseline_config": baseline_config,
        "best_config": best_cfg,
        "verdict": {"code": verdict, "reason": reason},
    }
    file_infos = _compute_file_infos(out_dir, skip_names={"manifest.json", "checksums.sha256", "evidence_pack.zip"})
    manifest_files = list(file_infos)
    manifest_files.append({"path": "./manifest.json", "size_bytes": None, "sha256": None})
    _write_manifest(out_dir, config=config_for_manifest, files=manifest_files)
    file_infos_final = _compute_file_infos(out_dir, skip_names={"checksums.sha256", "evidence_pack.zip"})
    _write_checksums(out_dir, file_infos_final)
    _write_zip_pack(out_dir, zip_name="evidence_pack.zip")


def _parse_list_str(value: str) -> list[str]:
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _parse_list_float(values: Sequence[str]) -> list[float]:
    out: list[float] = []
    for v in values:
        if not str(v).strip():
            continue
        out.append(float(v))
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ACCURACY-A1.2 isomers sweep (signal repair) and build an evidence pack.")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV, help="Canonical isomer truth CSV.")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    p.add_argument(
        "--edge_weight_modes",
        type=_parse_list_str,
        default="unweighted,bond_order,bond_order_delta_chi",
        help="Comma-separated list: unweighted,bond_order,bond_order_delta_chi",
    )
    p.add_argument(
        "--potential_modes",
        type=_parse_list_str,
        default="static,self_consistent",
        help="Comma-separated list: static,self_consistent",
    )
    p.add_argument(
        "--gammas",
        type=str,
        default="0.0,0.25,0.5,1.0",
        help="Comma-separated list of gamma values.",
    )
    p.add_argument(
        "--predictors",
        type=_parse_list_str,
        default="free_energy_beta,heat_trace_beta,logdet_shifted_eps",
        help="Comma-separated list: free_energy_beta,heat_trace_beta,logdet_shifted_eps,occupied_sum_k",
    )
    p.add_argument(
        "--betas",
        type=str,
        default="0.5,1.0,2.0",
        help="Comma-separated list of beta values (used by *_beta predictors).",
    )
    p.add_argument("--eps", type=float, default=1e-6, help="Epsilon for logdet_shifted_eps.")
    p.add_argument("--occ_k", type=int, default=5, help="k for occupied_sum_k predictor.")
    p.add_argument("--baseline_gamma", type=float, default=float(POTENTIAL_SCALE_GAMMA_DEFAULT), help="Gamma for baseline H_trace (A1.1).")
    p.add_argument("--signal_gate_spearman", type=float, default=0.20, help="Signal gate threshold for mean Spearman.")
    p.add_argument("--signal_gate_pairwise", type=float, default=0.60, help="Signal gate threshold for pairwise order accuracy.")
    args = p.parse_args(argv)
    args.gammas = _parse_list_float(_parse_list_str(args.gammas))
    args.betas = _parse_list_float(_parse_list_str(args.betas))
    args.edge_weight_modes = list(args.edge_weight_modes)
    args.potential_modes = list(args.potential_modes)
    args.predictors = list(args.predictors)
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_accuracy_a1_isomers_sweep(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        edge_weight_modes=list(args.edge_weight_modes),
        potential_modes=list(args.potential_modes),
        gammas=list(args.gammas),
        predictors=list(args.predictors),
        betas=list(args.betas),
        eps=float(args.eps),
        occ_k=int(args.occ_k),
        baseline_gamma=float(args.baseline_gamma),
        signal_gate_spearman=float(args.signal_gate_spearman),
        signal_gate_pairwise=float(args.signal_gate_pairwise),
    )
    print(f"wrote: {args.out_dir.as_posix()}")
    print(f"zip: {(args.out_dir / 'evidence_pack.zip').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

