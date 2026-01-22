from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from hetero2.chemgraph import ChemGraph
from hetero2.physics_operator import POTENTIAL_SCALE_GAMMA_DEFAULT, compute_physics_features


class AccuracyA1Error(ValueError):
    pass


DEFAULT_INPUT_CSV = Path("data/accuracy/isomer_truth.v1.csv")
DEFAULT_OUT_DIR = Path("out_accuracy_a1_isomers")


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
    # Hash text content normalized to LF to reduce cross-platform EOL noise.
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
        raise AccuracyA1Error("missing CSV header")
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise AccuracyA1Error(f"missing required columns: {missing}")


def _rankdata(values: list[float]) -> list[float]:
    # Average ranks for ties (1-based).
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


def run_accuracy_a1_isomers(
    *,
    input_csv: Path,
    out_dir: Path,
    potential_scale_gamma: float,
) -> None:
    if not input_csv.exists():
        raise AccuracyA1Error(f"missing input_csv: {input_csv}")

    reader = csv.DictReader(input_csv.read_text(encoding="utf-8").splitlines())
    _require_columns(reader.fieldnames, required=["id", "group_id", "smiles", "energy_rel_kcalmol"])
    input_rows = [dict(r) for r in reader]
    if not input_rows:
        raise AccuracyA1Error("input_csv has no data rows")

    records: list[dict[str, object]] = []
    for row in input_rows:
        mid = str(row.get("id") or "").strip()
        gid = str(row.get("group_id") or "").strip()
        smiles = str(row.get("smiles") or "").strip()
        if not mid or not gid or not smiles:
            raise AccuracyA1Error(f"invalid row (id/group_id/smiles required): {row}")
        truth_rel = float(str(row.get("energy_rel_kcalmol") or "").strip())

        g = ChemGraph(smiles=smiles)
        adj = np.asarray(g.adjacency(), dtype=float)
        types = tuple(int(x) for x in g.heavy_atom_types())
        feats = compute_physics_features(
            adjacency=adj,
            bonds=None,
            types=types,
            physics_mode="hamiltonian",
            edge_weight_mode="unweighted",
            potential_scale_gamma=float(potential_scale_gamma),
        )
        pred_raw = float(feats.get("H_trace")) if feats.get("H_trace") != "" else float("nan")
        if not math.isfinite(pred_raw):
            raise AccuracyA1Error(f"non-finite pred_raw(H_trace) for id={mid}")

        records.append(
            {
                "id": mid,
                "group_id": gid,
                "smiles": smiles,
                "truth_rel_kcalmol": float(truth_rel),
                "pred_raw": float(pred_raw),
                "pred_rel": float("nan"),
                "n_heavy_atoms": int(g.n_heavy_atoms()),
            }
        )

    # Group and normalize (relative-to-min) inside each group_id.
    by_group: dict[str, list[dict[str, object]]] = {}
    for rec in records:
        by_group.setdefault(str(rec["group_id"]), []).append(rec)

    for gid, group in by_group.items():
        if len(group) < 2:
            raise AccuracyA1Error(f"group_id has <2 rows: {gid}")
        min_pred = min(float(r["pred_raw"]) for r in group)
        min_truth = min(float(r["truth_rel_kcalmol"]) for r in group)
        for r in group:
            r["pred_rel"] = float(float(r["pred_raw"]) - float(min_pred))
            r["truth_rel_kcalmol"] = float(float(r["truth_rel_kcalmol"]) - float(min_truth))

    # Compute per-group metrics.
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

    metrics: dict[str, object] = {
        "config": {
            "input_csv": input_csv.as_posix(),
            "input_sha256_normalized": _sha256_text_normalized(input_csv),
            "potential_scale_gamma": float(potential_scale_gamma),
            "physics_mode": "hamiltonian",
            "edge_weight_mode": "unweighted",
            "pred_proxy": "H_trace",
        },
        "dataset": {
            "rows_total": int(len(records)),
            "groups_total": int(len(by_group)),
        },
        "metrics": {
            "mean_spearman_pred_vs_truth": float(statistics.fmean(spearmans)) if spearmans else float("nan"),
            "pairwise_order_accuracy_overall": (float(pairwise_correct) / float(pairwise_total)) if pairwise_total else float("nan"),
            "top1_accuracy_mean": float(top1_hits) / float(len(by_group)) if by_group else float("nan"),
        },
        "per_group": per_group,
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write summary.csv (row-level), metrics.json, index.md.
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "group_id", "smiles", "n_heavy_atoms", "truth_rel_kcalmol", "pred_raw", "pred_rel"]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in sorted(records, key=lambda rr: (str(rr["group_id"]), str(rr["id"]))):
            w.writerow({k: r.get(k, "") for k in fieldnames})

    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    report = [
        "# ACCURACY-A1 (Isomers) — baseline operator score\n",
        "",
        f"- rows_total: {metrics['dataset']['rows_total']}",
        f"- groups_total: {metrics['dataset']['groups_total']}",
        f"- mean_spearman_pred_vs_truth: {metrics['metrics']['mean_spearman_pred_vs_truth']}",
        f"- pairwise_order_accuracy_overall: {metrics['metrics']['pairwise_order_accuracy_overall']}",
        f"- top1_accuracy_mean: {metrics['metrics']['top1_accuracy_mean']}",
        "",
        "Config:",
        "```json",
        json.dumps(metrics["config"], ensure_ascii=False, sort_keys=True, indent=2),
        "```",
        "",
    ]
    (out_dir / "index.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    # Pack (manifest + checksums + zip) per release checklist.
    config = dict(metrics["config"])
    file_infos = _compute_file_infos(out_dir, skip_names={"manifest.json", "checksums.sha256", "evidence_pack.zip"})
    manifest_files = list(file_infos)
    manifest_files.append({"path": "./manifest.json", "size_bytes": None, "sha256": None})
    _write_manifest(out_dir, config=config, files=manifest_files)
    file_infos_final = _compute_file_infos(out_dir, skip_names={"checksums.sha256", "evidence_pack.zip"})
    _write_checksums(out_dir, file_infos_final)
    _write_zip_pack(out_dir, zip_name="evidence_pack.zip")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ACCURACY-A1 isomers baseline (H_trace proxy) and build an evidence pack.")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV, help="Canonical isomer truth CSV.")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    p.add_argument("--potential_scale_gamma", type=float, default=float(POTENTIAL_SCALE_GAMMA_DEFAULT), help="Scale gamma for V (default: project default).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_accuracy_a1_isomers(input_csv=args.input_csv, out_dir=args.out_dir, potential_scale_gamma=float(args.potential_scale_gamma))
    print(f"wrote: {args.out_dir.as_posix()}")
    print(f"zip: {(args.out_dir / 'evidence_pack.zip').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
