from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _require_rdkit_draw():
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem, Draw  # type: ignore
        from rdkit import DataStructs  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("RDKit (with Draw) required for hetero2 report images. Install: pip install -e \".[dev,chem]\"") from exc
    return Chem, AllChem, Draw, DataStructs


def _tanimoto_scores(orig_smiles: str, decoys: Iterable[Dict[str, object]]) -> Dict[str, float]:
    Chem, AllChem, _, DataStructs = _require_rdkit_draw()
    orig = Chem.MolFromSmiles(orig_smiles)
    if orig is None:
        return {}
    fp_orig = AllChem.GetMorganFingerprintAsBitVect(orig, 2, nBits=2048)
    scores: Dict[str, float] = {}
    for d in decoys:
        s = str(d.get("smiles", ""))
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        scores[str(d.get("hash", s))] = float(DataStructs.FingerprintSimilarity(fp_orig, fp))
    return scores


def _draw_png(smiles: str, path: Path) -> None:
    _, AllChem, Draw, _ = _require_rdkit_draw()
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=(400, 300))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _decoy_table(decoys: List[Dict[str, object]], tanimoto: Dict[str, float], physchem_ref: Dict[str, float]) -> List[str]:
    keys = ["mw", "logp", "tpsa"]
    lines = [
        "",
        "Top hard negatives (by Tanimoto ascending)",
        "| idx | smiles | tanimoto | Δmw | Δlogp | Δtpsa | rings | aromatic_rings |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    def delta(val: float, ref: float) -> float:
        return float(val - ref)
    decoys_sorted = sorted(decoys, key=lambda d: tanimoto.get(str(d.get("hash", "")), math.inf))
    for idx, d in enumerate(decoys_sorted[:6], start=1):
        smi = str(d.get("smiles", ""))
        h = str(d.get("hash", smi))
        t = tanimoto.get(h, math.nan)
        phys = d.get("physchem", {}) or {}
        ring_info = d.get("ring_info", {}) or {}
        vals = [delta(float(phys.get(k, math.nan)), float(physchem_ref.get(k, math.nan))) for k in keys]
        lines.append(
            f"| {idx} | `{smi}` | {t:.3f} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {ring_info.get('n_rings','')} | {ring_info.get('n_aromatic_rings','')} |"
        )
    return lines


def render_report_v2(
    payload: Dict[str, object],
    *,
    out_path: str = "aspirin_report.md",
    assets_dir: str | Path | None = None,
) -> str:
    out_file = Path(out_path).resolve()
    stem = out_file.stem
    assets_dir_path = Path(assets_dir) if assets_dir is not None else out_file.with_name(f"{stem}_assets")
    decoys = payload.get("decoys", []) or []
    ring_info = payload.get("ring_info", {}) or {}
    physchem = payload.get("physchem", {}) or {}
    hardness = payload.get("hardness", {}) or {}
    warnings = payload.get("warnings", []) or []
    audit = payload.get("audit", {}) or {}
    neg = audit.get("neg_controls", {}) if isinstance(audit, dict) else {}
    run = payload.get("run", {}) or {}
    smiles = str(payload.get("smiles", ""))

    # Images
    try:
        _draw_png(smiles, assets_dir_path / "input.png")
        tanimoto = _tanimoto_scores(smiles, decoys)
        decoys_sorted = sorted(decoys, key=lambda d: tanimoto.get(str(d.get("hash", "")), math.inf))
        for idx, d in enumerate(decoys_sorted[:6], start=1):
            _draw_png(str(d.get("smiles", "")), assets_dir_path / f"decoy_{idx:02d}.png")
    except Exception:
        tanimoto = {}

    lines = [
        "HETERO-2 Report (hetero2_pipeline.v1)",
        "",
        "Summary",
        f"- Verdict: {neg.get('verdict', '')}",
        f"- Slack: {neg.get('slack', '')}",
        f"- Gate: {neg.get('gate', '')}",
        f"- Margin: {neg.get('margin', '')}",
        f"- Decoys generated: {len(decoys)}",
    ]
    if payload.get("decoy_stats"):
        stats = payload["decoy_stats"]
        if isinstance(stats, dict) and "sanitize_fail" in stats and "attempts" in stats:
            rate = 1.0 - (float(stats.get("sanitize_fail", 0)) / float(stats.get("attempts", 1)))
            lines.append(f"- Sanitize pass rate: {rate:.3f}")
    lines.extend(
        [
            "",
            "Rings",
            f"- n_rings: {ring_info.get('n_rings', '')}",
            f"- n_aromatic_rings: {ring_info.get('n_aromatic_rings', '')}",
            "",
            "PhysChem",
            f"- MW: {physchem.get('mw', '')}",
            f"- LogP: {physchem.get('logp', '')}",
            f"- TPSA: {physchem.get('tpsa', '')}",
            f"- HBD: {physchem.get('hbd', '')}",
            f"- HBA: {physchem.get('hba', '')}",
            f"- QED: {physchem.get('qed', '') if 'qed' in physchem else 'n/a'}",
            "",
            "Hardness",
            f"- Tanimoto Morgan min: {hardness.get('min', '')}",
            f"- Tanimoto Morgan median: {hardness.get('median', '')}",
        ]
    )

    lines.append("")
    lines.append("Warnings")
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- none")

    # Images section
    lines.extend(
        [
            "",
            "## Molecule",
            f"![input]({assets_dir_path.name}/input.png)",
            "",
            "## Example Hard Negatives",
        ]
    )
    for idx in range(1, 7):
        img = assets_dir_path / f"decoy_{idx:02d}.png"
        if img.exists():
            lines.append(f"![decoy_{idx:02d}]({assets_dir_path.name}/decoy_{idx:02d}.png)")

    # Tables
    if tanimoto:
        lines.extend(_decoy_table(decoys, tanimoto, physchem))

    lines.extend(
        [
            "",
            "Repro",
            f"- schema_version: {payload.get('schema_version', '')}",
            f"- seed: {run.get('seed', '')}",
            f"- timestamp: {run.get('timestamp', '')}",
            f"- cmd: {run.get('cmd', '')}",
        ]
    )
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out_file)
