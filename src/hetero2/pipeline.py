from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

from hetero1a.api import run_audit
from hetero2.chemgraph import ChemGraph
from hetero2.decoys_rewire import generate_rewire_decoys
from hetero2.guardrails import MAX_ATOMS_DEFAULT, preflight_smiles
from hetero2.spectral import compute_stability_metrics, laplacian_eigvals


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _mock_score_from_hash(hash_text: str) -> float:
    val = int(hash_text[:12], 16)
    return float(val) / float(16**12 - 1)


def _tanimoto_stats(smiles_orig: str, decoys: List[Dict[str, object]]) -> Dict[str, float]:
    Chem, AllChem = _require_rdkit_fps()
    orig = Chem.MolFromSmiles(smiles_orig)
    if orig is None:
        return {"min": float("nan"), "median": float("nan")}
    fp_orig = AllChem.GetMorganFingerprintAsBitVect(orig, 2, nBits=2048)
    sims: List[float] = []
    for d in decoys:
        smi = str(d.get("smiles", ""))
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sim = float(_require_rdkit_ds().FingerprintSimilarity(fp_orig, fp))
        sims.append(sim)
    if not sims:
        return {"min": float("nan"), "median": float("nan")}
    sims_sorted = sorted(sims)
    mid = len(sims_sorted) // 2
    if len(sims_sorted) % 2 == 0:
        median = (sims_sorted[mid - 1] + sims_sorted[mid]) / 2.0
    else:
        median = sims_sorted[mid]
    return {"min": float(min(sims_sorted)), "median": float(median)}


def _physchem_delta_mean(orig: Dict[str, float], decoys: List[Dict[str, object]]) -> Dict[str, float]:
    keys = [k for k in orig if isinstance(orig[k], (int, float))]
    acc = {k: 0.0 for k in keys}
    count = 0
    for d in decoys:
        phys = d.get("physchem", {}) or {}
        count += 1
        for k in keys:
            if k not in phys:
                continue
            acc[k] += abs(float(orig[k]) - float(phys[k]))
    if count == 0:
        return {k: float("nan") for k in keys}
    return {k: float(acc[k] / count) for k in keys}


def _require_rdkit_fps():
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("RDKit is required for hetero2 pipeline. Install: pip install -e \".[dev,chem]\"") from exc
    return Chem, AllChem


def _require_rdkit_ds():
    try:
        from rdkit import DataStructs  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("RDKit is required for hetero2 pipeline. Install: pip install -e \".[dev,chem]\"") from exc
    return DataStructs


def _skip_payload(smiles: str, *, warnings: List[str], seed: int, timestamp: str, reason: str) -> Dict[str, object]:
    ts = timestamp
    return {
        "schema_version": "hetero2_pipeline.v1",
        "smiles": smiles,
        "ring_info": {},
        "physchem": {},
        "decoys": [],
        "decoy_stats": {},
        "score_mode": "skip",
        "audit": {"neg_controls": {"verdict": "SKIP", "gate": "", "slack": "", "margin": ""}},
        "warnings": sorted(set(warnings)),
        "hardness": {},
        "physchem_delta_mean": {},
        "spectral": {},
        "run": {"seed": int(seed), "timestamp": ts, "cmd": ["hetero2.pipeline.v2"]},
        "skip": {"reason": reason},
    }


def run_pipeline_v2(
    smiles: str,
    *,
    k_decoys: int = 20,
    seed: int = 0,
    timestamp: str = "",
    max_attempts: int | None = None,
    score_mode: str = "mock",
    scores_input: str | None = None,
    guardrails_max_atoms: int = MAX_ATOMS_DEFAULT,
    guardrails_require_connected: bool = True,
) -> Dict[str, object]:
    ts = timestamp.strip() or _utc_now_iso()
    preflight = preflight_smiles(smiles, max_atoms=guardrails_max_atoms, require_connected=guardrails_require_connected)
    if not preflight.ok:
        return _skip_payload(preflight.canonical_smiles, warnings=preflight.warnings, seed=seed, timestamp=ts, reason=preflight.skip_reason or "guardrail")

    if score_mode == "external_scores" and not scores_input:
        warnings = list(preflight.warnings)
        warnings.append("skip:missing_scores_input")
        return _skip_payload(preflight.canonical_smiles, warnings=warnings, seed=seed, timestamp=ts, reason="missing_scores_input")

    cg = ChemGraph(preflight.canonical_smiles)
    eigvals = laplacian_eigvals(cg.laplacian())
    spectral_metrics = compute_stability_metrics(eigvals)
    decoys_result = generate_rewire_decoys(
        cg.canonical_smiles,
        k=k_decoys,
        seed=seed,
        max_attempts=max_attempts,
        lock_aromatic=True,
        allow_ring_bonds=False,
    )
    decoys = decoys_result.decoys
    if len(decoys) == 0:
        warnings = []
        warnings.extend(preflight.warnings)
        warnings.extend(decoys_result.warnings)
        warnings.append("skip:no_decoys_generated")
        warnings = sorted(set(warnings))
        return _skip_payload(
            cg.canonical_smiles,
            warnings=warnings,
            seed=seed,
            timestamp=ts,
            reason="no_decoys_generated",
        )

    scores: Dict[str, Dict[str, float]] = {}
    scores_input_id = ""
    effective_score_mode = "mock"
    if score_mode == "external_scores" and scores_input:
        data = json.loads(Path(scores_input).read_text(encoding="utf-8"))
        scores_input_id = Path(scores_input).name
        scores = {h: {"score": float(v.get("score", 0.0)), "weight": float(v.get("weight", 1.0))} for h, v in (data.get("decoys") or {}).items()}
        orig_score = float((data.get("original") or {}).get("score", 1.0))
        orig_weight = float((data.get("original") or {}).get("weight", 1.0))
        effective_score_mode = "external_scores"
    else:
        for d in decoys:
            h = str(d["hash"])
            scores[h] = {"score": _mock_score_from_hash(h), "weight": 1.0}
        orig_score = 1.0
        orig_weight = 1.0

    items = [{"label": 1, "score": orig_score, "weight": orig_weight}]
    missing_scores = 0
    for d in decoys:
        h = str(d["hash"])
        if h not in scores:
            missing_scores += 1
            continue
        entry = scores[h]
        items.append({"label": 0, "score": float(entry["score"]), "weight": float(entry["weight"])})

    audit_payload = {"dataset_id": f"hetero2:{cg.canonical_smiles}", "items": items}
    audit_result = run_audit(audit_payload, seed=int(seed), timestamp=ts, cmd_argv=["hetero2.pipeline.v2"])

    warnings = []
    warnings.extend(preflight.warnings)
    warnings.extend(decoys_result.warnings)
    warnings.extend(audit_result.get("warnings", []))
    if missing_scores > 0 and effective_score_mode == "external_scores":
        warnings.append(f"missing_scores_for_some_decoys:{missing_scores}")
    warnings = sorted(set(warnings))

    hardness = _tanimoto_stats(cg.canonical_smiles, decoys)
    physchem_delta = _physchem_delta_mean(cg.physchem(), decoys)

    return {
        "schema_version": "hetero2_pipeline.v1",
        "smiles": cg.canonical_smiles,
        "ring_info": cg.ring_info(),
        "physchem": cg.physchem(),
        "decoys": decoys,
        "decoy_stats": decoys_result.stats,
        "score_mode": effective_score_mode,
        "audit": audit_result,
        "warnings": warnings,
        "hardness": hardness,
        "physchem_delta_mean": physchem_delta,
        "spectral": spectral_metrics,
        "run": {"seed": int(seed), "timestamp": ts, "cmd": ["hetero2.pipeline.v2"]},
    }
