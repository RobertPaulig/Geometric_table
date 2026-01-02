from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class _Input:
    mol_id: str
    node_types: List[str]
    edges: List[Tuple[int, int]]
    k: int
    seed: int
    timestamp: str
    max_valence: Dict[str, int]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalized_cmd(argv: Sequence[str]) -> List[str]:
    # Keep a stable command string for audits/tests: drop `--out <path>` and `--input <path>`.
    out: List[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in {"--out", "--input"}:
            skip_next = True
            continue
        out.append(token)
    return out


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HETERO-1A decoys generator (tree graph, degree-preserving).")
    ap.add_argument("--input", required=True, help="Path to JSON input.")
    ap.add_argument("--out", default="", help="Output JSON path (default: stdout).")
    ap.add_argument("--seed", type=int, default=None, help="Override seed from input JSON.")
    ap.add_argument("--timestamp", default=None, help="Override timestamp from input JSON (ISO).")
    ap.add_argument("--min_dist_to_original", type=float, default=0.0, help="Minimum distance to original tree.")
    ap.add_argument("--min_pair_dist", type=float, default=0.0, help="Minimum pairwise distance between decoys.")
    ap.add_argument("--max_attempts", type=int, default=None, help="Maximum attempts to generate decoys.")
    return ap.parse_args(list(argv) if argv is not None else None)


def _load_input(payload: Dict[str, object], *, args: argparse.Namespace) -> _Input:
    mol_id = str(payload.get("mol_id", "")).strip()
    if not mol_id:
        raise ValueError("input JSON must contain non-empty 'mol_id'")

    node_types = payload.get("node_types", None)
    if not isinstance(node_types, list) or not node_types:
        raise ValueError("input JSON must contain non-empty 'node_types'")
    node_types = [str(x) for x in node_types]

    edges_raw = payload.get("edges", None)
    if not isinstance(edges_raw, list):
        raise ValueError("input JSON must contain list 'edges'")
    edges: List[Tuple[int, int]] = []
    for row in edges_raw:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("each edge must be [u,v]")
        u, v = int(row[0]), int(row[1])
        edges.append((u, v))

    k = int(payload.get("k", 0))
    if k <= 0:
        raise ValueError("input JSON must contain positive 'k'")

    seed = int(payload.get("seed", 0))
    if args.seed is not None:
        seed = int(args.seed)

    timestamp = str(payload.get("timestamp", "")).strip()
    if args.timestamp is not None:
        timestamp = str(args.timestamp).strip()
    if not timestamp:
        timestamp = _utc_now_iso()

    max_valence_raw = payload.get("max_valence", None)
    if not isinstance(max_valence_raw, dict) or not max_valence_raw:
        raise ValueError("input JSON must contain non-empty 'max_valence'")
    max_valence = {str(k): int(v) for k, v in max_valence_raw.items()}

    return _Input(
        mol_id=mol_id,
        node_types=node_types,
        edges=edges,
        k=k,
        seed=seed,
        timestamp=timestamp,
        max_valence=max_valence,
    )


def _edge_hash(edges: Sequence[Tuple[int, int]]) -> str:
    pairs = sorted((min(a, b), max(a, b)) for a, b in edges)
    text = ";".join(f"{a}-{b}" for a, b in pairs)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _degree_sequence(n: int, edges: Sequence[Tuple[int, int]]) -> List[int]:
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def _validate_tree(n: int, edges: Sequence[Tuple[int, int]]) -> None:
    if n <= 1:
        raise ValueError("n must be >= 2")
    if len(edges) != n - 1:
        raise ValueError("edges must form a tree (n-1 edges)")
    seen = set()
    for u, v in edges:
        if u == v:
            raise ValueError("self-loop in edges")
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("edge index out of range")
        key = (min(u, v), max(u, v))
        if key in seen:
            raise ValueError("duplicate edge")
        seen.add(key)
    # Connectivity check via BFS
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    stack = [0]
    seen_nodes = {0}
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in seen_nodes:
                seen_nodes.add(nxt)
                stack.append(nxt)
    if len(seen_nodes) != n:
        raise ValueError("edges are not connected")


def _prufer_from_degree(deg: Sequence[int], rng: random.Random) -> List[int]:
    seq: List[int] = []
    for i, d in enumerate(deg):
        if d < 1:
            raise ValueError("degree must be >=1")
        seq.extend([i] * (d - 1))
    rng.shuffle(seq)
    return seq


def _tree_from_prufer(seq: Sequence[int]) -> List[Tuple[int, int]]:
    n = len(seq) + 2
    degree = [1] * n
    for x in seq:
        degree[x] += 1

    import heapq

    leaves = [i for i, d in enumerate(degree) if d == 1]
    heapq.heapify(leaves)
    edges: List[Tuple[int, int]] = []
    for x in seq:
        leaf = heapq.heappop(leaves)
        edges.append((leaf, x))
        degree[leaf] -= 1
        degree[x] -= 1
        if degree[x] == 1:
            heapq.heappush(leaves, x)
    a = heapq.heappop(leaves)
    b = heapq.heappop(leaves)
    edges.append((a, b))
    return edges


def _degree_ok(deg: Sequence[int], node_types: Sequence[str], max_valence: Dict[str, int]) -> bool:
    for i, d in enumerate(deg):
        t = node_types[i]
        if t not in max_valence:
            return False
        if d > int(max_valence[t]):
            return False
    return True


def _edges_set(edges: Sequence[Tuple[int, int]]) -> set[Tuple[int, int]]:
    return set((min(a, b), max(a, b)) for a, b in edges)


def _edge_dist(a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]], n: int) -> float:
    # Normalized symmetric difference for trees with same n.
    ea = _edges_set(a)
    eb = _edges_set(b)
    diff = ea.symmetric_difference(eb)
    return float(len(diff)) / float(2 * (n - 1))


def _generate_decoys(
    *,
    edges: Sequence[Tuple[int, int]],
    node_types: Sequence[str],
    k: int,
    seed: int,
    max_valence: Dict[str, int],
    min_dist_to_original: float,
    min_pair_dist: float,
    max_attempts: int | None,
) -> Tuple[List[List[Tuple[int, int]]], List[str], Dict[str, int], List[float]]:
    n = len(node_types)
    orig_deg = _degree_sequence(n, edges)
    if not _degree_ok(orig_deg, node_types, max_valence):
        raise ValueError("input violates max_valence constraints")
    orig_hash = _edge_hash(edges)
    orig_edges_norm = sorted((min(a, b), max(a, b)) for a, b in edges)

    rng = random.Random(seed)
    decoys: List[List[Tuple[int, int]]] = []
    seen = {orig_hash}
    warnings: List[str] = []
    attempts = 0
    rejected_duplicate = 0
    rejected_too_close_to_original = 0
    rejected_too_close_to_existing = 0
    dists_to_original: List[float] = []

    max_attempts_eff = max_attempts if max_attempts is not None else max(100, k * 200)
    while len(decoys) < k and attempts < max_attempts_eff:
        attempts += 1
        seq = _prufer_from_degree(orig_deg, rng)
        new_edges = _tree_from_prufer(seq)
        new_hash = _edge_hash(new_edges)
        if new_hash in seen:
            rejected_duplicate += 1
            continue
        new_deg = _degree_sequence(n, new_edges)
        if new_deg != orig_deg:
            continue
        if not _degree_ok(new_deg, node_types, max_valence):
            continue
        new_edges_norm = sorted((min(a, b), max(a, b)) for a, b in new_edges)
        dist_to_orig = _edge_dist(new_edges_norm, orig_edges_norm, n)
        if dist_to_orig < min_dist_to_original:
            rejected_too_close_to_original += 1
            continue
        if min_pair_dist > 0.0:
            too_close = False
            for existing in decoys:
                if _edge_dist(new_edges_norm, existing, n) < min_pair_dist:
                    too_close = True
                    break
            if too_close:
                rejected_too_close_to_existing += 1
                continue
        seen.add(new_hash)
        decoys.append(new_edges_norm)
        dists_to_original.append(dist_to_orig)

    if len(decoys) < k:
        warnings.append("not_enough_unique_decoys")
    if len(decoys) < k and (min_dist_to_original > 0.0 or min_pair_dist > 0.0):
        warnings.append("could_not_generate_k_decoys_under_constraints")

    stats = {
        "attempts": attempts,
        "rejected_duplicate": rejected_duplicate,
        "rejected_too_close_to_original": rejected_too_close_to_original,
        "rejected_too_close_to_existing": rejected_too_close_to_existing,
    }
    return decoys, warnings, stats, dists_to_original


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    data = _load_input(payload, args=args)

    n = len(data.node_types)
    _validate_tree(n, data.edges)

    decoys, warnings, stats, dists_to_original = _generate_decoys(
        edges=data.edges,
        node_types=data.node_types,
        k=data.k,
        seed=data.seed,
        max_valence=data.max_valence,
        min_dist_to_original=float(args.min_dist_to_original),
        min_pair_dist=float(args.min_pair_dist),
        max_attempts=int(args.max_attempts) if args.max_attempts is not None else None,
    )

    pairwise_dists: List[float] = []
    for i in range(len(decoys)):
        for j in range(i + 1, len(decoys)):
            pairwise_dists.append(_edge_dist(decoys[i], decoys[j], n))

    def _summary(values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}
        return {
            "min": float(min(values)),
            "mean": float(sum(values) / len(values)),
            "max": float(max(values)),
        }

    overlap_vals: List[float] = []
    orig_edges_norm = sorted((min(a, b), max(a, b)) for a, b in data.edges)
    orig_set = _edges_set(orig_edges_norm)
    for edges in decoys:
        overlap = len(orig_set.intersection(_edges_set(edges)))
        overlap_vals.append(float(overlap) / float(n - 1))

    out = {
        "schema_version": "hetero_decoys.v1",
        "mol_id": data.mol_id,
        "n": n,
        "k_requested": int(data.k),
        "k_generated": int(len(decoys)),
        "constraints": {"preserve_degree": True, "preserve_types": True},
        "metrics": {
            "dist_to_original": _summary(dists_to_original),
            "pairwise_dist": {
                "min": _summary(pairwise_dists)["min"],
                "mean": _summary(pairwise_dists)["mean"],
            },
            "edge_overlap_to_original_mean": float(sum(overlap_vals) / len(overlap_vals)) if overlap_vals else float("nan"),
        },
        "filter": {
            "min_dist_to_original": float(args.min_dist_to_original),
            "min_pair_dist": float(args.min_pair_dist),
            "attempts": int(stats["attempts"]),
            "rejected_too_close_to_original": int(stats["rejected_too_close_to_original"]),
            "rejected_too_close_to_existing": int(stats["rejected_too_close_to_existing"]),
            "rejected_duplicate": int(stats["rejected_duplicate"]),
        },
        "decoys": [{"edges": edges, "hash": _edge_hash(edges)} for edges in decoys],
        "warnings": warnings,
        "run": {
            "seed": int(data.seed),
            "timestamp": data.timestamp,
            "cmd": _normalized_cmd(sys.argv),
        },
    }

    text = json.dumps(out, ensure_ascii=False, sort_keys=True, indent=2) + os.linesep
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
