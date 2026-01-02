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


def _generate_decoys(
    *,
    edges: Sequence[Tuple[int, int]],
    node_types: Sequence[str],
    k: int,
    seed: int,
    max_valence: Dict[str, int],
) -> Tuple[List[List[Tuple[int, int]]], List[str]]:
    n = len(node_types)
    orig_deg = _degree_sequence(n, edges)
    if not _degree_ok(orig_deg, node_types, max_valence):
        raise ValueError("input violates max_valence constraints")
    orig_hash = _edge_hash(edges)

    rng = random.Random(seed)
    decoys: List[List[Tuple[int, int]]] = []
    seen = {orig_hash}
    warnings: List[str] = []

    max_attempts = max(100, k * 50)
    attempts = 0
    while len(decoys) < k and attempts < max_attempts:
        attempts += 1
        seq = _prufer_from_degree(orig_deg, rng)
        new_edges = _tree_from_prufer(seq)
        new_hash = _edge_hash(new_edges)
        if new_hash in seen:
            continue
        new_deg = _degree_sequence(n, new_edges)
        if new_deg != orig_deg:
            continue
        if not _degree_ok(new_deg, node_types, max_valence):
            continue
        seen.add(new_hash)
        decoys.append(sorted((min(a, b), max(a, b)) for a, b in new_edges))

    if len(decoys) < k:
        warnings.append("not_enough_unique_decoys")
    return decoys, warnings


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    data = _load_input(payload, args=args)

    n = len(data.node_types)
    _validate_tree(n, data.edges)

    decoys, warnings = _generate_decoys(
        edges=data.edges,
        node_types=data.node_types,
        k=data.k,
        seed=data.seed,
        max_valence=data.max_valence,
    )

    out = {
        "schema_version": "hetero_decoys.v1",
        "mol_id": data.mol_id,
        "n": n,
        "k_requested": int(data.k),
        "k_generated": int(len(decoys)),
        "constraints": {"preserve_degree": True, "preserve_types": True},
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

