import json
import subprocess
import sys
from pathlib import Path


def _edges_to_degree(n: int, edges: list[list[int]]) -> list[int]:
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def _is_tree(n: int, edges: list[list[int]]) -> bool:
    if len(edges) != n - 1:
        return False
    seen = set()
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)
        adj[u].append(v)
        adj[v].append(u)
    stack = [0]
    visited = {0}
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return len(visited) == n


def test_decoys_deterministic_and_constraints(tmp_path: Path) -> None:
    input_path = Path("tests/data/hetero_tree_min.json")
    out1 = tmp_path / "out1.json"
    out2 = tmp_path / "out2.json"
    base = [
        sys.executable,
        "-m",
        "analysis.chem.decoys",
        "--input",
        str(input_path),
        "--out",
    ]
    subprocess.run([*base, str(out1)], check=True)
    subprocess.run([*base, str(out2)], check=True)

    a = json.loads(out1.read_text(encoding="utf-8"))
    b = json.loads(out2.read_text(encoding="utf-8"))
    assert a == b

    assert a["k_generated"] == a["k_requested"]
    assert a["warnings"] == []

    n = a["n"]
    max_valence = json.loads(input_path.read_text(encoding="utf-8"))["max_valence"]
    orig_edges = json.loads(input_path.read_text(encoding="utf-8"))["edges"]
    orig_deg = _edges_to_degree(n, orig_edges)

    for item in a["decoys"]:
        edges = item["edges"]
        assert _is_tree(n, edges)
        deg = _edges_to_degree(n, edges)
        assert deg == orig_deg
        for i, d in enumerate(deg):
            t = a["constraints"]["preserve_types"]
            assert t is True
            node_type = json.loads(input_path.read_text(encoding="utf-8"))["node_types"][i]
            assert d <= max_valence[node_type]
