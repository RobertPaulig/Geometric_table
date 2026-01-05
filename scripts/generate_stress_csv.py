#!/usr/bin/env python3
import argparse
import random


def make_too_large_smiles(n_heavy: int = 210) -> str:
    # Linear alkane with >200 heavy atoms.
    return "C" * n_heavy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rnd = random.Random(args.seed)

    # Fixed mix similar to spec: 90% OK, 5% too_large, 2.5% disconnected,
    # 2.08% invalid, 0.42% missing.
    n = args.rows
    n_too_large = max(1, int(0.05 * n))
    n_disconnected = max(1, int(0.025 * n))
    n_invalid = max(1, int(0.0208 * n))
    n_missing = max(1, int(0.0042 * n))
    n_ok = n - (n_too_large + n_disconnected + n_invalid + n_missing)
    if n_ok < 0:
        raise ValueError("rows too small for the fixed mix; increase --rows")

    rows = []

    # OK: simple valid SMILES.
    ok_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCOC", "CC(C)O"]
    for i in range(n_ok):
        rows.append((f"ok_{i:05d}", rnd.choice(ok_smiles)))

    # too_large.
    big = make_too_large_smiles(210)
    for i in range(n_too_large):
        rows.append((f"too_large_{i:05d}", big))

    # disconnected: dot-separated fragments.
    for i in range(n_disconnected):
        rows.append((f"disconnected_{i:05d}", "CC.CC"))

    # invalid.
    bad_smiles = ["C1", "(", "notasmiles", "C(C", "C=)C"]
    for i in range(n_invalid):
        rows.append((f"invalid_{i:05d}", rnd.choice(bad_smiles)))

    # missing_smiles: empty.
    for i in range(n_missing):
        rows.append((f"missing_smiles_{i:05d}", ""))

    rnd.shuffle(rows)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("id,smiles\n")
        for row_id, smi in rows:
            f.write(f"{row_id},{smi}\n")


if __name__ == "__main__":
    main()
