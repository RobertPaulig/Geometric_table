from __future__ import annotations

import pandas as pd

# Pauling electronegativities for first-row d-block (approximate)
PAULING_D_BLOCK = {
    21: ("Sc", 1.36),
    22: ("Ti", 1.54),
    23: ("V", 1.63),
    24: ("Cr", 1.66),
    25: ("Mn", 1.55),
    26: ("Fe", 1.83),
    27: ("Co", 1.88),
    28: ("Ni", 1.91),
    29: ("Cu", 1.90),
    30: ("Zn", 1.65),
}

# Mapping between chi_spec and chi_Pauling from fit_chi_to_pauling
A = 0.5
B = 1.41


def pauling_to_chi_spec(chi_p: float) -> float:
    return (chi_p - B) / A


E_PORT_D_BLOCK = 6.0


def derive_DA_from_chi_spec(chi_spec: float, e_port: float):
    if e_port <= 0:
        e_port = 1e-6
    d_index = max(-chi_spec, 0.0) / e_port
    a_index = max(chi_spec, 0.0) / e_port
    return d_index, a_index


def classify_role(d_index: float, a_index: float) -> str:
    D = d_index
    A = a_index
    if D > 0.20 and A < 0.05:
        return "terminator"
    if D > 0.05 and A < 0.15:
        return "bridge"
    if A > 0.05:
        return "hub"
    return "inert"


def main(indices_path: str | None = None) -> None:
    if indices_path is None:
        indices_path = "data/element_indices_v4.csv"
    df = pd.read_csv(indices_path)

    rows = []
    for Z, (El, chi_p) in PAULING_D_BLOCK.items():
        chi_spec = pauling_to_chi_spec(chi_p)
        d_index, a_index = derive_DA_from_chi_spec(chi_spec, E_PORT_D_BLOCK)
        role = classify_role(d_index, a_index)
        rows.append(
            {
                "Z": Z,
                "El": El,
                "role": role,
                "period": 4,
                "chi_spec": chi_spec,
                "E_port": E_PORT_D_BLOCK,
                "D_index": d_index,
                "A_index": a_index,
            }
        )

    df_d = pd.DataFrame(rows)
    df_no_d = df[~df["Z"].isin(df_d["Z"])]
    df_all = pd.concat([df_no_d, df_d], ignore_index=True).sort_values("Z")

    out_name = "element_indices_with_dblock.csv"
    df_all.to_csv(out_name, index=False)

    print(f"Saved extended table with d-block to {out_name}\n")
    print("[D-BLOCK SUMMARY]")
    print(df_d[["Z", "El", "role", "chi_spec", "D_index", "A_index"]])


if __name__ == "__main__":
    main()

