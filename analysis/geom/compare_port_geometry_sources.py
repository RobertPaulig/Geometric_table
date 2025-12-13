from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import argparse

import matplotlib.pyplot as plt
import numpy as np

from core.geom_atoms import PERIODIC_TABLE
from core.port_geometry_spectral import canonical_port_vectors, ws_sp_gap, hybrid_strength
from core.thermo_config import get_current_thermo_config, override_thermo_config
from core.spectral_density_ws import WSRadialParams
from analysis.thermo_cli import add_thermo_args, apply_thermo_from_args


def compare_for_elements(names):
    cfg = get_current_thermo_config()

    rows = []
    matches = 0
    for name in names:
        atom = PERIODIC_TABLE.get(name)
        if atom is None:
            continue
        # legacy-ярлык из справочника
        base_label = atom.port_geometry
        # inferred при текущем ThermoConfig
        inferred_label = atom.effective_port_geometry(cfg)
        vecs = atom.port_vectors(cfg)
        if inferred_label == base_label:
            matches += 1

        # спектральные признаки для отладки (WSGeom-профиль из ThermoConfig)
        params = WSRadialParams(
            R_max=cfg.ws_geom_R_max,
            R_well=cfg.ws_geom_R_well,
            V0=cfg.ws_geom_V0,
            N_grid=cfg.ws_geom_N_grid,
            ell=0,
            state_index=0,
        )
        gap = ws_sp_gap(atom.Z, params)
        h = hybrid_strength(gap, cfg.ws_geom_gap_ref, cfg.ws_geom_gap_scale)

        rows.append((name, base_label, inferred_label, vecs, gap, h))

    for name, base_label, inferred_label, vecs, gap, h in rows:
        print(
            f"{name:>2}: base={base_label:>10} -> inferred={inferred_label:>10}, "
            f"ports={vecs.shape[0]}, gap={gap:.4f}, h={h:.4f}"
        )

    print(f"Matches with legacy (B,C,N,O,Si,P,S): {matches}/{len(rows)}")

    # Простейший сферический scatter для одного элемента (например, C)
    target = "C"
    row = next((r for r in rows if r[0] == target), None)
    if row is None:
        return
    _, _, inferred_label, vecs, _, _ = row
    if vecs.size == 0:
        return

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    xs, ys, zs = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    ax.scatter(xs, ys, zs, s=60)
    for i in range(vecs.shape[0]):
        ax.plot([0, xs[i]], [0, ys[i]], [0, zs[i]], "k-", alpha=0.5)
    ax.set_title(f"Port directions for {target} ({inferred_label})")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "port_geometry_compare_C.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved 3D port directions plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare legacy vs spectral port geometry under current ThermoConfig."
    )
    add_thermo_args(parser)
    args = parser.parse_args()
    apply_thermo_from_args(args)

    print("Effective ThermoConfig (geom compare):")
    print(asdict(get_current_thermo_config()))

    names = ["B", "C", "N", "O", "Si", "P", "S"]
    compare_for_elements(names)


if __name__ == "__main__":
    main()
