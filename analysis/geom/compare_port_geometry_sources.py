from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.geom_atoms import AtomGraph, PERIODIC_TABLE
from core.port_geometry_spectral import canonical_port_vectors, ws_sp_gap, hybrid_strength
from core.thermo_config import ThermoConfig, override_thermo_config
from core.spectral_density_ws import WSRadialParams


def compare_for_elements(names):
    cfg_legacy = ThermoConfig(coupling_port_geometry=0.0, port_geometry_source="legacy")
    cfg_spectral = ThermoConfig(
        coupling_port_geometry=1.0,
        port_geometry_source="ws_sp_gap",
    )

    rows = []
    matches = 0
    for name in names:
        atom = PERIODIC_TABLE.get(name)
        if atom is None:
            continue
        with override_thermo_config(cfg_legacy):
            base_label = atom.port_geometry
        with override_thermo_config(cfg_spectral):
            inferred_label = atom.effective_port_geometry(cfg_spectral)
            vecs = atom.port_vectors(cfg_spectral)
        if inferred_label == base_label:
            matches += 1

        # спектральные признаки для отладки
        params = WSRadialParams()
        gap = ws_sp_gap(atom.Z, params)
        h = hybrid_strength(gap, cfg_spectral.ws_geom_gap_scale)

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
    names = ["B", "C", "N", "O", "Si", "P", "S"]
    compare_for_elements(names)


if __name__ == "__main__":
    main()
