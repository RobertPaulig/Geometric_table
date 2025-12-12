from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.density_models import beta_effective
from core.geom_atoms import toy_ldos_radial
from core.spectral_density_ws import WSRadialParams, make_ws_rho3d_interpolator
from core.thermo_config import ThermoConfig, override_thermo_config, set_current_thermo_config
from core.geom_atoms import estimate_atom_energy_fdm
from core.fdm import FDMIntegrator, make_tensor_grid_ifs


def _make_gaussian_density_fn(Z: int, thermo: ThermoConfig):
    beta = beta_effective(
        Z,
        thermo.coupling_density,
        model=thermo.density_model,
        blend=thermo.density_blend,
        Z_ref=thermo.density_Z_ref,
    )

    def rho_gauss(r: np.ndarray) -> np.ndarray:
        return toy_ldos_radial(r, beta)

    return rho_gauss, beta


def compare_for_Z(Z: int) -> None:
    thermo = ThermoConfig(coupling_density=1.0)
    set_current_thermo_config(thermo)

    rho_gauss_fn, beta = _make_gaussian_density_fn(Z, thermo)

    params = WSRadialParams()
    rho_ws_fn = make_ws_rho3d_interpolator(Z, params)

    # 1D radial profile
    r_max_plot = min(params.R_max, 10.0)
    radii = np.linspace(0.0, r_max_plot, 256)
    # embed radii into 3D to call toy_ldos_radial
    r3 = np.zeros((radii.size, 3), dtype=float)
    r3[:, 0] = radii
    rho_gauss = rho_gauss_fn(r3)

    I_target = (math.pi / beta) ** 1.5 if beta > 0.0 else 0.0
    rho_ws = rho_ws_fn(radii) * I_target

    # FDM energies
    thermo_gauss = ThermoConfig(coupling_density=1.0, density_source="gaussian")
    thermo_ws = ThermoConfig(
        coupling_density=1.0,
        coupling_density_shape=1.0,
        density_source="ws_radial",
    )

    with override_thermo_config(thermo_gauss):
        e_gauss = estimate_atom_energy_fdm(Z, 1.0)
    with override_thermo_config(thermo_ws):
        e_ws = estimate_atom_energy_fdm(Z, 1.0)

    print(f"Z={Z}: beta={beta:.4g}, E_fdm_gauss={e_gauss:.4g}, E_fdm_ws={e_ws:.4g}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(radii, rho_gauss, label="Gaussian proxy", lw=2)
    ax.plot(radii, rho_ws, label="WS radial (scaled)", lw=2)
    ax.set_xlabel("r")
    ax.set_ylabel("rho(r)")
    ax.set_title(f"Density comparison for Z={Z}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"spectral_density_compare_Z{Z}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main() -> None:
    Z_list = [1, 6, 14, 26]
    for Z in Z_list:
        compare_for_Z(Z)


if __name__ == "__main__":
    main()
