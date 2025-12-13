from __future__ import annotations

import math
from pathlib import Path
from dataclasses import asdict
import argparse

import matplotlib.pyplot as plt
import numpy as np

from core.density_models import beta_effective
from core.geom_atoms import toy_ldos_radial, cube_to_ball
from core.spectral_density_ws import (
    WSRadialParams,
    make_ws_rho3d_interpolator,
    make_ws_rho3d_with_diagnostics,
)
from core.thermo_config import (
    ThermoConfig,
    override_thermo_config,
    set_current_thermo_config,
    get_current_thermo_config,
)
from core.geom_atoms import estimate_atom_energy_fdm
from core.fdm import FDMIntegrator, make_tensor_grid_ifs
from analysis.thermo_cli import add_thermo_args, apply_thermo_from_args


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
    # Используем текущий ThermoConfig из глобального состояния
    thermo = get_current_thermo_config()

    rho_gauss_fn, beta = _make_gaussian_density_fn(Z, thermo)

    params = WSRadialParams()
    rho_ws_fn, diag = make_ws_rho3d_with_diagnostics(Z, params)

    # 1D radial profile
    r_max_plot = min(params.R_max, 10.0)
    radii = np.linspace(0.0, r_max_plot, 256)
    # embed radii into 3D to call toy_ldos_radial
    r3 = np.zeros((radii.size, 3), dtype=float)
    r3[:, 0] = radii
    rho_gauss = rho_gauss_fn(r3)

    I_target = (math.pi / beta) ** 1.5 if beta > 0.0 else 0.0
    rho_ws = rho_ws_fn(radii) * I_target

    # Диагностика box-aware нормировки (без изменения основной логики энергий)
    def I_box(R: float, beta_val: float) -> float:
        if beta_val <= 0.0:
            return 0.0
        sqrt_term = math.sqrt(math.pi / float(beta_val))
        erf_term = math.erf(math.sqrt(float(beta_val)) * float(R))
        Ix = sqrt_term * erf_term
        return Ix ** 3

    dim = 3
    ifs = make_tensor_grid_ifs(dim=dim, base=2)
    fdm_mass = FDMIntegrator(ifs)

    R_default = 4.0
    R_eff = max(R_default, 1.2 * float(diag.r_99))

    def integrand_ws_mass(r: np.ndarray) -> np.ndarray:
        radii_mass = np.sqrt(np.sum(r * r, axis=1))
        return rho_ws_fn(radii_mass)

    mean_ws_raw = fdm_mass.integrate(
        integrand_ws_mass,
        depth=4,
        dim=dim,
        transform=lambda u: cube_to_ball(u, R_eff),
    )
    volume_eff = (2.0 * R_eff) ** 3
    M_ws_box_raw = mean_ws_raw * volume_eff

    I_box_val = I_box(R_eff, beta)
    if I_box_val > 0.0:
        scale_ws = I_box_val / max(M_ws_box_raw, 1e-30)
        M_ws_box_scaled = M_ws_box_raw * scale_ws
        mass_ratio_box = M_ws_box_scaled / I_box_val
    else:
        scale_ws = 0.0
        M_ws_box_scaled = 0.0
        mass_ratio_box = float("nan")

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

    ratio = e_ws / e_gauss if e_gauss != 0 else float("nan")

    print(
        f"Z={Z}: beta={beta:.4g}, "
        f"R_eff={R_eff:.3f}, r_mean={diag.r_mean:.3f}, r_rms={diag.r_rms:.3f}, r_99={diag.r_99:.3f}, "
        f"I_target={I_target:.4g}, I_box={I_box_val:.4g}, "
        f"M_ws_box_raw={M_ws_box_raw:.4g}, scale_ws={scale_ws:.4g}, mass_ratio_box={mass_ratio_box:.4g}, "
        f"E_fdm_gauss={e_gauss:.4g}, E_fdm_ws={e_ws:.4g}, ratio_E={ratio:.4g}"
    )

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
    parser = argparse.ArgumentParser(
        description="Compare Gaussian vs WS spectral density for selected Z."
    )
    add_thermo_args(parser)
    args = parser.parse_args()
    apply_thermo_from_args(args)

    print("Effective ThermoConfig (spectral_density):")
    print(asdict(get_current_thermo_config()))

    Z_list = [1, 6, 14, 26]
    for Z in Z_list:
        compare_for_Z(Z)


if __name__ == "__main__":
    main()
