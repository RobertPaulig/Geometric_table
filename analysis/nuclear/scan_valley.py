from __future__ import annotations

import argparse

from analysis.nuclear_cli import apply_nuclear_config_if_provided
from core.nuclear_bands import find_best_N_for_Z


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nuclear-config",
        type=str,
        default=None,
        help="Path to nuclear config (YAML/JSON); baseline used if omitted.",
    )
    parser.add_argument(
        "--z-min",
        type=int,
        default=8,
        help="Minimal Z to scan valley for.",
    )
    parser.add_argument(
        "--z-max",
        type=int,
        default=82,
        help="Maximum Z to scan valley for.",
    )
    args = parser.parse_args(argv)

    apply_nuclear_config_if_provided(args.nuclear_config)

    # лёгкие и средние ядра: от O до Pb
    for Z in range(args.z_min, args.z_max + 1):
        N_min = Z
        N_max = int(Z * 1.7)
        N_best, F_best = find_best_N_for_Z(Z, N_min, N_max)
        A = Z + N_best
        ratio = N_best / Z
        print(
            f"Z={Z:3d}: N_best={N_best:3d}, "
            f"N/Z={ratio:5.2f}, A={A:4d}, F={F_best:8.2f}"
        )


if __name__ == "__main__":
    main()
