from __future__ import annotations

import argparse

from analysis.nuclear_cli import apply_nuclear_config_if_provided
from core.nuclear_island import set_magic_mode, MAGIC_Z, MAGIC_N


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Configure nuclear shell profile (--nuclear-config) "
            "and magic-number mode (--magic)."
        )
    )
    parser.add_argument(
        "--nuclear-config",
        type=str,
        default=None,
        help="Path to nuclear shell config (YAML/JSON).",
    )
    parser.add_argument(
        "--magic",
        type=str,
        choices=["legacy", "ws"],
        default="legacy",
        help="Magic-number mode: legacy neutron magic vs WS spectral magic.",
    )
    parser.add_argument(
        "--print-magic",
        action="store_true",
        help="Print current Z/N magic numbers after applying mode.",
    )

    args = parser.parse_args(argv)

    apply_nuclear_config_if_provided(args.nuclear_config)
    set_magic_mode(args.magic)

    if args.print_magic:
        print(f"Proton magic numbers Z: {list(MAGIC_Z)}")
        print(f"Neutron magic numbers N: {list(MAGIC_N)}")


if __name__ == "__main__":
    main()

