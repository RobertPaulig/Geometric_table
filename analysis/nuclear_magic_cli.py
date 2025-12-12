from __future__ import annotations

import argparse

from analysis.nuclear_cli import apply_nuclear_config_if_provided
from core.nuclear_magic import (
    get_magic_numbers,
    load_magic_from_yaml,
    set_magic_mode,
    set_magic_numbers,
)


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
        choices=["legacy", "ws", "custom"],
        default="legacy",
        help="Magic-number mode: legacy, ws, или custom (из YAML-конфига).",
    )
    parser.add_argument(
        "--magic-config",
        type=str,
        default=None,
        help="YAML с полями Z/N для режима --magic=custom.",
    )
    parser.add_argument(
        "--print-magic",
        action="store_true",
        help="Print current Z/N magic numbers after applying mode.",
    )

    args = parser.parse_args(argv)

    apply_nuclear_config_if_provided(args.nuclear_config)

    if args.magic == "custom":
        if not args.magic_config:
            raise SystemExit("--magic=custom требует --magic-config=path/to/magic.yaml")
        magic = load_magic_from_yaml(args.magic_config)
        set_magic_numbers(magic)
    else:
        set_magic_mode(args.magic)
        magic = get_magic_numbers()

    if args.print_magic:
        print(f"Proton magic numbers Z: {list(magic.Z)}")
        print(f"Neutron magic numbers N: {list(magic.N)}")


if __name__ == "__main__":
    main()
