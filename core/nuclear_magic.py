from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MagicSet:
    Z: tuple[int, ...]
    N: tuple[int, ...]


LEGACY_MAGIC = MagicSet(
    Z=(2, 8, 20, 28, 50, 82),
    N=(2, 8, 20, 28, 50, 82, 126),
)

WS_MAGIC = MagicSet(
    Z=(2, 8, 20, 28, 50, 82),
    N=(2, 8, 14, 20, 28, 50, 82, 126),
)


_current_magic: MagicSet = LEGACY_MAGIC


def set_magic_mode(mode: str) -> None:
    global _current_magic
    mode = mode.lower()
    if mode == "legacy":
        _current_magic = LEGACY_MAGIC
    elif mode == "ws":
        _current_magic = WS_MAGIC
    else:
        raise ValueError(f"Unknown magic mode: {mode}")


def get_magic_numbers() -> MagicSet:
    return _current_magic

