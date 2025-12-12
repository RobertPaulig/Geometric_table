from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


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


def set_magic_numbers(magic: MagicSet) -> None:
    """
    Установить произвольный набор magic-чисел (R&D-режим).
    """
    global _current_magic
    _current_magic = magic


def load_magic_from_yaml(path: str | Path) -> MagicSet:
    """
    Загрузить MagicSet из YAML-файла.

    Ожидается структура вида:

        Z: [2, 8, 20, ...]
        N: [2, 8, 20, ...]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Magic YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data: Any = yaml.safe_load(f)

    if not isinstance(data, Mapping):
        raise ValueError(f"Magic YAML must be a mapping, got {type(data)!r}")

    try:
        z_seq = tuple(int(z) for z in data["Z"])
        n_seq = tuple(int(n) for n in data["N"])
    except KeyError as e:
        raise ValueError(f"Magic YAML missing key: {e!r}") from e

    return MagicSet(Z=z_seq, N=n_seq)
