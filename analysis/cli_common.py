from __future__ import annotations

import argparse
from typing import Iterable, Optional


def add_common_output_args(parser: argparse.ArgumentParser) -> None:
    """
    Общие флаги для скриптов, у которых потенциально может быть нестандартный
    вывод (результаты/картинки). Пока просто задел под будущее.
    """
    # сейчас ничего не добавляем, оставляем как расширяемую точку
    # parser.add_argument("--results-dir", ...) — можно будет включить позже
    return


def parse_argv(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """
    Просто обёртка вокруг parser.parse_args(argv), оставлена для симметрии,
    если захочешь централизованно модифицировать поведение.
    """
    raise RuntimeError("cli_common.parse_argv: не вызывай напрямую")

