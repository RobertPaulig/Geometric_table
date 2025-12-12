from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# Базовые директории проекта
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


class MissingDataError(RuntimeError):
    """Поднимается, если ожидаемый CSV в data/ не найден."""


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR


def data_path(*parts: str) -> Path:
    """Унифицированный путь к data/…"""
    return DATA_DIR.joinpath(*parts)


def results_path(*parts: str) -> Path:
    """Унифицированный путь к results/… (создаёт директорию при необходимости)."""
    return ensure_results_dir().joinpath(*parts)


def read_data_csv(
    name_or_path: str | Path,
    *,
    required: bool = True,
    expected_columns: Optional[Iterable[str]] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Прочитать CSV из data/.

    name_or_path:
        - 'foo.csv' → берётся data/foo.csv
        - 'subdir/foo.csv' → data/subdir/foo.csv
        - произвольный Path → используется как есть.
    """
    path = Path(name_or_path)
    if not path.is_absolute():
        path = data_path(str(path))

    if not path.exists():
        msg = f"[ANALYSIS-IO] Expected data CSV not found: {path}"
        if required:
            raise MissingDataError(msg)
        print(msg)
        return pd.DataFrame()

    df = pd.read_csv(path, **read_csv_kwargs)

    if expected_columns is not None:
        missing = [c for c in expected_columns if c not in df.columns]
        if missing:
            raise MissingDataError(
                f"[ANALYSIS-IO] {path} is missing columns: {missing} "
                f"(has: {list(df.columns)})"
            )

    return df


def write_result_csv(df: pd.DataFrame, name_or_path: str, **to_csv_kwargs) -> Path:
    """
    Записать CSV в results/… с логом пути.
    """
    path = Path(name_or_path)
    if not path.is_absolute():
        path = results_path(str(path))

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **to_csv_kwargs)
    print(f"[ANALYSIS-IO] Saved CSV: {path}")
    return path


def write_text_result(text: str, name_or_path: str) -> Path:
    """
    Записать текстовый отчёт в results/… с логом пути.
    """
    path = Path(name_or_path)
    if not path.is_absolute():
        path = results_path(str(path))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
    print(f"[ANALYSIS-IO] Saved text: {path}")
    return path

