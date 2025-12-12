from __future__ import annotations

from pathlib import Path
from typing import Iterable

from analysis.io_utils import results_path, write_text_result


def write_growth_txt(
    name: str,
    lines: Iterable[str],
    header: str | None = None,
) -> Path:
    """
    Унифицированная запись текстовых отчётов growth-скриптов.

    name      — логическое имя отчёта, без пути (например, "cycle_stats").
    lines     — список строк без перевода строки.
    header    — опциональный хедер (будет добавлен в начале файла).
    """
    out_path = results_path(f"{name}.txt")
    chunks: list[str] = []
    if header:
        chunks.append(header.rstrip("\n"))
        chunks.append("")  # пустая строка после заголовка
    chunks.extend(lines)
    body = "\n".join(chunks) + "\n"
    write_text_result(body, out_path)
    print(f"[GROWTH-REPORT] wrote {out_path}")
    return out_path
