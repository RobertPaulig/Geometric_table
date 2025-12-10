from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List


DEFAULT_OUTPUT_NAME = "project_snapshot.txt"


EXCLUDED_DIRS = {
    ".git",
    ".vscode",
    "__pycache__",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
}

# Extensions that are typically binary or not useful for code/architecture review
EXCLUDED_EXTENSIONS = {
    ".txt",  # user requested to skip source txt files
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".svg",
    ".ico",
    ".pyc",
    ".pyo",
    ".pyd",
    ".dll",
    ".so",
    ".dylib",
    ".exe",
}


def should_exclude_file(path: Path, output_file: Path) -> bool:
    """Return True if this file should be skipped."""
    name = path.name

    # Skip the output file itself to avoid self-inclusion
    if path.resolve() == output_file.resolve():
        return True

    # Skip anything starting with "name3" in any directory
    if name.startswith("name3"):
        return True

    # Skip by extension
    if path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True

    return False


def iter_project_files(root: Path, output_file: Path) -> Iterable[Path]:
    """Yield project files respecting exclude rules."""
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            # Directory-level exclusions are handled via the traversal in build_tree;
            # here we only yield files.
            continue
        if should_exclude_file(path, output_file):
            continue
        # Also skip files in excluded directories
        if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def build_tree(root: Path) -> str:
    """Build a simple text tree of the project with exclusions."""
    lines: List[str] = []
    root_name = root.name
    lines.append(f"{root_name}/")

    def walk(dir_path: Path, prefix: str = "") -> None:
        entries = []
        for entry in dir_path.iterdir():
            # Exclude directories
            if entry.is_dir() and entry.name in EXCLUDED_DIRS:
                continue
            # Exclude name3* files at tree level as well
            if entry.is_file() and entry.name.startswith("name3"):
                continue
            # Exclude typical binary/non-code files at tree level
            if entry.is_file() and entry.suffix.lower() in EXCLUDED_EXTENSIONS:
                continue
            entries.append(entry)

        entries.sort(key=lambda p: (p.is_file(), p.name.lower()))

        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            line = f"{prefix}{connector}{entry.name}"
            lines.append(line)
            if entry.is_dir():
                extension_prefix = "    " if i == len(entries) - 1 else "│   "
                walk(entry, prefix + extension_prefix)

    walk(root)
    return "\n".join(lines)


def create_snapshot(root: Path, output_file: Path) -> None:
    tree = build_tree(root)

    sections: List[str] = []
    sections.append("# PROJECT TREE\n")
    sections.append(tree)
    sections.append("\n\n# FILE CONTENTS\n")

    for file_path in iter_project_files(root, output_file):
        rel = file_path.relative_to(root)
        sections.append(f"\n\n===== FILE: {rel.as_posix()} =====\n")
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Skip non-text files silently
            continue
        sections.append(text)

    output_file.write_text("".join(sections), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect project structure and source files into a single text snapshot "
            "for architectural review."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output txt file name (default: {DEFAULT_OUTPUT_NAME})",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root to scan (default: current directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_file = Path(args.output).resolve()
    create_snapshot(root, output_file)
    print(f"Snapshot written to: {output_file}")


if __name__ == "__main__":
    main()

