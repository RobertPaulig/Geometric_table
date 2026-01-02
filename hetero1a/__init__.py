from importlib import metadata

from .api import render_report, run_audit, run_decoys, run_pipeline

try:
    __version__ = metadata.version("hetero1a")
except Exception:
    __version__ = "0.1.1"

__all__ = ["run_audit", "run_decoys", "run_pipeline", "render_report"]
