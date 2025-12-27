from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(f"[CI-SMOKE] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    theta_json = repo_root / "analysis" / "chem" / "configs" / "hetero_theta_best.json"
    if not theta_json.exists():
        raise FileNotFoundError(f"hetero_theta_best.json not found at {theta_json}")
    _run([sys.executable, "-m", "pytest", "-q"])
    _run(
        [
            sys.executable,
            "-m",
            "analysis.chem.hetero_validation_suite",
            "--theta_json",
            str(theta_json),
            "--fp_exclude_energy_like",
            "--formulas",
            "C4H10O",
            "C4H11N",
            "--out_dir",
            str(repo_root / "results" / "hetero_suite_ci_smoke"),
            "--stub_prefix",
            "ci_smoke",
        ]
    )
    print("[CI-SMOKE] done")


if __name__ == "__main__":
    main()
