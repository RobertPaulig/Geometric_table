import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_dockerfile_exists() -> None:
    assert Path("Dockerfile").exists()


RUN = os.environ.get("HETERO2_DOCKER_SMOKE", "") == "1"


@pytest.mark.skipif(
    (not RUN) or (shutil.which("docker") is None),
    reason="docker smoke validated by CI job docker-smoke; set HETERO2_DOCKER_SMOKE=1 to run locally",
)
def test_docker_smoke_local(tmp_path: Path) -> None:
    # Light local smoke: build and run hetero2 demo if docker available
    image = "hetero2:test-smoke"
    subprocess.run(["docker", "build", "-t", image, "."], check=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    subprocess.run(
        ["docker", "run", "--rm", "-v", f"{out_dir}:/out", image, "hetero2-demo-aspirin", "--out_dir", "/out/aspirin"],
        check=True,
    )
    assert (out_dir / "aspirin" / "aspirin_report.md").exists()
    assert (out_dir / "aspirin" / "aspirin_assets").exists()
