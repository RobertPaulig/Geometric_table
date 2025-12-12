from __future__ import annotations

from core.thermo_config import ThermoConfig, load_thermo_config


def test_thermo_defaults(tmp_path) -> None:
    p = tmp_path / "thermo.yaml"
    p.write_text("{}", encoding="utf-8")
    cfg = load_thermo_config(str(p))
    assert cfg == ThermoConfig()


def test_thermo_load_embedded_section(tmp_path) -> None:
    p = tmp_path / "experiment.yaml"
    p.write_text(
        """
growth:
  max_depth: 4
thermo:
  temperature: 2.0
  coupling_delta_F: 1.0
  experiment_name: "hot_run"
""",
        encoding="utf-8",
    )
    cfg = load_thermo_config(str(p))
    assert cfg.temperature == 2.0
    assert cfg.coupling_delta_F == 1.0
    assert cfg.experiment_name == "hot_run"

