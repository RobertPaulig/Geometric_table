from __future__ import annotations


def test_eq_runner_start_topology_alias() -> None:
    from analysis.chem.chem_validation_eq_runner import _parse_args

    cfg_a = _parse_args(["--start_topology", "n_hexane"])
    cfg_b = _parse_args(["--start_topologies", "n_hexane"])

    assert cfg_a.start_topologies == cfg_b.start_topologies == ("n_hexane",)

