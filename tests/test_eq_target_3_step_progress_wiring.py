import analysis.chem.eq_target_3_scan as scan


def test_eq_target_3_step_progress_wiring(monkeypatch):
    seen = []

    def fake_run_fixed_n_tree_mcmc(**kwargs):
        seen.append(kwargs)
        assert kwargs["step_heartbeat_every"] == 2
        assert kwargs["step_heartbeat"] is not None
        # Return minimal shape-compatible outputs.
        samples = [{"topology": "tree:0-1", "energy": 0.0}] * 2

        class Summary:
            energy_cache_hits = 0
            energy_cache_misses = 0
            steps = kwargs["steps"]

        return samples, Summary()

    monkeypatch.setattr(scan, "run_fixed_n_tree_mcmc", fake_run_fixed_n_tree_mcmc)
    scan.main(
        [
            "--N",
            "15",
            "--steps_grid",
            "10",
            "--chains",
            "1",
            "--thin",
            "1",
            "--start_specs",
            "path",
            "--progress",
            "--no-step_progress",
            "--step_heartbeat_every",
            "2",
        ]
    )
    assert len(seen) == 1

