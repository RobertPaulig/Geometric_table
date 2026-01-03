from analysis.experimental.ray_audit import RayAuditor


def test_ray_auditor_calibration_sum() -> None:
    auditor = RayAuditor(size=120, num_rays=120)
    assert auditor.check_calibration() == 8299
