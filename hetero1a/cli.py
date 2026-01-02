from analysis.chem.audit import main as audit_main
from analysis.chem.decoys import main as decoys_main
from analysis.chem.pipeline import main as pipeline_main
from analysis.chem.report import main as report_main


def hetero_audit() -> int:
    return audit_main()


def hetero_decoys() -> int:
    return decoys_main()


def hetero_pipeline() -> int:
    return pipeline_main()


def hetero_report() -> int:
    return report_main()
